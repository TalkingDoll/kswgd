# This is a Gaussian example as a proof of concept.
import numpy as np
from scipy.linalg import svd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
from grad_ker1 import grad_ker1
from K_tar_eval import K_tar_eval

# Set random seed for reproducibility (optional)
np.random.seed(0)

# Sample 500 points from a 3D Gaussian (as in the MATLAB code)
n = 500
d = 2
lambda_ = 1
u = np.random.normal(0, 1, (n, d))
u[:, 0] = lambda_ * u[:, 0]
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(n, 1)) * 1/100 + 99/100
u_trans = u / u_norm
X_tar = r * u_trans
n = X_tar.shape[0]
 
# Scheme 1: Euclidean KDE-score (Euler–Maruyama) to generate X_tar_next for EDMD
# - Estimate score ∇log q(x) using Gaussian KDE with global bandwidth (median distance)
# - One Euler–Maruyama step: x_next = x + Δt * score(x) + sqrt(2Δt) * ξ
dt_edmd = 1e-3
# Pairwise squared distances for bandwidth and weights
diffs_edmd = X_tar[:, None, :] - X_tar[None, :, :]
dist2_edmd = np.sum(diffs_edmd ** 2, axis=2)
h_edmd = np.sqrt(np.median(dist2_edmd) + 1e-12)
W_edmd = np.exp(-dist2_edmd / (2.0 * (h_edmd ** 2)))
sumW_edmd = np.sum(W_edmd, axis=1, keepdims=True) + 1e-12
weighted_means_edmd = (W_edmd @ X_tar) / sumW_edmd
score_eucl = (weighted_means_edmd - X_tar) / (h_edmd ** 2)
xi_edmd = np.random.normal(0.0, 1.0, size=X_tar.shape)
X_tar_next = X_tar + dt_edmd * score_eucl + np.sqrt(2.0 * dt_edmd) * xi_edmd

# # Scheme 2: Manifold (sphere) via projected Euclidean score — commented out for toggling
# # (Start from Euclidean KDE-score, then project drift and noise to the tangent space, and renormalize)
# dt_edmd = 1e-2
# diffs_edmd = X_tar[:, None, :] - X_tar[None, :, :]
# dist2_edmd = np.sum(diffs_edmd ** 2, axis=2)
# h_edmd = np.sqrt(np.median(dist2_edmd) + 1e-12)
# W_edmd = np.exp(-dist2_edmd / (2.0 * (h_edmd ** 2)))
# sumW_edmd = np.sum(W_edmd, axis=1, keepdims=True) + 1e-12
# weighted_means_edmd = (W_edmd @ X_tar) / sumW_edmd
# score_eucl = (weighted_means_edmd - X_tar) / (h_edmd ** 2)
# # Project drift and noise to the tangent space of the unit sphere
# X_norm = X_tar / (np.linalg.norm(X_tar, axis=1, keepdims=True) + 1e-12)
# proj = np.eye(X_tar.shape[1])[None, :, :] - X_norm[:, :, None] * X_norm[:, None, :]
# score_tan = np.einsum('nij,ni->nj', proj, score_eucl)
# xi_edmd = np.random.normal(0.0, 1.0, size=X_tar.shape)
# xi_tan = xi_edmd - (np.sum(X_norm * xi_edmd, axis=1, keepdims=True)) * X_norm
# X_step = X_norm + dt_edmd * score_tan + np.sqrt(2.0 * dt_edmd) * xi_tan
# X_tar_next = X_step / (np.linalg.norm(X_step, axis=1, keepdims=True) + 1e-12)

# Quick visualization: X_tar vs X_tar_next (Scheme 1)
if d == 2:
    plt.figure()
    plt.scatter(X_tar[:, 0], X_tar[:, 1], s=10, c='C0', label='X_tar')
    plt.scatter(X_tar_next[:, 0], X_tar_next[:, 1], s=10, c='C1', label='X_tar_next')
    plt.legend()
    plt.axis('equal')
    plt.title('X_tar vs X_tar_next (Scheme 1)')
    plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='X_tar', c='C0', s=10)
    ax.scatter(X_tar_next[:, 0], X_tar_next[:, 1], X_tar_next[:, 2], label='X_tar_next', c='C1', s=10)
    ax.legend()
    ax.set_title('X_tar vs X_tar_next (Scheme 1)')
    plt.show()

"""
Kernel EDMD (KDMD) setup
- Define several common kernels; keep one active (others commented).
- Build Gram matrices K_xx = k(X_tar, X_tar), K_xy = k(X_tar, X_tar_next).
- Compute the sample-space Koopman matrix K_kernel_edmd = (K_xx + gamma I)^{-1} K_xy (n×n).
- Optionally symmetrize to enforce a reversible/self-adjoint approximation for a real spectrum (recommended for Green operator construction).
Remarks:
- The EDMD dictionary and feature-space generalized eigenproblem are removed; everything is in sample space via kernels.
"""

def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Returns ||X_i - Y_j||^2 matrix
    x2 = np.sum(X**2, axis=1, keepdims=True)
    y2 = np.sum(Y**2, axis=1, keepdims=True).T
    return x2 + y2 - 2.0 * (X @ Y.T)

# --- Kernels (choose ONE activation section below) ---

def kernel_rbf(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian RBF: k(x,y) = exp(-||x-y||^2 / (2 sigma^2))."""
    D2 = pairwise_sq_dists(X, Y)
    s2 = max(sigma**2, 1e-12)
    return np.exp(-D2 / (2.0 * s2))

def kernel_laplacian(X: np.ndarray, Y: np.ndarray, ell: float) -> np.ndarray:
    """Laplacian: k(x,y) = exp(-||x-y||_2 / ell)."""
    D2 = pairwise_sq_dists(X, Y)
    D = np.sqrt(np.maximum(D2, 0.0))
    l = max(ell, 1e-12)
    return np.exp(-D / l)

def kernel_polynomial(X: np.ndarray, Y: np.ndarray, degree: int = 3, c: float = 1.0) -> np.ndarray:
    """Polynomial: k(x,y) = (x·y + c)^degree (degree >= 1)."""
    return np.power(np.maximum(X @ Y.T + c, 0.0), int(max(1, degree)))

def kernel_matern32(X: np.ndarray, Y: np.ndarray, ell: float) -> np.ndarray:
    """Matérn ν=3/2: k(r) = (1 + sqrt(3)r/ell) exp(-sqrt(3)r/ell)."""
    D2 = pairwise_sq_dists(X, Y)
    r = np.sqrt(np.maximum(D2, 0.0))
    l = max(ell, 1e-12)
    z = np.sqrt(3.0) * r / l
    return (1.0 + z) * np.exp(-z)

# --- Activation: choose ONE kernel for KDMD (others commented) ---

# RBF (active): bandwidth chosen from median pairwise distance
med_d2 = float(np.median(pairwise_sq_dists(X_tar, X_tar)))
sigma_kedmd = np.sqrt(max(med_d2, 1e-12))
K_xx = kernel_rbf(X_tar, X_tar, sigma_kedmd)
K_xy = kernel_rbf(X_tar, X_tar_next, sigma_kedmd)

# # Laplacian (example)
# med_d2 = float(np.median(pairwise_sq_dists(X_tar, X_tar)))
# ell_kedmd = np.sqrt(max(med_d2, 1e-12))
# K_xx = kernel_laplacian(X_tar, X_tar, ell_kedmd)
# K_xy = kernel_laplacian(X_tar, X_tar_next, ell_kedmd)

# # Polynomial (example)
# K_xx = kernel_polynomial(X_tar, X_tar, degree=3, c=1.0)
# K_xy = kernel_polynomial(X_tar, X_tar_next, degree=3, c=1.0)

# # Matérn ν=3/2 (example)
# med_d2 = float(np.median(pairwise_sq_dists(X_tar, X_tar)))
# ell_kedmd = np.sqrt(max(med_d2, 1e-12))
# K_xx = kernel_matern32(X_tar, X_tar, ell_kedmd)
# K_xy = kernel_matern32(X_tar, X_tar_next, ell_kedmd)

# KDMD Koopman matrix in sample space (n×n)
# Use ridge on K_xx and build K_kernel_edmd without further normalization;
# we'll treat it as the DMPS data_kernel downstream.
gamma_ridge = 1e-6
K_xx_reg = K_xx + gamma_ridge * np.eye(K_xx.shape[0])
K_kernel_edmd = np.linalg.solve(K_xx_reg, K_xy)
print("K_kernel_edmd shape:", K_kernel_edmd.shape)

# Safety: ensure finite and nonnegative values before DMPS-style normalization
K_kernel_edmd = np.nan_to_num(K_kernel_edmd, nan=0.0, posinf=0.0, neginf=0.0)
minK = float(np.min(K_kernel_edmd))
if minK < 0.0:
    # Shift so the minimum is near zero; keep a tiny floor to avoid exact zeros
    K_kernel_edmd = K_kernel_edmd - minK + 1e-12

# Form the anisotropic graph Laplacian
sq_tar = np.sum(X_tar ** 2, axis=1)
H = sq_tar[:, None] + sq_tar[None, :] - 2 * (X_tar @ X_tar.T)
# Bandwidth for data kernel (used by kernels/gradients); decoupled from KDMD/DMPS normalization
epsilon_kern = 0.5 * np.median(H) / (np.log(n + 1) + 1e-12)

# Treat KDMD operator as the raw data_kernel for DMPS-style normalization
data_kernel = K_kernel_edmd
p_x = np.sqrt(np.sum(data_kernel, axis=1) + 1e-12)
p_y = p_x.copy()
# Normalize kernel (for grad_ker1 and K_tar_eval downstream)
data_kernel_norm = data_kernel / p_x[:, None] / p_y[None, :]
D_y = np.sum(data_kernel_norm, axis=0) + 1e-12

# Random-walk symmetric normalization (as in DMPS)
rw_kernel = 0.5 * (data_kernel_norm / D_y + data_kernel_norm / D_y[:, None])
# Sanitize numerical issues
rw_kernel = np.nan_to_num(rw_kernel, nan=0.0, posinf=0.0, neginf=0.0)

# DMPS-style spectrum via SVD on rw_kernel
phi, s, _ = svd(rw_kernel)

# Green's operator weights: for Koopman eigenvalues s_k, use w_k = dt_edmd / (1 - s_k)
# Small Tikhonov on the gap; drop the constant (largest) mode
gap = 1.0 - s
reg_mu = 1e-6
inv_lambda = dt_edmd / (gap + reg_mu)
if inv_lambda.size > 0:
    inv_lambda[0] = 0.0

# Optional truncation (can set a smaller number like 200 if desired)
K_max = phi.shape[1]
phi_trunc = phi[:, :K_max]
w_trunc = inv_lambda[:K_max]

# Run algorithm
iter = 1000
h = 50
m = 700
u = np.random.normal(0, 1, (m, d))
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
u_trans = u / u_norm
x_init = r * u_trans
x_init = x_init[x_init[:, 1] > 0.7, :]
m = x_init.shape[0]
x_t = np.zeros((m, d, iter))
x_t[:, :, 0] = x_init

p_tar = np.sum(data_kernel, axis=0)
D = np.sum(data_kernel / np.sqrt(p_tar) / np.sqrt(p_tar)[:, None], axis=1)

sum_x = np.zeros((m, d))
for t in range(iter - 1):
    grad_matrix = grad_ker1(x_t[:, :, t], X_tar, p_tar, sq_tar, D, epsilon_kern)
    cross_matrix = K_tar_eval(X_tar, x_t[:, :, t], p_tar, sq_tar, D, epsilon_kern)
    for i in range(d):
        sum_x[:, i] = np.sum(grad_matrix[:, :, i] @ phi_trunc @ np.diag(w_trunc) @ phi_trunc.T @ cross_matrix, axis=1)
    x_t[:, :, t + 1] = x_t[:, :, t] - h / m * sum_x

# Plotting results
if d == 2:
    plt.figure()
    plt.plot(X_tar[:, 0], X_tar[:, 1], 'o', label='Target')
    plt.plot(x_t[:, 0, 0], x_t[:, 1, 0], 'o', label='Init')
    plt.plot(x_t[:, 0, -1], x_t[:, 1, -1], 'o', label='Final')
    plt.legend()
    plt.title('2D Results')
    plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='Target')
    ax.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax.legend()
    plt.title('3D Results')
    plt.show()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax2.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax2.legend()
    plt.title('3D Final State')
    plt.show()

# Plot matrix (scatter matrix)
pd.plotting.scatter_matrix(
    pd.DataFrame(X_tar),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle('Scatter Matrix of X_tar')
plt.show()

pd.plotting.scatter_matrix(
    pd.DataFrame(x_t[:, :, -1]),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle('Scatter Matrix of x_t (final)')
plt.show()
