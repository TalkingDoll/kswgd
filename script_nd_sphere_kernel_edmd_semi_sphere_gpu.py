# This is a Gaussian example as a proof of concept.
import numpy as np
from scipy.linalg import svd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
from grad_ker1 import grad_ker1
from K_tar_eval import K_tar_eval
import time
import sys
from typing import Optional

# ---------------------- GPU Config (optional) ----------------------
USE_GPU = False  # set True to attempt GPU acceleration via CuPy (falls back to CPU if unavailable)
try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    GPU_AVAILABLE = False
USE_GPU = bool(USE_GPU and GPU_AVAILABLE)

# ---------------------- Timing & Progress Utilities ----------------------
def _fmt_secs(s: float) -> str:
    s = max(0.0, float(s))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

def _print_phase(name: str, t_start: float) -> float:
    dt = time.time() - t_start
    print(f"[TIMER] {name}: {dt:.3f}s")
    return time.time()

def _print_progress(curr: int, total: int, elapsed: float, eta: float, prefix: str = "") -> None:
    total = max(1, int(total))
    curr = min(max(0, int(curr)), total)
    bar_len = 30
    filled = int(bar_len * curr / total)
    filled = min(filled, bar_len)
    bar = ("=" * filled) + (">" if filled < bar_len else "") + ("." * max(0, bar_len - filled - (0 if filled == bar_len else 1)))
    pct = 100.0 * curr / total
    msg = f"\r{prefix}[{bar}] {pct:5.1f}% | iter {curr}/{total} | elapsed {_fmt_secs(elapsed)} | eta {_fmt_secs(eta)}"
    print(msg, end="", flush=True)

RUN_START = time.time()
_t = time.time()
print(f"[DEVICE] GPU {'ENABLED' if USE_GPU else 'DISABLED'}")

def to_xp(arr: np.ndarray):
    if USE_GPU:
        return cp.asarray(arr)
    return arr

def to_np(arr):
    if USE_GPU:
        return cp.asnumpy(arr)
    return arr

# Set random seed for reproducibility (optional)
np.random.seed(0)

# Sample 500 points from a 3D Gaussian (as in the MATLAB code)
n = 500
d = 20
lambda_ = 1
u = np.random.normal(0, 1, (n, d))
u[:, 0] = lambda_ * u[:, 0]
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(n, 1)) * 1/100 + 99/100
u_trans = u / u_norm

# ---------------------- Hemisphere folding for target ----------------------
# Choose hemisphere axis (index). Here axis 1 means keep the half-space with positive y.
axis_hemi = 1
n_axis = np.zeros((d,))
n_axis[axis_hemi] = 1.0

def reflect_to_hemisphere(U: np.ndarray, n_vec: np.ndarray) -> np.ndarray:
    """Reflect points across the plane orthogonal to n_vec so dot(U, n_vec) >= 0.
    This preserves perpendicular components and folds full-sphere to hemisphere.
    """
    dots = U @ n_vec
    mask = dots < 0
    if not np.any(mask):
        return U
    U_ref = U.copy()
    U_ref[mask] = U_ref[mask] - 2.0 * (dots[mask][:, None]) * n_vec[None, :]
    return U_ref

u_trans_hemi = reflect_to_hemisphere(u_trans, n_axis)
X_tar = r * u_trans_hemi
n = X_tar.shape[0]
 
# Scheme 1: Euclidean KDE-score (Euler–Maruyama) to generate X_tar_next for EDMD
# - Estimate score ∇log q(x) using Gaussian KDE with global bandwidth (median distance)
# - One Euler–Maruyama step: x_next = x + Δt * score(x) + sqrt(2Δt) * ξ
dt_edmd = 1e-2
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
_t = _print_phase("EDMD Scheme 1 (KDE-score + one step)", _t)

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
    plt.title('X_tar vs X_tar_next (Scheme 1, hemisphere)')
    plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='X_tar', c='C0', s=10)
    ax.scatter(X_tar_next[:, 0], X_tar_next[:, 1], X_tar_next[:, 2], label='X_tar_next', c='C1', s=10)
    ax.legend()
    ax.set_title('X_tar vs X_tar_next (Scheme 1, hemisphere)')
    plt.show()
_t = _print_phase("Quick visualization", _t)

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

# # RBF (active): bandwidth chosen from median pairwise distance
# med_d2 = float(np.median(pairwise_sq_dists(X_tar, X_tar)))
# sigma_kedmd = np.sqrt(max(med_d2, 1e-12))
# K_xx = kernel_rbf(X_tar, X_tar, sigma_kedmd)
# K_xy = kernel_rbf(X_tar, X_tar_next, sigma_kedmd)

# # Laplacian (example)
# med_d2 = float(np.median(pairwise_sq_dists(X_tar, X_tar)))
# ell_kedmd = np.sqrt(max(med_d2, 1e-12))
# K_xx = kernel_laplacian(X_tar, X_tar, ell_kedmd)
# K_xy = kernel_laplacian(X_tar, X_tar_next, ell_kedmd)

# Polynomial (example)
K_xx = kernel_polynomial(X_tar, X_tar, degree=10, c=1.0)
K_xy = kernel_polynomial(X_tar, X_tar_next, degree=10, c=1.0)
_t = _print_phase("KDMD Gram matrices (polynomial)", _t)

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
_t = _print_phase("Solve (K_xx+gamma I)^{-1} K_xy", _t)

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
_t = _print_phase("DMPS-style normalization (density + symmetric RW)", _t)

# DMPS-style spectrum via SVD on rw_kernel
phi, s, _ = svd(rw_kernel)
_t = _print_phase("SVD on rw_kernel", _t)

# Green's operator weights: for Koopman eigenvalues s_k, use w_k = dt_edmd / (1 - s_k)
# Small Tikhonov on the gap; drop the constant (largest) mode
gap = 1.0 - s
reg_mu = 1e-6
inv_lambda = dt_edmd / (gap + reg_mu)
if inv_lambda.size > 0:
    inv_lambda[0] = 0.0
_t = _print_phase("Green weights setup (drop constant)", _t)

# DMPS-style thresholding: drop tiny modes (like MATLAB tol=1e-6)
tol = 1e-8
keep_mask = s >= tol
keep_idx = np.where(keep_mask)[0]
if keep_idx.size == 0:
    # Fallback to keep at least the first mode (it will be zero-weighted anyway)
    keep_idx = np.array([0], dtype=int)
phi_trunc = phi[:, keep_idx]
w_trunc = inv_lambda[keep_idx]
print(f"Kept spectral modes: {phi_trunc.shape[1]} of {phi.shape[1]} (tol={tol})")
_t = _print_phase("Spectral truncation (threshold)", _t)

# Move spectral components to GPU if enabled
if USE_GPU:
    phi_trunc_gpu = cp.asarray(phi_trunc)
    w_trunc_gpu = cp.asarray(w_trunc)

# Run algorithm
iter = 1000
h = 20
m = 700
u = np.random.normal(0, 1, (m, d))
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
u_trans = u / u_norm

# Fold initial directions to the same hemisphere
u_trans_hemi = reflect_to_hemisphere(u_trans, n_axis)
x_init = r * u_trans_hemi
x_init = x_init[x_init[:, 1] > 0.2, :]
m = x_init.shape[0]
x_t = np.zeros((m, d, iter))
x_t[:, :, 0] = x_init
_t = _print_phase("LAWGD init (particles & buffers)", _t)

p_tar = np.sum(data_kernel, axis=0)
D = np.sum(data_kernel / np.sqrt(p_tar) / np.sqrt(p_tar)[:, None], axis=1)

sum_x = np.zeros((m, d))

# ---------------------- Iteration Loop with Progress Bar ----------------------
loop_start = time.time()
total_loop = max(1, iter - 1)
last_print = 0.0
for t in range(iter - 1):
    grad_matrix = grad_ker1(x_t[:, :, t], X_tar, p_tar, sq_tar, D, epsilon_kern)
    cross_matrix = K_tar_eval(X_tar, x_t[:, :, t], p_tar, sq_tar, D, epsilon_kern)
    # Apply low-rank Green once: M = phi * diag(w) * (phi^T @ cross)
    if USE_GPU:
        cross_gpu = cp.asarray(cross_matrix)
        tmp = phi_trunc_gpu.T @ cross_gpu            # (k x n) @ (n x m) -> (k x m)
        tmp = (w_trunc_gpu[:, None]) * tmp           # (k x 1) .* (k x m)
        M_gpu = phi_trunc_gpu @ tmp                  # (n x k) @ (k x m) -> (n x m)
        M = cp.asnumpy(M_gpu)
    else:
        tmp = phi_trunc.T @ cross_matrix             # (k x n) @ (n x m)
        tmp = (w_trunc[:, None]) * tmp               # (k x m)
        M = phi_trunc @ tmp                          # (n x m)
    for i in range(d):
        # sum over target samples dimension using precomputed M
        sum_x[:, i] = np.sum(grad_matrix[:, :, i] @ M, axis=1)
    x_t[:, :, t + 1] = x_t[:, :, t] - h / m * sum_x
    # Progress bar update (throttled)
    done = t + 1
    elapsed = time.time() - RUN_START
    avg_iter_time = (time.time() - loop_start) / max(1, done)
    eta = avg_iter_time * (total_loop - done)
    # Print every ~1% or every 0.5s
    if done == total_loop or (done % max(1, total_loop // 100) == 0) or (time.time() - last_print > 0.5):
        _print_progress(done, total_loop, elapsed, eta, prefix="[LAWGD] ")
        last_print = time.time()

# Ensure the progress line ends cleanly
print()
print(f"[TIMER] RUN total (pre-plot): {time.time() - RUN_START:.3f}s")

# Plotting results
if d == 2:
    plt.figure()
    plt.plot(X_tar[:, 0], X_tar[:, 1], 'o', label='Target')
    plt.plot(x_t[:, 0, 0], x_t[:, 1, 0], 'o', label='Init')
    plt.plot(x_t[:, 0, -1], x_t[:, 1, -1], 'o', label='Final')
    plt.legend()
    plt.title('2D Results (hemisphere)')
    plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='Target')
    ax.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax.legend()
    plt.title('3D Results (hemisphere)')
    plt.show()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax2.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax2.legend()
    plt.title('3D Final State (hemisphere)')
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
