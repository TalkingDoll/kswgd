# This is a Gaussian example as a proof of concept.
import numpy as np
from scipy.linalg import svd
from scipy.linalg import eigh  # may be unused after GPU refactor
import matplotlib.pyplot as plt
import pandas as pd
import time
from typing import Optional

# GPU-aware gradient / kernel eval (xp-aware NumPy/CuPy implementations)
from grad_ker1_gpu import grad_ker1  # type: ignore
from K_tar_eval_gpu import K_tar_eval  # type: ignore

# ---------------- GPU Configuration ---------------- #
USE_GPU = True  # toggle; will auto-disable if CuPy unavailable
try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    GPU_AVAILABLE = False
USE_GPU = bool(USE_GPU and GPU_AVAILABLE)
xp = cp if USE_GPU else np

def to_xp(a):
    if USE_GPU:
        return cp.asarray(a)
    return np.asarray(a)

def to_np(a):
    if USE_GPU:
        return np.asarray(cp.asnumpy(a))  # type: ignore
    return a

if USE_GPU:
    try:
        dev = cp.cuda.Device()  # type: ignore
        props = cp.cuda.runtime.getDeviceProperties(dev.id)  # type: ignore
        name = props.get('name', b'')
        if isinstance(name, (bytes, bytearray)):
            name = name.decode(errors='ignore')
        print(f"[DEVICE] GPU ENABLED | id={dev.id} | {name} | CuPy {cp.__version__}")  # type: ignore
    except Exception:
        print("[DEVICE] GPU ENABLED (device info unavailable)")
else:
    print("[DEVICE] GPU DISABLED -> using NumPy CPU backend")

# ---------------- Timing & Progress Utilities ---------------- #
_t0_global = time.time()
_last_phase_time = _t0_global

def _fmt_secs(sec: float) -> str:
    if sec < 1e-6:
        return f"{sec*1e9:.1f}ns"
    if sec < 1e-3:
        return f"{sec*1e6:.1f}µs"
    if sec < 1:
        return f"{sec*1e3:.2f}ms"
    if sec < 60:
        return f"{sec:.2f}s"
    m = int(sec // 60)
    return f"{m}m{sec-60*m:.1f}s"

def _print_phase(name: str):
    global _last_phase_time
    now = time.time()
    phase = now - _last_phase_time
    total = now - _t0_global
    print(f"[TIMER] {name:<28} | phase {_fmt_secs(phase):>8} | total {_fmt_secs(total):>8}")
    _last_phase_time = now

def _print_progress(t: int, total: int, prefix: str = "Iter"):
    if total <= 0:
        return
    frac = t / total
    bar_len = 30
    filled = int(bar_len * frac)
    bar = '#' * filled + '-' * (bar_len - filled)
    print(f"\r[{prefix}] {t:>5}/{total:<5} |{bar}| {frac*100:5.1f}%", end='')

print("[INFO] KDMD full-sphere run start (GPU backend = {} )".format('CuPy' if USE_GPU else 'NumPy'))
_print_phase("start")

# Set random seed for reproducibility (optional)
np.random.seed(0)

# Sample 500 points from a d-dimensional Gaussian (as in the MATLAB code)
n = 500
d = 2
lambda_ = 1
u = np.random.normal(0, 1, (n, d)).astype(np.float32)
u[:, 0] = lambda_ * u[:, 0]
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = (np.sqrt(np.random.rand(n, 1)) * 1/100 + 99/100).astype(np.float32)
u_trans = u / u_norm
X_tar = (r * u_trans).astype(np.float32)
n = X_tar.shape[0]
if USE_GPU:
    X_tar = to_xp(X_tar)
 
# Scheme 1: Euclidean KDE-score (Euler–Maruyama) to generate X_tar_next for EDMD
# - Estimate score ∇log q(x) using Gaussian KDE with global bandwidth (median distance)
# - One Euler–Maruyama step: x_next = x + Δt * score(x) + sqrt(2Δt) * ξ
dt_edmd = 1e-1
# Pairwise squared distances for bandwidth and weights (xp backend)
diffs_edmd = X_tar[:, None, :] - X_tar[None, :, :]
dist2_edmd = xp.sum(diffs_edmd ** 2, axis=2)
h_edmd = float(xp.sqrt(xp.median(dist2_edmd) + 1e-12))
W_edmd = xp.exp(-dist2_edmd / (2.0 * (h_edmd ** 2)))
sumW_edmd = xp.sum(W_edmd, axis=1, keepdims=True) + 1e-12
weighted_means_edmd = (W_edmd @ X_tar) / sumW_edmd
score_eucl = (weighted_means_edmd - X_tar) / (h_edmd ** 2)
xi_edmd = to_xp(np.random.normal(0.0, 1.0, size=to_np(X_tar).shape).astype(np.float32))
X_tar_next = X_tar + dt_edmd * score_eucl + xp.sqrt(2.0 * dt_edmd) * xi_edmd
_print_phase("KDE score & EM step")

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
    X_tar_plot = to_np(X_tar)
    X_tar_next_plot = to_np(X_tar_next)
    plt.figure()
    plt.scatter(X_tar_plot[:, 0], X_tar_plot[:, 1], s=10, c='C0', label='X_tar')
    plt.scatter(X_tar_next_plot[:, 0], X_tar_next_plot[:, 1], s=10, c='C1', label='X_tar_next')
    plt.legend(); plt.axis('equal'); plt.title('X_tar vs X_tar_next (Scheme 1)')
    plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    X_tar_plot = to_np(X_tar)
    X_tar_next_plot = to_np(X_tar_next)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar_plot[:, 0], X_tar_plot[:, 1], X_tar_plot[:, 2], label='X_tar', c='C0', s=10)
    ax.scatter(X_tar_next_plot[:, 0], X_tar_next_plot[:, 1], X_tar_next_plot[:, 2], label='X_tar_next', c='C1', s=10)
    ax.legend(); ax.set_title('X_tar vs X_tar_next (Scheme 1)')
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

def pairwise_sq_dists(X, Y):
    x2 = xp.sum(X**2, axis=1, keepdims=True)
    y2 = xp.sum(Y**2, axis=1, keepdims=True).T
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

# """Activate chosen kernel (polynomial here) and build Gram matrices."""
# # Polynomial (example)
# K_xx = kernel_polynomial(X_tar, X_tar, degree=5, c=1.0)
# K_xy = kernel_polynomial(X_tar, X_tar_next, degree=5, c=1.0)
# _print_phase("KDMD Gram matrices (polynomial)")

# # Matérn ν=3/2 (example)
# med_d2 = float(np.median(pairwise_sq_dists(X_tar, X_tar)))
# ell_kedmd = np.sqrt(max(med_d2, 1e-12))
# K_xx = kernel_matern32(X_tar, X_tar, ell_kedmd)
# K_xy = kernel_matern32(X_tar, X_tar_next, ell_kedmd)

# KDMD Koopman matrix in sample space (n×n)
# Use ridge on K_xx and build K_kernel_edmd without further normalization;
# we'll treat it as the DMPS data_kernel downstream.
gamma_ridge = 1e-6
eye_xx = xp.eye(K_xx.shape[0], dtype=K_xx.dtype)
K_xx_reg = K_xx + gamma_ridge * eye_xx
K_kernel_edmd = xp.linalg.solve(K_xx_reg, K_xy)
print("K_kernel_edmd shape:", tuple(K_kernel_edmd.shape))
_print_phase("KDMD Koopman solve")

# Safety: ensure finite and nonnegative values before DMPS-style normalization
K_kernel_edmd = np.nan_to_num(K_kernel_edmd, nan=0.0, posinf=0.0, neginf=0.0)
minK = float(np.min(K_kernel_edmd))
if minK < 0.0:
    # Shift so the minimum is near zero; keep a tiny floor to avoid exact zeros
    K_kernel_edmd = K_kernel_edmd - minK + 1e-12

# Form the anisotropic graph Laplacian
sq_tar = xp.sum(X_tar ** 2, axis=1)
H = sq_tar[:, None] + sq_tar[None, :] - 2 * (X_tar @ X_tar.T)
epsilon_kern = float(0.5 * xp.median(H) / (np.log(n + 1) + 1e-12))

# Treat KDMD operator as the raw data_kernel for DMPS-style normalization
data_kernel = K_kernel_edmd
p_x = xp.sqrt(xp.sum(data_kernel, axis=1) + 1e-12)
p_y = p_x.copy()
data_kernel_norm = data_kernel / p_x[:, None] / p_y[None, :]
D_y = xp.sum(data_kernel_norm, axis=0) + 1e-12

# Random-walk symmetric normalization (as in DMPS)
rw_kernel = 0.5 * (data_kernel_norm / D_y + data_kernel_norm / D_y[:, None])
if USE_GPU:
    rw_kernel = cp.nan_to_num(rw_kernel, nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore
else:
    rw_kernel = np.nan_to_num(rw_kernel, nan=0.0, posinf=0.0, neginf=0.0)
_print_phase("DMPS-style normalization (density + symmetric RW)")

# DMPS-style spectrum via SVD on rw_kernel
if USE_GPU:
    phi, s, _ = xp.linalg.svd(rw_kernel, full_matrices=False)
else:
    phi, s, _ = svd(to_np(rw_kernel), full_matrices=False)
    phi = to_xp(phi)
    s = to_xp(s)
_print_phase("SVD rw_kernel")

# Green's operator weights: for Koopman eigenvalues s_k, use w_k = dt_edmd / (1 - s_k)
# Small Tikhonov on the gap; drop the constant (largest) mode
gap = 1.0 - s
reg_mu = 1e-6
inv_lambda = (dt_edmd / (gap + reg_mu)).astype(phi.dtype)
if inv_lambda.size > 0:
    inv_lambda[0] = 0.0
_print_phase("Green weights setup (drop constant)")

# Spectral truncation (threshold on singular values)
tol = 1e-8
keep_mask = s >= tol
keep_idx = xp.where(keep_mask)[0]
if keep_idx.size == 0:
    keep_idx = xp.asarray([0])
phi_trunc = phi[:, keep_idx]
w_trunc = inv_lambda[keep_idx]
if not USE_GPU:
    kept = keep_idx.shape[0]
else:
    kept = int(to_np(keep_idx).shape[0])
print(f"Kept spectral modes: {kept} of {phi.shape[1]} (tol={tol})")
_print_phase("Spectral truncation (threshold)")

# Run algorithm
iter = 1000
h = 20
m = 700
u = np.random.normal(0, 1, (m, d)).astype(np.float32)
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
u_trans = u / u_norm
x_init = r * u_trans
x_init = x_init[x_init[:, 1] > 0.8, :]
m = x_init.shape[0]
x_t = xp.zeros((m, d, iter), dtype=phi_trunc.dtype)
x_t[:, :, 0] = to_xp(x_init.astype(np.float32))

p_tar = xp.sum(data_kernel, axis=0)
D = xp.sum(data_kernel / xp.sqrt(p_tar) / xp.sqrt(p_tar)[:, None], axis=1)
_print_phase("Setup particles & density")

phi_w = (phi_trunc * w_trunc)
prog_every = max(1, (iter - 1)//25)
iter_start = time.time()
for t in range(iter - 1):
    x_curr = x_t[:, :, t]
    grad_matrix = grad_ker1(x_curr, X_tar, p_tar, sq_tar, D, epsilon_kern)
    cross_matrix = K_tar_eval(X_tar, x_curr, p_tar, sq_tar, D, epsilon_kern)
    cross_sum = xp.sum(cross_matrix, axis=1)
    v_spec = phi_trunc.T @ cross_sum
    s_vec = phi_w @ v_spec
    sum_x = xp.einsum('mnd,n->md', grad_matrix, s_vec)
    x_t[:, :, t + 1] = x_curr - (h / m) * sum_x
    if (t % prog_every) == 0 or t == iter - 2:
        _print_progress(t + 1, iter - 1, prefix="Descent")
print()
_print_phase("Iterations finished")
print(f"[INFO] Iter loop wall time: {_fmt_secs(time.time() - iter_start)}")

# Plotting results
X_tar_host = to_np(X_tar)
X_tar_next_host = to_np(X_tar_next)
x_t_host = to_np(x_t)
if d == 2:
    plt.figure()
    plt.plot(X_tar_host[:, 0], X_tar_host[:, 1], 'o', label='Target')
    plt.plot(x_t_host[:, 0, 0], x_t_host[:, 1, 0], 'o', label='Init')
    plt.plot(x_t_host[:, 0, -1], x_t_host[:, 1, -1], 'o', label='Final')
    plt.legend(); plt.title('2D Results'); plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar_host[:, 0], X_tar_host[:, 1], X_tar_host[:, 2], label='Target')
    ax.scatter(x_t_host[:, 0, 0], x_t_host[:, 1, 0], x_t_host[:, 2, 0], label='Init')
    ax.scatter(x_t_host[:, 0, -1], x_t_host[:, 1, -1], x_t_host[:, 2, -1], label='Final')
    ax.legend(); plt.title('3D Results'); plt.show()
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_t_host[:, 0, 0], x_t_host[:, 1, 0], x_t_host[:, 2, 0], label='Init')
    ax2.scatter(x_t_host[:, 0, -1], x_t_host[:, 1, -1], x_t_host[:, 2, -1], label='Final')
    ax2.legend(); plt.title('3D Final State'); plt.show()

# Plot matrix (scatter matrix)
pd.plotting.scatter_matrix(
    pd.DataFrame(X_tar_host),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle('Scatter Matrix of X_tar')
plt.show()

pd.plotting.scatter_matrix(
    pd.DataFrame(x_t_host[:, :, -1]),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle('Scatter Matrix of x_t (final)')
plt.show()
