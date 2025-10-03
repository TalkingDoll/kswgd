# This is a Gaussian example as a proof of concept.
import numpy as np
from scipy.linalg import svd
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import pandas as pd
from grad_ker1_gpu import grad_ker1
from K_tar_eval_gpu import K_tar_eval
import time
import sys
from typing import Optional
import os

# ---------------------- GPU Config (optional) ----------------------
USE_GPU = True  # set True to attempt GPU acceleration via CuPy (falls back to CPU if unavailable)
try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    GPU_AVAILABLE = False
USE_GPU = bool(USE_GPU and GPU_AVAILABLE)

# Optional: use ResKoopNet (neural Koopman) instead of KDMD to build Koopman operator
USE_RESKOOPNET = True
SLICED_W1_NUM_PROJ = 64  # set >0 to also report a sliced W1 metric (lightweight diagnostic)

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
    SUPPRESS_TIMERS = {
        "Quick visualization",
        "Kernel-EDMD dual operator (Σ^+_γ Q^T Â Q Σ^+_γ)",
        "DMPS-style normalization (density + symmetric RW)",
        "SVD on rw_kernel",
        "Green weights setup (drop constant)",
        "Spectral truncation (threshold)",
    }
    if name not in SUPPRESS_TIMERS:
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
    msg = f"{prefix}[{bar}] {pct:5.1f}% | iter {curr}/{total} | elapsed {_fmt_secs(elapsed)} | eta {_fmt_secs(eta)}"
    # Clear the previous line completely, then print the new one without newline.
    # This avoids residual characters when the new message is shorter and prevents extra lines.
    global _LAST_PROGRESS_LEN
    try:
        prev = int(_LAST_PROGRESS_LEN)
    except Exception:
        prev = 0
    clear = "\r" + (" " * max(0, prev)) + "\r"
    print(clear, end="")
    print(msg, end="", flush=True)
    _LAST_PROGRESS_LEN = len(msg)

RUN_START = time.time()
_t = time.time()
if USE_GPU:
    try:
        dev = cp.cuda.Device()  # type: ignore
        props = cp.cuda.runtime.getDeviceProperties(dev.id)  # type: ignore
        name = props.get('name', '')
        if isinstance(name, (bytes, bytearray)):
            name = name.decode(errors='ignore')
        print(f"[DEVICE] GPU ENABLED | id={dev.id} | {name} | CuPy {cp.__version__}")  # type: ignore
    except Exception:
        print("[DEVICE] GPU ENABLED (device info unavailable)")
else:
    print("[DEVICE] GPU DISABLED")

# Track length of the last printed progress line to clear it before updating
_LAST_PROGRESS_LEN = 0

# Figure output directory (same folder as this script)
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figure_kernel_edmd_test')
os.makedirs(FIG_DIR, exist_ok=True)

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
d = 2
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
 
# # Scheme 1: Euclidean KDE-score (Euler–Maruyama) to generate X_tar_next for EDMD
# # - Estimate score ∇log q(x) using Gaussian KDE with global bandwidth (median distance)
# # - One Euler–Maruyama step: x_next = x + Δt * score(x) + sqrt(2Δt) * ξ
# dt_edmd = 1e-2
# # Pairwise squared distances for bandwidth and weights
# diffs_edmd = X_tar[:, None, :] - X_tar[None, :, :]
# dist2_edmd = np.sum(diffs_edmd ** 2, axis=2)
# h_edmd = np.sqrt(np.median(dist2_edmd) + 1e-12)
# W_edmd = np.exp(-dist2_edmd / (2.0 * (h_edmd ** 2)))
# sumW_edmd = np.sum(W_edmd, axis=1, keepdims=True) + 1e-12
# weighted_means_edmd = (W_edmd @ X_tar) / sumW_edmd
# score_eucl = (weighted_means_edmd - X_tar) / (h_edmd ** 2)
# xi_edmd = np.random.normal(0.0, 1.0, size=X_tar.shape)
# X_tar_next = X_tar + dt_edmd * score_eucl + np.sqrt(2.0 * dt_edmd) * xi_edmd
# _t = _print_phase("EDMD Scheme 1 (KDE-score + one step)", _t)

# Scheme 2: Manifold (sphere) via projected Euclidean score — commented out for toggling
# (Start from Euclidean KDE-score, then project drift and noise to the tangent space, and renormalize)
dt_edmd = 0.15
diffs_edmd = X_tar[:, None, :] - X_tar[None, :, :]
dist2_edmd = np.sum(diffs_edmd ** 2, axis=2)
h_edmd = np.sqrt(np.median(dist2_edmd) + 1e-12)
W_edmd = np.exp(-dist2_edmd / (2.0 * (h_edmd ** 2)))
sumW_edmd = np.sum(W_edmd, axis=1, keepdims=True) + 1e-12
weighted_means_edmd = (W_edmd @ X_tar) / sumW_edmd
score_eucl = (weighted_means_edmd - X_tar) / (h_edmd ** 2)
# Project drift and noise to the tangent space of the unit sphere
X_norm = X_tar / (np.linalg.norm(X_tar, axis=1, keepdims=True) + 1e-12)
proj = np.eye(X_tar.shape[1])[None, :, :] - X_norm[:, :, None] * X_norm[:, None, :]
score_tan = np.einsum('nij,ni->nj', proj, score_eucl)
xi_edmd = np.random.normal(0.0, 1.0, size=X_tar.shape)
xi_tan = xi_edmd - (np.sum(X_norm * xi_edmd, axis=1, keepdims=True)) * X_norm
X_step = X_norm + dt_edmd * score_tan + np.sqrt(2.0 * dt_edmd) * xi_tan
X_tar_next = X_step / (np.linalg.norm(X_step, axis=1, keepdims=True) + 1e-12)

# Quick visualization: X_tar vs X_tar_next (Scheme 1)
if d == 2:
    fig = plt.figure()
    plt.scatter(X_tar[:, 0], X_tar[:, 1], s=10, c='C0', label='X_tar')
    plt.scatter(X_tar_next[:, 0], X_tar_next[:, 1], s=10, c='C1', label='X_tar_next')
    plt.legend()
    plt.axis('equal')
    plt.title('X_tar vs X_tar_next (Scheme 1, hemisphere)')
    fig.savefig(os.path.join(FIG_DIR, 'quick_vis_2d.png'), dpi=200, bbox_inches='tight')
else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='X_tar', c='C0', s=10)
    ax.scatter(X_tar_next[:, 0], X_tar_next[:, 1], X_tar_next[:, 2], label='X_tar_next', c='C1', s=10)
    ax.legend()
    ax.set_title('X_tar vs X_tar_next (Scheme 1, hemisphere)')
    fig.savefig(os.path.join(FIG_DIR, 'quick_vis_3d.png'), dpi=200, bbox_inches='tight')
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

# ---------------------- ResKoopNet Koopman ----------------------
# Build Koopman operator via ResKoopNet and assemble an n×n sample-space operator
import torch  # type: ignore
from solver_resdmd_torch import KoopmanNNTorch, KoopmanSolverTorch  # type: ignore
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Dictionary size and NN width (kept moderate to avoid long runtime in-script)
# dic_size = min(256, max(32, d + 16))
dic_size = 30
# net_width = min(256, max(64, d // 2))
net_width = dic_size
basis_function = KoopmanNNTorch(input_size=d, layer_sizes=[net_width, net_width, net_width], n_psi_train=dic_size - (d + 1)).to(device)
# Provide a valid checkpoint path (required by the solver when saving best model)
ckpt_path = os.path.join(FIG_DIR, 'reskoopnet_ckpt.torch')
solver = KoopmanSolverTorch(dic=basis_function, target_dim=d, reg=0.1, checkpoint_file=ckpt_path)
# Train/validate split
n_all = X_tar.shape[0]
split = max(1, int(0.8 * n_all))
data_x_train = X_tar[:split].astype(np.float64, copy=False)
data_y_train = X_tar_next[:split].astype(np.float64, copy=False)
data_x_valid = X_tar[split:].astype(np.float64, copy=False)
data_y_valid = X_tar_next[split:].astype(np.float64, copy=False)
data_train = [data_x_train, data_y_train]
data_valid = [data_x_valid, data_y_valid]
# Build solver (keep epochs modest to fit script runtime)
solver.build(data_train=data_train,
             data_valid=data_valid,
             epochs=8,
             batch_size=256,
             lr=1e-4,
             log_interval=20,
             lr_decay_factor=.8)
# Extract Koopman matrix and features, then assemble an n×n sample-space operator
# Use the learned Koopman matrix with features computed on ALL target points
Koopman_matrix_K = solver.K.detach().cpu().numpy()
with torch.no_grad():
    X_tar_tensor = torch.from_numpy(X_tar.astype(np.float64, copy=False)).to(device)
    Psi_X_all_t = basis_function(X_tar_tensor)
Psi_X_all = Psi_X_all_t.detach().cpu().numpy()
K_reskoopnet = Psi_X_all @ Koopman_matrix_K @ Psi_X_all.T
print("K_reskoopnet shape:", K_reskoopnet.shape)
_t = _print_phase("ResKoopNet Koopman operator (sample-space) build", _t)

# Removed legacy KDMD kernel code (no longer used)

# Choose operator source (ResKoopNet)
K_op = K_reskoopnet

# Safety: ensure finite and nonnegative values before DMPS-style normalization
K_op = np.nan_to_num(K_op, nan=0.0, posinf=0.0, neginf=0.0)
minK = float(np.min(K_op))
if minK < 0.0:
    # Shift so the minimum is near zero; keep a tiny floor to avoid exact zeros
    K_op = K_op - minK + 1e-12

# Align target arrays length with operator dimension if they differ
n_op = K_op.shape[0]
if n_op != X_tar.shape[0]:
    print(f"[WARN] Operator size ({n_op}) != number of target points ({X_tar.shape[0]}). Truncating target arrays to match operator.")
    X_tar = X_tar[:n_op, :]
    n = X_tar.shape[0]

# Form the anisotropic graph Laplacian
sq_tar = np.sum(X_tar ** 2, axis=1)
H = sq_tar[:, None] + sq_tar[None, :] - 2 * (X_tar @ X_tar.T)
# Bandwidth for data kernel (used by kernels/gradients); decoupled from KDMD/DMPS normalization
epsilon_kern = 0.5 * np.median(H) / (np.log(n + 1) + 1e-12)

# Treat the selected operator as the raw data_kernel for DMPS-style normalization
data_kernel = K_op
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
tol = 1e-6
keep_mask = s >= tol
keep_idx = np.where(keep_mask)[0]
if keep_idx.size == 0:
    # Fallback to keep at least the first mode (it will be zero-weighted anyway)
    keep_idx = np.array([0], dtype=int)
phi_trunc = phi[:, keep_idx].astype(np.float32, copy=False)
w_trunc = inv_lambda[keep_idx].astype(np.float32, copy=False)
print(f"Kept spectral modes: {phi_trunc.shape[1]} of {phi.shape[1]} (tol={tol})")
_t = _print_phase("Spectral truncation (threshold)", _t)

xp = np
if USE_GPU:
    phi_trunc_gpu = cp.asarray(phi_trunc)
    w_trunc_gpu = cp.asarray(w_trunc)
    xp = cp

# Run algorithm
iter = 1000
h = 5
m = 700
u = np.random.normal(0, 1, (m, d))
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
u_trans = u / u_norm

# Fold initial directions to the same hemisphere
u_trans_hemi = reflect_to_hemisphere(u_trans, n_axis)
x_init = r * u_trans_hemi
x_init = x_init[x_init[:, 1] > 0.7, :]
m = x_init.shape[0]
x_t = np.zeros((m, d, iter), dtype=np.float32)
x_t[:, :, 0] = x_init.astype(np.float32, copy=False)
_t = _print_phase("LAWGD init (particles & buffers)", _t)

p_tar = np.sum(data_kernel, axis=0).astype(np.float32, copy=False)
D = np.sum(data_kernel / np.sqrt(p_tar) / np.sqrt(p_tar)[:, None], axis=1).astype(np.float32, copy=False)

sum_x = np.zeros((m, d))

# If using GPU, stage constants and state on GPU once to avoid repeated transfers
if USE_GPU:
    X_tar_xp = cp.asarray(X_tar.astype(np.float32, copy=False))
    p_tar_xp = cp.asarray(p_tar.astype(np.float32, copy=False))
    sq_tar_xp = cp.asarray(sq_tar.astype(np.float32, copy=False))
    D_xp = cp.asarray(D.astype(np.float32, copy=False))
    x_t_gpu = cp.asarray(x_t)

# ---------------------- Iteration Loop with Progress Bar ----------------------
loop_start = time.time()
total_loop = max(1, iter - 1)
last_print = 0.0
for t in range(iter - 1):
    # Use xp-aware functions: when USE_GPU, these run on CuPy and stay on GPU.
    if USE_GPU:
        x_t_slice = x_t_gpu[:, :, t]
        grad_matrix = grad_ker1(x_t_slice, X_tar_xp, p_tar_xp, sq_tar_xp, D_xp, epsilon_kern)  # (m,n,d)
        cross_matrix = K_tar_eval(X_tar_xp, x_t_slice, p_tar_xp, sq_tar_xp, D_xp, epsilon_kern)  # (n,m)
        tmp = phi_trunc_gpu.T @ cross_matrix           # (k x m)
        tmp = (w_trunc_gpu[:, None]) * tmp             # (k x m)
        M = phi_trunc_gpu @ tmp                        # (n x m)
        sum_x_gpu = cp.einsum('mnd,nm->md', grad_matrix, M)
        x_t_gpu[:, :, t + 1] = x_t_slice - (h / m) * sum_x_gpu
    else:
        x_t_slice = x_t[:, :, t]
        grad_matrix = grad_ker1(x_t_slice, X_tar, p_tar, sq_tar, D, epsilon_kern)
        cross_matrix = K_tar_eval(X_tar, x_t_slice, p_tar, sq_tar, D, epsilon_kern)
        tmp = (phi_trunc.T @ cross_matrix)
        tmp = (w_trunc[:, None]) * tmp
        M = phi_trunc @ tmp
        sum_x = np.einsum('mnd,nm->md', grad_matrix, M)
        x_t[:, :, t + 1] = x_t[:, :, t] - (h / m) * sum_x
    # Progress bar update (throttled)
    done = t + 1
    elapsed = time.time() - RUN_START
    avg_iter_time = (time.time() - loop_start) / max(1, done)
    eta = avg_iter_time * (total_loop - done)
    # Print every ~1% (and at the end) to avoid terminal spam
    if done == total_loop or (done % max(1, total_loop // 100) == 0):
        _print_progress(done, total_loop, elapsed, eta, prefix="[LAWGD] ")
        last_print = time.time()

# Ensure the progress line ends cleanly
print()
if USE_GPU:
    # Bring back to host for plotting
    x_t = cp.asnumpy(x_t_gpu)
print(f"[TIMER] RUN total (pre-plot): {time.time() - RUN_START:.3f}s")

#############################################
# Plotting results with progress bar
#############################################
plot_total_start = time.time()
_tp = time.time()

PLOT_STEPS = []
if d == 2:
    PLOT_STEPS.append("main_2d")
else:
    PLOT_STEPS.extend(["main_3d", "main_3d_final"])
PLOT_STEPS.extend(["scatter_X_tar", "scatter_x_t_final"])  # common scatter matrices
_plot_last_len = 0

def _print_plot_progress(k: int, total: int, label: str):
    global _plot_last_len
    total = max(1, total)
    k = min(k, total)
    frac = k / total
    bar_len = 28
    filled = int(bar_len * frac)
    bar = '=' * filled + ('>' if filled < bar_len else '') + '.' * max(0, bar_len - filled - (0 if filled == bar_len else 1))
    msg = f"[PLOT] [{bar}] {frac*100:5.1f}% | step {k}/{total} | {label}"
    clear = "\r" + (" " * _plot_last_len) + "\r"
    print(clear + msg, end="", flush=True)
    _plot_last_len = len(msg)

plot_step_total = len(PLOT_STEPS)
_plot_state = {"idx": 0}

def _advance(label: str):
    _plot_state["idx"] += 1
    _print_plot_progress(_plot_state["idx"], plot_step_total, label)

# --- Main figure(s)
if d == 2:
    fig = plt.figure()
    plt.plot(X_tar[:, 0], X_tar[:, 1], 'o', label='Target')
    plt.plot(x_t[:, 0, 0], x_t[:, 1, 0], 'o', label='Init')
    plt.plot(x_t[:, 0, -1], x_t[:, 1, -1], 'o', label='Final')
    plt.legend(); plt.title('2D Results (hemisphere)')
    fig.savefig(os.path.join(FIG_DIR, 'results_2d.png'), dpi=200, bbox_inches='tight')
    _tp = _print_phase("Plot main results (2D)", _tp)
    _advance("main_2d")
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='Target')
    ax.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax.legend(); plt.title('3D Results (hemisphere)')
    fig.savefig(os.path.join(FIG_DIR, 'results_3d.png'), dpi=200, bbox_inches='tight')
    _tp = _print_phase("Plot main results (3D)", _tp)
    _advance("main_3d")
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax2.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax2.legend(); plt.title('3D Final State (hemisphere)')
    fig2.savefig(os.path.join(FIG_DIR, 'results_3d_final.png'), dpi=200, bbox_inches='tight')
    _tp = _print_phase("Plot main results (3D final)", _tp)
    _advance("main_3d_final")

#############################################
# Select up to 10 best dimensions by per-dim KL over iterations
#############################################
try:
    k_max = min(10, d)
    # Target stats
    mu_tar_sel = X_tar.mean(axis=0)
    var_tar_sel = X_tar.var(axis=0)
    var_tar_sel = np.maximum(var_tar_sel, 1e-12)
    # History stats over iterations: shapes (iter, d)
    mu_hist = x_t.mean(axis=0).T
    var_hist = x_t.var(axis=0).T
    var_hist = np.maximum(var_hist, 1e-12)
    # Per-dim KL(target||x_t[.,t]) across time
    # KL_j(t) = 0.5 * [ var_tar_j/var_t_j + (mu_t_j - mu_tar_j)^2 / var_t_j - 1 + log(var_t_j/var_tar_j) ]
    ratio = var_tar_sel[None, :] / var_hist
    diff2 = (mu_hist - mu_tar_sel[None, :])**2 / var_hist
    kl_per_dim_hist = 0.5 * (ratio + diff2 - 1.0 + np.log(var_hist / var_tar_sel[None, :]))
    kl_per_dim_hist = np.nan_to_num(kl_per_dim_hist, nan=np.inf, posinf=np.inf, neginf=np.inf)
    best_kl_per_dim = np.min(kl_per_dim_hist, axis=0)
    idx_sorted = np.argsort(best_kl_per_dim)
    idx_sel = idx_sorted[:k_max]
    print(f"[PLOT] Using top-{len(idx_sel)} dims by per-dim min KL over iterations: {idx_sel.tolist()}")
except Exception as _e:
    # Fallback: first k_max dims
    idx_sel = np.arange(min(10, d))
    print(f"[PLOT] Fallback selecting first {len(idx_sel)} dims (error computing KL ranks: {_e})")

# --- Scatter matrix X_tar (selected dims)
pd.plotting.scatter_matrix(
    pd.DataFrame(X_tar[:, idx_sel], columns=[f"dim_{j}" for j in idx_sel]),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle(f'Scatter Matrix of X_tar (top {len(idx_sel)} dims)')
fig = plt.gcf()
fig.savefig(os.path.join(FIG_DIR, 'scatter_matrix_X_tar.png'), dpi=200, bbox_inches='tight')
_tp = _print_phase("Plot scatter matrix X_tar (top dims)", _tp)
_advance("scatter_X_tar")

# --- Scatter matrix final x_t (selected dims)
pd.plotting.scatter_matrix(
    pd.DataFrame(x_t[:, idx_sel, -1], columns=[f"dim_{j}" for j in idx_sel]),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle(f'Scatter Matrix of x_t (final, top {len(idx_sel)} dims)')
fig = plt.gcf()
fig.savefig(os.path.join(FIG_DIR, 'scatter_matrix_x_t_final.png'), dpi=200, bbox_inches='tight')
_tp = _print_phase("Plot scatter matrix x_t_final (top dims)", _tp)
_advance("scatter_x_t_final")

# Finish plotting progress bar line
print()
print(f"[TIMER] Plotting total: {time.time() - plot_total_start:.3f}s")

# # Show once at the end (Scheme A) and print save location
# print(f"[FIGURES] Saved all figures to: {FIG_DIR}")

# ---------------- KL Divergence Report (diagonal Gaussian approximation) ----------------
# We approximate each distribution by a diagonal Gaussian fitted to samples.
# For X ~ N(mu_X, diag(sigma_X^2)), Y ~ N(mu_Y, diag(sigma_Y^2)):
#   KL(X||Y) = 0.5 * sum_j [ (sigma_X_j^2 / sigma_Y_j^2) + ((mu_Y_j - mu_X_j)^2 / sigma_Y_j^2) - 1 + log(sigma_Y_j^2 / sigma_X_j^2) ]
# We report KL(target||final), KL(final||target) and their symmetric average.
def _kl_diag(mu_p, var_p, mu_q, var_q):
    var_p = np.maximum(var_p, 1e-12)
    var_q = np.maximum(var_q, 1e-12)
    ratio = var_p / var_q
    diff2 = (mu_q - mu_p)**2 / var_q
    return 0.5 * np.sum(ratio + diff2 - 1.0 + np.log(var_q / var_p))
try:
    X_fin = x_t[:, :, -1]
    mu_tar = X_tar.mean(axis=0)
    mu_fin = X_fin.mean(axis=0)
    var_tar = X_tar.var(axis=0)
    var_fin = X_fin.var(axis=0)
    kl_tar_fin = _kl_diag(mu_tar, var_tar, mu_fin, var_fin)
    print(f"[KL] KL(target||final)= {kl_tar_fin:.6e}")
    print(f"[KL] Average KL(target||final)= {kl_tar_fin/d:.6e}")
except Exception as _e:
    print(f"[KL] Skipped (error: {_e})")

# Optional: Sliced Wasserstein-1 metric (no extra deps). Approximates W1 by averaging 1D W1 over random projections.
def _w1_1d(x: np.ndarray, y: np.ndarray, L: Optional[int] = None) -> float:
    # Use quantile grids to handle unequal sample sizes robustly
    n1 = x.shape[0]; n2 = y.shape[0]
    if L is None:
        L = min(n1, n2)
    u = (np.arange(L) + 0.5) / max(L, 1)
    qx = np.quantile(x, u, method='linear')
    qy = np.quantile(y, u, method='linear')
    return float(np.mean(np.abs(qx - qy)))

def sliced_wasserstein_1(X: np.ndarray, Y: np.ndarray, num_proj: int = 64, seed: Optional[int] = None) -> float:
    if num_proj <= 0:
        return float('nan')
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    acc = 0.0
    for _ in range(num_proj):
        v = rng.normal(0.0, 1.0, size=(d,))
        nv = np.linalg.norm(v) + 1e-12
        v = v / nv
        x_proj = X @ v
        y_proj = Y @ v
        acc += _w1_1d(x_proj, y_proj)
    return acc / num_proj

try:
    if SLICED_W1_NUM_PROJ and SLICED_W1_NUM_PROJ > 0:
        X_fin = x_t[:, :, -1]
        sw1 = sliced_wasserstein_1(X_tar, X_fin, num_proj=SLICED_W1_NUM_PROJ, seed=0)
        print(f"[W1] Sliced W1 (num_proj={SLICED_W1_NUM_PROJ})= {sw1:.6e}")
except Exception as _e:
    print(f"[W1] Skipped (error: {_e})")

plt.show()
