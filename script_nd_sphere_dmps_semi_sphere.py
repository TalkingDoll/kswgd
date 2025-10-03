# This is a Gaussian example as a proof of concept.
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import pandas as pd
import time

# --------------- Optional GPU backend (CuPy) ---------------
USE_GPU = True  # set False to force CPU even if CuPy is available
try:  # Attempt to import cupy
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except Exception:
    cp = None  # type: ignore
    GPU_AVAILABLE = False
USE_GPU = bool(USE_GPU and GPU_AVAILABLE)
if USE_GPU:
    from grad_ker1_gpu import grad_ker1  # xp-aware versions
    from K_tar_eval_gpu import K_tar_eval
else:
    from grad_ker1 import grad_ker1      # CPU fallbacks
    from K_tar_eval import K_tar_eval

# ---------------- Timing / Progress Utilities (added for monitoring) ----------------
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

_LAST_PROGRESS_LEN = 0
def _print_progress(curr: int, total: int, start_time: float, prefix: str = "") -> None:
    global _LAST_PROGRESS_LEN
    total = max(1, int(total))
    curr = min(max(0, curr), total)
    bar_len = 30
    filled = int(bar_len * curr / total)
    filled = min(filled, bar_len)
    bar = ("=" * filled) + (">" if filled < bar_len else "") + ("." * max(0, bar_len - filled - (0 if filled == bar_len else 1)))
    pct = 100.0 * curr / total
    elapsed = time.time() - start_time
    avg = elapsed / max(1, curr)
    eta = avg * (total - curr)
    msg = f"{prefix}[{bar}] {pct:5.1f}% | iter {curr}/{total} | elapsed {_fmt_secs(elapsed)} | eta {_fmt_secs(eta)}"
    prev = _LAST_PROGRESS_LEN
    clear = "\r" + (" " * prev) + "\r"
    print(clear, end="")
    print(msg, end="", flush=True)
    _LAST_PROGRESS_LEN = len(msg)

# Set random seed for reproducibility (optional)
np.random.seed(0)

"""
DMPS (Diffusion Maps style) particle descent script.
Modified: add optional hemisphere sampling so that target samples live on a semi-sphere.
Direct hemisphere sampling (instead of reflection) by rejecting negative-axis directions.
Toggle via USE_HEMISPHERE. When False, behavior reverts to original full-sphere version.
"""

# ---------------- Hemisphere configuration ----------------
USE_HEMISPHERE = True          # set False to recover original full-sphere behavior
axis_hemi = 1                  # keep directions with positive component along this axis

def sample_hemisphere_directions(n: int, d: int, axis: int, lambda_: float = 1.0) -> np.ndarray:
    """Directly sample unit directions on a hemisphere by rejection sampling.
    Args:
        n: number of samples desired
        d: dimension
        axis: integer axis index to enforce positive (0-indexed)
        lambda_: anisotropy parameter for stretching axis 0
    Returns:
        U: (n, d) unit directions with U[:, axis] >= 0
    """
    samples = []
    while len(samples) < n:
        # Over-sample to reduce rejection rounds
        batch_size = max(n - len(samples), int((n - len(samples)) * 2.5))  # ~2x expected
        u = np.random.normal(0, 1, (batch_size, d))
        if lambda_ != 1.0:
            u[:, 0] = lambda_ * u[:, 0]
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        u_trans = u / (u_norm + 1e-12)
        # Keep only hemisphere samples
        valid = u_trans[:, axis] >= 0
        samples.append(u_trans[valid, :])
    U = np.vstack(samples)[:n, :]
    return U

print(f"[DEVICE] {'GPU' if USE_GPU else 'CPU'} mode active")

# ---------------- Target sample generation ----------------
n = 500
d = 3
lambda_ = 1
_t = time.time()
if USE_HEMISPHERE:
    u_trans = sample_hemisphere_directions(n, d, axis_hemi, lambda_)
    _t = _print_phase("Target sample generation (direct hemisphere sampling)", _t)
else:
    u = np.random.normal(0, 1, (n, d))
    u[:, 0] = lambda_ * u[:, 0]
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    u_trans = u / (u_norm + 1e-12)
    _t = _print_phase("Target sample generation (full sphere)", _t)
r = np.sqrt(np.random.rand(n, 1)) * 1/100 + 99/100
X_tar = r * u_trans
n = X_tar.shape[0]

# Form the anisotropic graph Laplacian
sq_tar = np.sum(X_tar ** 2, axis=1)
H = sq_tar[:, None] + sq_tar[None, :] - 2 * (X_tar @ X_tar.T)
epsilon = 0.5 * np.median(H) / np.log(n + 1)
def ker(X):
    sq_tar = np.sum(X ** 2, axis=1)
    return np.exp(-(sq_tar[:, None] + sq_tar[None, :] - 2 * (X @ X.T)) / (2 * epsilon))

data_kernel = ker(X_tar)
_t = _print_phase("Base kernel (Gaussian) build", _t)
p_x = np.sqrt(np.sum(data_kernel, axis=1))
p_y = p_x.copy()
# Normalize kernel
data_kernel_norm = data_kernel / p_x[:, None] / p_y[None, :]
D_y = np.sum(data_kernel_norm, axis=0)

# Match MATLAB: 0.5*(A ./ D_y + A ./ D_y') where D_y is a row vector.
# First term divides columns by D_y (broadcast over last axis),
# second term explicitly divides rows by D_y (reshape as column vector).
rw_kernel = 0.5 * (data_kernel_norm / D_y + data_kernel_norm / D_y[:, None])
_t = _print_phase("Random-walk symmetric normalization", _t)
phi, s, _ = svd(rw_kernel)
_t = _print_phase("SVD on rw_kernel", _t)
lambda_ns = s
lambda_ = -lambda_ns + 1
inv_lambda = np.zeros_like(lambda_)
inv_lambda[1:] = 1 / lambda_[1:]
inv_lambda = inv_lambda * epsilon
inv_K = phi @ np.diag(inv_lambda) @ phi.T
_t = _print_phase("Primary inverse-like weights (inv_lambda)", _t)

tol = 1e-6
lambda_ns_mod = np.copy(lambda_ns)
lambda_ns_mod[lambda_ns_mod < tol] = 0
below_tol = np.sum(lambda_ns < tol)
above_tol = n - below_tol
reg = 0.001
lambda_ns_inv = np.zeros_like(lambda_ns)
mask = lambda_ns >= tol
lambda_ns_inv[mask] = epsilon / (lambda_ns[mask] + reg)
inv_K_ns = phi @ np.diag(lambda_ns_inv) @ phi.T
_t = _print_phase("Regularized inverse weights (lambda_ns_inv)", _t)

# ---------------- Particle descent (initialization) ----------------
iter = 1000
h = 15
m = 700
if USE_HEMISPHERE:
    # Directly sample on hemisphere with additional spherical cap constraint
    # Over-sample to ensure enough points after filtering
    u_trans = sample_hemisphere_directions(int(m * 1.5), d, axis_hemi, lambda_=1.0)
    r = np.sqrt(np.random.rand(u_trans.shape[0], 1)) * 1/100 + 99/100
    x_init = r * u_trans
    # Keep a spherical cap: y > 0.7
    x_init = x_init[x_init[:, 1] > 0.7, :]
    if x_init.shape[0] > m:
        x_init = x_init[:m, :]  # trim to exactly m
    m = x_init.shape[0]
else:
    u = np.random.normal(0, 1, (m, d))
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
    u_trans = u / (u_norm + 1e-12)
    x_init = r * u_trans
    x_init = x_init[x_init[:, 1] > 0.7, :]
    m = x_init.shape[0]
x_t = np.zeros((m, d, iter), dtype=np.float64)  # Use float64 for numerical precision
x_t[:, :, 0] = x_init
_t = _print_phase("Particle initialization", _t)

p_tar = np.sum(data_kernel, axis=0)
D = np.sum(data_kernel / np.sqrt(p_tar) / np.sqrt(p_tar)[:, None], axis=1)

inv_K_ns_s_ns = phi @ np.diag(lambda_ns_inv * inv_lambda * lambda_ns_inv) @ phi.T
lambda_s_s_ns = inv_lambda * inv_lambda * lambda_ns_inv
lambda_s_s_ns = lambda_s_s_ns[:above_tol]
lambda_ns_s_ns = lambda_ns_inv * inv_lambda * lambda_ns_inv
lambda_ns_s_ns = lambda_ns_s_ns[:above_tol]

sum_x = np.zeros((m, d))
loop_start = time.time()
total_loop = iter - 1
if USE_GPU:
    # Stage constants on GPU (keep float64 for precision)
    X_tar_gpu = cp.asarray(X_tar)
    p_tar_gpu = cp.asarray(p_tar)
    sq_tar_gpu = cp.asarray(sq_tar)
    D_gpu = cp.asarray(D)
    phi_gpu = cp.asarray(phi[:, :above_tol])
    lambda_ns_s_ns_gpu = cp.asarray(lambda_ns_s_ns)
    x_t_gpu = cp.asarray(x_t)
    diag_lambda_gpu = cp.diag(lambda_ns_s_ns_gpu)
    # Iteration loop (GPU)
    for t in range(iter - 1):
        x_slice = x_t_gpu[:, :, t]
        grad_matrix = grad_ker1(x_slice, X_tar_gpu, p_tar_gpu, sq_tar_gpu, D_gpu, epsilon)           # (m,n,d)
        cross_matrix = K_tar_eval(X_tar_gpu, x_slice, p_tar_gpu, sq_tar_gpu, D_gpu, epsilon)         # (n,m)
        # Compute spectral contraction efficiently:
        # phi[:, :k] @ diag(lambda) @ phi[:, :k].T @ cross_matrix  => (phi * lambda) @ (phi.T @ cross_matrix)
        tmp = phi_gpu.T @ cross_matrix                    # (k,m)
        tmp = (lambda_ns_s_ns_gpu[:, None]) * tmp         # weight each mode
        M = phi_gpu @ tmp                                 # (n,m)
        # einsum over gradient tensor and M
        sum_x_gpu = cp.einsum('mnd,nm->md', grad_matrix, M)
        x_t_gpu[:, :, t + 1] = x_slice - (h / m) * sum_x_gpu
        done = t + 1
        if done == total_loop or (done % max(1, total_loop // 100) == 0):
            _print_progress(done, total_loop, loop_start, prefix="[DMPS] ")
    x_t = cp.asnumpy(x_t_gpu)
else:
    for t in range(iter - 1):
        grad_matrix = grad_ker1(x_t[:, :, t], X_tar, p_tar, sq_tar, D, epsilon)
        cross_matrix = K_tar_eval(X_tar, x_t[:, :, t], p_tar, sq_tar, D, epsilon)
        # Precompute spectral matrix once per iteration
        phi_trunc = phi[:, :above_tol]
        diag_lambda = np.diag(lambda_ns_s_ns)
        tmp = phi_trunc.T @ cross_matrix            # (k,m)
        tmp = (lambda_ns_s_ns[:, None]) * tmp       # mode weighting
        M = phi_trunc @ tmp                         # (n,m)
        # Vectorized gradient contraction
        sum_x = np.einsum('mnd,nm->md', grad_matrix, M)
        x_t[:, :, t + 1] = x_t[:, :, t] - (h / m) * sum_x
        done = t + 1
        if done == total_loop or (done % max(1, total_loop // 100) == 0):
            _print_progress(done, total_loop, loop_start, prefix="[DMPS] ")

print()  # newline after progress bar
_t = _print_phase("Iteration loop total", loop_start)

# ---------------- Plotting with progress & timing ----------------
plot_total_start = time.time()
_tp = time.time()

# Define ordered plot steps (mirrors pattern from kernel_edmd semi-sphere GPU script)
PLOT_STEPS = []
if d == 2:
    PLOT_STEPS.append("main_2d")
else:
    PLOT_STEPS.extend(["main_3d", "main_3d_final"])
PLOT_STEPS.extend(["scatter_X_tar", "scatter_x_t_final"])  # scatter matrices

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
    plt.figure()
    plt.plot(X_tar[:, 0], X_tar[:, 1], 'o', label='Target')
    plt.plot(x_t[:, 0, 0], x_t[:, 1, 0], 'o', label='Init')
    plt.plot(x_t[:, 0, -1], x_t[:, 1, -1], 'o', label='Final')
    plt.legend(); plt.title('2D Results (hemisphere)' if USE_HEMISPHERE else '2D Results (full sphere)')
    _tp = _print_phase("Plot main results (2D)", _tp)
    _advance("main_2d")
else:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], label='Target')
    ax.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax.legend(); plt.title('3D Results (hemisphere)' if USE_HEMISPHERE else '3D Results (full sphere)')
    _tp = _print_phase("Plot main results (3D)", _tp)
    _advance("main_3d")
    fig2 = plt.figure(); ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], label='Init')
    ax2.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], label='Final')
    ax2.legend(); plt.title('3D Final State (hemisphere)' if USE_HEMISPHERE else '3D Final State (full sphere)')
    _tp = _print_phase("Plot main results (3D final)", _tp)
    _advance("main_3d_final")

#############################################
# Select up to 10 best dimensions by per-dim KL over iterations
#############################################
try:
    k_max = min(10, d)
    mu_tar_sel = X_tar.mean(axis=0)
    var_tar_sel = np.maximum(X_tar.var(axis=0), 1e-12)
    # Over time stats (iter x d)
    mu_hist = x_t.mean(axis=0).T
    var_hist = np.maximum(x_t.var(axis=0).T, 1e-12)
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
fig.savefig('scatter_matrix_X_tar.png', dpi=200, bbox_inches='tight')
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
fig.savefig('scatter_matrix_x_t_final.png', dpi=200, bbox_inches='tight')
_tp = _print_phase("Plot scatter matrix x_t_final (top dims)", _tp)
_advance("scatter_x_t_final")

# Finish plotting progress bar line
print()
print(f"[TIMER] Plotting total: {time.time() - plot_total_start:.3f}s")

# ---------------- Wasserstein-1 Distance (Standard/Exact) ----------------
# Using POT (Python Optimal Transport) library for exact W1 computation
# Install: pip install POT
# W1 is the Earth Mover's Distance (EMD) with uniform weights
print("\n[W1] Computing exact Wasserstein-1 distance...")

try:
    import ot
    
    X_fin = x_t[:, :, -1]
    n_tar = X_tar.shape[0]
    n_fin = X_fin.shape[0]
    
    # Uniform distributions (equal mass per point)
    a = np.ones(n_tar) / n_tar
    b = np.ones(n_fin) / n_fin
    
    # Compute pairwise Euclidean distance matrix
    _t_dist = time.time()
    M = ot.dist(X_tar, X_fin, metric='euclidean')
    print(f"[W1] Distance matrix computed in {time.time() - _t_dist:.3f}s")
    
    # Solve exact optimal transport problem (EMD algorithm)
    _t_emd = time.time()
    w1_exact = ot.emd2(a, b, M)
    print(f"[W1] EMD solved in {time.time() - _t_emd:.3f}s")
    
    print(f"[W1] Exact Wasserstein-1 distance = {w1_exact:.6e}")
    
except ImportError:
    print("[W1] ERROR: POT library not installed. Install with: pip install POT")
    print("[W1] Falling back to basic statistics...")
    try:
        X_fin = x_t[:, :, -1]
        mu_tar = X_tar.mean(axis=0)
        mu_fin = X_fin.mean(axis=0)
        mse_mean = np.mean((mu_tar - mu_fin)**2)
        print(f"[STAT] MSE(means)= {mse_mean:.6e}")
        print(f"[STAT] Target mean: {mu_tar}")
        print(f"[STAT] Final mean: {mu_fin}")
    except Exception as _e2:
        print(f"[STAT] Statistics failed: {_e2}")
except Exception as _e:
    print(f"[W1] Failed to compute W1 (error: {_e})")

plt.show()
