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
Modified: add optional hemisphere folding so that target samples live on a semi-sphere
just like the kernel+EDMD script (mirror full-sphere directions into a chosen half-space).
Toggle via USE_HEMISPHERE. When False, behavior reverts to original full-sphere version.
"""

# ---------------- Hemisphere configuration (mirrors the kernel_edmd semi-sphere script) ----------------
USE_HEMISPHERE = True          # set False to recover original full-sphere behavior
axis_hemi = 1                  # keep directions with positive component along this axis

def reflect_to_hemisphere(U: np.ndarray, axis: int) -> np.ndarray:
    """Reflect points across the hyperplane x_axis = 0 so that the axis-th coordinate becomes non-negative.
    Equivalent to folding full sphere into a hemisphere (Neumann reflection / method of images).
    Args:
        U: (n,d) unit (or arbitrary) direction vectors.
        axis: integer axis index to enforce non-negative.
    Returns:
        U_ref: reflected copy with U_ref[:, axis] >= 0 (up to floating error).
    """
    if axis < 0 or axis >= U.shape[1]:
        return U
    mask = U[:, axis] < 0
    if not np.any(mask):
        return U
    U_ref = U.copy()
    U_ref[mask, axis] *= -1.0
    return U_ref

print(f"[DEVICE] {'GPU' if USE_GPU else 'CPU'} mode active")

# ---------------- Target sample generation ----------------
n = 500
d = 100
lambda_ = 1
u = np.random.normal(0, 1, (n, d))
u[:, 0] = lambda_ * u[:, 0]
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(n, 1)) * 1/100 + 99/100
u_trans = u / (u_norm + 1e-12)
_t = time.time()
if USE_HEMISPHERE:
    u_trans = reflect_to_hemisphere(u_trans, axis_hemi)
_t = _print_phase("Target sample generation (with optional hemisphere)", _t)
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
h = 25
m = 700
u = np.random.normal(0, 1, (m, d))
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
u_trans = u / (u_norm + 1e-12)
if USE_HEMISPHERE:
    u_trans = reflect_to_hemisphere(u_trans, axis_hemi)
x_init = r * u_trans
# Keep a spherical cap (original constraint) but now guaranteed in hemisphere if enabled.
x_init = x_init[x_init[:, 1] > 0.25, :]
m = x_init.shape[0]
x_t = np.zeros((m, d, iter), dtype=np.float32)
x_t[:, :, 0] = x_init.astype(np.float32, copy=False)
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
    # Stage constants on GPU
    X_tar_gpu = cp.asarray(X_tar.astype(np.float32, copy=False))
    p_tar_gpu = cp.asarray(p_tar.astype(np.float32, copy=False))
    sq_tar_gpu = cp.asarray(sq_tar.astype(np.float32, copy=False))
    D_gpu = cp.asarray(D.astype(np.float32, copy=False))
    phi_gpu = cp.asarray(phi[:, :above_tol].astype(np.float32, copy=False))
    lambda_ns_s_ns_gpu = cp.asarray(lambda_ns_s_ns.astype(np.float32, copy=False))
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

# --- Scatter matrix X_tar
pd.plotting.scatter_matrix(
    pd.DataFrame(X_tar),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle('Scatter Matrix of X_tar')
_tp = _print_phase("Plot scatter matrix X_tar", _tp)
_advance("scatter_X_tar")

# --- Scatter matrix final x_t
pd.plotting.scatter_matrix(
    pd.DataFrame(x_t[:, :, -1]),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle('Scatter Matrix of x_t (final)')
_tp = _print_phase("Plot scatter matrix x_t_final", _tp)
_advance("scatter_x_t_final")

# Finish plotting progress bar line
print()
print(f"[TIMER] Plotting total: {time.time() - plot_total_start:.3f}s")

# ---------------- KL Divergence Report (diagonal Gaussian approximation) ----------------
# We approximate target and final particle sets by diagonal Gaussians.
# KL(P||Q) for diagonal Gaussians:
#   KL = 0.5 * sum_j [ (var_P_j / var_Q_j) + ( (mu_Q_j - mu_P_j)^2 / var_Q_j ) - 1 + log(var_Q_j / var_P_j) ]
# We print KL(target||final), KL(final||target), and symmetric average.
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

plt.show()
