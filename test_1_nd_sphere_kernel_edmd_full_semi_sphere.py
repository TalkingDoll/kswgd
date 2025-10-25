# ============================================================
# Kernel EDMD + LAWGD Implementation
# ============================================================
# This script implements Kernel Extended Dynamic Mode Decomposition (EDMD)
# combined with Langevin-Adjusted Wasserstein Gradient Descent (LAWGD).
#
# Key differences from standard DMPS:
# 1. Uses Kernel-EDMD to learn the Koopman operator from time-evolved data pairs
# 2. X_tar_next is generated via manifold-constrained Langevin dynamics:
#    - KDE-based score estimation: ∇log π(x)
#    - Tangent space projection for manifold constraint
#    - Euler-Maruyama update with diffusion coefficient √2
#    - Renormalization to stay on the sphere
#    - Reflecting boundary (for semi-circle) to keep data in physical domain
# 3. Koopman operator: K = K_xy @ (K_xx + γI)^{-1}
# 4. Subsequent DMPS-style normalization and spectral decomposition
# 5. LAWGD particle transport: NO artificial boundary (data-driven learning)
#
# ============================================================
# Boundary Conditions Philosophy
# ============================================================
# CRITICAL DISTINCTION:
#
# 1. DATA GENERATION (X_tar, X_tar_next):
#    - Represents the TRUE PHYSICAL SYSTEM
#    - For semi-circle: Apply reflecting boundary (y < 0 → y > 0)
#    - Reason: The real system HAS this boundary constraint
#    - This is NOT artificial - it's the ground truth dynamics
#
# 2. LAWGD PARTICLE TRANSPORT (x_t iteration):
#    - DATA-DRIVEN LEARNING process
#    - NO artificial boundary operations
#    - Reason: We want the Koopman operator to LEARN the boundary behavior
#    - If Koopman learns correctly, particles will naturally stay in domain
#    - Adding artificial boundaries would defeat the purpose of learning
#
# Philosophy: The algorithm should learn FROM data (which has boundaries),
# not be artificially constrained DURING learning (which is cheating).
# ============================================================

import numpy as np
from scipy.linalg import svd, eigh
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

print(f"[DEVICE] {'GPU' if USE_GPU else 'CPU'} mode active")

# ---------------- Timing / Progress Utilities ----------------
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
_t = time.time()

# ---------------- Configuration ----------------
USE_SEMICIRCLE = True  # Set False for full circle, True for semi-circle (upper half)
KERNEL_TYPE = 5  # 1: RBF, 2: Spherical, 3: Matérn, 4: Rational Quadratic, 5: Polynomial
# ⚠️ CHANGED: Using Spherical kernel to avoid symmetric bias from RBF kernel

# Sample 500 points from a circle or semi-circle
n = 500
d = 2
lambda_ = 1  # Anisotropy parameter (stretches along x-axis)

if USE_SEMICIRCLE:
    # ============================================================
    # METHOD 1: Rejection Sampling (same basis as full sphere)
    # ============================================================
    # This method uses the same Gaussian sampling + normalization as the full circle,
    # but filters to keep only points in the upper semi-circle (y >= 0).
    # Pros: Maintains same statistical properties as full circle method
    # Cons: Less efficient (rejects ~50% of samples)
    
    samples = []
    while len(samples) < n:
        # Over-sample to reduce number of rejection rounds
        batch_size = max(n - len(samples), int((n - len(samples)) * 2.5))
        
        # Gaussian sampling (same as full circle)
        u = np.random.normal(0, 1, (batch_size, d))
        u[:, 0] = lambda_ * u[:, 0]  # Apply anisotropy
        u_norm = np.linalg.norm(u, axis=1, keepdims=True)
        u_trans = u / (u_norm + 1e-12)
        
        # Rejection step: keep only upper semi-circle points
        valid = u_trans[:, 1] >= 0  # Filter for y >= 0
        samples.append(u_trans[valid, :])
    
    u_trans = np.vstack(samples)[:n, :]  # Take exactly n samples
    label = "semi-circle"
    
    # ============================================================
    # METHOD 2: Polar Coordinate Sampling 
    # ============================================================
    # This method directly samples angles in [0, π] and converts to Cartesian.
    # Pros: More efficient, more intuitive, flexible angle range
    # Cons: Different sampling method from full circle (not Gaussian-based)
    
    # theta = np.random.uniform(0, np.pi, (n, 1))
    # u_trans = np.hstack([
    #     np.cos(theta),  # x = cos(θ)
    #     np.sin(theta)   # y = sin(θ), ensures y >= 0
    # ])
    # 
    # # Apply anisotropy (stretch along x-axis)
    # u_trans[:, 0] = lambda_ * u_trans[:, 0]
    # u_norm = np.linalg.norm(u_trans, axis=1, keepdims=True)
    # u_trans = u_trans / (u_norm + 1e-12)
    # 
    # label = "semi-circle"
    
else:
    # ============================================================
    # METHOD 1: Gaussian Sampling (Marsaglia Method)
    # ============================================================
    # This method uses Gaussian sampling + normalization.
    # Pros: Generalizes to higher dimensions (3D, 4D, etc.)
    # Cons: Less intuitive, no explicit angle representation
    
    u = np.random.normal(0, 1, (n, d))
    u[:, 0] = lambda_ * u[:, 0]  # Apply anisotropy
    u_norm = np.linalg.norm(u, axis=1, keepdims=True)
    u_trans = u / (u_norm + 1e-12)
    
    label = "full circle"

    # ============================================================
    # METHOD 2: Polar Coordinate Sampling (RECOMMENDED)
    # ============================================================
    # Directly sample angles uniformly in [0, 2π] and convert to Cartesian.
    # Pros: Intuitive, efficient, explicitly angle-based
    # Cons: None for 2D case
    
    # theta = np.random.uniform(0, 2 * np.pi, (n, 1))
    # u_trans = np.hstack([
    #     np.cos(theta),  # x = cos(θ)
    #     np.sin(theta)   # y = sin(θ)
    # ])
    
    # # Apply anisotropy (stretch along x-axis) - Optional
    # # WARNING: This breaks the uniform angular distribution!
    # # After stretching and re-normalizing, points cluster near y-axis when lambda_ > 1
    # if lambda_ != 1:
    #     u_trans[:, 0] = lambda_ * u_trans[:, 0]
    #     u_norm = np.linalg.norm(u_trans, axis=1, keepdims=True)
    #     u_trans = u_trans / (u_norm + 1e-12)
    #     label = f"full circle (anisotropic λ={lambda_})"
    # else:
    #     label = "full circle (uniform)"

# Radial distance: slightly randomized around radius 1 (creates a thin annulus)
r = np.sqrt(np.random.rand(n, 1)) * 1/100 + 99/100
X_tar = r * u_trans
# _t = _print_phase(f"Target sample generation ({label})", _t)
n = X_tar.shape[0]

# ============================================================
# Kernel EDMD: Generate X_tar_next via manifold Langevin dynamics
# ============================================================
# ⚠️ CRITICAL FIX: Increase dt_edmd and noise to generate MORE boundary crossings
# This ensures Koopman operator learns boundary behavior!
# Previous: dt_edmd=1e-3 → only 0.4% particles crossed → insufficient training
# New: dt_edmd=5e-2 → expect 15-30% particles to cross → better boundary learning

# Step 1: KDE-based score estimation (drift term)
dt_edmd = 5e-2  # Sampling time
diffs_edmd = X_tar[:, None, :] - X_tar[None, :, :]  # Pairwise differences
dist2_edmd = np.sum(diffs_edmd ** 2, axis=2)  # Squared distances
h_edmd = np.sqrt(np.median(dist2_edmd) + 1e-12)  # KDE bandwidth
W_edmd = np.exp(-dist2_edmd / (2.0 * (h_edmd ** 2)))  # Gaussian weights
sumW_edmd = np.sum(W_edmd, axis=1, keepdims=True) + 1e-12
weighted_means_edmd = (W_edmd @ X_tar) / sumW_edmd  # Weighted average
score_eucl = (weighted_means_edmd - X_tar) / (h_edmd ** 2)  # Score function ∇log π(x)

# Step 2: Project to tangent space (manifold constraint)
X_norm = X_tar / (np.linalg.norm(X_tar, axis=1, keepdims=True) + 1e-12)
# Projection matrix: P = I - n⊗n^T (removes normal component)
proj = np.eye(X_tar.shape[1])[None, :, :] - X_norm[:, :, None] * X_norm[:, None, :]
# ============================================================
# SDE Interpretation: Choose ONE of the following options
# ============================================================

# Option 1: Itô SDE (original, no correction needed)
# - Interpretation: dX = P·∇log(π)dt + √2·P·dW (Itô form)
# - Discretization: Euler-Maruyama (consistent with Itô calculus)
# - No geometric correction term needed
score_tan = np.einsum('nij,ni->nj', proj, score_eucl)

# Option 2: Stratonovich SDE with Itô-Stratonovich correction
# - Interpretation: dX = P·∇log(π)dt + √2·P∘dW (Stratonovich form)
# - Since we use Euler-Maruyama (Itô discretization), need correction term
# - Correction: -(d-1)/2·n accounts for the difference between Itô and Stratonovich
# - For unit sphere S^(d-1): Stratonovich drift = Itô drift - (1/2)∇·(σσᵀ)
# geometric_drift = -(d - 1) / 2 * X_norm  # Shape: (n, d), Itô-Stratonovich correction
# score_tan = np.einsum('nij,ni->nj', proj, score_eucl) + dt_edmd * np.einsum('nij,ni->nj', proj, geometric_drift)

# ============================================================

# Step 3: Langevin update with manifold-projected noise
# INCREASED noise multiplier from sqrt(2*dt) to 3*sqrt(2*dt) for better boundary exploration
noise_multiplier = 3.0  # Amplify noise to ensure particles explore boundary region
xi_edmd = np.random.normal(0.0, 1.0, size=X_tar.shape)  # Gaussian noise
xi_tan = xi_edmd - (np.sum(X_norm * xi_edmd, axis=1, keepdims=True)) * X_norm  # Project noise
X_step = X_norm + dt_edmd * score_tan + noise_multiplier * np.sqrt(2.0 * dt_edmd) * xi_tan  # Euler-Maruyama

# Step 4: Renormalize to stay on the sphere
X_tar_next = X_step / (np.linalg.norm(X_step, axis=1, keepdims=True) + 1e-12)

# Step 5: Apply reflecting boundary for semi-circle (CRITICAL FIX!)
# ⚠️ IMPORTANT: This MUST be done BEFORE computing K_xy kernel matrix!
# Otherwise K_xy will contain information about the lower semi-circle.
if USE_SEMICIRCLE:
    # Check which particles crossed the boundary (y < 0)
    crossed_boundary = X_tar_next[:, 1] < 0
    num_crossed = np.sum(crossed_boundary)
    
    if num_crossed > 0:
        # Reflecting boundary: mirror the y-component
        X_tar_next[crossed_boundary, 1] = -X_tar_next[crossed_boundary, 1]
        print(f"[BOUNDARY FIX] Applied reflection to {num_crossed}/{n} points ({100*num_crossed/n:.1f}%) in X_tar_next")
        print(f"[BOUNDARY INFO] This teaches Koopman operator about boundary dynamics!")
        
        # Optional: Enhance boundary learning by duplicating near-boundary samples
        # This gives Koopman more training data about boundary behavior
        ENHANCE_BOUNDARY_LEARNING = False  # Set True to enable
        if ENHANCE_BOUNDARY_LEARNING:
            # Find points very close to boundary (y < 0.05)
            near_boundary = X_tar_next[:, 1] < 0.05
            num_near = np.sum(near_boundary)
            if num_near > 0:
                print(f"[BOUNDARY ENHANCE] Found {num_near} points near boundary - enhancing their representation")
    else:
        print(f"[BOUNDARY WARNING] No particles crossed boundary! Koopman won't learn boundary behavior.")
        print(f"[BOUNDARY HINT] Consider increasing dt_edmd or noise_multiplier")

# Verify X_tar_next is strictly on semi-circle after reflection
if USE_SEMICIRCLE:
    min_y = np.min(X_tar_next[:, 1])
    if min_y < -1e-10:  # Small tolerance for numerical errors
        print(f"[WARNING] X_tar_next still has points below y=0! min_y = {min_y:.6e}")
    else:
        print(f"[VERIFIED] X_tar_next strictly on semi-circle (min_y = {min_y:.6e})")

# _t = _print_phase("Kernel-EDMD: X_tar_next generation (manifold Langevin)", _t)

# Quick visualization: X_tar vs X_tar_next
# Get kernel name for display
kernel_names = {1: "RBF", 2: "Spherical", 3: "Matérn", 4: "Rational Quadratic"}
kernel_display = kernel_names.get(KERNEL_TYPE, "Unknown")

if d == 2:
    plt.figure(figsize=(8, 8))
    plt.scatter(X_tar[:, 0], X_tar[:, 1], s=10, c='C0', alpha=0.6, label='X_tar (current)')
    plt.scatter(X_tar_next[:, 0], X_tar_next[:, 1], s=10, c='C1', alpha=0.6, label='X_tar_next (evolved)')
    plt.legend()
    plt.axis('equal')
    plt.title(f'Kernel-EDMD: Time Evolution Pair\nKernel: {kernel_display}')
    plt.grid(True, alpha=0.3)
    plt.show(block=False)  # Non-blocking display

# ============================================================
# Step 5: Kernel Selection for Kernel EDMD
# ============================================================
# Note: KERNEL_TYPE is configured at the top of the file (Configuration section)
# CRITICAL: K_xy is computed HERE using the CORRECTED X_tar_next (after reflection)

# Compute bandwidth parameter (used by most kernels)
sq_tar = np.sum(X_tar ** 2, axis=1)
H = sq_tar[:, None] + sq_tar[None, :] - 2 * (X_tar @ X_tar.T)
epsilon = 0.5 * np.median(H) / (np.log(n + 1) + 1e-12)
length_scale = np.sqrt(np.median(H))  # Alternative scale parameter

# ============================================================
# Kernel Definitions
# ============================================================

def kernel1_rbf(X, Y, eps):
    """
    Kernel 1: RBF/Gaussian Kernel
    k(x,y) = exp(-||x-y||²/(2ε))
    
    Pros: Smooth, universal approximator
    Cons: Sensitive to bandwidth choice
    """
    sq_x = np.sum(X ** 2, axis=1)
    sq_y = np.sum(Y ** 2, axis=1)
    D2 = sq_x[:, None] + sq_y[None, :] - 2 * (X @ Y.T)
    return np.exp(-D2 / (2 * eps))

def kernel2_spherical(X, Y, theta_scale=1.0):
    """
    Kernel 2: Spherical/Geodesic Kernel
    k(x,y) = exp(-d_geodesic(x,y)² / (2·θ²))
    
    Uses geodesic distance on the manifold: d = arccos(x·y^T)
    
    Pros: Respects manifold geometry, ideal for circle/sphere data
    Cons: Requires normalized inputs
    """
    # Ensure normalized to unit sphere
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    
    # Cosine similarity
    cos_sim = X_norm @ Y_norm.T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # Numerical stability
    
    # Geodesic distance
    geodesic_dist = np.arccos(cos_sim)
    
    return np.exp(-geodesic_dist**2 / (2 * theta_scale**2))

def kernel3_matern(X, Y, length_scale=1.0, nu=1.5):
    """
    Kernel 3: Matérn Kernel (ν=1.5, once differentiable)
    k(x,y) = (1 + √3·d/ℓ) · exp(-√3·d/ℓ)
    
    Pros: Better generalization than RBF, controls smoothness
    Cons: Slightly more expensive to compute
    """
    sq_x = np.sum(X ** 2, axis=1)
    sq_y = np.sum(Y ** 2, axis=1)
    D2 = sq_x[:, None] + sq_y[None, :] - 2 * (X @ Y.T)
    D = np.sqrt(np.maximum(D2, 0))  # Euclidean distance
    
    if nu == 1.5:
        # Once differentiable case (most common)
        sqrt3_D = np.sqrt(3) * D / length_scale
        K = (1 + sqrt3_D) * np.exp(-sqrt3_D)
    elif nu == 2.5:
        # Twice differentiable case
        sqrt5_D = np.sqrt(5) * D / length_scale
        K = (1 + sqrt5_D + sqrt5_D**2 / 3) * np.exp(-sqrt5_D)
    else:
        # Fall back to exponential kernel for nu=0.5
        K = np.exp(-D / length_scale)
    
    return K

def kernel4_rational_quadratic(X, Y, alpha=1.0, length_scale=1.0):
    """
    Kernel 4: Rational Quadratic Kernel
    k(x,y) = (1 + ||x-y||²/(2α·ℓ²))^(-α)
    
    Equivalent to infinite mixture of RBF kernels with different length scales
    α → ∞: converges to RBF kernel
    
    Pros: Captures multi-scale features
    Cons: More parameters to tune
    """
    sq_x = np.sum(X ** 2, axis=1)
    sq_y = np.sum(Y ** 2, axis=1)
    D2 = sq_x[:, None] + sq_y[None, :] - 2 * (X @ Y.T)
    return (1 + D2 / (2 * alpha * length_scale**2)) ** (-alpha)

def kernel5_polynomial(X, Y, degree=3, coef0=1.0, gamma=None):
    """
    Kernel 5: Polynomial Kernel
    k(x,y) = (γ·⟨x,y⟩ + c₀)^d
    
    Parameters:
    - degree (d): Polynomial degree (typically 2-5)
    - coef0 (c₀): Independent term (typically 0 or 1)
    - gamma (γ): Scaling factor (if None, defaults to 1/d)
    
    Special cases:
    - d=1, c₀=0: Linear kernel
    - d=2: Quadratic kernel (captures pairwise feature interactions)
    - d=3: Cubic kernel (common choice)
    
    Pros: 
    - Computationally efficient (only inner products)
    - Explicit feature interactions up to degree d
    - Works well for data with polynomial structure
    - Non-stationary (not translation-invariant)
    
    Cons:
    - Sensitive to degree choice
    - Can grow unbounded for large inner products
    - Less smooth than RBF for high degrees
    - May have numerical issues if γ or c₀ are poorly chosen
    
    Note: For manifold data (unit sphere), ⟨x,y⟩ ∈ [-1,1], so kernel is bounded.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]  # Default: 1/dimension
    
    # Inner product matrix
    inner_prod = X @ Y.T
    
    # Polynomial kernel
    K = (gamma * inner_prod + coef0) ** degree
    
    return K

# ============================================================
# Select and Apply Kernel
# ============================================================
if KERNEL_TYPE == 1:
    kernel_name = "RBF"
    K_xx = kernel1_rbf(X_tar, X_tar, epsilon)
    K_xy = kernel1_rbf(X_tar, X_tar_next, epsilon)
elif KERNEL_TYPE == 2:
    kernel_name = "Spherical"
    theta_scale = 0.3  # Tune this parameter (0.1 to 1.0) - reduced for better locality
    K_xx = kernel2_spherical(X_tar, X_tar, theta_scale=theta_scale)
    K_xy = kernel2_spherical(X_tar, X_tar_next, theta_scale=theta_scale)
elif KERNEL_TYPE == 3:
    kernel_name = "Matérn"
    K_xx = kernel3_matern(X_tar, X_tar, length_scale=length_scale, nu=1.5)
    K_xy = kernel3_matern(X_tar, X_tar_next, length_scale=length_scale, nu=1.5)
elif KERNEL_TYPE == 4:
    kernel_name = "Rational Quadratic"
    alpha = 2.0  # Tune this parameter (1.0 to 5.0)
    K_xx = kernel4_rational_quadratic(X_tar, X_tar, alpha=alpha, length_scale=length_scale)
    K_xy = kernel4_rational_quadratic(X_tar, X_tar_next, alpha=alpha, length_scale=length_scale)
elif KERNEL_TYPE == 5:
    kernel_name = "Polynomial"
    poly_degree = 10  # Polynomial degree (typically 2-5)
    poly_coef0 = 1.0  # Independent term (0 or 1)
    poly_gamma = 1.0 / d  # Scaling factor (1/dimension)
    K_xx = kernel5_polynomial(X_tar, X_tar, degree=poly_degree, coef0=poly_coef0, gamma=poly_gamma)
    K_xy = kernel5_polynomial(X_tar, X_tar_next, degree=poly_degree, coef0=poly_coef0, gamma=poly_gamma)
else:
    raise ValueError(f"Invalid KERNEL_TYPE: {KERNEL_TYPE}. Choose 1, 2, 3, 4, or 5.")

print(f"[KERNEL] Using {kernel_name} kernel (Type {KERNEL_TYPE})")
_t = _print_phase(f"Kernel-EDMD: Gram matrices K_xx and K_xy ({kernel_name})", _t)

# Step 6: Compute Koopman operator via Kernel EDMD
# Formula: K_koopman = K_xy @ (K_xx + γI)^{-1}
# This approximates the Koopman operator for the dynamics X_t → X_{t+dt}
gamma_ridge = 1e-6  # Tikhonov regularization

# Use eigendecomposition for stable inversion: (K_xx + γI)^{-1} = Q @ diag(1/(λ+γ)) @ Q^T
evals_kxx, Q_kxx = eigh(K_xx)
evals_kxx = np.clip(evals_kxx, 0.0, None)  # Ensure non-negative
inv_evals = 1.0 / (evals_kxx + gamma_ridge)
# K_koopman = K_xy @ Q @ diag(inv_evals) @ Q^T
data_kernel = K_xy @ (Q_kxx @ (np.diag(inv_evals) @ Q_kxx.T))

# Safety: ensure finite values
data_kernel = np.nan_to_num(data_kernel, nan=0.0, posinf=0.0, neginf=0.0)
minK = float(np.min(data_kernel))
if minK < 0.0:
    data_kernel = data_kernel - minK + 1e-12  # Shift to non-negative
_t = _print_phase("Kernel-EDMD: Koopman operator K_xy @ (K_xx + γI)^{-1}", _t)

# ============================================================
# DMPS-style normalization (applied to Koopman operator)
# ============================================================
# Note: sq_tar is needed later for grad_ker1 and K_tar_eval functions
sq_tar = np.sum(X_tar ** 2, axis=1)  # Keep for downstream gradient computations

p_x = np.sqrt(np.sum(data_kernel, axis=1))
p_y = p_x.copy()
# Normalize kernel
data_kernel_norm = data_kernel / p_x[:, None] / p_y[None, :]
D_y = np.sum(data_kernel_norm, axis=0)

# Match MATLAB: 0.5*(A ./ D_y + A ./ D_y') where D_y is a row vector.
# First term divides columns by D_y (broadcast over last axis),
# second term explicitly divides rows by D_y (reshape as column vector).
rw_kernel = 0.5 * (data_kernel_norm / D_y + data_kernel_norm / D_y[:, None])
# _t = _print_phase("Random-walk symmetric normalization", _t)

# ============================================================
# Spectral Decomposition: Choose ONE method below
# ============================================================

# # --- Method 1: SVD (Original DMPS) ---
# phi, s, _ = svd(rw_kernel)
# _t = _print_phase("SVD on rw_kernel", _t)

# --- Method 2: Eigendecomposition (faster for symmetric matrices) ---
# Note: eigh returns eigenvalues in ascending order, need to reverse for consistency with SVD
evals_rw, evecs_rw = eigh(rw_kernel)
# Reverse to get descending order (largest eigenvalue first, like SVD)
phi = evecs_rw[:, ::-1]
s = evals_rw[::-1]

# Truncate eigenvalues: set values below threshold to zero
# This handles both negative eigenvalues (numerical errors) and near-zero eigenvalues (ill-conditioning)
tol_truncate = 1e-6
s_original = s.copy()  # Keep for analysis
s = np.where(s < tol_truncate, 0.0, s)
num_truncated = np.sum(s_original < tol_truncate)
print(f"[INFO] Truncated {num_truncated} eigenvalues < {tol_truncate:.0e} to zero")
_t = _print_phase("Eigendecomposition on rw_kernel", _t)

# ============================================================

lambda_ns = s

# ============================================================
# Eigenvalue Analysis
# ============================================================
print("\n[EIGENVALUE ANALYSIS]")
print(f"Total eigenvalues: {len(lambda_ns)}")
print(f"Largest eigenvalue: {lambda_ns[0]:.6f}")
print(f"Smallest eigenvalue (after clipping): {lambda_ns[-1]:.6e}")

# Count eigenvalues > 1
count_gt_1 = np.sum(lambda_ns > 1.0)
print(f"Eigenvalues > 1.0: {count_gt_1}")

# Count eigenvalues < 0 (should be 0 after clipping)
count_lt_0 = np.sum(lambda_ns < 0.0)
print(f"Eigenvalues < 0.0 (after clipping): {count_lt_0}")

# Count eigenvalues close to 0 (0 <= λ < 1e-6)
tol_small = 1e-6
count_near_0 = np.sum((lambda_ns >= 0.0) & (lambda_ns < tol_small))
print(f"Eigenvalues in [0, {tol_small:.0e}): {count_near_0}")

# Count eigenvalues in different ranges
count_tiny_to_1 = np.sum((lambda_ns >= tol_small) & (lambda_ns <= 1.0))
print(f"Eigenvalues in [{tol_small:.0e}, 1.0]: {count_tiny_to_1}")

# Show first few eigenvalues
print(f"\nFirst 10 eigenvalues:")
for i in range(min(10, len(lambda_ns))):
    print(f"  λ[{i}] = {lambda_ns[i]:.8f}")
print("=" * 60 + "\n")
# ============================================================

lambda_ = -lambda_ns + 1
inv_lambda = np.zeros_like(lambda_)
inv_lambda[1:] = 1 / lambda_[1:]
inv_lambda = inv_lambda * epsilon
inv_K = phi @ np.diag(inv_lambda) @ phi.T
# _t = _print_phase("Primary inverse-like weights (inv_lambda)", _t)

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
# _t = _print_phase("Regularized inverse weights (lambda_ns_inv)", _t)

# Run algorithm
iter = 1000
h = 2  # Reduced step size for better stability
m = 700
u = np.random.normal(0, 1, (m, d))
u_norm = np.linalg.norm(u, axis=1, keepdims=True)
r = np.sqrt(np.random.rand(m, 1)) * 1/100 + 99/100
u_trans = u / u_norm
x_init = r * u_trans

# Particle initialization strategy based on mode
if USE_SEMICIRCLE:
    # Semi-circle: only upper hemisphere (y > 0.9)
    x_init = x_init[x_init[:, 1] > 0.9, :]
else:
    # Full circle: both upper (y > 0.9) AND lower (y < -0.9) hemispheres
    x_init = x_init[x_init[:, 1] > 0.95, :]

m = x_init.shape[0]
print(f"[INFO] Initialized {m} particles ({'semi-circle' if USE_SEMICIRCLE else 'full circle'} mode)")
x_t = np.zeros((m, d, iter), dtype=np.float64)  # Use float64 for precision
x_t[:, :, 0] = x_init
# _t = _print_phase("Particle initialization", _t)

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
    
    # Iteration loop (GPU) - Data-driven, no artificial boundary
    for t in range(iter - 1):
        x_slice = x_t_gpu[:, :, t]
        grad_matrix = grad_ker1(x_slice, X_tar_gpu, p_tar_gpu, sq_tar_gpu, D_gpu, epsilon)
        cross_matrix = K_tar_eval(X_tar_gpu, x_slice, p_tar_gpu, sq_tar_gpu, D_gpu, epsilon)
        
        # Original full-sphere algorithm (per-dimension loop)
        sum_x_gpu = cp.zeros((m, d))
        for i in range(d):
            sum_x_gpu[:, i] = cp.sum(
                grad_matrix[:, :, i] @ phi_gpu @ diag_lambda_gpu @ phi_gpu.T @ cross_matrix,
                axis=1
            )
        
        x_proposed = x_slice - (h / m) * sum_x_gpu
        
        # Normalize to unit circle (no artificial reflection - let Koopman operator handle it)
        x_norm_proposed = cp.sqrt(cp.sum(x_proposed ** 2, axis=1, keepdims=True))
        x_t_gpu[:, :, t + 1] = x_proposed / (x_norm_proposed + 1e-12)
        
        done = t + 1
        if done == total_loop or (done % max(1, total_loop // 100) == 0):
            _print_progress(done, total_loop, loop_start, prefix="[Kernel-EDMD-GPU] ")
    
    x_t = cp.asnumpy(x_t_gpu)
else:
    # CPU iteration loop - Data-driven, no artificial boundary
    for t in range(iter - 1):
        grad_matrix = grad_ker1(x_t[:, :, t], X_tar, p_tar, sq_tar, D, epsilon)
        cross_matrix = K_tar_eval(X_tar, x_t[:, :, t], p_tar, sq_tar, D, epsilon)
        for i in range(d):
            sum_x[:, i] = np.sum(
                grad_matrix[:, :, i] @ phi[:, :above_tol] @ np.diag(lambda_ns_s_ns) @ phi[:, :above_tol].T @ cross_matrix,
                axis=1
            )
        
        x_proposed = x_t[:, :, t] - h / m * sum_x
        
        # Normalize to unit circle (no artificial reflection - let Koopman operator handle it)
        x_norm_proposed = np.sqrt(np.sum(x_proposed ** 2, axis=1, keepdims=True))
        x_t[:, :, t + 1] = x_proposed / (x_norm_proposed + 1e-12)
        
        done = t + 1
        if done == total_loop or (done % max(1, total_loop // 100) == 0):
            _print_progress(done, total_loop, loop_start, prefix="[Kernel-EDMD-CPU] ")

print()
_t = _print_phase("Iteration loop total", loop_start)

# Plotting results
if d == 2:
    plt.figure(figsize=(8, 8))
    plt.plot(X_tar[:, 0], X_tar[:, 1], 'o', markersize=4, alpha=0.6, label='Target')
    plt.plot(x_t[:, 0, 0], x_t[:, 1, 0], 'o', markersize=8, color='red', label='Init')  # Red solid circles
    plt.plot(x_t[:, 0, -1], x_t[:, 1, -1], 'o', markersize=10, markerfacecolor='none', markeredgecolor='magenta', markeredgewidth=1.2, alpha=0.6, label='Final')  # Green hollow circles
    plt.legend(fontsize=12)
    plt.title(f'2D Results - Kernel EDMD with {kernel_name}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_tar[:, 0], X_tar[:, 1], X_tar[:, 2], s=20, alpha=0.6, label='Target')
    ax.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], s=50, marker='o', color='red', label='Init')  # Red solid circles
    ax.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], s=100, marker='o', facecolors='none', edgecolors='green', linewidths=0.8, alpha=0.5, label='Final')  # Green hollow circles
    ax.legend(fontsize=12)
    plt.title(f'3D Results - Kernel EDMD with {kernel_name}')
    plt.show()
    
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(x_t[:, 0, 0], x_t[:, 1, 0], x_t[:, 2, 0], s=50, marker='o', color='red', label='Init')  # Red solid circles
    ax2.scatter(x_t[:, 0, -1], x_t[:, 1, -1], x_t[:, 2, -1], s=100, marker='o', facecolors='none', edgecolors='green', linewidths=0.8, alpha=0.5, label='Final')  # Green hollow circles
    ax2.legend(fontsize=12)
    plt.title(f'3D Final State - Kernel EDMD with {kernel_name}')
    plt.show()

# Plot matrix (scatter matrix)
pd.plotting.scatter_matrix(
    pd.DataFrame(X_tar),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle(f'Scatter Matrix of X_tar - {kernel_name} Kernel')
plt.show()

pd.plotting.scatter_matrix(
    pd.DataFrame(x_t[:, :, -1]),
    alpha=0.2,
    figsize=(6, 6),
    diagonal='hist',
    hist_kwds={'edgecolor': 'black'}
)
plt.suptitle(f'Scatter Matrix of x_t (final) - {kernel_name} Kernel')
plt.show()
