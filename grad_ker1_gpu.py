import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # type: ignore


def _xp_from(*arrays):
    """Return numpy or cupy based on input array types."""
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def grad_ker1(x_eval, x_tar, p_tar, sq_tar, D2, epsilon):
    """
    Python translation of grad_ker1.m, vectorized and xp-aware (NumPy/CuPy).
    x_eval: (m, d)
    x_tar: (n, d)
    p_tar: (n,)
    sq_tar: (n,)
    D2: (n,)
    epsilon: scalar bandwidth
    Returns:
        g: (m, n, d)
    """
    xp = _xp_from(x_eval, x_tar)
    # Ensure all arrays are on the same backend
    x_eval = xp.asarray(x_eval)
    x_tar = xp.asarray(x_tar)
    p_tar = xp.asarray(p_tar).reshape(-1)
    sq_tar = xp.asarray(sq_tar).reshape(-1)
    D2 = xp.asarray(D2).reshape(-1)
    eps = float(epsilon)
    m, d = x_eval.shape
    # K_cross and normalizations
    sq_x_eval = xp.sum(x_eval ** 2, axis=1)
    K_cross = xp.exp(-(sq_x_eval[:, None] + sq_tar[None, :] - 2 * (x_eval @ x_tar.T)) / (2 * eps))
    p_eval = xp.sum(K_cross, axis=1) + 1e-12
    sqrt_p_eval = xp.sqrt(p_eval)
    sqrt_p_tar = xp.sqrt(p_tar) + 1e-12
    M = K_cross / sqrt_p_eval[:, None] / sqrt_p_tar[None, :]
    D = xp.sum(M, axis=1) + 1e-12
    # Vectorized gradient across all dimensions
    diff = x_eval[:, None, :] - x_tar[None, :, :]          # (m, n, d)
    d_K_cross = (-(diff / eps)) * K_cross[:, :, None]      # (m, n, d)
    d_sqrt_p_eval = (1.0 / (2.0 * sqrt_p_eval))[:, None] * xp.sum(d_K_cross, axis=1)  # (m, d)
    dM = (d_K_cross * sqrt_p_eval[:, None, None] - K_cross[:, :, None] * d_sqrt_p_eval[:, None, :]) \
         / sqrt_p_tar[None, :, None] / p_eval[:, None, None]
    dD = xp.sum(dM, axis=1)                                # (m, d)
    g1 = (dM * D[:, None, None] - dD[:, None, :] * M[:, :, None]) / (D[:, None, None] ** 2) / eps
    g2 = dM / (D2[None, :, None] + 1e-12) / eps
    g = 0.5 * (g1 + g2)
    return g
