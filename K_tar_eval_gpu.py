import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None  # type: ignore


def _xp_from(*arrays):
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):
                return cp
    return np


def K_tar_eval(x_tar, x_eval, p_tar, sq_tar, D, epsilon):
    """
    Python translation of K_tar_eval.m (xp-aware NumPy/CuPy)
    x_tar: (n, d)
    x_eval: (m, d)
    p_tar: (n,)
    sq_tar: (n,)
    D: (n,)
    epsilon: scalar bandwidth
    Returns:
        K: (n, m)
    """
    xp = _xp_from(x_tar, x_eval)
    x_tar = xp.asarray(x_tar)
    x_eval = xp.asarray(x_eval)
    p_tar = xp.asarray(p_tar).reshape(-1)
    sq_tar = xp.asarray(sq_tar).reshape(-1)
    D = xp.asarray(D).reshape(-1)
    eps = float(epsilon)
    sq_x_eval = xp.sum(x_eval ** 2, axis=1)
    K_cross = xp.exp(-(sq_x_eval[None, :] + sq_tar[:, None] - 2 * (x_tar @ x_eval.T)) / (2 * eps))
    p_eval = xp.sum(K_cross, axis=0) + 1e-12
    M = K_cross / xp.sqrt(p_eval)[None, :] / xp.sqrt(p_tar + 1e-12)[:, None]
    K1 = M / (D[:, None] + 1e-12)
    K2 = M / (xp.sum(M, axis=0, keepdims=True) + 1e-12)
    K = 0.5 * (K2 + K1)
    return K
