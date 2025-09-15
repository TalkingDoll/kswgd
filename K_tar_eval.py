import numpy as np

def K_tar_eval(x_tar, x_eval, p_tar, sq_tar, D, epsilon):
    """
    Python translation of K_tar_eval.m
    x_tar: (n, d) target points
    x_eval: (m, d) evaluation points
    p_tar: (n,) target density (should be 1D)
    sq_tar: (n,) squared norms of X_tar
    D: (n,) normalization vector
    epsilon: scalar bandwidth
    Returns:
        K: (n, m) kernel matrix
    """
    sq_x_eval = np.sum(x_eval ** 2, axis=1)
    K_cross = np.exp(-(sq_x_eval[None, :] + sq_tar[:, None] - 2 * x_tar @ x_eval.T) / (2 * epsilon))
    p_eval = np.sum(K_cross, axis=0)
    M = K_cross / np.sqrt(p_eval)[None, :] / np.sqrt(p_tar)[:, None]
    K1 = M / D[:, None]
    K2 = M / np.sum(M, axis=0, keepdims=True)
    K = 0.5 * (K2 + K1)
    return K
