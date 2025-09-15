import numpy as np

def grad_ker1(x_eval, x_tar, p_tar, sq_tar, D2, epsilon):
    """
    Python translation of grad_ker1.m
    x_eval: (m, d) particles to evaluate
    x_tar: (n, d) target points
    p_tar: (n,) target density (should be 1D)
    sq_tar: (n,) squared norms of X_tar
    D2: (n,) normalization vector
    epsilon: scalar bandwidth
    Returns:
        g: (m, n, d) gradient tensor
    """
    m, d = x_eval.shape
    n = x_tar.shape[0]
    sq_x_eval = np.sum(x_eval ** 2, axis=1)
    p_tar = p_tar.reshape(-1)
    K_cross = np.exp(-(sq_x_eval[:, None] + sq_tar[None, :] - 2 * x_eval @ x_tar.T) / (2 * epsilon))
    p_eval = np.sum(K_cross, axis=1)
    M = K_cross / np.sqrt(p_eval)[:, None] / np.sqrt(p_tar)[None, :]
    D = np.sum(M, axis=1)
    g = np.zeros((m, n, d))
    for i in range(d):
        d_K_cross = (-(x_eval[:, i][:, None] - x_tar[:, i][None, :]) / epsilon) * K_cross
        d_sqrt_p_eval = 1.0 / (2 * np.sqrt(p_eval)) * np.sum(d_K_cross, axis=1)
        dM = (d_K_cross * np.sqrt(p_eval)[:, None] - K_cross * d_sqrt_p_eval[:, None]) / np.sqrt(p_tar)[None, :] / p_eval[:, None]
        dD = np.sum(dM, axis=1)
        g1 = (dM * D[:, None] - dD[:, None] * M) / (D[:, None] ** 2) / epsilon
        g2 = dM / D2[None, :] / epsilon
        g[:, :, i] = 0.5 * (g1 + g2)
    return g
