from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from K_tar_eval_gpu import K_tar_eval
from grad_ker1_gpu import grad_ker1

CHECKPOINT_DIR = Path("checkpoints")

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - depends on the local GPU environment
    cp = None  # type: ignore


def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    X2 = np.sum(X * X, axis=1)[:, None]
    Y2 = np.sum(Y * Y, axis=1)[None, :]
    return np.maximum(X2 + Y2 - 2.0 * X @ Y.T, 0.0)


def median_bandwidth(X: np.ndarray, scale: float = 0.5) -> float:
    D2 = pairwise_sq_dists(X)
    vals = D2[np.triu_indices_from(D2, k=1)]
    vals = vals[vals > 1e-14]
    med = float(np.median(vals)) if len(vals) else 1.0
    return max(scale * med / np.log(len(X) + 1.0), 1e-10)


def gaussian_kernel(X: np.ndarray, Y: np.ndarray, epsilon: float) -> np.ndarray:
    return np.exp(-pairwise_sq_dists(X, Y) / (2.0 * float(epsilon)))


def named_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    kernel_type: int | str = 1,
    epsilon: float | None = None,
    length_scale: float | None = None,
    theta_scale: float = 0.3,
    matern_nu: float = 1.5,
    rq_alpha: float = 2.0,
    polynomial_degree: int = 10,
    polynomial_coef0: float = 1.0,
    polynomial_gamma: float | None = None,
) -> tuple[np.ndarray, str]:
    """Kernel choices used by the legacy nd-sphere notebook."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    key = str(kernel_type).lower()
    aliases = {
        "1": "rbf",
        "2": "spherical",
        "3": "matern",
        "4": "rational_quadratic",
        "5": "polynomial",
    }
    key = aliases.get(key, key.replace("-", "_").replace(" ", "_"))
    if key in {"rbf", "gaussian"}:
        if epsilon is None:
            raise ValueError("epsilon is required for the RBF kernel.")
        return gaussian_kernel(X, Y, epsilon), "RBF"
    if key == "spherical":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
        cos_sim = np.clip(Xn @ Yn.T, -1.0, 1.0)
        geodesic = np.arccos(cos_sim)
        return np.exp(-(geodesic**2) / (2.0 * theta_scale**2)), "Spherical"
    D2 = pairwise_sq_dists(X, Y)
    length_scale = float(length_scale if length_scale is not None else np.sqrt(np.median(D2) + 1e-12))
    if key == "matern":
        if abs(float(matern_nu) - 1.5) > 1e-12:
            raise ValueError("Only the legacy Matern nu=1.5 kernel is implemented.")
        D = np.sqrt(np.maximum(D2, 0.0))
        scaled = np.sqrt(3.0) * D / length_scale
        return (1.0 + scaled) * np.exp(-scaled), "Matern"
    if key == "rational_quadratic":
        return (1.0 + D2 / (2.0 * float(rq_alpha) * length_scale**2)) ** (-float(rq_alpha)), "Rational Quadratic"
    if key == "polynomial":
        gamma = 1.0 / X.shape[1] if polynomial_gamma is None else float(polynomial_gamma)
        return (gamma * (X @ Y.T) + float(polynomial_coef0)) ** int(polynomial_degree), "Polynomial"
    raise ValueError(f"Unknown kernel_type: {kernel_type!r}")


@dataclass
class SpectralModel:
    X_target: np.ndarray
    epsilon: float
    p_target: np.ndarray
    sq_target: np.ndarray
    density_norm: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    weights: np.ndarray
    kernel_matrix: np.ndarray
    method: str

    @property
    def n_target(self) -> int:
        return int(self.X_target.shape[0])


@dataclass
class TransportResult:
    final: np.ndarray
    snapshots: dict[int, np.ndarray]
    step_norms: np.ndarray


@dataclass
class MetricTransportResult:
    final: np.ndarray
    snapshots: dict[int, np.ndarray]
    step_norms: np.ndarray
    metrics: dict[str, np.ndarray]


@dataclass
class KernelSpectrum2D:
    X_target: np.ndarray
    epsilon: float
    p_target: np.ndarray
    density_norm: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    lambda_generator: np.ndarray
    method: str


@dataclass
class LatentKSWGDResult:
    final: np.ndarray
    snapshots: dict[int, np.ndarray]
    step_norms: np.ndarray
    metrics: dict[str, np.ndarray]
    trajectory: np.ndarray
    n_modes: int


def _safe_density_norm(kernel_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p_target = np.sum(kernel_matrix, axis=0) + 1e-12
    sqrt_p = np.sqrt(p_target)
    density_norm = np.sum(kernel_matrix / sqrt_p[None, :] / sqrt_p[:, None], axis=1) + 1e-12
    return p_target, density_norm


def fit_spectral_model(
    X_target: np.ndarray,
    X_next: np.ndarray | None = None,
    *,
    epsilon: float | None = None,
    max_modes: int = 80,
    ridge: float = 1e-6,
    eigen_tol: float = 1e-6,
    kernel_scale: float = 0.5,
    normalize_weights: bool = False,
    method: str = "KSWGD",
) -> SpectralModel:
    """Fit the normalized spectral object used by the KSWGD update.

    If `X_next` is provided, the kernel matrix uses one-step target snapshots.
    If it is omitted, the target kernel itself is used, which is the
    DMPS-style object.
    """
    X_target = np.asarray(X_target, dtype=np.float64)
    epsilon = median_bandwidth(X_target, scale=kernel_scale) if epsilon is None else float(epsilon)

    K_xx = gaussian_kernel(X_target, X_target, epsilon)
    if X_next is None:
        data_kernel = K_xx.copy()
    else:
        X_next = np.asarray(X_next, dtype=np.float64)
        K_xy = gaussian_kernel(X_target, X_next, epsilon)
        evals, Q = np.linalg.eigh(K_xx)
        inv_evals = 1.0 / (np.clip(evals, 0.0, None) + float(ridge))
        data_kernel = K_xy @ (Q @ (inv_evals[:, None] * Q.T))
        data_kernel = np.nan_to_num(data_kernel, nan=0.0, posinf=0.0, neginf=0.0)
        min_value = float(np.min(data_kernel))
        if min_value < 0.0:
            data_kernel = data_kernel - min_value + 1e-12

    p_target, density_norm = _safe_density_norm(data_kernel)
    sqrt_p = np.sqrt(p_target)
    data_kernel_norm = data_kernel / sqrt_p[:, None] / sqrt_p[None, :]
    D_y = np.sum(data_kernel_norm, axis=0) + 1e-12
    rw_kernel = 0.5 * (data_kernel_norm / D_y[None, :] + data_kernel_norm / D_y[:, None])
    rw_kernel = 0.5 * (rw_kernel + rw_kernel.T)

    evals_rw, evecs_rw = np.linalg.eigh(rw_kernel)
    order = np.argsort(evals_rw)[::-1]
    eigenvalues = np.clip(evals_rw[order], 0.0, None)
    eigenvectors = evecs_rw[:, order]

    inv_generator = np.zeros_like(eigenvalues)
    generator_gap = 1.0 - eigenvalues
    valid_gap = np.abs(generator_gap) > eigen_tol
    inv_generator[valid_gap] = epsilon / generator_gap[valid_gap]
    inv_generator[0] = 0.0

    inv_diffusion = np.zeros_like(eigenvalues)
    valid_eigs = eigenvalues >= eigen_tol
    inv_diffusion[valid_eigs] = epsilon / (eigenvalues[valid_eigs] + 1e-3)

    weights = inv_diffusion * inv_generator * inv_diffusion
    valid_modes = np.flatnonzero(valid_eigs)
    valid_modes = valid_modes[: min(int(max_modes), len(valid_modes))]
    if len(valid_modes) == 0:
        valid_modes = np.arange(min(int(max_modes), len(eigenvalues)))
        weights = np.ones_like(eigenvalues)
        weights[0] = 0.0
    eigenvectors = eigenvectors[:, valid_modes]
    eigenvalues = eigenvalues[valid_modes]
    weights = weights[valid_modes]

    if normalize_weights and np.max(np.abs(weights)) > 0:
        weights = weights / np.max(np.abs(weights))

    return SpectralModel(
        X_target=X_target,
        epsilon=epsilon,
        p_target=p_target,
        sq_target=np.sum(X_target * X_target, axis=1),
        density_norm=density_norm,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        weights=weights,
        kernel_matrix=data_kernel,
        method=method,
    )


def fit_spectral_model_from_kernel(
    X_target: np.ndarray,
    data_kernel: np.ndarray,
    *,
    epsilon: float,
    max_modes: int | None = None,
    eigen_tol: float = 1e-6,
    reg: float = 1e-3,
    normalize_weights: bool = False,
    method: str = "KSWGD",
) -> SpectralModel:
    """Fit the KSWGD spectral object from a precomputed target kernel.

    This keeps the legacy notebooks' kernel construction intact while reusing
    the shared particle-transport implementation.
    """
    X_target = np.asarray(X_target, dtype=np.float64)
    data_kernel = np.asarray(data_kernel, dtype=np.float64)
    data_kernel = np.nan_to_num(data_kernel, nan=0.0, posinf=0.0, neginf=0.0)
    min_value = float(np.min(data_kernel))
    if min_value < 0.0:
        data_kernel = data_kernel - min_value + 1e-12

    p_target, density_norm = _safe_density_norm(data_kernel)
    sqrt_p = np.sqrt(p_target)
    data_kernel_norm = data_kernel / sqrt_p[:, None] / sqrt_p[None, :]
    D_y = np.sum(data_kernel_norm, axis=0) + 1e-12
    rw_kernel = 0.5 * (data_kernel_norm / D_y[None, :] + data_kernel_norm / D_y[:, None])
    rw_kernel = 0.5 * (rw_kernel + rw_kernel.T)

    evals_rw, evecs_rw = np.linalg.eigh(rw_kernel)
    order = np.argsort(evals_rw)[::-1]
    eigenvalues = evals_rw[order]
    eigenvalues = np.where(eigenvalues < eigen_tol, 0.0, eigenvalues)
    eigenvectors = np.real(evecs_rw[:, order])

    inv_generator = np.zeros_like(eigenvalues)
    generator_gap = 1.0 - eigenvalues
    valid_gap = np.abs(generator_gap) > eigen_tol
    valid_generator = np.flatnonzero(valid_gap)
    valid_generator = valid_generator[valid_generator != 0]
    inv_generator[valid_generator] = float(epsilon) / generator_gap[valid_generator]

    inv_diffusion = np.zeros_like(eigenvalues)
    valid_eigs = eigenvalues >= eigen_tol
    inv_diffusion[valid_eigs] = float(epsilon) / (eigenvalues[valid_eigs] + float(reg))
    weights = inv_diffusion * inv_generator * inv_diffusion

    valid_modes = np.flatnonzero(valid_eigs)
    if max_modes is None:
        max_modes = len(valid_modes)
    valid_modes = valid_modes[: min(int(max_modes), len(valid_modes))]
    if len(valid_modes) == 0:
        valid_modes = np.arange(min(int(max_modes or len(eigenvalues)), len(eigenvalues)))
        weights = np.ones_like(eigenvalues)
        weights[0] = 0.0
    eigenvectors = eigenvectors[:, valid_modes]
    eigenvalues = eigenvalues[valid_modes]
    weights = weights[valid_modes]

    if normalize_weights and np.max(np.abs(weights)) > 0:
        weights = weights / np.max(np.abs(weights))

    return SpectralModel(
        X_target=X_target,
        epsilon=float(epsilon),
        p_target=p_target,
        sq_target=np.sum(X_target * X_target, axis=1),
        density_norm=density_norm,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        weights=weights,
        kernel_matrix=data_kernel,
        method=method,
    )


def fit_kernel_edmd_model(
    X_target: np.ndarray,
    X_next: np.ndarray,
    *,
    kernel_type: int | str = 5,
    epsilon: float | None = None,
    ridge: float = 1e-6,
    max_modes: int | None = None,
    eigen_tol: float = 1e-6,
    method: str = "KSWGD",
    **kernel_kwargs,
) -> tuple[SpectralModel, str]:
    """Build the legacy kernel-EDMD spectral model used in extra_test/test_1."""
    X_target = np.asarray(X_target, dtype=np.float64)
    X_next = np.asarray(X_next, dtype=np.float64)
    D2 = pairwise_sq_dists(X_target)
    if epsilon is None:
        epsilon = 0.5 * float(np.median(D2)) / (np.log(len(X_target) + 1.0) + 1e-12)
    length_scale = float(np.sqrt(np.median(D2) + 1e-12))
    K_xx, kernel_name = named_kernel_matrix(
        X_target,
        X_target,
        kernel_type=kernel_type,
        epsilon=float(epsilon),
        length_scale=length_scale,
        **kernel_kwargs,
    )
    K_xy, _ = named_kernel_matrix(
        X_target,
        X_next,
        kernel_type=kernel_type,
        epsilon=float(epsilon),
        length_scale=length_scale,
        **kernel_kwargs,
    )
    evals, Q = np.linalg.eigh(K_xx)
    evals = np.clip(evals, 0.0, None)
    inv = 1.0 / (evals + float(ridge))
    data_kernel = K_xy @ ((Q * inv[None, :]) @ Q.T)
    model = fit_spectral_model_from_kernel(
        X_target,
        data_kernel,
        epsilon=float(epsilon),
        max_modes=max_modes,
        eigen_tol=eigen_tol,
        method=method,
    )
    return model, kernel_name


def _use_gpu(use_gpu: bool | str) -> bool:
    if use_gpu == "auto":
        return cp is not None
    return bool(use_gpu and cp is not None)


def _project_torus_backend(X, R: float, r: float, xp):
    theta = xp.arctan2(X[:, 1], X[:, 0])
    rho = xp.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    phi = xp.arctan2(X[:, 2], rho - R)
    x = (R + r * xp.cos(phi)) * xp.cos(theta)
    y = (R + r * xp.cos(phi)) * xp.sin(theta)
    z = r * xp.sin(phi)
    return xp.stack([x, y, z], axis=1)


def _project_torus_gradient_backend(gradient, X, R: float, r: float, xp):
    theta = xp.arctan2(X[:, 1], X[:, 0])
    rho = xp.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    phi = xp.arctan2(X[:, 2], rho - R)
    normal = xp.stack(
        [xp.cos(phi) * xp.cos(theta), xp.cos(phi) * xp.sin(theta), xp.sin(phi)],
        axis=1,
    )
    normal = normal / (xp.linalg.norm(normal, axis=1, keepdims=True) + 1e-12)
    return gradient - xp.sum(gradient * normal, axis=1, keepdims=True) * normal


def _clip_by_norm(update, max_step_norm: float | None, xp):
    if max_step_norm is None or max_step_norm <= 0:
        return update
    norms = xp.linalg.norm(update, axis=1, keepdims=True)
    if getattr(xp, "__name__", "") == "torch":
        factors = xp.clamp(float(max_step_norm) / (norms + 1e-300), max=1.0)
    else:
        factors = xp.minimum(1.0, float(max_step_norm) / (norms + 1e-300))
    return update * factors


def torch_device(device: str | None = None):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_checkpoint_parent(path: str | Path | None) -> str | None:
    if path is None:
        return None
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_path)


def train_sdmd_solver(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    input_size: int,
    layer_sizes: list[int],
    n_psi_train: int,
    target_dim: int,
    delta_t: float,
    reg: float = 0.1,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 1e-5,
    fnn_batch_size: int = 32,
    checkpoint_file: str = "checkpoints/sdmd_koopman.torch",
    fnn_checkpoint_file: str = "checkpoints/sdmd_fnn.torch",
    a_b_file: str = "checkpoints/sdmd_ab.jbl",
    train_fraction: float = 0.7,
    device: str | None = None,
    verbose: bool = True,
):
    """Train the same neural dictionary solver used by the original notebooks."""
    import solver_sdmd_torch_gpu
    from solver_sdmd_torch_gpu import KoopmanNNTorch, KoopmanSolverTorch

    dev = torch_device(device)
    solver_sdmd_torch_gpu.device = str(dev)
    checkpoint_file = ensure_checkpoint_parent(checkpoint_file)
    fnn_checkpoint_file = ensure_checkpoint_parent(fnn_checkpoint_file)
    a_b_file = ensure_checkpoint_parent(a_b_file)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    cut = int(float(train_fraction) * len(X))
    data_train = [X[:cut], Y[:cut]]
    data_valid = [X[cut + 1 :], Y[cut + 1 :]]

    basis_function = KoopmanNNTorch(
        input_size=int(input_size),
        layer_sizes=list(layer_sizes),
        n_psi_train=int(n_psi_train),
    ).to(dev).double()
    solver = KoopmanSolverTorch(
        dic=basis_function,
        target_dim=int(target_dim),
        reg=float(reg),
        checkpoint_file=checkpoint_file,
        fnn_checkpoint_file=fnn_checkpoint_file,
        a_b_file=a_b_file,
        generator_batch_size=2,
        fnn_batch_size=int(fnn_batch_size),
        delta_t=float(delta_t),
    )
    if verbose:
        print(f"SDMD training data: train={data_train[0].shape}, valid={data_valid[0].shape}, device={dev}")
    solver.build_with_generator(
        data_train=data_train,
        data_valid=data_valid,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        log_interval=10,
        lr_decay_factor=0.8,
    )
    solver.dic = solver.dic.double().to(dev)
    return solver


def _kernel_eval_torch(X_query, X_data, epsilon: float, p_target, density_norm):
    import torch

    sq_query = torch.sum(X_query**2, dim=1)
    sq_data = torch.sum(X_data**2, dim=1)
    H = sq_query[:, None] + sq_data[None, :] - 2.0 * (X_query @ X_data.T)
    K_raw = torch.exp(-H / (2.0 * float(epsilon)))
    p_query = torch.sqrt(torch.sum(K_raw, dim=1, keepdim=True) + 1e-14)
    K_norm = K_raw / p_query / p_target[None, :]
    return K_norm / density_norm[None, :]


def _kernel_grad_torch(X_query, X_data, epsilon: float, p_target):
    import torch

    diff = X_query[:, None, :] - X_data[None, :, :]
    sq_query = torch.sum(X_query**2, dim=1)
    sq_data = torch.sum(X_data**2, dim=1)
    H = sq_query[:, None] + sq_data[None, :] - 2.0 * (X_query @ X_data.T)
    K_raw = torch.exp(-H / (2.0 * float(epsilon)))
    grad_raw = -K_raw[:, :, None] * diff / float(epsilon)
    p_query = torch.sqrt(torch.sum(K_raw, dim=1, keepdim=True) + 1e-14)
    return grad_raw / p_query[:, :, None] / p_target[None, :, None]


def _kernel_eval_and_grad_torch(X_query, X_data, epsilon: float, p_target, density_norm):
    import torch

    diff = X_query[:, None, :] - X_data[None, :, :]
    sq_query = torch.sum(X_query**2, dim=1)
    sq_data = torch.sum(X_data**2, dim=1)
    H = sq_query[:, None] + sq_data[None, :] - 2.0 * (X_query @ X_data.T)
    K_raw = torch.exp(-H / (2.0 * float(epsilon)))
    p_query = torch.sqrt(torch.sum(K_raw, dim=1, keepdim=True) + 1e-14)
    K_norm = K_raw / p_query / p_target[None, :]
    grad_raw = -K_raw[:, :, None] * diff / float(epsilon)
    grad = grad_raw / p_query[:, :, None] / p_target[None, :, None]
    return K_norm / density_norm[None, :], grad


def _repulsive_force_torch(X, epsilon: float):
    import torch

    diff = X[:, None, :] - X[None, :, :]
    sq = torch.sum(X**2, dim=1)
    H = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    K = torch.exp(-H / (2.0 * float(epsilon)))
    grad = -K[:, :, None] * diff / float(epsilon)
    return torch.mean(grad, dim=1)


def fit_diffusion_map_kernel_spectrum_2d(
    X_target: np.ndarray,
    *,
    dt: float,
    max_modes: int | None = 1000,
    lowrank_niter: int = 5,
    device: str | None = None,
    method: str = "DMPS",
) -> KernelSpectrum2D:
    """Restore the original diffusion-map spectrum used by test 2 DMPS.

    Only the leading spectrum is needed downstream, so by default this uses a
    truncated SVD for the largest 1000 singular values/vectors instead of a full
    decomposition of the 9999-by-9999 random-walk kernel.
    """
    import torch

    dev = torch_device(device)
    X_target = np.asarray(X_target, dtype=np.float64)
    X_gpu = torch.as_tensor(X_target, dtype=torch.float64, device=dev)
    epsilon = 2.0 * float(dt)
    sq = torch.sum(X_gpu**2, dim=1)
    H = sq[:, None] + sq[None, :] - 2.0 * (X_gpu @ X_gpu.T)
    K = torch.exp(-H / (2.0 * epsilon))
    p_target = torch.sqrt(torch.sum(K, dim=1) + 1e-14)
    K_norm = K / p_target[:, None] / p_target[None, :]
    density_norm = torch.sum(K_norm, dim=0) + 1e-14
    rw_kernel = K_norm / density_norm[:, None]
    n = rw_kernel.shape[0]
    if max_modes is None or int(max_modes) >= n:
        phi, s, _ = torch.linalg.svd(rw_kernel)
    else:
        torch.manual_seed(0)
        q = min(int(max_modes), n)
        phi, s, _ = torch.svd_lowrank(rw_kernel, q=q, niter=int(lowrank_niter))
        order = torch.argsort(s, descending=True)
        s = s[order]
        phi = phi[:, order]
    eigenvalues = s.detach().cpu().numpy()
    eigenvectors = phi.detach().cpu().numpy()
    lambda_generator = (eigenvalues - 1.0) / epsilon
    p_np = p_target.detach().cpu().numpy()
    d_np = density_norm.detach().cpu().numpy()
    del X_gpu, sq, H, K, K_norm, density_norm, rw_kernel, phi, s
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return KernelSpectrum2D(
        X_target=X_target,
        epsilon=epsilon,
        p_target=p_np,
        density_norm=d_np,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_generator=lambda_generator,
        method=method,
    )


def sdmd_kernel_spectrum_2d(
    solver,
    X_target: np.ndarray,
    *,
    base_kernel: KernelSpectrum2D,
    dt: float,
    method: str = "KSWGD",
) -> KernelSpectrum2D:
    eigenvalues = np.real(np.asarray(solver.eigenvalues).T).reshape(-1)
    eigenvectors = np.real(solver.eigenfunctions(np.asarray(X_target, dtype=np.float64)))
    lambda_generator = (eigenvalues - 1.0) / float(dt)
    return KernelSpectrum2D(
        X_target=np.asarray(X_target, dtype=np.float64),
        epsilon=float(base_kernel.epsilon),
        p_target=np.asarray(base_kernel.p_target, dtype=np.float64),
        density_norm=np.asarray(base_kernel.density_norm, dtype=np.float64),
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_generator=lambda_generator,
        method=method,
    )


def run_kernel_spectrum_transport_2d(
    model: KernelSpectrum2D,
    X_init: np.ndarray,
    *,
    n_iter: int,
    step_size: float,
    alpha_repulsive: float = 0.5,
    snapshot_steps: Iterable[int] | None = None,
    record_interval: int = 100,
    n_skip_modes: int = 1,
    eig_threshold: float = 0.01,
    fixed_mode_count: int | None = None,
    potential=None,
    well_centers: np.ndarray | None = None,
    well_radius: float = 0.4,
    device: str | None = None,
    progress: bool = True,
) -> MetricTransportResult:
    """Run the original target-kernel spectral transport used in test 2."""
    import torch

    dev = torch_device(device)
    X_init = np.asarray(X_init, dtype=np.float64)
    X_target = torch.as_tensor(model.X_target, dtype=torch.float64, device=dev)
    p_target = torch.as_tensor(model.p_target, dtype=torch.float64, device=dev)
    density_norm = torch.as_tensor(model.density_norm, dtype=torch.float64, device=dev)

    eigvals_after_skip = np.asarray(model.eigenvalues).reshape(-1)[int(n_skip_modes) :]
    if fixed_mode_count is None:
        k_modes = int(np.sum(eigvals_after_skip.real > float(eig_threshold)))
    else:
        k_modes = min(int(fixed_mode_count), len(eigvals_after_skip))
    if k_modes <= 0:
        raise RuntimeError(f"No modes selected for {model.method}.")

    start = int(n_skip_modes)
    end = start + k_modes
    eigvecs = torch.as_tensor(np.real(model.eigenvectors[:, start:end]), dtype=torch.float64, device=dev)
    lambda_selected = np.asarray(model.lambda_generator).reshape(-1)[start:end]
    lambda_inv = np.zeros(k_modes, dtype=np.complex128)
    mask = np.abs(lambda_selected) > 1e-8
    lambda_inv[mask] = 1.0 / lambda_selected[mask]
    lambda_inv_t = torch.as_tensor(lambda_inv, dtype=torch.complex128, device=dev)

    metric_steps = set(range(0, int(n_iter) + 1, int(record_interval))) if record_interval else {0, int(n_iter)}
    requested = {0, int(n_iter), *(int(s) for s in (snapshot_steps or [])), *metric_steps}
    requested = {s for s in requested if 0 <= s <= int(n_iter)}
    snapshots: dict[int, np.ndarray] = {0: X_init.copy()}
    step_norms = np.zeros(int(n_iter), dtype=np.float64)
    X_gpu = torch.as_tensor(X_init, dtype=torch.float64, device=dev)

    if well_centers is None:
        well_centers = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)

    metrics = {"steps": [], "well_coverage": [], "avg_potential": [], "movement_rate": []}

    def record_metrics(step: int, current: np.ndarray, movement: float = 0.0) -> None:
        dists = np.linalg.norm(current[:, None, :] - well_centers[None, :, :], axis=2)
        coverage = 100.0 * np.mean(np.min(dists, axis=1) <= float(well_radius))
        avg_potential = float(np.mean(potential.potential(current))) if potential is not None else np.nan
        metrics["steps"].append(step)
        metrics["well_coverage"].append(coverage)
        metrics["avg_potential"].append(avg_potential)
        metrics["movement_rate"].append(movement)

    record_metrics(0, X_init)
    report_every = max(1, int(n_iter) // 6)

    for step in range(1, int(n_iter) + 1):
        K_curr, grad_K = _kernel_eval_and_grad_torch(X_gpu, X_target, model.epsilon, p_target, density_norm)
        coeffs = eigvecs.T @ K_curr.T
        coeffs_inv = lambda_inv_t[:, None] * coeffs.to(torch.complex128)
        f_inv = eigvecs @ coeffs_inv.real
        driving = torch.sum(grad_K * f_inv.T[:, :, None], dim=1)
        repulsive = _repulsive_force_torch(X_gpu, model.epsilon)
        update = float(step_size) * (driving + float(alpha_repulsive) * repulsive)
        X_gpu = X_gpu - update
        step_norms[step - 1] = float(torch.mean(torch.linalg.norm(update, dim=1)).detach().cpu().item())

        current_np = None
        if step in requested:
            current_np = X_gpu.detach().cpu().numpy()
            snapshots[step] = current_np.copy()
        if step in metric_steps:
            if current_np is None:
                current_np = X_gpu.detach().cpu().numpy()
            record_metrics(step, current_np, step_norms[step - 1])
        if progress and (step == int(n_iter) or step % report_every == 0):
            print(f"{model.method}: step {step}/{int(n_iter)}, mean step norm={step_norms[step - 1]:.3e}")

    for key in metrics:
        metrics[key] = np.asarray(metrics[key])
    final = X_gpu.detach().cpu().numpy()
    return MetricTransportResult(final=final.copy(), snapshots=snapshots, step_norms=step_norms, metrics=metrics)


def build_target_kernel_torch(
    X_train: np.ndarray,
    *,
    c_epsilon: float = 0.5,
    X_reference: np.ndarray | None = None,
    reference_scale: float = 0.5,
    epsilon_override: float | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Build the normalized target kernel used by the original Allen-Cahn notebooks."""
    import torch

    dev = torch_device(device)
    X_train = np.asarray(X_train, dtype=np.float64)
    X_train_gpu = torch.as_tensor(X_train, dtype=torch.float64, device=dev)
    n = X_train.shape[0]
    sq_train = torch.sum(X_train_gpu**2, dim=1)
    H = sq_train[:, None] + sq_train[None, :] - 2.0 * (X_train_gpu @ X_train_gpu.T)
    epsilon_target = float((torch.median(H) * (float(c_epsilon) / np.log(n + 1))).detach().cpu().item())
    epsilon_target = max(epsilon_target, 1e-10)
    epsilon_reference = None
    if X_reference is not None:
        X_ref_gpu = torch.as_tensor(np.asarray(X_reference, dtype=np.float64), dtype=torch.float64, device=dev)
        sq_ref = torch.sum(X_ref_gpu**2, dim=1)
        H_ref = sq_ref[:, None] + sq_train[None, :] - 2.0 * (X_ref_gpu @ X_train_gpu.T)
        epsilon_reference = float((float(reference_scale) * torch.median(H_ref)).detach().cpu().item())
        epsilon_reference = max(epsilon_reference, 1e-10)
    if epsilon_override is not None:
        epsilon = float(epsilon_override)
    elif epsilon_reference is not None:
        epsilon = max(epsilon_target, epsilon_reference)
    else:
        epsilon = epsilon_target
    epsilon = max(epsilon, 1e-10)
    data_kernel = torch.exp(-H / (2.0 * epsilon))
    p_x = torch.sqrt(torch.sum(data_kernel, dim=1) + 1e-14)
    data_kernel_norm = data_kernel / p_x[:, None] / p_x[None, :]
    D_y = torch.sum(data_kernel_norm, dim=0) + 1e-14
    return {
        "X_train_gpu": X_train_gpu,
        "p_x_gpu": p_x,
        "D_y_gpu": D_y,
        "epsilon": epsilon,
        "epsilon_target": epsilon_target,
        "epsilon_reference": epsilon_reference,
        "epsilon_override": epsilon_override,
        "n_train": n,
        "device": str(dev),
    }


def target_nn_distance(Z_particles: np.ndarray, Z_target: np.ndarray, max_target: int = 2000) -> float:
    Z_particles = np.asarray(Z_particles, dtype=np.float64)
    Z_target = np.asarray(Z_target, dtype=np.float64)
    if Z_target.shape[0] > int(max_target):
        ref = Z_target[np.linspace(0, Z_target.shape[0] - 1, int(max_target)).astype(int)]
    else:
        ref = Z_target
    diff = Z_particles[:, None, :] - ref[None, :, :]
    return float(np.mean(np.min(np.sqrt(np.sum(diff * diff, axis=2)), axis=1)))


def run_latent_kswgd_with_spectrum(
    *,
    method_name: str,
    Z_init: np.ndarray,
    Z_target: np.ndarray,
    target_kernel: dict[str, object],
    eigvecs_on_target: np.ndarray,
    lambda_gen_full_values: np.ndarray,
    eigvals_for_mode_selection: np.ndarray | None = None,
    fixed_mode_count: int | None = 60,
    eig_threshold_value: float = 1e-3,
    n_skip_modes: int = 1,
    n_iter: int = 1500,
    step_size: float = 0.02,
    record_every: int = 100,
    snapshot_steps: Iterable[int] | None = None,
    adaptive_step: bool = True,
    target_initial_movement: float = 0.02,
    max_step_multiplier: float = 1e13,
    max_particle_step: float | None = 0.05,
    device: str | None = None,
    progress: bool = True,
) -> LatentKSWGDResult:
    """Restore the original latent Allen-Cahn target-kernel KSWGD loop."""
    import torch

    dev = torch_device(device or str(target_kernel.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    Z_init = np.asarray(Z_init, dtype=np.float64)
    Z_target = np.asarray(Z_target, dtype=np.float64)
    if eigvals_for_mode_selection is None:
        eigvals_for_mode_selection = lambda_gen_full_values
    eigvals_after_skip = np.asarray(eigvals_for_mode_selection).reshape(-1)[int(n_skip_modes) :]
    if fixed_mode_count is not None:
        k_modes = min(int(fixed_mode_count), len(eigvals_after_skip))
    else:
        k_modes = int(np.sum(eigvals_after_skip.real > float(eig_threshold_value)))
    if k_modes <= 0:
        raise RuntimeError("No spectral modes selected for KSWGD.")

    mode_start = int(n_skip_modes)
    mode_end = mode_start + k_modes
    lambda_selected = np.asarray(lambda_gen_full_values).reshape(-1)[mode_start:mode_end]
    lambda_inv = np.zeros(k_modes, dtype=np.complex128)
    mask = np.abs(lambda_selected) > 1e-8
    lambda_inv[mask] = 1.0 / lambda_selected[mask]

    eigvecs_selected = torch.as_tensor(
        np.real(eigvecs_on_target[:, mode_start:mode_end]),
        dtype=torch.float64,
        device=dev,
    )
    lambda_inv_gpu = torch.as_tensor(lambda_inv, dtype=torch.complex128, device=dev)
    X_train_gpu = target_kernel["X_train_gpu"]
    p_x_gpu = target_kernel["p_x_gpu"]
    D_y_gpu = target_kernel["D_y_gpu"]
    if str(X_train_gpu.device) != str(dev):
        X_train_gpu = X_train_gpu.to(dev)
        p_x_gpu = p_x_gpu.to(dev)
        D_y_gpu = D_y_gpu.to(dev)
    epsilon_val = float(target_kernel["epsilon"])

    n_particles, latent_dim = Z_init.shape
    Z_traj = np.zeros((n_particles, latent_dim, int(n_iter) + 1), dtype=np.float64)
    Z_traj[:, :, 0] = Z_init.copy()
    requested = {0, int(n_iter), *(int(s) for s in (snapshot_steps or []))}
    requested = {s for s in requested if 0 <= s <= int(n_iter)}
    snapshots = {0: Z_init.copy()}
    step_norms = np.zeros(int(n_iter), dtype=np.float64)

    def compute_force(Z_curr_np: np.ndarray):
        Z_curr_gpu = torch.as_tensor(Z_curr_np, dtype=torch.float64, device=dev)
        K_curr = _kernel_eval_torch(Z_curr_gpu, X_train_gpu, epsilon_val, p_x_gpu, D_y_gpu)
        coeffs = eigvecs_selected.T @ K_curr.T
        coeffs_inv = lambda_inv_gpu[:, None] * coeffs.to(torch.complex128)
        f_inv = eigvecs_selected @ coeffs_inv.real
        grad_K = _kernel_grad_torch(Z_curr_gpu, X_train_gpu, epsilon_val, p_x_gpu)
        force = torch.sum(grad_K * f_inv.T[:, :, None], dim=1)
        return force, K_curr

    with torch.no_grad():
        initial_force, K0 = compute_force(Z_init)
        initial_kernel_mean = float(K0.mean().detach().cpu().item())
        initial_kernel_max = float(K0.max().detach().cpu().item())
        initial_force_norm = float(torch.mean(torch.linalg.norm(initial_force, dim=1)).detach().cpu().item())

    effective_step_size = float(step_size)
    step_multiplier = 1.0
    if adaptive_step:
        raw_initial_movement = max(float(step_size) * initial_force_norm, 1e-300)
        step_multiplier = min(float(max_step_multiplier), float(target_initial_movement) / raw_initial_movement)
        effective_step_size = float(step_size) * step_multiplier

    metrics = {
        "steps": [0],
        "target_nn_distance": [target_nn_distance(Z_init, Z_target)],
        "mean_norm": [float(np.mean(np.linalg.norm(Z_init, axis=1)))],
        "movement_rate": [0.0],
        "kernel_mean": [initial_kernel_mean],
        "kernel_max": [initial_kernel_max],
        "force_norm": [initial_force_norm],
        "effective_step_size": [effective_step_size],
    }

    print(f"{method_name}: particles={n_particles}, latent_dim={latent_dim}, modes={k_modes}, epsilon={epsilon_val:.6e}")
    print(f"initial K mean/max={initial_kernel_mean:.3e}/{initial_kernel_max:.3e}; effective step={effective_step_size:.3e}")
    report_every = max(1, int(n_iter) // 6)
    for it in range(int(n_iter)):
        Z_curr = Z_traj[:, :, it]
        force, _ = compute_force(Z_curr)
        update = -effective_step_size * force
        if max_particle_step is not None and max_particle_step > 0:
            norms = torch.linalg.norm(update, dim=1, keepdim=True)
            update = update * torch.clamp(float(max_particle_step) / (norms + 1e-300), max=1.0)
        Z_curr_gpu = torch.as_tensor(Z_curr, dtype=torch.float64, device=dev)
        Z_next = (Z_curr_gpu + update).detach().cpu().numpy()
        Z_traj[:, :, it + 1] = Z_next
        step_norms[it] = float(np.mean(np.linalg.norm(Z_next - Z_curr, axis=1)))
        step = it + 1
        if step in requested:
            snapshots[step] = Z_next.copy()
        if step % int(record_every) == 0 or step == int(n_iter):
            with torch.no_grad():
                force_next, K_next = compute_force(Z_next)
            metrics["steps"].append(step)
            metrics["target_nn_distance"].append(target_nn_distance(Z_next, Z_target))
            metrics["mean_norm"].append(float(np.mean(np.linalg.norm(Z_next, axis=1))))
            metrics["movement_rate"].append(step_norms[it])
            metrics["kernel_mean"].append(float(K_next.mean().detach().cpu().item()))
            metrics["kernel_max"].append(float(K_next.max().detach().cpu().item()))
            metrics["force_norm"].append(float(torch.mean(torch.linalg.norm(force_next, dim=1)).detach().cpu().item()))
            metrics["effective_step_size"].append(effective_step_size)
        if progress and (step == int(n_iter) or step % report_every == 0):
            print(f"{method_name}: step {step}/{int(n_iter)}, mean step norm={step_norms[it]:.3e}")

    for key in metrics:
        metrics[key] = np.asarray(metrics[key])
    final = Z_traj[:, :, int(n_iter)].copy()
    snapshots[int(n_iter)] = final.copy()
    return LatentKSWGDResult(final=final, snapshots=snapshots, step_norms=step_norms, metrics=metrics, trajectory=Z_traj, n_modes=k_modes)


def run_kswgd(
    model: SpectralModel,
    X_init: np.ndarray,
    *,
    n_iter: int,
    step_size: float,
    snapshot_steps: Iterable[int] | None = None,
    use_gpu: bool | str = "auto",
    torus: tuple[float, float] | None = None,
    project_fn=None,
    gradient_project_fn=None,
    max_step_norm: float | None = None,
    normalize_by_particles: bool = True,
    adaptive_initial_movement: float | None = None,
    max_step_multiplier: float = 1e13,
    progress: bool = True,
) -> TransportResult:
    """Transport particles with the fitted KSWGD/DMPS spectral model."""
    snapshot_steps = sorted({0, int(n_iter), *(int(s) for s in (snapshot_steps or []))})
    snapshot_steps = [s for s in snapshot_steps if 0 <= s <= int(n_iter)]
    snapshot_set = set(snapshot_steps)

    gpu = _use_gpu(use_gpu)
    xp = cp if gpu else np
    to_numpy = cp.asnumpy if gpu else (lambda x: np.asarray(x))

    X_target = xp.asarray(model.X_target)
    p_target = xp.asarray(model.p_target)
    sq_target = xp.asarray(model.sq_target)
    density_norm = xp.asarray(model.density_norm)
    phi = xp.asarray(model.eigenvectors)
    weights = xp.asarray(model.weights)
    spectral_matrix = phi @ (weights[:, None] * phi.T)
    X = xp.asarray(np.asarray(X_init, dtype=np.float64))
    m = int(X.shape[0])
    d = int(X.shape[1])

    snapshots: dict[int, np.ndarray] = {0: to_numpy(X).copy()}
    step_norms = np.zeros(int(n_iter), dtype=np.float64)
    report_every = max(1, int(n_iter) // 5)
    adaptive_multiplier = 1.0

    for step in range(1, int(n_iter) + 1):
        grad_matrix = grad_ker1(X, X_target, p_target, sq_target, density_norm, model.epsilon)
        cross_matrix = K_tar_eval(X_target, X, p_target, sq_target, density_norm, model.epsilon)
        weighted_cross_sum = spectral_matrix @ xp.sum(cross_matrix, axis=1)
        driving = xp.empty((m, d), dtype=X.dtype)
        for j in range(d):
            driving[:, j] = grad_matrix[:, :, j] @ weighted_cross_sum

        normalization = m if normalize_by_particles else 1.0
        update = (float(step_size) / normalization) * driving
        if torus is not None:
            update = _project_torus_gradient_backend(update, X, torus[0], torus[1], xp)
        elif gradient_project_fn is not None:
            update = xp.asarray(gradient_project_fn(to_numpy(update), to_numpy(X)))

        if adaptive_initial_movement is not None:
            if step == 1:
                mean_norm = float(to_numpy(xp.mean(xp.linalg.norm(update, axis=1))))
                adaptive_multiplier = min(float(max_step_multiplier), float(adaptive_initial_movement) / (mean_norm + 1e-300))
            update = adaptive_multiplier * update

        update = _clip_by_norm(update, max_step_norm, xp)
        X = X - update
        if torus is not None:
            X = _project_torus_backend(X, torus[0], torus[1], xp)
        elif project_fn is not None:
            X = xp.asarray(project_fn(to_numpy(X)))

        step_norms[step - 1] = float(to_numpy(xp.mean(xp.linalg.norm(update, axis=1))))
        if step in snapshot_set:
            snapshots[step] = to_numpy(X).copy()
        if progress and (step == int(n_iter) or step % report_every == 0):
            print(f"{model.method}: step {step}/{int(n_iter)}, mean step norm={step_norms[step - 1]:.3e}")

    return TransportResult(final=to_numpy(X).copy(), snapshots=snapshots, step_norms=step_norms)


def run_nn_spectral_kswgd_2d(
    solver,
    eigenvalues: np.ndarray,
    X_init: np.ndarray,
    *,
    dt: float,
    n_iter: int,
    step_size: float,
    alpha_repulsive: float = 0.5,
    repulsive_epsilon: float | None = None,
    snapshot_steps: Iterable[int] | None = None,
    record_interval: int = 100,
    n_skip_modes: int = 1,
    eig_threshold: float = 0.01,
    fixed_mode_count: int | None = None,
    max_step_norm: float | None = None,
    potential=None,
    well_centers: np.ndarray | None = None,
    well_radius: float = 0.4,
    device: str | None = None,
    progress: bool = True,
) -> MetricTransportResult:
    """Run the legacy 2D neural-dictionary spectral KSWGD update.

    This is used by the quadruple-well experiment, where the original notebook
    used gradients of learned spectral eigenfunctions rather than gradients of
    the target kernel itself.
    """
    import torch

    dev = torch_device(device or str(next(solver.dic.parameters()).device))
    dic = solver.dic.double().to(dev).eval()
    solver.dic = dic
    eigenvalues = np.asarray(eigenvalues).reshape(-1).real
    X0 = np.asarray(X_init, dtype=np.float64)
    with torch.no_grad():
        psi_dim = int(dic(torch.zeros((1, X0.shape[1]), dtype=torch.float64, device=dev)).shape[1])
    eigvecs = np.asarray(solver.eigenvectors)
    if eigvecs.shape[0] != psi_dim and eigvecs.shape[1] == psi_dim:
        eigvecs = eigvecs.T

    max_modes_available = min(len(eigenvalues), eigvecs.shape[1])
    after_skip = eigenvalues[int(n_skip_modes) : max_modes_available]
    if fixed_mode_count is None:
        k_modes = int(np.sum(after_skip > float(eig_threshold)))
    else:
        k_modes = min(int(fixed_mode_count), len(after_skip))
    if k_modes <= 0:
        raise RuntimeError("No spectral modes selected for KSWGD.")

    mode_indices = np.arange(int(n_skip_modes), int(n_skip_modes) + k_modes)
    lambda_gen = (eigenvalues[mode_indices] - 1.0) / float(dt)
    lambda_inv = np.zeros_like(lambda_gen, dtype=np.complex128)
    mask = np.abs(lambda_gen) > 1e-8
    lambda_inv[mask] = 1.0 / lambda_gen[mask]
    lambda_inv_t = torch.as_tensor(lambda_inv.real, dtype=torch.float64, device=dev)
    eigvecs_selected = torch.as_tensor(
        np.real(eigvecs[:, mode_indices]),
        dtype=torch.float64,
        device=dev,
    )

    if repulsive_epsilon is None:
        repulsive_epsilon = 2.0 * float(dt)
    if well_centers is None:
        well_centers = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)

    metric_steps = set(range(0, int(n_iter) + 1, int(record_interval))) if record_interval else {0, int(n_iter)}
    requested = {0, int(n_iter), *(int(s) for s in (snapshot_steps or [])), *metric_steps}
    requested = {s for s in requested if 0 <= s <= int(n_iter)}
    snapshots: dict[int, np.ndarray] = {0: X0.copy()}
    step_norms = np.zeros(int(n_iter), dtype=np.float64)
    X_gpu = torch.as_tensor(X0, dtype=torch.float64, device=dev)

    metrics = {"steps": [], "well_coverage": [], "avg_potential": [], "movement_rate": []}

    def record_metrics(step: int, current: np.ndarray, movement: float = 0.0) -> None:
        dists = np.linalg.norm(current[:, None, :] - well_centers[None, :, :], axis=2)
        coverage = 100.0 * np.mean(np.min(dists, axis=1) <= float(well_radius))
        if potential is None:
            avg_potential = np.nan
        elif hasattr(potential, "potential"):
            avg_potential = float(np.mean(potential.potential(current)))
        elif callable(potential):
            avg_potential = float(np.mean(potential(current)))
        else:
            avg_potential = np.nan
        metrics["steps"].append(step)
        metrics["well_coverage"].append(coverage)
        metrics["avg_potential"].append(avg_potential)
        metrics["movement_rate"].append(movement)

    record_metrics(0, X0)
    report_every = max(1, int(n_iter) // 6)

    for step in range(1, int(n_iter) + 1):
        X_curr = X_gpu.detach().requires_grad_(True)
        psi_basis = dic(X_curr)
        psi_values = psi_basis @ eigvecs_selected
        grad_psi = torch.zeros((X0.shape[0], k_modes, X0.shape[1]), dtype=torch.float64, device=dev)
        for k in range(k_modes):
            if X_curr.grad is not None:
                X_curr.grad.zero_()
            psi_values[:, k].sum().backward(retain_graph=True)
            grad_psi[:, k, :] = X_curr.grad.detach()
        driving = torch.sum((psi_values.detach() * lambda_inv_t)[..., None] * grad_psi, dim=1)
        repulsive = _repulsive_force_torch(X_curr.detach(), float(repulsive_epsilon))
        update = float(step_size) * (driving + float(alpha_repulsive) * repulsive)
        update = _clip_by_norm(update, max_step_norm, torch)
        X_gpu = X_curr.detach() - update.detach()
        step_norms[step - 1] = float(torch.mean(torch.linalg.norm(update, dim=1)).detach().cpu().item())

        current_np = None
        if step in requested:
            current_np = X_gpu.detach().cpu().numpy()
            snapshots[step] = current_np.copy()
        if step in metric_steps:
            if current_np is None:
                current_np = X_gpu.detach().cpu().numpy()
            record_metrics(step, current_np, step_norms[step - 1])
        if progress and (step == int(n_iter) or step % report_every == 0):
            print(f"KSWGD: step {step}/{int(n_iter)}, mean step norm={step_norms[step - 1]:.3e}")

    for key in metrics:
        metrics[key] = np.asarray(metrics[key])
    final = X_gpu.detach().cpu().numpy()
    return MetricTransportResult(final=final.copy(), snapshots=snapshots, step_norms=step_norms, metrics=metrics)
