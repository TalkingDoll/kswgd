from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class TorusConfig:
    R: float = 2.0
    r: float = 0.8
    n_target: int = 800
    n_particles: int = 2000
    n_iter: int = 5000
    step_size: float = 0.005
    dt_edmd: float = 0.05
    theta_init_center: float = 0.0
    phi_init_center: float = np.pi / 2
    theta_spread: float = 0.3
    phi_spread: float = 0.3
    seed: int = 42


def angles_to_cartesian(theta: np.ndarray, phi: np.ndarray, R: float, r: float) -> np.ndarray:
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return np.column_stack([x, y, z])


def cartesian_to_angles(X: np.ndarray, R: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    theta = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2 * np.pi)
    rho = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    phi = np.mod(np.arctan2(X[:, 2], rho - R), 2 * np.pi)
    return theta, phi


def sample_torus(config: TorusConfig, n: int | None = None, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n or config.n_target)
    gen = rng(config.seed if seed is None else seed)
    theta_samples: list[np.ndarray] = []
    phi_samples: list[np.ndarray] = []
    max_density = config.R + config.r
    count = 0
    while count < n:
        batch = max(256, int((n - count) * 1.7))
        theta = gen.uniform(0, 2 * np.pi, batch)
        phi = gen.uniform(0, 2 * np.pi, batch)
        accept_prob = (config.R + config.r * np.cos(phi)) / max_density
        keep = gen.uniform(0, 1, batch) < accept_prob
        theta_samples.append(theta[keep])
        phi_samples.append(phi[keep])
        count += int(np.sum(keep))
    theta_tar = np.concatenate(theta_samples)[:n]
    phi_tar = np.concatenate(phi_samples)[:n]
    X_tar = angles_to_cartesian(theta_tar, phi_tar, config.R, config.r)
    return X_tar, theta_tar, phi_tar


def torus_sde_step(theta: np.ndarray, phi: np.ndarray, config: TorusConfig, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen = rng(config.seed + 11 if seed is None else seed)
    g_theta = config.R + config.r * np.cos(phi)
    sigma_theta = np.sqrt(2.0) / g_theta
    sigma_phi = np.sqrt(2.0) / config.r
    ito_correction_theta = np.sin(phi) / (config.r * g_theta**2)
    dW1 = gen.normal(0.0, np.sqrt(config.dt_edmd), size=theta.shape)
    dW2 = gen.normal(0.0, np.sqrt(config.dt_edmd), size=phi.shape)
    theta_next = np.mod(theta + ito_correction_theta * config.dt_edmd + sigma_theta * dW1, 2 * np.pi)
    phi_next = np.mod(phi + sigma_phi * dW2, 2 * np.pi)
    X_next = angles_to_cartesian(theta_next, phi_next, config.R, config.r)
    return X_next, theta_next, phi_next


def init_torus_particles(config: TorusConfig, seed: int | None = None) -> np.ndarray:
    gen = rng(config.seed + 23 if seed is None else seed)
    theta = np.mod(
        config.theta_init_center + gen.uniform(-config.theta_spread, config.theta_spread, config.n_particles),
        2 * np.pi,
    )
    phi = np.mod(
        config.phi_init_center + gen.uniform(-config.phi_spread, config.phi_spread, config.n_particles),
        2 * np.pi,
    )
    return angles_to_cartesian(theta, phi, config.R, config.r)


def project_to_torus(X: np.ndarray, R: float, r: float) -> np.ndarray:
    theta, phi = cartesian_to_angles(X, R, r)
    return angles_to_cartesian(theta, phi, R, r)


def project_torus_gradient(gradient: np.ndarray, X: np.ndarray, R: float, r: float) -> np.ndarray:
    theta, phi = cartesian_to_angles(X, R, r)
    e_theta = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)])
    e_phi = np.column_stack([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)])
    return (np.sum(gradient * e_theta, axis=1, keepdims=True) * e_theta) + (
        np.sum(gradient * e_phi, axis=1, keepdims=True) * e_phi
    )


@dataclass
class QuadrupleWellConfig:
    n_target: int = 10000
    n_particles: int = 3000
    n_iter: int = 3000
    step_size: float = 0.05
    dt: float = 0.01
    burnin: int = 1000
    substeps: int = 10
    temperature: float = 1.0
    a: float = 1.0
    b: float = 1.0
    coupling: float = 0.0
    seed: int = 42
    bounds: tuple[float, float] = (-2.0, 2.0)


class FourWellPotential:
    centers = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=float)

    def __init__(self, temperature: float = 1.0, a: float = 1.0, b: float = 1.0, coupling: float = 0.0):
        self.temperature = float(temperature)
        self.a = float(a)
        self.b = float(b)
        self.coupling = float(coupling)

    def potential(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        x, y = X[..., 0], X[..., 1]
        return self.a * (x**2 - 1.0) ** 2 + self.b * (y**2 - 1.0) ** 2 + self.coupling * x * y

    def V(self, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        if y is None:
            return self.potential(x)
        pts = np.stack([x, y], axis=-1)
        return self.potential(pts)

    def grad(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        x, y = X[..., 0], X[..., 1]
        return np.stack(
            [
                4.0 * self.a * x * (x**2 - 1.0) + self.coupling * y,
                4.0 * self.b * y * (y**2 - 1.0) + self.coupling * x,
            ],
            axis=-1,
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.grad(X) / self.temperature

    def grad_V(self, X: np.ndarray) -> np.ndarray:
        return self.grad(X)

    def stationary_density(self, x: np.ndarray, y: np.ndarray, x_range=(-3, 3), y_range=(-3, 3), n_grid: int = 500):
        V_val = self.V(x, y)
        unnormalized = np.exp(-V_val / self.temperature)
        x_int = np.linspace(x_range[0], x_range[1], n_grid)
        y_int = np.linspace(y_range[0], y_range[1], n_grid)
        dx = (x_range[1] - x_range[0]) / n_grid
        dy = (y_range[1] - y_range[0]) / n_grid
        X_int, Y_int = np.meshgrid(x_int, y_int)
        Z = np.sum(np.exp(-self.V(X_int, Y_int) / self.temperature)) * dx * dy
        return unnormalized / Z

    def density_on_grid(self, n_grid: int = 180, bounds: tuple[float, float] = (-2.0, 2.0)):
        lo, hi = bounds
        xs = np.linspace(lo, hi, n_grid)
        ys = np.linspace(lo, hi, n_grid)
        Xg, Yg = np.meshgrid(xs, ys)
        pts = np.column_stack([Xg.ravel(), Yg.ravel()])
        V = self.potential(pts).reshape(n_grid, n_grid)
        rho = np.exp(-V / self.temperature)
        rho /= np.trapezoid(np.trapezoid(rho, ys, axis=0), xs)
        return Xg, Yg, V, rho


def simulate_langevin_target(config: QuadrupleWellConfig, potential: FourWellPotential | None = None) -> tuple[np.ndarray, np.ndarray]:
    potential = potential or FourWellPotential(config.temperature, a=config.a, b=config.b, coupling=config.coupling)
    gen = rng(config.seed)
    X = gen.normal(0.0, 0.8, size=(config.n_target + config.burnin + 1, 2))
    x = X[0]
    samples = []
    next_samples = []
    dt_sub = config.dt / max(1, config.substeps)
    noise = np.sqrt(2.0 * config.temperature * dt_sub)
    total = config.n_target + config.burnin + 1
    for i in range(total):
        prev = x.copy()
        for _ in range(max(1, config.substeps)):
            x = x - dt_sub * potential.grad(x) + noise * gen.normal(size=2)
            x = np.clip(x, config.bounds[0] * 2, config.bounds[1] * 2)
        if i >= config.burnin and len(samples) < config.n_target:
            samples.append(prev.copy())
            next_samples.append(x.copy())
    return np.asarray(samples), np.asarray(next_samples)


def generate_quadruple_langevin_trajectory(
    config: QuadrupleWellConfig,
    potential: FourWellPotential | None = None,
    *,
    return_trajectory: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the original consecutive Langevin time-series target for test 2."""
    potential = potential or FourWellPotential(config.temperature, a=config.a, b=config.b, coupling=config.coupling)
    gen = np.random.RandomState(config.seed)
    x = gen.randn(2) * 0.5
    trajectory = []
    dt_sub = config.dt / max(1, config.substeps)
    noise = np.sqrt(2.0 * config.temperature * dt_sub)
    total = int(config.burnin + config.n_target)
    for i in range(total):
        for _ in range(max(1, config.substeps)):
            x = x - dt_sub * potential.grad(x.reshape(1, -1))[0] + noise * gen.randn(2)
        if i >= config.burnin:
            trajectory.append(x.copy())
    trajectory_arr = np.asarray(trajectory, dtype=np.float64)
    X_target = trajectory_arr[:-1].copy()
    X_next = trajectory_arr[1:].copy()
    if return_trajectory:
        return X_target, X_next, trajectory_arr
    return X_target, X_next


def langevin_step_batch(
    X: np.ndarray,
    config: QuadrupleWellConfig,
    potential: FourWellPotential | None = None,
    seed: int | None = None,
) -> np.ndarray:
    potential = potential or FourWellPotential(config.temperature, a=config.a, b=config.b, coupling=config.coupling)
    gen = rng(config.seed + 17 if seed is None else seed)
    X_next = np.asarray(X, dtype=np.float64).copy()
    dt_sub = config.dt / max(1, config.substeps)
    noise = np.sqrt(2.0 * config.temperature * dt_sub)
    for _ in range(max(1, config.substeps)):
        X_next = X_next - dt_sub * potential.grad(X_next) + noise * gen.normal(size=X_next.shape)
        X_next = np.clip(X_next, config.bounds[0] * 2, config.bounds[1] * 2)
    return X_next


def sample_quadruple_boltzmann_grid(
    config: QuadrupleWellConfig,
    potential: FourWellPotential | None = None,
    n: int | None = None,
    grid_size: int = 320,
    seed: int | None = None,
) -> np.ndarray:
    potential = potential or FourWellPotential(config.temperature, a=config.a, b=config.b, coupling=config.coupling)
    n = int(config.n_target if n is None else n)
    gen = rng(config.seed if seed is None else seed)
    lo, hi = config.bounds
    xs = np.linspace(lo, hi, grid_size)
    ys = np.linspace(lo, hi, grid_size)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    rho = np.exp(-potential.potential(pts) / config.temperature)
    prob = rho / np.sum(rho)
    idx = gen.choice(len(pts), size=n, replace=True, p=prob)
    dx = (hi - lo) / max(grid_size - 1, 1)
    return pts[idx] + gen.uniform(-0.5 * dx, 0.5 * dx, size=(n, 2))


def init_quadruple_particles(config: QuadrupleWellConfig) -> np.ndarray:
    gen = rng(config.seed + 31)
    centers = FourWellPotential.centers
    particles: list[np.ndarray] = []
    count = 0
    while count < config.n_particles:
        cand = gen.uniform(config.bounds[0], config.bounds[1], size=(max(512, config.n_particles), 2))
        dist = np.linalg.norm(cand[:, None, :] - centers[None, :, :], axis=2)
        keep = cand[np.min(dist, axis=1) > 0.4]
        particles.append(keep)
        count += len(keep)
    return np.concatenate(particles, axis=0)[: config.n_particles]


def init_quadruple_particles_uniform_outside(
    config: QuadrupleWellConfig,
    *,
    radius: float = 0.4,
    seed: int | None = None,
) -> np.ndarray:
    """Original test-2 initialization: draw candidates uniformly, then remove well interiors."""
    gen = np.random.RandomState(config.seed if seed is None else seed)
    lo, hi = config.bounds
    candidates = gen.uniform(low=[lo, lo], high=[hi, hi], size=(config.n_particles, 2))
    dists = np.linalg.norm(candidates[:, None, :] - FourWellPotential.centers[None, :, :], axis=2)
    return candidates[np.min(dists, axis=1) > float(radius)]


def well_coverage(X: np.ndarray, radius: float = 0.45) -> float:
    X = np.asarray(X)
    dists = np.linalg.norm(X[:, None, :] - FourWellPotential.centers[None, :, :], axis=2)
    covered = np.any(dists < radius, axis=0)
    return 100.0 * float(np.mean(covered))


@dataclass
class AllenCahnConfig:
    sigma: float = 1.4142
    N: int = 128
    n_snapshots: int = 900
    burnin: int = 25
    dt: float = 2e-5
    epsilon: float = 0.01
    diffusion: float = 0.001
    seed: int = 29
    latent_dim: int = 16
    n_particles: int = 300
    n_iter: int = 1500
    step_size: float = 0.08


class StochasticAllenCahn2D:
    def __init__(self, config: AllenCahnConfig):
        self.config = config
        self.dx = 1.0 / config.N

    def laplacian(self, U: np.ndarray) -> np.ndarray:
        return (
            np.roll(U, 1, axis=0)
            + np.roll(U, -1, axis=0)
            + np.roll(U, 1, axis=1)
            + np.roll(U, -1, axis=1)
            - 4.0 * U
        ) / (self.dx**2)

    def step(self, U: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        cfg = self.config
        reaction = (U - U**3) / (cfg.epsilon**2)
        drift = cfg.diffusion * self.laplacian(U) + reaction
        noise = cfg.sigma * np.sqrt(cfg.dt) * gen.normal(size=U.shape)
        return np.clip(U + cfg.dt * drift + noise, -1.5, 1.5)

    def simulate(self) -> np.ndarray:
        cfg = self.config
        gen = rng(cfg.seed)
        U = gen.uniform(-0.5, 0.5, size=(cfg.N, cfg.N))
        fields = []
        for t in range(cfg.burnin + cfg.n_snapshots):
            U = self.step(U, gen)
            if t >= cfg.burnin:
                fields.append(U.copy())
        return np.asarray(fields)


def pca_encode(fields: np.ndarray, latent_dim: int) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], dict[str, np.ndarray]]:
    X = fields.reshape(fields.shape[0], -1)
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    components = vt[:latent_dim]
    Z = Xc @ components.T

    def decode(Z_new: np.ndarray) -> np.ndarray:
        flat = np.asarray(Z_new) @ components + mean
        return flat.reshape((-1,) + fields.shape[1:])

    return Z, decode, {"mean": mean, "components": components}


def sample_ac_initial_fields(
    n_samples: int,
    N: int = 128,
    mode: str = "uniform_fields",
    low: float = -0.5,
    high: float = 0.5,
    noise_std: float = 0.15,
    seed: int | None = None,
) -> np.ndarray:
    gen = rng(seed)
    if mode == "uniform_fields":
        fields = gen.uniform(low, high, size=(n_samples, N, N))
    elif mode == "zero_gaussian_fields":
        fields = gen.normal(0.0, noise_std, size=(n_samples, N, N))
    elif mode == "constant_zero_fields":
        fields = np.zeros((n_samples, N, N), dtype=np.float64)
    else:
        raise ValueError(f"Unknown initial field mode: {mode}")
    return fields.astype(np.float32)


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, optim, DataLoader, TensorDataset


def _make_ac_autoencoder(input_size: int, latent_dim: int):
    torch, nn, _, _, _ = _torch_modules()

    class ACEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            flat_size = input_size * input_size
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, latent_dim),
            )

        def forward(self, x):
            return self.net(x)

    class ACDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            flat_size = input_size * input_size
            self.net = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, flat_size),
                nn.Tanh(),
            )

        def forward(self, z):
            return self.net(z).view(-1, input_size, input_size)

    class ACAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ACEncoder()
            self.decoder = ACDecoder()

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return ACAutoencoder()


def train_ac_autoencoder(
    fields: np.ndarray,
    latent_dim: int,
    n_epochs: int = 120,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[object, float, float, str]:
    torch, nn, optim, DataLoader, TensorDataset = _torch_modules()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    fields = np.asarray(fields, dtype=np.float32)
    n_samples, N, _ = fields.shape
    flat = fields.reshape(n_samples, N * N)
    data_min = float(flat.min())
    data_max = float(flat.max())
    data_range = data_max - data_min + 1e-8
    scaled = 2.0 * (flat - data_min) / data_range - 1.0

    tensor = torch.as_tensor(scaled, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    ae = _make_ac_autoencoder(N, latent_dim).to(device).float()
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ae.train()
    for epoch in range(int(n_epochs)):
        total = 0.0
        for (batch,) in loader:
            batch = batch.view(-1, N, N)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = ae(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * batch.shape[0]
        if verbose and ((epoch + 1) % 25 == 0 or epoch == 0 or epoch + 1 == int(n_epochs)):
            print(f"AE epoch {epoch + 1}/{n_epochs}: loss={total / n_samples:.6e}")
    ae.eval()
    return ae, data_min, data_max, device


def encode_ac_fields(ae, fields: np.ndarray, data_min: float, data_max: float, device: str, batch_size: int = 256) -> np.ndarray:
    torch, _, _, _, _ = _torch_modules()
    fields = np.asarray(fields, dtype=np.float32)
    n_samples, N, _ = fields.shape
    flat = fields.reshape(n_samples, N * N)
    scaled = 2.0 * (flat - data_min) / (data_max - data_min + 1e-8) - 1.0
    out = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            tensor = torch.as_tensor(scaled[start : start + batch_size], dtype=torch.float32, device=device).view(-1, N, N)
            out.append(ae.encoder(tensor).detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def decode_ac_latent(
    ae,
    z: np.ndarray,
    N: int,
    data_min: float,
    data_max: float,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    torch, _, _, _, _ = _torch_modules()
    z = np.asarray(z, dtype=np.float32)
    decoded = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            tensor = torch.as_tensor(z[start : start + batch_size], dtype=torch.float32, device=device)
            decoded.append(ae.decoder(tensor).detach().cpu().numpy())
    scaled = np.concatenate(decoded, axis=0).reshape(len(z), N * N)
    fields = (scaled + 1.0) / 2.0 * (data_max - data_min + 1e-8) + data_min
    return fields.reshape(-1, N, N)


def ac_latent_bandwidth(
    Z_target: np.ndarray,
    Z_reference: np.ndarray | None = None,
    c_epsilon: float = 0.5,
    reference_scale: float = 0.2,
) -> float:
    Z_target = np.asarray(Z_target, dtype=np.float64)
    sq = np.sum(Z_target * Z_target, axis=1)
    H = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z_target @ Z_target.T, 0.0)
    eps_target = max(float(np.median(H) * c_epsilon / np.log(len(Z_target) + 1.0)), 1e-10)
    if Z_reference is None:
        return eps_target
    Z_reference = np.asarray(Z_reference, dtype=np.float64)
    sq_ref = np.sum(Z_reference * Z_reference, axis=1)
    H_ref = np.maximum(sq_ref[:, None] + sq[None, :] - 2.0 * Z_reference @ Z_target.T, 0.0)
    eps_ref = max(float(reference_scale * np.median(H_ref)), 1e-10)
    return max(eps_target, eps_ref)


def init_latent_particles(Z: np.ndarray, n_particles: int, seed: int | None = None, scale: float = 0.35) -> np.ndarray:
    gen = rng(seed)
    center = Z.mean(axis=0)
    std = Z.std(axis=0) + 1e-6
    return center + scale * gen.normal(size=(n_particles, Z.shape[1])) * std
