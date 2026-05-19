from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray


@dataclass
class ParticleResult:
    final: Array
    snapshots: dict[int, Array]
    accept_rate: float | None = None


def pairwise_sq_dists(X: Array, Y: Array | None = None) -> Array:
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    return np.maximum(np.sum(X * X, axis=1)[:, None] + np.sum(Y * Y, axis=1)[None, :] - 2.0 * X @ Y.T, 0.0)


def median_bandwidth(X: Array, Y: Array | None = None, scale: float = 1.0) -> float:
    D2 = pairwise_sq_dists(X, Y)
    vals = D2.ravel() if Y is not None else D2[np.triu_indices_from(D2, k=1)]
    vals = vals[vals > 1e-14]
    med = float(np.median(vals)) if len(vals) else 1.0
    return max(scale * med / (2.0 * np.log(max(len(X), 2))), 1e-10)


def kde_score(X: Array, target: Array, bandwidth: float | None = None) -> Array:
    X = np.asarray(X, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    bandwidth = median_bandwidth(target) if bandwidth is None else float(bandwidth)
    K = np.exp(-pairwise_sq_dists(X, target) / (2.0 * bandwidth))
    W = K / (np.sum(K, axis=1, keepdims=True) + 1e-12)
    return (W @ target - X) / bandwidth


def rbf_particle_velocity(X: Array, score: Array, bandwidth: float | None = None, matrix: Array | None = None) -> Array:
    X = np.asarray(X, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    n = len(X)
    bandwidth = median_bandwidth(X) if bandwidth is None else float(bandwidth)
    K = np.exp(-pairwise_sq_dists(X) / (2.0 * bandwidth))
    grad_term = (X * np.sum(K, axis=0)[:, None] - K.T @ X) / bandwidth
    velocity = (K @ score + grad_term) / max(n, 1)
    if matrix is not None:
        velocity = velocity @ matrix
    return velocity


def _snapshot_steps(n_iter: int, steps: Iterable[int] | None) -> list[int]:
    return sorted({0, int(n_iter), *(int(s) for s in (steps or []))})


def _clip_update(update: Array, max_norm: float | None) -> Array:
    if max_norm is None or max_norm <= 0:
        return update
    norms = np.linalg.norm(update, axis=1, keepdims=True)
    return update * np.minimum(1.0, float(max_norm) / (norms + 1e-300))


def run_ula(
    X_init: Array,
    *,
    score_fn: Callable[[Array], Array],
    n_iter: int,
    step_size: float,
    noise_scale: float = 1.0,
    snapshot_steps: Iterable[int] | None = None,
    project_fn: Callable[[Array], Array] | None = None,
    max_step_norm: float | None = None,
    seed: int = 0,
) -> ParticleResult:
    gen = np.random.default_rng(seed)
    X = np.asarray(X_init, dtype=np.float64).copy()
    steps = _snapshot_steps(n_iter, snapshot_steps)
    snapshots = {0: X.copy()}
    for step in range(1, int(n_iter) + 1):
        drift = float(step_size) * score_fn(X)
        drift = _clip_update(drift, max_step_norm)
        noise = float(noise_scale) * np.sqrt(2.0 * float(step_size)) * gen.normal(size=X.shape)
        X = X + drift + noise
        if project_fn is not None:
            X = project_fn(X)
        if step in steps:
            snapshots[step] = X.copy()
    return ParticleResult(final=X, snapshots=snapshots)


def run_svgd(
    X_init: Array,
    *,
    score_fn: Callable[[Array], Array],
    n_iter: int,
    step_size: float,
    snapshot_steps: Iterable[int] | None = None,
    project_fn: Callable[[Array], Array] | None = None,
    matrix: Array | None = None,
    adagrad: bool = True,
    max_step_norm: float | None = None,
) -> ParticleResult:
    X = np.asarray(X_init, dtype=np.float64).copy()
    steps = _snapshot_steps(n_iter, snapshot_steps)
    snapshots = {0: X.copy()}
    acc = np.zeros_like(X)
    for step in range(1, int(n_iter) + 1):
        velocity = rbf_particle_velocity(X, score_fn(X), matrix=matrix)
        if adagrad:
            acc += velocity * velocity
            update = float(step_size) * velocity / (np.sqrt(acc) + 1e-8)
        else:
            update = float(step_size) * velocity
        update = _clip_update(update, max_step_norm)
        X = X + update
        if project_fn is not None:
            X = project_fn(X)
        if step in steps:
            snapshots[step] = X.copy()
    return ParticleResult(final=X, snapshots=snapshots)


def run_matrix_svgd(
    X_init: Array,
    *,
    score_fn: Callable[[Array], Array],
    n_iter: int,
    step_size: float,
    snapshot_steps: Iterable[int] | None = None,
    project_fn: Callable[[Array], Array] | None = None,
    preconditioner: Array | None = None,
    reg: float = 5e-2,
    max_step_norm: float | None = None,
) -> ParticleResult:
    X0 = np.asarray(X_init, dtype=np.float64)
    if preconditioner is None:
        cov = np.cov(X0.T)
        preconditioner = cov + reg * np.eye(X0.shape[1])
    return run_svgd(
        X0,
        score_fn=score_fn,
        n_iter=n_iter,
        step_size=step_size,
        snapshot_steps=snapshot_steps,
        project_fn=project_fn,
        matrix=preconditioner,
        max_step_norm=max_step_norm,
    )


def run_mala(
    X_init: Array,
    *,
    log_prob_fn: Callable[[Array], Array],
    score_fn: Callable[[Array], Array],
    n_iter: int,
    step_size: float,
    snapshot_steps: Iterable[int] | None = None,
    seed: int = 0,
) -> ParticleResult:
    gen = np.random.default_rng(seed)
    X = np.asarray(X_init, dtype=np.float64).copy()
    steps = _snapshot_steps(n_iter, snapshot_steps)
    snapshots = {0: X.copy()}
    accepted = 0
    total = 0
    h = float(step_size)
    noise_sd = np.sqrt(2.0 * h)
    for step in range(1, int(n_iter) + 1):
        score_x = score_fn(X)
        mean_forward = X + h * score_x
        proposal = mean_forward + noise_sd * gen.normal(size=X.shape)
        score_y = score_fn(proposal)
        mean_backward = proposal + h * score_y
        log_q_xy = -np.sum((proposal - mean_forward) ** 2, axis=1) / (4.0 * h)
        log_q_yx = -np.sum((X - mean_backward) ** 2, axis=1) / (4.0 * h)
        log_alpha = log_prob_fn(proposal) + log_q_yx - log_prob_fn(X) - log_q_xy
        accept = np.log(gen.uniform(size=len(X))) < np.minimum(log_alpha, 0.0)
        X[accept] = proposal[accept]
        accepted += int(np.sum(accept))
        total += len(X)
        if step in steps:
            snapshots[step] = X.copy()
    return ParticleResult(final=X, snapshots=snapshots, accept_rate=100.0 * accepted / max(total, 1))


def run_hmc(
    X_init: Array,
    *,
    log_prob_fn: Callable[[Array], Array],
    score_fn: Callable[[Array], Array],
    n_iter: int,
    step_size: float,
    leapfrog_steps: int = 5,
    snapshot_steps: Iterable[int] | None = None,
    seed: int = 0,
) -> ParticleResult:
    gen = np.random.default_rng(seed)
    X = np.asarray(X_init, dtype=np.float64).copy()
    steps = _snapshot_steps(n_iter, snapshot_steps)
    snapshots = {0: X.copy()}
    accepted = 0
    total = 0
    eps = float(step_size)
    for step in range(1, int(n_iter) + 1):
        P0 = gen.normal(size=X.shape)
        X_prop = X.copy()
        P = P0 + 0.5 * eps * score_fn(X_prop)
        for lf in range(int(leapfrog_steps)):
            X_prop = X_prop + eps * P
            if lf != int(leapfrog_steps) - 1:
                P = P + eps * score_fn(X_prop)
        P = P + 0.5 * eps * score_fn(X_prop)
        current_H = -log_prob_fn(X) + 0.5 * np.sum(P0 * P0, axis=1)
        proposed_H = -log_prob_fn(X_prop) + 0.5 * np.sum(P * P, axis=1)
        log_alpha = current_H - proposed_H
        accept = np.log(gen.uniform(size=len(X))) < np.minimum(log_alpha, 0.0)
        X[accept] = X_prop[accept]
        accepted += int(np.sum(accept))
        total += len(X)
        if step in steps:
            snapshots[step] = X.copy()
    return ParticleResult(final=X, snapshots=snapshots, accept_rate=100.0 * accepted / max(total, 1))


def run_ldvi(
    X_init: Array,
    *,
    score_fn: Callable[[Array], Array],
    n_iter: int,
    step_size: float,
    snapshot_steps: Iterable[int] | None = None,
    seed: int = 0,
    project_fn: Callable[[Array], Array] | None = None,
    max_step_norm: float | None = None,
) -> ParticleResult:
    gen = np.random.default_rng(seed)
    X = np.asarray(X_init, dtype=np.float64).copy()
    steps = _snapshot_steps(n_iter, snapshot_steps)
    snapshots = {0: X.copy()}
    for step in range(1, int(n_iter) + 1):
        tau = step / max(int(n_iter), 1)
        update = float(step_size) * (1.0 - 0.5 * tau) * score_fn(X)
        update += 0.02 * np.sqrt(max(1.0 - tau, 0.0)) * gen.normal(size=X.shape)
        update = _clip_update(update, max_step_norm)
        X = X + update
        if project_fn is not None:
            X = project_fn(X)
        if step in steps:
            snapshots[step] = X.copy()
    return ParticleResult(final=X, snapshots=snapshots)


def run_torus_ula(
    theta_init: Array,
    phi_init: Array,
    *,
    R: float,
    r: float,
    n_iter: int,
    dt: float,
    seed: int = 0,
    include_surface_drift: bool = False,
    snapshot_steps: Iterable[int] | None = None,
    to_cartesian: Callable[[Array, Array, float, float], Array] | None = None,
) -> ParticleResult:
    if to_cartesian is None:
        from systems import angles_to_cartesian

        to_cartesian = angles_to_cartesian
    gen = np.random.default_rng(seed)
    theta = np.asarray(theta_init, dtype=np.float64).copy()
    phi = np.asarray(phi_init, dtype=np.float64).copy()
    steps = _snapshot_steps(n_iter, snapshot_steps)
    snapshots = {0: to_cartesian(theta, phi, R, r)}
    for step in range(1, int(n_iter) + 1):
        g_theta = R + r * np.cos(phi)
        drift_theta = np.sin(phi) / (r * g_theta**2) if include_surface_drift else 0.0
        theta = np.mod(theta + drift_theta * dt + np.sqrt(2.0 * dt) * gen.normal(size=theta.shape) / g_theta, 2.0 * np.pi)
        phi = np.mod(phi + np.sqrt(2.0 * dt) * gen.normal(size=phi.shape) / r, 2.0 * np.pi)
        if step in steps:
            snapshots[step] = to_cartesian(theta, phi, R, r)
    return ParticleResult(final=to_cartesian(theta, phi, R, r), snapshots=snapshots)


def run_torus_flow_matching(
    theta_init: Array,
    phi_init: Array,
    theta_target: Array,
    phi_target: Array,
    *,
    R: float,
    r: float,
    n_epochs: int = 20,
    batch_size: int = 256,
    n_steps: int = 20,
    hidden: int = 64,
    seed: int = 0,
    generalized: bool = False,
) -> ParticleResult:
    """Lightweight RFM/GFM baseline on angular coordinates of the flat torus."""
    import torch
    import torch.nn as nn

    from systems import angles_to_cartesian

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X0 = torch.tensor(np.column_stack([theta_init, phi_init]), dtype=torch.float32, device=device)
    X1 = torch.tensor(np.column_stack([theta_target, phi_target]), dtype=torch.float32, device=device)

    def logmap(x, y):
        return torch.atan2(torch.sin(y - x), torch.cos(y - x))

    def expmap(x, u):
        return torch.remainder(x + u, 2.0 * np.pi)

    class VectorField(nn.Module):
        def __init__(self):
            super().__init__()
            extra_time = 2 if generalized else 1
            self.net = nn.Sequential(
                nn.Linear(2 * 8 + extra_time, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2),
            )

        def features(self, x):
            feats = []
            for freq in range(1, 5):
                feats.extend([torch.sin(freq * x), torch.cos(freq * x)])
            return torch.cat(feats, dim=-1)

        def forward(self, x, t, s=None):
            if t.dim() == 1:
                t = t[:, None]
            parts = [self.features(x), t]
            if generalized:
                if s is None:
                    s = t
                if s.dim() == 1:
                    s = s[:, None]
                parts.append(s)
            return self.net(torch.cat(parts, dim=-1))

    model = VectorField().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_target = X1.shape[0]
    for _ in range(int(n_epochs)):
        perm = torch.randperm(n_target, device=device)
        for start in range(0, n_target, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            x1 = X1[idx]
            src_idx = torch.randint(0, X0.shape[0], (x1.shape[0],), device=device)
            x0 = X0[src_idx]
            t = torch.rand(x1.shape[0], device=device) * 0.98 + 0.01
            u = logmap(x0, x1)
            xt = expmap(x0, t[:, None] * u)
            if generalized:
                s = torch.rand_like(t) * t
                pred = model(xt, t, s)
            else:
                pred = model(xt, t)
            loss = torch.mean((pred - u) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        x = X0.clone()
        times = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=device)
        for k in range(int(n_steps)):
            s = times[k].expand(x.shape[0])
            t = times[k + 1].expand(x.shape[0])
            dt = times[k + 1] - times[k]
            v = model(x, t, s) if generalized else model(x, s)
            x = expmap(x, dt * v)

    arr = x.detach().cpu().numpy()
    final = angles_to_cartesian(arr[:, 0], arr[:, 1], R, r)
    return ParticleResult(final=final, snapshots={0: angles_to_cartesian(theta_init, phi_init, R, r), n_steps: final})


def torus_phi_tv(X: Array, R: float, r: float, bins: int = 40) -> float:
    from systems import cartesian_to_angles

    _, phi = cartesian_to_angles(X, R, r)
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    empirical = np.histogram(phi, bins=edges)[0] / len(phi)
    a = edges[:-1]
    b = edges[1:]
    expected = (R * (b - a) + r * (np.sin(b) - np.sin(a))) / (2 * np.pi * R)
    expected = np.maximum(expected, 0.0)
    expected = expected / np.sum(expected)
    return float(0.5 * np.sum(np.abs(empirical - expected)))


def mmd_rbf(X: Array, Y: Array, bandwidth: float | None = None) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if bandwidth is None:
        bandwidth = median_bandwidth(np.vstack([X, Y]), scale=2.0)
    Kxx = np.exp(-pairwise_sq_dists(X) / (2.0 * bandwidth))
    Kyy = np.exp(-pairwise_sq_dists(Y) / (2.0 * bandwidth))
    Kxy = np.exp(-pairwise_sq_dists(X, Y) / (2.0 * bandwidth))
    mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def sliced_wasserstein(X: Array, Y: Array, n_projections: int = 128, seed: int = 0) -> float:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    gen = np.random.default_rng(seed)
    directions = gen.normal(size=(int(n_projections), X.shape[1]))
    directions = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
    values = []
    q = np.linspace(0.0, 1.0, max(len(X), len(Y)))
    for direction in directions:
        px = np.sort(X @ direction)
        py = np.sort(Y @ direction)
        px = np.interp(q, np.linspace(0.0, 1.0, len(px)), px)
        py = np.interp(q, np.linspace(0.0, 1.0, len(py)), py)
        values.append(np.mean((px - py) ** 2))
    return float(np.sqrt(np.mean(values)))


def well_occupancy(X: Array, centers: Array, radius: float = 0.45) -> Array:
    dists = np.linalg.norm(np.asarray(X)[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
    return np.sum(dists < radius, axis=0)


def particle_well_coverage(X: Array, centers: Array, radius: float = 0.4) -> float:
    dists = np.linalg.norm(np.asarray(X)[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
    return float(100.0 * np.mean(np.min(dists, axis=1) <= float(radius)))


def well_density_percentages(X: Array, centers: Array, radius: float = 0.4) -> Array:
    dists = np.linalg.norm(np.asarray(X)[:, None, :] - np.asarray(centers)[None, :, :], axis=2)
    return 100.0 * np.sum(dists <= float(radius), axis=0) / max(len(X), 1)


def compute_kl_divergence_kde(
    particles: Array,
    potential,
    *,
    x_range: tuple[float, float] = (-2.0, 2.0),
    y_range: tuple[float, float] = (-2.0, 2.0),
    n_grid: int = 50,
) -> float:
    from scipy.stats import gaussian_kde

    x_eval = np.linspace(x_range[0], x_range[1], n_grid)
    y_eval = np.linspace(y_range[0], y_range[1], n_grid)
    X_eval, Y_eval = np.meshgrid(x_eval, y_eval)
    positions = np.vstack([X_eval.ravel(), Y_eval.ravel()])
    dx = (x_range[1] - x_range[0]) / n_grid
    dy = (y_range[1] - y_range[0]) / n_grid
    V_grid = potential.V(X_eval, Y_eval) if hasattr(potential, "V") else potential.potential(np.column_stack([X_eval.ravel(), Y_eval.ravel()])).reshape(X_eval.shape)
    pi_unnorm = np.exp(-V_grid / getattr(potential, "temperature", 1.0))
    pi_grid = pi_unnorm / (np.sum(pi_unnorm) * dx * dy)
    pi_flat = pi_grid.ravel()
    try:
        kde = gaussian_kde(np.asarray(particles).T, bw_method="scott")
        q_flat = kde(positions)
        q_flat = q_flat / (np.sum(q_flat) * dx * dy)
    except Exception:
        return 100.0
    eps = 1e-10
    q_safe = np.maximum(q_flat, eps)
    pi_safe = np.maximum(pi_flat, eps)
    kl = np.sum(q_flat * np.log(q_safe / pi_safe)) * dx * dy
    return float(max(kl, 0.0))
