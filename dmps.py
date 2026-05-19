from __future__ import annotations

import numpy as np

from kswgd import (
    KernelSpectrum2D,
    MetricTransportResult,
    SpectralModel,
    TransportResult,
    fit_diffusion_map_kernel_spectrum_2d,
    fit_spectral_model,
    run_kernel_spectrum_transport_2d,
    run_kswgd,
)


def fit_dmps(
    X_target: np.ndarray,
    *,
    epsilon: float | None = None,
    max_modes: int = 80,
    kernel_scale: float = 0.5,
    normalize_weights: bool = False,
) -> SpectralModel:
    return fit_spectral_model(
        X_target,
        X_next=None,
        epsilon=epsilon,
        max_modes=max_modes,
        kernel_scale=kernel_scale,
        normalize_weights=normalize_weights,
        method="DMPS",
    )


def run_dmps(
    model: SpectralModel,
    X_init: np.ndarray,
    *,
    n_iter: int,
    step_size: float,
    snapshot_steps=None,
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
    return run_kswgd(
        model,
        X_init,
        n_iter=n_iter,
        step_size=step_size,
        snapshot_steps=snapshot_steps,
        use_gpu=use_gpu,
        torus=torus,
        project_fn=project_fn,
        gradient_project_fn=gradient_project_fn,
        max_step_norm=max_step_norm,
        normalize_by_particles=normalize_by_particles,
        adaptive_initial_movement=adaptive_initial_movement,
        max_step_multiplier=max_step_multiplier,
        progress=progress,
    )


def fit_diffusion_map_dmps_2d(
    X_target: np.ndarray,
    *,
    dt: float,
    max_modes: int | None = 1000,
    lowrank_niter: int = 5,
    device: str | None = None,
) -> KernelSpectrum2D:
    return fit_diffusion_map_kernel_spectrum_2d(
        X_target,
        dt=dt,
        max_modes=max_modes,
        lowrank_niter=lowrank_niter,
        device=device,
        method="DMPS",
    )


def run_diffusion_map_dmps_2d(
    model: KernelSpectrum2D,
    X_init: np.ndarray,
    *,
    n_iter: int,
    step_size: float,
    alpha_repulsive: float = 0.5,
    snapshot_steps=None,
    record_interval: int = 100,
    potential=None,
    well_centers=None,
    well_radius: float = 0.4,
    device: str | None = None,
    progress: bool = True,
) -> MetricTransportResult:
    return run_kernel_spectrum_transport_2d(
        model,
        X_init,
        n_iter=n_iter,
        step_size=step_size,
        alpha_repulsive=alpha_repulsive,
        snapshot_steps=snapshot_steps,
        record_interval=record_interval,
        potential=potential,
        well_centers=well_centers,
        well_radius=well_radius,
        device=device,
        progress=progress,
    )
