from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from plotting_utils import finish_layout, polish_axes, save_figure

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    display = lambda fig: None


def torus_surface(R: float, r: float, n_theta: int = 72, n_phi: int = 36):
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi)
    X = (R + r * np.cos(PH)) * np.cos(TH)
    Y = (R + r * np.cos(PH)) * np.sin(TH)
    Z = r * np.sin(PH)
    return X, Y, Z


def set_pi_ticks(ax) -> None:
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def clean_tick_labels(ax) -> None:
    for axis in [ax.xaxis, ax.yaxis]:
        ticks = axis.get_ticklocs()
        labels = []
        for value in ticks:
            try:
                labels.append(f"{float(value):.10f}".rstrip("0").rstrip("."))
            except Exception:
                labels.append(str(value))
        if axis is ax.xaxis:
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)


def plot_torus_triptych(
    X_target: np.ndarray,
    X_initial: np.ndarray,
    X_final: np.ndarray,
    *,
    R: float,
    r: float,
    method: str,
    stem: str | Path,
    caption: str,
    final_color: str = "magenta",
) -> None:
    Xs, Ys, Zs = torus_surface(R, r)
    fig = plt.figure(figsize=(18.2, 5.25))
    panels = [
        ("Target", X_target, "blue", 3, 0.70),
        ("Initial", X_initial, "red", 9, 0.86),
        ("Final", X_final, final_color, 9, 0.86),
    ]
    for i, (title, points, color, size, alpha) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        ax.plot_surface(Xs, Ys, Zs, alpha=0.20, cmap="viridis", edgecolor="none")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=alpha, edgecolors="none")
        ax.set_title(title, pad=8, fontsize=18)
        ax.set_xlabel("X", labelpad=5, fontsize=13)
        ax.set_ylabel("Y", labelpad=5, fontsize=13)
        ax.set_zlabel("Z", labelpad=5, fontsize=13)
        ax.set_box_aspect([1, 1, 0.42])
        ax.view_init(elev=24, azim=-55)
        ax.tick_params(labelsize=9, pad=1)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.98, wspace=0.00)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_torus_angular(
    theta_target: np.ndarray,
    phi_target: np.ndarray,
    theta_initial: np.ndarray,
    phi_initial: np.ndarray,
    theta_final: np.ndarray,
    phi_final: np.ndarray,
    *,
    method: str,
    stem: str | Path,
    caption: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 6.1))
    panel_data = [
        ("Target (θ, φ)", theta_target, phi_target, "C0", 10, "filled"),
        ("Initial (θ, φ)", theta_initial, phi_initial, "red", 18, "filled"),
        ("Final (θ, φ)", theta_final, phi_final, "magenta", 18, "open"),
    ]
    for ax, (title, theta, phi, color, size, style) in zip(axes, panel_data):
        if style == "open":
            ax.scatter(theta, phi, s=size, facecolors="none", edgecolors=color, linewidths=0.8, alpha=0.72)
        else:
            ax.scatter(theta, phi, s=size, c=color, alpha=0.56, edgecolors="none")
        ax.set_title(title, pad=16)
        ax.set_xlabel("θ")
        ax.set_ylabel("φ")
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.28)
        set_pi_ticks(ax)
    fig.suptitle(f"Angular Coordinate Distributions - {method}", y=0.98)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.12, top=0.82, wspace=0.30)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_torus_overlap_matrix(
    X_target: np.ndarray,
    X_final: np.ndarray,
    *,
    method: str,
    stem: str | Path,
    caption: str,
    mode: str = "frequency",
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 11.0))
    columns = ["x", "y", "z"]
    df_target = pd.DataFrame(X_target, columns=columns)
    df_final = pd.DataFrame(X_final, columns=columns)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                target = df_target[columns[i]].values
                final = df_final[columns[i]].values
                if mode == "kde":
                    lo = min(float(target.min()), float(final.min()))
                    hi = max(float(target.max()), float(final.max()))
                    grid = np.linspace(lo, hi, 200)
                    ax.fill_between(grid, gaussian_kde(target)(grid), alpha=0.34, color="blue", label="Target")
                    ax.fill_between(grid, gaussian_kde(final)(grid), alpha=0.34, color="red", label="Final")
                    ax.plot(grid, gaussian_kde(target)(grid), color="blue", linewidth=1.5)
                    ax.plot(grid, gaussian_kde(final)(grid), color="red", linewidth=1.5)
                else:
                    counts_target, edges = np.histogram(target, bins=15, density=True)
                    counts_final, _ = np.histogram(final, bins=edges, density=True)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.fill_between(centers, counts_target, alpha=0.34, color="blue", label="Target")
                    ax.fill_between(centers, counts_final, alpha=0.34, color="red", label="Final")
                    ax.plot(centers, counts_target, color="blue", linewidth=1.5, marker="o", markersize=3)
                    ax.plot(centers, counts_final, color="red", linewidth=1.5, marker="o", markersize=3)
            else:
                ax.scatter(df_target[columns[j]], df_target[columns[i]], alpha=0.26, color="blue", s=5, label="Target")
                ax.scatter(df_final[columns[j]], df_final[columns[i]], alpha=0.26, color="red", s=5, label="Final")
            clean_tick_labels(ax)
            if i == 2:
                ax.set_xlabel(columns[j])
            if j == 0:
                ax.set_ylabel(columns[i])
    for i in range(3):
        axes[i, 2].set_xlim(-1, 1)
    axes[2, 0].set_ylim(-1, 1)
    axes[2, 1].set_ylim(-1, 1)
    fig.suptitle(f"Overlapping Scatter Matrix (Target vs Final) - {method}", y=0.975)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.010), frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.130, top=0.925, wspace=0.24, hspace=0.24)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def _safe_kde(points: np.ndarray, positions: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return np.zeros(shape)
    try:
        return gaussian_kde(points.T, bw_method="scott")(positions).reshape(shape)
    except np.linalg.LinAlgError:
        points = points + 1e-4 * np.random.default_rng(0).normal(size=points.shape)
        return gaussian_kde(points.T, bw_method="scott")(positions).reshape(shape)


def plot_quad_snapshots(
    snapshots_by_method: Mapping[str, Mapping[int, np.ndarray]],
    *,
    potential,
    bounds: tuple[float, float],
    stem: str | Path,
    caption: str,
) -> None:
    first_steps = sorted(next(iter(snapshots_by_method.values())).keys())
    Xg, Yg, _, rho = potential.density_on_grid(n_grid=170, bounds=bounds)
    n_rows = len(snapshots_by_method)
    n_cols = len(first_steps)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.25 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    vmax = np.percentile(rho, 99.5)
    for row, (method, snapshots) in enumerate(snapshots_by_method.items()):
        for col, step in enumerate(first_steps):
            ax = axes[row, col]
            ax.contourf(Xg, Yg, rho, levels=24, cmap="Greys", alpha=0.60, vmin=0, vmax=vmax)
            pts = snapshots[step]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.66, edgecolors="none")
            ax.set_title(f"{method}, step {step}", pad=14)
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            if row == n_rows - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
            polish_axes(ax, equal=True)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.92, wspace=0.18, hspace=0.34)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_quad_heatmaps(
    density_sources: Mapping[str, np.ndarray | str],
    *,
    potential,
    bounds: tuple[float, float],
    stem: str | Path,
    caption: str,
    n_grid: int = 150,
    vmax_source: str | None = None,
    cmap_name: str = "viridis",
    density_gamma: float = 1.25,
    vmax_percentile: float = 95.0,
) -> None:
    lo, hi = bounds
    xs = np.linspace(lo, hi, n_grid)
    ys = np.linspace(lo, hi, n_grid)
    Xg, Yg = np.meshgrid(xs, ys)
    positions = np.vstack([Xg.ravel(), Yg.ravel()])
    _, _, _, boltzmann = potential.density_on_grid(n_grid=n_grid, bounds=bounds)
    density_maps = {}
    for name, source in density_sources.items():
        density_maps[name] = boltzmann if isinstance(source, str) and source == "boltzmann" else _safe_kde(source, positions, Xg.shape)
    if vmax_source is not None and vmax_source in density_maps:
        vmax = float(np.percentile(density_maps[vmax_source], vmax_percentile))
    else:
        vmax = max(float(np.percentile(v, vmax_percentile)) for v in density_maps.values())
    vmax = max(vmax, 1e-12)
    levels = np.linspace(0.0, vmax, 64)
    norm = colors.PowerNorm(gamma=float(density_gamma), vmin=0.0, vmax=vmax, clip=True)
    cmap = plt.get_cmap(cmap_name).copy()
    n = len(density_maps)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.35 * n_cols, 4.85 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    im = None
    for idx, (ax, (name, density)) in enumerate(zip(axes, density_maps.items())):
        density_plot = np.clip(density, 0.0, vmax)
        im = ax.contourf(Xg, Yg, density_plot, levels=levels, cmap=cmap, norm=norm)
        ax.contour(Xg, Yg, density_plot, levels=levels[::6], colors="white", alpha=0.30, linewidths=0.45)
        ax.set_title(name, pad=12, fontsize=17)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("x")
        if idx % n_cols == 0:
            ax.set_ylabel("y")
        polish_axes(ax, equal=True)
    for ax in axes[len(density_maps) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.24, top=0.88, wspace=0.045, hspace=0.34)
    if im is not None:
        cax = fig.add_axes([0.24, 0.085, 0.52, 0.026])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks(np.linspace(0.0, vmax, 6))
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar.set_label("Density", labelpad=8)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_quad_well_bars(
    snapshots_by_method: Mapping[str, Mapping[int, np.ndarray]],
    *,
    centers: np.ndarray,
    radius: float,
    stem: str | Path,
    caption: str,
) -> None:
    methods = list(snapshots_by_method)
    steps = sorted(next(iter(snapshots_by_method.values())).keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.3 * len(methods), 4.9), sharey=True)
    axes = np.atleast_1d(axes)
    labels = ["(-1,-1)", "(-1,1)", "(1,-1)", "(1,1)"]
    x = np.arange(len(labels))
    width = 0.78 / len(steps)
    for ax, method in zip(axes, methods):
        for k, step in enumerate(steps):
            pts = snapshots_by_method[method][step]
            dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
            counts = np.sum(dists < radius, axis=0)
            ax.bar(x + (k - (len(steps) - 1) / 2) * width, counts, width=width, label=f"step {step}")
        ax.set_title(method, pad=14)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        if ax is axes[0]:
            ax.set_ylabel("particle count")
        polish_axes(ax, grid=True)
    axes[-1].legend(frameon=False, loc="upper right")
    finish_layout(fig, top=0.88, w_pad=1.1)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def summarize_fields(name: str, fields: np.ndarray) -> str:
    fields = np.asarray(fields)
    return f"{name}: mean={fields.mean():+.3f}, std={fields.std():.3f}, frac(|u|>0.8)={np.mean(np.abs(fields) > 0.8):.2%}"


def plot_quad_pair_snapshots(
    snapshots_by_method: Mapping[str, Mapping[int, np.ndarray]],
    *,
    X_target: np.ndarray,
    potential,
    bounds: tuple[float, float],
    centers: np.ndarray,
    radius: float,
    stem: str | Path,
    caption: str,
) -> None:
    Xg, Yg, _, rho = potential.density_on_grid(n_grid=170, bounds=bounds)
    methods = list(snapshots_by_method)
    steps = sorted(next(iter(snapshots_by_method.values())).keys())
    fig, axes = plt.subplots(len(methods), len(steps), figsize=(5.15 * len(steps), 4.85 * len(methods)), sharex=True, sharey=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    for row, method in enumerate(methods):
        for col, step in enumerate(steps):
            ax = axes[row, col]
            ax.contourf(Xg, Yg, rho, levels=30, cmap="Blues", alpha=0.70)
            ax.scatter(X_target[:, 0], X_target[:, 1], s=2.2, c="lightgray", alpha=0.16, edgecolors="none")
            pts = snapshots_by_method[method][step]
            if step == 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="red", marker="o", label="Initial", zorder=5, edgecolors="none")
            else:
                ax.scatter(pts[:, 0], pts[:, 1], s=13, facecolors="none", edgecolors="magenta", linewidths=0.8, label=f"Step {step}", zorder=8)
            for center in centers:
                ax.add_patch(
                    Circle(
                        center,
                        radius,
                        fill=False,
                        edgecolor="limegreen",
                        linewidth=2.2,
                        linestyle="--",
                        zorder=20,
                    )
                )
            ax.set_title(f"{method}: step {step}", pad=12)
            if row == len(methods) - 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            polish_axes(ax, equal=True, grid=True)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.075, top=0.91, wspace=0.10, hspace=0.24)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_quad_movement_rate(
    metrics_by_method: Mapping[str, Mapping[str, np.ndarray]],
    *,
    stem: str | Path,
    caption: str,
    mark_step: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.7))
    colors = {"DMPS": "#E74C3C", "KSWGD": "#3498DB"}
    markers = {"DMPS": "o", "KSWGD": "s"}
    for method, metrics in metrics_by_method.items():
        steps = np.asarray(metrics["steps"])
        movement = np.asarray(metrics["movement_rate"])
        ax.plot(steps[1:], movement[1:], linewidth=2.4, marker=markers.get(method, "o"), markersize=4, markevery=5, label=method, color=colors.get(method))
    if mark_step is not None:
        ax.axvline(x=mark_step, color="gray", linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlabel("Iteration Steps")
    ax.set_ylabel("Movement Rate (Avg Displacement)")
    ax.set_title("Particle Stability", pad=14)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", framealpha=0.9)
    polish_axes(ax)
    finish_layout(fig, top=0.90)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_quad_well_density_evolution(
    snapshots_by_method: Mapping[str, Mapping[int, np.ndarray]],
    *,
    centers: np.ndarray,
    radius: float,
    stem: str | Path,
    caption: str,
) -> None:
    methods = list(snapshots_by_method)
    steps = sorted(next(iter(snapshots_by_method.values())).keys())
    labels = ["Well 1\n(-1,-1)", "Well 2\n(-1,1)", "Well 3\n(1,-1)", "Well 4\n(1,1)"]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
    fig, axes = plt.subplots(1, len(methods), figsize=(7.8 * len(methods), 5.9), sharey=True)
    axes = np.atleast_1d(axes)
    x_base = np.arange(len(steps))
    width = 0.18
    for ax, method in zip(axes, methods):
        densities = np.zeros((4, len(steps)))
        for t_idx, step in enumerate(steps):
            pts = snapshots_by_method[method][step]
            dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
            densities[:, t_idx] = 100.0 * np.sum(dists <= radius, axis=0) / max(len(pts), 1)
        for w_idx in range(4):
            ax.bar(x_base + (w_idx - 1.5) * width, densities[w_idx], width, label=labels[w_idx], color=colors[w_idx], edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axhline(y=25, color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"{method}: 4-Well Density Evolution", pad=12)
        ax.set_xlabel("Iteration Steps")
        if ax is axes[0]:
            ax.set_ylabel("Particle Density (%)")
        ax.set_xticks(x_base)
        ax.set_xticklabels([str(s) for s in steps], rotation=0)
        ax.set_ylim(0, 35)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=10, ncol=2)
        polish_axes(ax)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.11, top=0.88, wspace=0.12)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_quad_kl_divergence(
    kl_by_method: Mapping[str, np.ndarray],
    steps: Iterable[int],
    *,
    training_kl: float,
    stem: str | Path,
    caption: str,
    mark_step: int | None = None,
) -> None:
    steps = list(steps)
    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    colors = {"DMPS": "#E74C3C", "KSWGD": "#3498DB"}
    markers = {"DMPS": "o", "KSWGD": "s"}
    for method, values in kl_by_method.items():
        ax.plot(steps, values, linewidth=2.4, marker=markers.get(method, "o"), markersize=4.5, markevery=5, label=method, color=colors.get(method))
    ax.axhline(y=training_kl, color="green", linestyle="-", linewidth=2.3, label="Training Data", alpha=0.82)
    ax.fill_between(steps, 0, training_kl, color="green", alpha=0.10)
    if mark_step is not None:
        ax.axvline(x=mark_step, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("Iteration Steps")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence vs Boltzmann Distribution", pad=14)
    ax.set_xlim(min(steps), max(steps))
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=12, loc="upper right", framealpha=0.9)
    polish_axes(ax)
    finish_layout(fig, top=0.90)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_ac_field_grid(
    samples_by_label: Mapping[str, np.ndarray],
    *,
    stem: str | Path,
    caption: str,
    count: int = 4,
) -> None:
    labels = list(samples_by_label)
    n_rows = len(labels)
    fig, axes = plt.subplots(n_rows, count, figsize=(2.85 * count, 2.55 * n_rows))
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]
    vmax = max(float(np.max(np.abs(np.asarray(v)[:count]))) for v in samples_by_label.values())
    vmax = max(vmax, 1e-3)
    im = None
    for row, label in enumerate(labels):
        arr = np.asarray(samples_by_label[label])
        for col in range(count):
            ax = axes[row, col]
            im = ax.imshow(arr[col], cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"Sample {col + 1}", pad=8, fontsize=16)
            if col == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=30, fontsize=15)
    fig.subplots_adjust(left=0.13, right=0.855, bottom=0.065, top=0.90, wspace=0.11, hspace=0.24)
    if im is not None:
        cax = fig.add_axes([0.895, 0.17, 0.016, 0.64])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("u", labelpad=10)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar.ax.tick_params(labelsize=12)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


AC_METHOD_STYLES = {
    "KSWGD": {
        "color": "#1f77b4",
        "linestyle": (0, (4, 2)),
        "linewidth": 2.8,
        "marker": "o",
        "markevery": 12,
        "markersize": 3.6,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "alpha": 0.98,
        "zorder": 6,
    },
    "ULA": {
        "color": "#ff7f0e",
        "linestyle": "-",
        "linewidth": 2.2,
        "alpha": 0.72,
        "zorder": 3,
    },
    "SVGD": {
        "color": "#2ca02c",
        "linestyle": "-.",
        "linewidth": 2.2,
        "alpha": 0.86,
        "zorder": 4,
    },
    "Matrix SVGD": {
        "color": "#d62728",
        "linestyle": ":",
        "linewidth": 2.7,
        "alpha": 0.92,
        "zorder": 5,
    },
}


def plot_ac_histogram_timeline(
    target_fields: np.ndarray,
    generated_by_step: Mapping[int, np.ndarray],
    *,
    method: str,
    stem: str | Path,
    caption: str,
    include_initial: bool = True,
) -> None:
    bins = np.linspace(-1.5, 1.5, 120)
    centers = 0.5 * (bins[:-1] + bins[1:])
    target_hist, _ = np.histogram(np.asarray(target_fields).ravel(), bins=bins, density=True)
    target_hist = gaussian_filter1d(target_hist, sigma=2)
    steps = sorted(generated_by_step)
    if not include_initial:
        steps = [s for s in steps if s != 0]
    n_cols = min(4, len(steps))
    n_rows = int(np.ceil(len(steps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.65 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    for idx, (ax, step) in enumerate(zip(axes, steps)):
        gen_hist, _ = np.histogram(np.asarray(generated_by_step[step]).ravel(), bins=bins, density=True)
        gen_hist = gaussian_filter1d(gen_hist, sigma=2)
        style = dict(AC_METHOD_STYLES.get(method, {"color": "#ff7f0e", "linewidth": 2.5}))
        color = style.get("color", "#ff7f0e")
        ax.plot(centers, target_hist, color="black", linewidth=2.5, label="Target")
        ax.plot(centers, gen_hist, label=method, **style)
        ax.fill_between(centers, gen_hist, color=color, alpha=0.12)
        ax.set_title(f"iteration {step}", pad=14)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("u")
        if idx % n_cols == 0:
            ax.set_ylabel("Density")
        ax.set_xlim(-1.5, 1.5)
        polish_axes(ax, grid=True)
    for ax in axes[len(steps) :]:
        ax.axis("off")
    axes[0].legend(frameon=True, loc="upper left")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.90, wspace=0.20, hspace=0.32)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_ac_method_histograms(
    target_fields: np.ndarray,
    method_to_fields: Mapping[str, np.ndarray],
    *,
    stem: str | Path,
    caption: str,
) -> None:
    bins = np.linspace(-1.5, 1.5, 120)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    target_hist, _ = np.histogram(np.asarray(target_fields).ravel(), bins=bins, density=True)
    ax.plot(centers, gaussian_filter1d(target_hist, sigma=2), color="black", linewidth=2.6, label="Target")
    for name, fields in method_to_fields.items():
        hist, _ = np.histogram(np.asarray(fields).ravel(), bins=bins, density=True)
        ax.plot(centers, gaussian_filter1d(hist, sigma=2), linewidth=2.2, label=name)
    ax.set_title("Pixel-value distributions", pad=16)
    ax.set_xlabel("u")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    polish_axes(ax, grid=True)
    finish_layout(fig, top=0.88)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)


def plot_ac_method_timeline_overlay(
    target_fields: np.ndarray,
    generated_by_method_step: Mapping[str, Mapping[int, np.ndarray]],
    *,
    stem: str | Path,
    caption: str,
    include_initial: bool = True,
) -> None:
    bins = np.linspace(-1.5, 1.5, 120)
    centers = 0.5 * (bins[:-1] + bins[1:])
    target_hist, _ = np.histogram(np.asarray(target_fields).ravel(), bins=bins, density=True)
    target_hist = gaussian_filter1d(target_hist, sigma=2)

    methods = list(generated_by_method_step)
    if not methods:
        raise ValueError("generated_by_method_step must contain at least one method")
    steps = sorted(next(iter(generated_by_method_step.values())))
    if not include_initial:
        steps = [s for s in steps if s != 0]

    draw_methods = [method for method in methods if method != "KSWGD"]
    if "KSWGD" in methods:
        draw_methods.append("KSWGD")
    n_cols = min(4, len(steps))
    n_rows = int(np.ceil(len(steps) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.3 * n_cols, 4.85 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()

    for idx, (ax, step) in enumerate(zip(axes, steps)):
        ax.plot(centers, target_hist, color="black", linewidth=2.7, label="Target")
        for method in draw_methods:
            fields = generated_by_method_step[method][step]
            hist, _ = np.histogram(np.asarray(fields).ravel(), bins=bins, density=True)
            hist = gaussian_filter1d(hist, sigma=2)
            ax.plot(centers, hist, label=method, **AC_METHOD_STYLES.get(method, {"linewidth": 2.2}))
        ax.set_title(f"iteration {step}", pad=14)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("u")
        if idx % n_cols == 0:
            ax.set_ylabel("Density")
        ax.set_xlim(-1.5, 1.5)
        polish_axes(ax, grid=True)

    for ax in axes[len(steps) :]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_order = ["Target", "KSWGD", "ULA", "SVGD", "Matrix SVGD"]
    labels = [label for label in legend_order if label in handle_by_label]
    handles = [handle_by_label[label] for label in labels]
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 0.018))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.135, top=0.90, wspace=0.20, hspace=0.32)
    display(fig)
    save_figure(fig, stem, caption)
    plt.close(fig)
