from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


FIG_ROOT = Path("figures")


def setup_plot_style(scale: float = 1.5) -> None:
    """Use readable defaults for notebook figures and exported PDF/PNG files."""
    base = 10.5 * scale
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.titlesize": base + 2,
            "axes.labelsize": base,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "legend.fontsize": base - 1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_figure_files(root: str | Path = FIG_ROOT) -> int:
    """Delete generated files under figures while keeping the directory tree."""
    root = Path(root)
    if not root.exists():
        root.mkdir(parents=True)
        return 0
    count = 0
    for item in root.rglob("*"):
        if item.is_file():
            item.unlink()
            count += 1
    return count


def save_figure(fig, stem: str | Path, caption: str, *, dpi: int = 300) -> tuple[Path, Path]:
    """Save a figure as PDF and PNG, then print a caption below it."""
    stem = Path(stem)
    ensure_dir(stem.parent)
    if stem.suffix.lower() in {".png", ".pdf"}:
        stem = stem.with_suffix("")
    png = stem.parent / f"{stem.name}.png"
    pdf = stem.parent / f"{stem.name}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    print(f"Caption: {caption}")
    print(f"Saved: {png} and {pdf}")
    return png, pdf


def polish_axes(ax, *, equal: bool = False, grid: bool = False) -> None:
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if grid:
        ax.grid(True, alpha=0.24, linestyle="--", linewidth=0.7)
    ax.tick_params(direction="in", length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def finish_layout(fig, *, top: float = 0.91, h_pad: float = 1.0, w_pad: float = 1.0) -> None:
    fig.tight_layout(rect=[0.0, 0.0, 1.0, top], h_pad=h_pad, w_pad=w_pad)


def caption_list(items: Iterable[str]) -> str:
    return " ".join(str(item).strip() for item in items if str(item).strip())
