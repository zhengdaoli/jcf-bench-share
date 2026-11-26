#!/usr/bin/env python3
"""Render main Figure: frame-axis geometry (cosine heatmap + cross-model bars).

Fold-averaged pairwise cosines among the Direct-refusal axis and five frame
axes, read from fold_training_axes.npz for three Qwen checkpoints.
Style matches make_figure1_reframing.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT_DIR = HERE / "figs/main_figure4"
PAPER_FIG_DIR = ROOT / "AuthorKit27/Figures"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
MODEL_COLORS = (BLUE, ORANGE, GREEN)
INK = "#22262B"
GRIDC = "#E7EAEE"
COSINE_CMAP = LinearSegmentedColormap.from_list(
    "paper_cosine",
    (BLUE, "#F7F7F7", ORANGE),
)

AXIS_KEYS = (
    "direct_refusal",
    "schema",
    "task_frame",
    "batch",
    "multi_vs_single_sbq",
    "unsafe_field",
)
AXIS_LABELS = (
    "Direct\nrefusal",
    "SBQ",
    "Semantic",
    "Batch",
    "Multi-\nrecord",
    "Field",
)
SHORT_FRAME = ("SBQ", "Semantic", "Batch", "Multi", "Field")

COSINE_STATS = OUT_DIR / "mean_fold_cosines.json"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_cosine_matrices() -> dict[str, np.ndarray]:
    stats = json.loads(COSINE_STATS.read_text(encoding="utf-8"))
    if stats["axes"] != list(AXIS_KEYS):
        raise ValueError("The published cosine statistics use different axes.")
    return {name: np.asarray(matrix, dtype=float) for name, matrix in stats["models"].items()}


def style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)


def save_panel(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / name
    for target in (stem.with_suffix(".pdf"), PAPER_FIG_DIR / f"{name}.pdf"):
        fig.savefig(target, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=220)


def make_cosine_matrix_panel(heat: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=-0.2, vcenter=0.0, vmax=1.0)
    im = ax.imshow(heat, cmap=COSINE_CMAP, norm=norm, aspect="equal")
    ax.set_xticks(range(len(AXIS_LABELS)))
    ax.set_yticks(range(len(AXIS_LABELS)))
    ax.set_xticklabels(AXIS_LABELS, fontsize=6.2)
    ax.set_yticklabels(AXIS_LABELS, fontsize=6.2)
    for i in range(len(AXIS_KEYS)):
        for j in range(len(AXIS_KEYS)):
            value = heat[i, j]
            color = "white" if abs(value) > 0.55 else INK
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.8,
                color=color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6.2, length=2.0)
    cbar.set_label("cosine", fontsize=7.0)
    return fig


def make_cross_model_panel(matrices: dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    x = np.arange(len(SHORT_FRAME))
    width = 0.26
    for offset, (name, matrix) in enumerate(matrices.items()):
        vals = matrix[0, 1:]
        ax.bar(
            x + (offset - 1) * width,
            vals,
            width=width,
            color=MODEL_COLORS[offset],
            zorder=3,
            label=name,
        )
    ax.axhline(0.0, color=INK, lw=0.7, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_FRAME, fontsize=6.6)
    ax.set_ylim(-0.08, 0.28)
    ax.set_ylabel("Cosine with Direct-refusal")
    ax.legend(frameon=False, loc="upper right", handlelength=1.0, borderaxespad=0.1)
    ax.grid(axis="y", color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    style_axis(ax)
    return fig


def main() -> None:
    matrices = load_cosine_matrices()
    panel_a = make_cosine_matrix_panel(matrices["Qwen2.5-7B"])
    panel_b = make_cross_model_panel(matrices)
    save_panel(panel_a, "main_frame_geometry_a")
    save_panel(panel_b, "main_frame_geometry_b")
    plt.close(panel_a)
    plt.close(panel_b)
    print(f"wrote separate geometry panels to {OUT_DIR} and {PAPER_FIG_DIR}")


if __name__ == "__main__":
    main()
