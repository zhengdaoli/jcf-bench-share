#!/usr/bin/env python3
"""Render the Figure 1 teaser for the reframing paper.

All 100 matched pairs (200 points) are projected onto the axes of the fold
that held each pair out, so every point is out-of-fold. Categories are shown
with colour (refusal outcome) and marker shape (operational frame) only:

* shape  = frame    (circle = Direct, diamond = JSON SBQ)
* colour = outcome  (orange = refusal, green = non-refusal)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
COORDS = HERE / "figs/main_figure1/allfold_heldout_axis_coordinates.csv"
OUT_DIR = HERE / "figs/main_figure1"
PAPER_FIG_DIR = HERE.parent.parent / "AuthorKit27/Figures"

REFUSAL = "#D55E00"
SUCCESS = "#009E73"
INK = "#22262B"
GRIDC = "#E7EAEE"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 9.0,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def main() -> None:
    df = pd.read_csv(COORDS)
    dx, dy = df["direct_schema_x"], df["direct_refusal_y"]
    fx, fy = df["framed_schema_x"], df["framed_refusal_y"]
    d_lab = df["direct_label"].to_numpy(int)
    f_lab = df["framed_label"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)

    ax.axhline(0, color="#D3D9E0", lw=0.7, zorder=0)
    ax.axvline(0, color="#D3D9E0", lw=0.7, zorder=0)
    ax.grid(color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # Direct: hollow circles, edge colour = outcome
    for lab, col in ((0, SUCCESS), (1, REFUSAL)):
        m = d_lab == lab
        ax.scatter(dx[m], dy[m], s=17, marker="o", facecolors="white",
                   edgecolors=col, linewidths=0.9, alpha=0.9, zorder=3)
    # Reframed: filled diamonds, fill = outcome
    for lab, col in ((0, SUCCESS), (1, REFUSAL)):
        m = f_lab == lab
        ax.scatter(fx[m], fy[m], s=19, marker="D", facecolors=col,
                   edgecolors="white", linewidths=0.35, alpha=0.9, zorder=4)

    ax.set_xlabel("SBQ-frame axis")
    ax.set_ylabel("Refusal axis")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)

    ax.set_xlim(-32, 13)
    ax.set_ylim(-16.5, 15)

    # Legend inside the empty band between the two clusters; entries are the
    # actual markers, so each row is one category (shape x colour).
    def marker(m, fc, ec, lw):
        return Line2D([], [], marker=m, linestyle="none", markerfacecolor=fc,
                      markeredgecolor=ec, markeredgewidth=lw, markersize=6.0)

    handles = [
        marker("o", "white", REFUSAL, 1.1),
        marker("o", "white", SUCCESS, 1.1),
        marker("D", REFUSAL, "white", 0.4),
        marker("D", SUCCESS, "white", 0.4),
    ]
    labels = [
        "Direct,\nrefusal",
        "Direct,\nnon-refusal",
        "Reframed,\nrefusal",
        "Reframed,\nnon-refusal",
    ]
    leg = ax.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.002),
        ncol=4, frameon=False, handletextpad=0.3, columnspacing=0.9,
        handlelength=1.0, borderaxespad=0.1, fontsize=6.4,
    )
    leg.set_zorder(6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "main_paired_reframing_projection"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "main_paired_reframing_projection.pdf", bbox_inches="tight")
    print(f"wrote {stem.with_suffix('.pdf')} and .png (+ copied to {PAPER_FIG_DIR})")


if __name__ == "__main__":
    main()
