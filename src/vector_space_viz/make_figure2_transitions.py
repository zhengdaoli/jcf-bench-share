#!/usr/bin/env python3
"""Render paired transitions for lexical attack success rate.

Empty or unparsed responses count as attack failures. Counts cover 100 fixed
queries per model. Panels:

(a) 01 transitions (attack failure to success)
(b) 10 transitions (attack success to failure)
(c) net ASR change (01 - 10)

Style matches make_figure1_reframing.py (colours, fonts, line weights).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "figs/main_figure2"
PAPER_FIG_DIR = HERE.parent.parent / "AuthorKit27/Figures"

REFUSAL = "#C6413B"
SUCCESS = "#0E8C86"
INK = "#22262B"
GRIDC = "#E7EAEE"
LIGHT_SUCCESS = "#7FC4C0"
LIGHT_REFUSAL = "#E0A49F"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODELS = ["Gemini-2.5-Flash", "GPT-5-Chat", "DeepSeek Chat", "Qwen-Plus"]
SHORT = ["Gemini", "GPT-5", "DeepSeek", "Qwen-Plus"]

# Paired lexical-ASR counts over 100 queries.
T01 = {"D\u2192S": [82, 17, 93, 57], "S\u2192M": [7, 40, 1, 40]}
T10 = {"D\u2192S": [1, 25, 0, 0], "S\u2192M": [1, 2, 0, 0]}


def style_axis(ax) -> None:
    ax.grid(axis="x", color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)


def grouped_bars(ax, data, dark, light, title, xmax) -> None:
    y = np.arange(len(SHORT))[::-1]
    h = 0.34
    ax.barh(y + h / 2 + 0.02, data["D\u2192S"], height=h, color=dark,
            zorder=3, label="Direct \u2192 SBQ Single")
    ax.barh(y - h / 2 - 0.02, data["S\u2192M"], height=h, color=light,
            zorder=3, label="SBQ Single \u2192 Multi")
    for yy, v in zip(y + h / 2 + 0.02, data["D\u2192S"]):
        ax.text(v + 1.2, yy, str(v), va="center", fontsize=6.6, color=INK)
    for yy, v in zip(y - h / 2 - 0.02, data["S\u2192M"]):
        ax.text(v + 1.2, yy, str(v), va="center", fontsize=6.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(SHORT)
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left")
    style_axis(ax)


def main() -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.15), constrained_layout=True
    )

    grouped_bars(
        axes[0], T01, SUCCESS, LIGHT_SUCCESS,
        "(a) attack-success gains (0 \u2192 1)", 104,
    )
    axes[0].set_xlabel("queries out of 100")
    # Dark = D->S, light = S->M in every panel; use neutral legend patches.
    handles = [Patch(facecolor="#5B6570"), Patch(facecolor="#C3CAD2")]
    labels = ["Direct \u2192 SBQ Single (dark)", "SBQ Single \u2192 Multi (light)"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06), handlelength=1.2,
               columnspacing=1.6, handletextpad=0.5)

    grouped_bars(
        axes[1], T10, REFUSAL, LIGHT_REFUSAL,
        "(b) attack-success losses (1 \u2192 0)", 32,
    )
    axes[1].set_xlabel("queries out of 100")

    # (c) net change in attack success rate.
    ax = axes[2]
    y = np.arange(len(SHORT))[::-1]
    h = 0.34
    net_ds = np.array(T01["D\u2192S"]) - np.array(T10["D\u2192S"])
    net_sm = np.array(T01["S\u2192M"]) - np.array(T10["S\u2192M"])
    ax.barh(y + h / 2 + 0.02, net_ds, height=h, color=SUCCESS, zorder=3)
    ax.barh(y - h / 2 - 0.02, net_sm, height=h, color=LIGHT_SUCCESS, zorder=3)
    for yy, v in zip(y + h / 2 + 0.02, net_ds):
        off = 1.5 if v >= 0 else -1.5
        ha = "left" if v >= 0 else "right"
        ax.text(v + off, yy, f"{v:+d}", va="center", ha=ha,
                fontsize=6.6, color=INK)
    for yy, v in zip(y - h / 2 - 0.02, net_sm):
        ax.text(v + 1.5, yy, f"{v:+d}", va="center", fontsize=6.6, color=INK)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(SHORT)
    ax.set_xlim(-28, 104)
    ax.set_xlabel("net change (points)")
    ax.set_title("(c) net ASR change", loc="left")
    style_axis(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "main_paired_transitions"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "main_paired_transitions.pdf",
                bbox_inches="tight")
    print(f"wrote {stem.with_suffix('.pdf')} and .png (+ copied to {PAPER_FIG_DIR})")


if __name__ == "__main__":
    main()
