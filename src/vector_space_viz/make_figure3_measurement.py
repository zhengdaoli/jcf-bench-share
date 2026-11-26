#!/usr/bin/env python3
"""Render the lexical attack-success measurement audit.

(a) decision-path coverage over the main black-box trials
(b) Gemini format audit
(c) GPT-5-Chat batch sweep

Missing or unparsed responses count as attack failures. Judge availability does
not enter the lexical ASR.
Style matches make_figure1_reframing.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "figs/main_figure3"
PAPER_FIG_DIR = HERE.parent.parent / "AuthorKit27/Figures"

REFUSAL = "#C6413B"
SUCCESS = "#0E8C86"
INK = "#22262B"
GRIDC = "#E7EAEE"
NEUTRAL = "#9AA3AC"

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

# (a) lexical decision paths over 4,800 trial records.
PROV = [("Refusal phrase", 2405, REFUSAL),
        ("No refusal phrase", 859, SUCCESS),
        ("Empty or skipped", 1536, "#9AA3AC")]

# (b) Gemini format audit, batch size 10, one attempt, 100 queries
FORMATS = ["Plain text", "Bullets", "JSON", "Partial JSON", "YAML", "XML",
           "Md table"]
VALID = [90, 100, 57, 40, 50, 100, 96]
FORMAT_ASR = [0, 0, 55, 39, 48, 97, 93]

# (c) GPT-5-Chat batch sweep
BATCH = [1, 2, 4, 6, 8, 10, 16, 32, 64]
B_VALID = [100, 100, 100, 100, 100, 100, 100, 36, 3]
B_ASR = [12, 19, 16, 40, 38, 51, 47, 18, 3]


def style_axis(ax, grid_axis="y") -> None:
    ax.grid(axis=grid_axis, color=GRIDC, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=INK)


def main() -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [0.85, 1.15, 1.0]},
    )

    # (a) decision-path coverage
    ax = axes[0]
    y = np.arange(len(PROV))[::-1]
    vals = [v for _, v, _ in PROV]
    cols = [c for _, _, c in PROV]
    ax.barh(y, vals, height=0.55, color=cols, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 80, yy, str(v), va="center", fontsize=7.0, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in PROV], fontsize=6.6)
    ax.set_xlim(0, 5350)
    ax.set_xlabel("trial records (of 4,800)")
    ax.set_title("(a) lexical outcome paths", loc="left",
                 fontsize=7.8)
    style_axis(ax, grid_axis="x")

    # (b) Gemini format audit
    ax = axes[1]
    x = np.arange(len(FORMATS))
    w = 0.38
    ax.bar(x - w / 2, VALID, width=w, color=NEUTRAL, zorder=3,
           label="valid extraction")
    ax.bar(x + w / 2, FORMAT_ASR, width=w, color=SUCCESS, zorder=3,
           label="attack success")
    ax.set_xticks(x)
    ax.set_xticklabels(FORMATS, rotation=38, ha="right", fontsize=6.2)
    ax.set_ylim(0, 112)
    ax.set_ylabel("count (of 100)")
    ax.set_title("(b) Gemini format audit", loc="left", fontsize=7.8)
    ax.legend(frameon=False, loc="upper left", handlelength=1.0,
              borderaxespad=0.1)
    style_axis(ax)

    # (c) GPT-5-Chat batch sweep
    ax = axes[2]
    xi = np.arange(len(BATCH))
    ax.plot(xi, B_VALID, "-o", color=NEUTRAL, ms=3.4, lw=1.2, zorder=3,
            label="valid extraction (%)")
    ax.plot(xi, B_ASR, "-D", color=SUCCESS, ms=3.2, lw=1.2, zorder=4,
            label="ASR (%)")
    ax.axvspan(6.5, 8.4, color="#F6E3E2", zorder=0)
    ax.text(7.45, 78, "extraction\ncollapse", ha="center", fontsize=6.4,
            color=REFUSAL)
    ax.set_xticks(xi)
    ax.set_xticklabels(BATCH, fontsize=6.6)
    ax.set_ylim(0, 112)
    ax.set_xlabel("records per call")
    ax.set_title("(c) GPT-5-Chat batch sweep", loc="left", fontsize=7.8)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2,
              borderaxespad=0.1)
    style_axis(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / "main_measurement_audit"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "main_measurement_audit.pdf",
                bbox_inches="tight")
    print(f"wrote {stem.with_suffix('.pdf')} and .png (+ copied to {PAPER_FIG_DIR})")


if __name__ == "__main__":
    main()
