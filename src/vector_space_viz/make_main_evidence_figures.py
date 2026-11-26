#!/usr/bin/env python3
"""Create main-paper behavior, control, and intervention figures."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "AuthorKit27" / "Figures"

MODELS = ("Qwen2.5-7B", "Qwen2.5-0.5B", "Qwen3-4B")
MODEL_COLORS = ("#0072B2", "#D55E00", "#009E73")
REFUSAL_COLOR = "#D55E00"
NONREFUSAL_COLOR = "#009E73"

FORMAT_ASR = np.asarray(
    [
        [59.5, 73.5, 81.5, 85.0, 80.0, 75.0],
        [88.5, 87.5, 93.0, 70.0, 60.0, 60.0],
        [83.0, 60.0, 66.0, 100.0, 85.0, 100.0],
    ]
)

FAMILY_ASR = np.asarray(
    [
        [30.0, 38.0, 26.0],
        [60.0, 100.0, 60.0],
        [100.0, 100.0, 100.0],
        [97.0, 100.0, 82.0],
        [100.0, 100.0, 72.0],
        [100.0, 100.0, 100.0],
        [71.5, 89.7, 69.7],
        [80.0, 63.3, 95.0],
    ]
)

STEERING_COUNTS = np.asarray(
    [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [1, 4, 0, 1, 5],
        [1, 6, 1, 0, 5],
    ]
)

PATCHING_COUNTS = {
    "All positions": [2, 7, 0],
    "Prompt positions": [2, 11, 7],
    "First position": [2, 2, 13],
    "Generated positions": [2, 14, 0],
}

PAIRED_TSNE = (
    ROOT
    / "src/vector_space_viz/figs/main_figure1/paired_tsne_coordinates.csv"
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_family_asr() -> None:
    prompt_forms = (
        "Direct",
        "Direct Batch",
        "Translation",
        "Summary",
        "Rewrite",
        "Extraction",
        "SBQ Single",
        "SBQ Multi",
    )
    offsets = (-0.23, 0.0, 0.23)
    bar_height = 0.20
    y = np.arange(len(prompt_forms))

    fig, ax = plt.subplots(figsize=(3.25, 2.10))
    for model_index, (model, color, offset) in enumerate(
        zip(MODELS, MODEL_COLORS, offsets)
    ):
        values = FAMILY_ASR[:, model_index]
        ax.barh(
            y + offset,
            values,
            height=bar_height,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=model,
            zorder=2,
        )
    ax.axhline(1.5, color="#C9CFD6", linewidth=0.65)
    ax.axhline(5.5, color="#C9CFD6", linewidth=0.65)
    ax.set_xlim(0, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Phrase-based attack success rate (%)")
    ax.set_yticks(y, prompt_forms)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E1E5EA", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#606870")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, width=0.6, color="#606870")
    fig.subplots_adjust(left=0.27, right=0.99, top=0.85, bottom=0.20)
    save(fig, "main_family_asr")


def plot_format_asr() -> None:
    conditions = (
        "Single JSON",
        "Single Table",
        "Single XML",
        "Multi JSON",
        "Multi Table",
        "Multi XML",
    )
    offsets = (-0.23, 0.0, 0.23)
    bar_height = 0.20
    y = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=(3.25, 1.95))
    for model_index, (model, color, offset) in enumerate(
        zip(MODELS, MODEL_COLORS, offsets)
    ):
        ax.barh(
            y + offset,
            FORMAT_ASR[model_index],
            height=bar_height,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=model,
            zorder=2,
        )

    ax.axhline(2.5, color="#C9CFD6", linewidth=0.65)
    ax.set_xlim(0, 106)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Phrase-based attack success rate (%)")
    ax.set_yticks(y, conditions)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E1E5EA", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#606870")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, width=0.6, color="#606870")
    fig.subplots_adjust(left=0.27, right=0.99, top=0.83, bottom=0.21)
    save(fig, "main_format_asr")


def plot_batch_controls() -> None:
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(3.25, 4.15),
        gridspec_kw={"height_ratios": [0.95, 1.05], "hspace": 0.58},
    )

    conditions = (
        "Direct",
        "Direct\nBatch",
        "Plain\nBatch",
        "Bullet\nBatch",
        "SBQ\nSingle",
        "SBQ\nMulti",
    )
    asr = np.asarray([9, 10, 1, 11, 90, 96])
    colors = (
        "#0072B2",
        "#0072B2",
        "#009E73",
        "#009E73",
        "#D55E00",
        "#D55E00",
    )
    x = np.arange(len(conditions))
    ax0.bar(x, asr, color=colors, width=0.66, edgecolor="white", linewidth=0.5)
    for index, value in enumerate(asr):
        ax0.text(index, value + 2.5, f"{value}", ha="center", fontsize=7.0, fontweight="bold")
    ax0.text(2.5, 41, "Parser failures\n23/30 and 25/30", ha="center", color="#5B6570", fontsize=6.7)
    ax0.set_xticks(x, conditions)
    ax0.set_ylim(0, 105)
    ax0.set_yticks([0, 25, 50, 75, 100])
    ax0.set_ylabel("Attack success rate (%)")
    ax0.set_title("(a) Gemini structural controls", loc="left", fontweight="bold")
    ax0.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    ax0.set_axisbelow(True)

    batch_sizes = np.asarray([1, 2, 4, 6, 8, 10, 16, 32, 64])
    positions = np.arange(len(batch_sizes))
    batch_asr = np.asarray([12, 19, 16, 40, 38, 51, 47, 18, 3])
    valid = np.asarray([100, 100, 100, 100, 100, 100, 100, 36, 3])
    ax1.axvspan(6.5, 8.5, color="#F1F3F5", zorder=0)
    ax1.plot(
        positions,
        batch_asr,
        marker="o",
        color="#0072B2",
        linewidth=1.6,
        markersize=4.2,
        label="Attack success rate",
    )
    ax1.plot(
        positions,
        valid,
        marker="s",
        color="#D55E00",
        linewidth=1.6,
        markersize=4.0,
        label="Valid extraction",
    )
    ax1.set_xticks(positions, [str(value) for value in batch_sizes])
    ax1.set_ylim(0, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_xlabel("Records per prompt")
    ax1.set_ylabel("Queries (%)")
    ax1.set_title("(b) GPT batch-size control", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    ax1.set_axisbelow(True)
    ax1.legend(loc="lower left", frameon=False, ncol=2, columnspacing=0.9)

    for ax in (ax0, ax1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#606870")
        ax.spines["bottom"].set_color("#606870")
        ax.tick_params(length=2.5, width=0.6, color="#606870")

    fig.subplots_adjust(left=0.17, right=0.99, top=0.96, bottom=0.10)
    save(fig, "main_batch_controls")


def plot_paired_tsne() -> None:
    with PAIRED_TSNE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    refusal_color = REFUSAL_COLOR
    nonrefusal_color = NONREFUSAL_COLOR
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    for row in rows:
        direct = (float(row["direct_x"]), float(row["direct_y"]))
        reframed = (float(row["framed_x"]), float(row["framed_y"]))
        ax.plot(
            [direct[0], reframed[0]],
            [direct[1], reframed[1]],
            color="#AEB6BF",
            linewidth=0.45,
            alpha=0.22,
            zorder=1,
        )
        direct_color = refusal_color if int(row["direct_label"]) else nonrefusal_color
        reframed_color = refusal_color if int(row["framed_label"]) else nonrefusal_color
        ax.scatter(
            *direct,
            s=22,
            marker="o",
            facecolor="white",
            edgecolor=direct_color,
            linewidth=0.8,
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            *reframed,
            s=24,
            marker="D",
            facecolor=reframed_color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=3,
        )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor="#5B6570",
            linewidth=0,
            label="Direct",
        ),
        plt.Line2D(
            [],
            [],
            marker="D",
            markersize=4.5,
            markerfacecolor="#5B6570",
            markeredgecolor="white",
            linewidth=0,
            label="Reframed",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            markersize=4.5,
            markerfacecolor=refusal_color,
            markeredgecolor=refusal_color,
            linewidth=0,
            label="Refusal",
        ),
        plt.Line2D(
            [],
            [],
            marker="s",
            markersize=4.5,
            markerfacecolor=nonrefusal_color,
            markeredgecolor=nonrefusal_color,
            linewidth=0,
            label="Non-refusal",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.99, bottom=0.02)
    save(fig, "main_paired_tsne")


def plot_intervention_diagnostics() -> None:
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.45),
        gridspec_kw={"width_ratios": [1.05, 0.95], "wspace": 0.33},
    )

    ax0.imshow(STEERING_COUNTS, cmap="Blues", vmin=0, vmax=6, aspect="auto")
    for row in range(STEERING_COUNTS.shape[0]):
        for column in range(STEERING_COUNTS.shape[1]):
            value = int(STEERING_COUNTS[row, column])
            ax0.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value >= 4 else "#20262D",
                fontsize=7.3,
                fontweight="bold",
            )
    ax0.set_xticks(
        np.arange(5),
        ["Baseline", "All", "Prompt", "First", "Generated"],
        rotation=20,
        ha="right",
    )
    ax0.set_yticks(
        np.arange(5),
        ["Translation", "Rewriting", "Extraction", "SBQ Single", "SBQ Multi"],
    )
    ax0.tick_params(length=0)
    ax0.set_title("(a) Joint steering refusals out of 12", loc="left", fontweight="bold")

    gamma = np.asarray([0.0, 0.5, 1.0])
    markers = ("o", "s", "D", "^")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for (label, values), marker, color in zip(
        PATCHING_COUNTS.items(),
        markers,
        colors,
    ):
        ax1.plot(
            gamma,
            values,
            marker=marker,
            color=color,
            linewidth=1.7,
            markersize=4.4,
            label=label,
        )
    ax1.set_xlim(-0.04, 1.04)
    ax1.set_ylim(-0.5, 15.5)
    ax1.set_xticks(gamma, ["Baseline", r"$\gamma=.5$", r"$\gamma=1$"])
    ax1.set_yticks([0, 5, 10, 15])
    ax1.set_ylabel("Refusals out of 60")
    ax1.set_title("(b) Activation patching by strength", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("#606870")
    ax1.spines["bottom"].set_color("#606870")
    ax1.tick_params(length=2.5, width=0.6, color="#606870")
    ax1.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8)

    fig.subplots_adjust(left=0.12, right=0.995, top=0.90, bottom=0.24)
    save(fig, "main_intervention_diagnostics")


def main() -> None:
    configure_style()
    plot_family_asr()
    plot_format_asr()
    plot_batch_controls()
    plot_paired_tsne()
    plot_intervention_diagnostics()


if __name__ == "__main__":
    main()
