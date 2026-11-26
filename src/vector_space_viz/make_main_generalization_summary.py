#!/usr/bin/env python3
"""Create the main behavior-and-representation summary figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.analyze_lexical_frame_transitions import load_outcomes
from src.vector_space_viz.make_figure4_frame_geometry import load_cosine_matrices


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "AuthorKit27" / "Figures"
CONTROL_DIR = ROOT / "src" / "outputs" / "control"

BLACK_BOX_MODELS = {
    "Gemini": "gemini_25",
    "GPT-5": "gpt-5-chat",
    "DeepSeek": "deepseek",
    "Qwen-Plus": "qwen-plus",
}

BLACK_BOX_CONDITIONS = (
    ("direct_single", 1),
    ("single_record_sbq", 1),
    ("multi_record_sbq", 10),
)

def lexical_asr(
    model_key: str,
    condition_token: str,
    batch_size: int,
) -> float:
    path = (
        CONTROL_DIR
        / model_key
        / (
            f"{model_key}_exp1_{condition_token}_"
            f"batch{batch_size}_attack3_evaluation.json"
        )
    )
    _, cumulative = load_outcomes(path)
    return 100.0 * sum(outcomes[-1] for outcomes in cumulative) / len(cumulative)


BLACK_BOX_ASR = {
    label: [
        lexical_asr(model_key, condition_token, batch_size)
        for condition_token, batch_size in BLACK_BOX_CONDITIONS
    ]
    for label, model_key in BLACK_BOX_MODELS.items()
}

WHITE_BOX_ASR = {
    "Qwen2.5-7B": [30.0, 71.5, 80.0],
    "Qwen2.5-0.5B": [38.0, 89.7, 63.3],
    "Qwen3-4B": [26.0, 69.7, 95.0],
}

AUC = {
    "Qwen2.5-7B": (0.611, 0.838),
    "Qwen2.5-0.5B": (0.669, 0.794),
    "Qwen3-4B": (0.624, 0.747),
}

AUC_DELTA = {
    "Qwen2.5-7B": 0.228,
    "Qwen2.5-0.5B": 0.125,
    "Qwen3-4B": 0.122,
}

QWEN_COLORS = {
    "Qwen2.5-7B": "#0072B2",
    "Qwen2.5-0.5B": "#D55E00",
    "Qwen3-4B": "#009E73",
}

FRAME_COLORS = ("#5B6570", "#D55E00", "#009E73")
FRAME_LABELS = ("Direct", "SBQ Single", "SBQ Multi")

GEOMETRY_AXES = (
    ("schema", "SBQ"),
    ("task_frame", "Semantic"),
    ("batch", "Batch"),
    ("multi_vs_single_sbq", "Multi"),
    ("unsafe_field", "Field"),
)

def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(ax: plt.Axes, *, y_label: str) -> None:
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#606870")
    ax.spines["bottom"].set_color("#606870")
    ax.tick_params(length=2.5, width=0.6, color="#606870")


def plot_grouped_bars(
    ax: plt.Axes,
    data: dict[str, list[float]],
    labels: list[str],
    *,
    y_label: str,
) -> None:
    models = list(data)
    x = np.arange(len(models))
    width = 0.24
    style_axis(ax, y_label=y_label)
    for frame_index, color in enumerate(FRAME_COLORS):
        values = [data[model][frame_index] for model in models]
        ax.bar(
            x + (frame_index - 1) * width,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=6.0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for color, label in zip(FRAME_COLORS, FRAME_LABELS)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=4.7,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.1,
    )


def plot_blackbox_asr_panel(ax: plt.Axes) -> None:
    plot_grouped_bars(
        ax,
        BLACK_BOX_ASR,
        ["Gem.", "GPT-5", "Deep.", "Qwen+"],
        y_label="Attack success rate (%)",
    )


def plot_phrase_asr_panel(ax: plt.Axes) -> None:
    plot_grouped_bars(
        ax,
        WHITE_BOX_ASR,
        ["Q2.5\n7B", "Q2.5\n0.5B", "Q3\n4B"],
        y_label="Attack success rate (%)",
    )


def plot_auc_panel(ax: plt.Axes) -> None:
    models = list(AUC)
    model_labels = ["Q2.5-7B", "Q2.5-0.5B", "Q3-4B"]
    y = np.arange(len(models))[::-1]

    ax.set_xlim(0.48, 0.88)
    ax.set_ylim(-0.65, len(models) - 0.35)
    ax.set_yticks(y, model_labels)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Refusal-label prediction AUC")
    ax.axvline(
        0.5,
        color="#7A838C",
        linewidth=0.75,
        linestyle=(0, (2, 2)),
        zorder=1,
    )
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#606870")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.5, width=0.6, color="#606870")

    for yi, model in zip(y, models):
        direct, frame = AUC[model]
        color = QWEN_COLORS[model]
        ax.plot([direct, frame], [yi, yi], color="#AEB6BF", linewidth=2.2, zorder=1)
        ax.scatter(
            direct,
            yi,
            s=28,
            facecolor="white",
            edgecolor="#5B6570",
            linewidth=1.2,
            zorder=3,
        )
        ax.scatter(
            frame,
            yi,
            s=32,
            marker="D",
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
        )
        ax.text(
            (direct + frame) / 2,
            yi + 0.18,
            f"+{AUC_DELTA[model]:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color="#30363D",
        )

def plot_geometry_panel(ax: plt.Axes) -> None:
    x = np.arange(len(GEOMETRY_AXES))
    offsets = [-0.15, 0.0, 0.15]
    markers = ["o", "s", "D"]

    ax.axhline(0.0, color="#606870", linewidth=0.65, zorder=1)
    for offset, marker, (model, matrix) in zip(
        offsets, markers, load_cosine_matrices().items()
    ):
        values = matrix[0, 1:]
        ax.scatter(
            x + offset,
            values,
            s=29,
            marker=marker,
            color=QWEN_COLORS[model],
            edgecolor="white",
            linewidth=0.55,
            label=model,
            zorder=3,
        )

    ax.set_xlim(-0.45, len(GEOMETRY_AXES) - 0.55)
    ax.set_ylim(-0.03, 0.18)
    ax.set_xticks(x, ["SBQ", "Sem.", "Batch", "Multi", "Field"])
    ax.tick_params(axis="x", labelsize=6.0, labelrotation=25)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_ylabel("Cosine with Direct refusal")
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#606870")
    ax.spines["bottom"].set_color("#606870")
    ax.tick_params(length=2.5, width=0.6, color="#606870")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=5.0,
        handletextpad=0.25,
        labelspacing=0.2,
        borderaxespad=0.0,
    )
def save_panel(plotter, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(1.78, 1.95))
    plotter(ax)
    fig.subplots_adjust(left=0.24, right=0.985, top=0.98, bottom=0.25)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panels = (
        (plot_blackbox_asr_panel, "main_summary_blackbox_asr"),
        (plot_phrase_asr_panel, "main_summary_phrase_asr"),
        (plot_auc_panel, "main_summary_transfer"),
        (plot_geometry_panel, "main_summary_geometry"),
    )
    for plotter, stem in panels:
        save_panel(plotter, stem)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.15, 2.2),
        gridspec_kw={"wspace": 0.58},
    )
    for ax, (plotter, _) in zip(axes, panels):
        plotter(ax)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.98, bottom=0.25)
    pdf_path = OUTPUT_DIR / "main_generalization_summary.pdf"
    png_path = OUTPUT_DIR / "main_generalization_summary.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for _, stem in panels:
        print(OUTPUT_DIR / f"{stem}.pdf")
    print(pdf_path)
    print(png_path)


if __name__ == "__main__":
    main()
