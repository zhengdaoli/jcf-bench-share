#!/usr/bin/env python3
"""Render the figures that carry the main-paper argument."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "AuthorKit27" / "Figures"

CHARCOAL = "#22272E"
GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
PALE_GRAY = "#F4F6F7"
ORANGE = "#D55E00"
BLUE = "#0072B2"
SKY = "#4C9BC6"
GREEN = "#008C72"
DEEP_GREEN = "#356859"
AMBER = "#E5A000"
MAGENTA = "#CC79A7"

MODELS = ["Gemini", "GPT-5-Chat", "DeepSeek", "Qwen-Plus"]
MODEL_SHORT = ["Gemini", "GPT-5", "DeepSeek", "Qwen+"]
MODEL_COLORS = [ORANGE, BLUE, GREEN, MAGENTA]

OPEN_MODELS = ["Qwen2.5-7B", "Qwen2.5-0.5B", "Qwen3-4B"]
OPEN_COLORS = [BLUE, GREEN, MAGENTA]
OPEN_MARKERS = ["o", "s", "D"]

PAIRED_PROJECTION_DIR = (
    ROOT / "src" / "vector_space_viz" / "figs" / "paired_reframing_projection_layer19_fold0"
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 8.6,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.png", dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_paired() -> dict[tuple[str, str], dict[str, str]]:
    path = ROOT / "src" / "outputs" / "control" / "paired_analysis" / "paired_frame_transitions.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    normalized = {
        "Gemini": "Gemini",
        "GPT-5-Chat": "GPT-5-Chat",
        "DeepSeek": "DeepSeek",
        "Qwen-Plus": "Qwen-Plus",
    }
    return {(normalized[row["model"]], row["contrast"]): row for row in rows}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def plot_overview() -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(3, 98, "One query. Three operational frames.", fontsize=9.0, fontweight="bold", color=CHARCOAL, va="top")

    card_specs = [
        (76, "Direct", "answer the query  qᵢ", GRAY),
        (47, "SBQ Single", "complete one record for  qᵢ", ORANGE),
        (18, "SBQ Multi", "complete ten records in shared context", BLUE),
    ]
    card_x, card_width, card_height = 4, 92, 14
    for y, title, detail, color in card_specs:
        ax.add_patch(
            patches.FancyBboxPatch(
                (card_x, y),
                card_width,
                card_height,
                boxstyle="round,pad=0.35,rounding_size=0.8",
                linewidth=1.2,
                edgecolor=color,
                facecolor="white",
            )
        )
        ax.text(card_x + 3, y + 7, title, fontsize=7.4, fontweight="bold", color=CHARCOAL, va="center")
        ax.text(card_x + 34, y + 7, detail, fontsize=6.5, color=CHARCOAL, va="center")

    comparisons = [
        (69, ORANGE, "D→S composite", "82 | 19 | 93 | 57"),
        (40, BLUE, "S→M composite", "7 | 40 | 1 | 40"),
    ]
    for y, color, title, counts in comparisons:
        ax.annotate("", xy=(13, y - 5), xytext=(13, y + 4), arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.3})
        ax.text(21, y, title, fontsize=6.6, color=color, fontweight="bold", va="center")
        ax.text(60, y, counts, fontsize=7.2, color=color, fontweight="bold", va="center")

    ax.text(4, 9.5, "01 count order  Gemini | GPT-5 | DeepSeek | Qwen+", fontsize=5.7, color=GRAY)
    ax.text(4, 3.0, "Same query identity. Model-specific transition locus.", fontsize=6.4, color=CHARCOAL, fontweight="bold")

    save_figure(fig, "main_overview")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_reframing_projection_candidates() -> None:
    transition_colors = {
        "1→0": ORANGE,
        "1→1": BLUE,
        "0→0": GRAY,
        "0→1": GREEN,
    }
    axis_rows = load_csv_rows(PAIRED_PROJECTION_DIR / "paired_fold_axis_coordinates.csv")
    axis_counts: dict[str, int] = {}
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    for row in axis_rows:
        transition = row["transition"]
        axis_counts[transition] = axis_counts.get(transition, 0) + 1
        color = transition_colors[transition]
        start = (float(row["direct_refusal_x"]), float(row["direct_refusal_y"]))
        end = (float(row["framed_schema_x"]), float(row["framed_refusal_y"]))
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 0.85 if transition != "1→0" else 1.15,
                "alpha": 0.46 if transition != "1→0" else 0.78,
                "mutation_scale": 6,
            },
            zorder=1,
        )
        ax.scatter(*start, s=23, marker="o", facecolor="white", edgecolor=color, linewidth=0.9, zorder=3)
        ax.scatter(*end, s=25, marker="D", facecolor=color, edgecolor="white", linewidth=0.45, zorder=4)

    for transition in ["1→0", "1→1", "0→0", "0→1"]:
        if transition not in axis_counts:
            continue
        ax.plot(
            [],
            [],
            color=transition_colors[transition],
            marker="D",
            linewidth=1.1,
            markersize=4,
            label=f"{transition}  ({axis_counts[transition]})",
        )
    ax.set_xlabel("SBQ-frame-axis projection")
    ax.set_ylabel("Direct-refusal-axis projection")
    ax.set_title("Reframing shifts held-out representations", loc="left", fontsize=8.2, fontweight="bold")
    ax.grid(color=LIGHT_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=5.9,
        handlelength=1.5,
        labelspacing=0.25,
        columnspacing=0.8,
        title="Direct→SBQ outcome",
        title_fontsize=5.9,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
    )
    ax.text(
        0.98,
        0.98,
        "20/20 move along SBQ frame\nmedian shift  +27.9",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color=GREEN,
        fontweight="bold",
    )
    save_figure(fig, "main_paired_reframing_projection")

    tsne_rows = load_csv_rows(PAIRED_PROJECTION_DIR / "paired_tsne_coordinates.csv")
    tsne_counts: dict[str, int] = {}
    fig, ax = plt.subplots(figsize=(3.25, 2.55))
    for row in tsne_rows:
        transition = row["transition"]
        tsne_counts[transition] = tsne_counts.get(transition, 0) + 1
        color = transition_colors[transition]
        start = (float(row["direct_x"]), float(row["direct_y"]))
        end = (float(row["framed_x"]), float(row["framed_y"]))
        highlighted = transition == "1→0"
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=0.65 if highlighted else 0.35,
            alpha=0.33 if highlighted else 0.11,
            zorder=1,
        )
        ax.scatter(*start, s=13, marker="o", facecolor="white", edgecolor=color, linewidth=0.55, alpha=0.78, zorder=2)
        ax.scatter(*end, s=14, marker="D", facecolor=color, edgecolor="white", linewidth=0.25, alpha=0.78, zorder=3)
    for transition in ["1→0", "1→1", "0→0", "0→1"]:
        ax.plot(
            [],
            [],
            color=transition_colors[transition],
            marker="D",
            linewidth=1.0,
            markersize=3.8,
            label=f"{transition}  ({tsne_counts[transition]})",
        )
    ax.set_title("Paired PCA→t-SNE trajectories", loc="left", fontsize=8.2, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=5.8, handlelength=1.4, columnspacing=0.8)
    ax.text(
        0.98,
        0.98,
        "100 query pairs\nlayout is non-metric",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=GRAY,
    )
    save_figure(fig, "appendix_paired_tsne")


def plot_transition_fingerprints() -> None:
    paired = load_paired()
    contrasts = [("D_to_S", "D → SBQ Single"), ("S_to_M", "SBQ Single → SBQ Multi")]
    fig = plt.figure(figsize=(7.1, 2.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], wspace=0.36, hspace=0.72)
    bar_axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    scatter_ax = fig.add_subplot(grid[:, 1])

    net_values: dict[str, list[int]] = {model: [] for model in MODELS}
    for axis_index, ((contrast_key, title), ax) in enumerate(zip(contrasts, bar_axes)):
        positives = [int(paired[(model, contrast_key)]["paired_01_count"]) for model in MODELS]
        reverses = [int(paired[(model, contrast_key)]["paired_10_count"]) for model in MODELS]
        y = np.arange(len(MODELS))
        ax.barh(y, positives, color=ORANGE, height=0.52, label="endpoint −→+ (01)")
        ax.barh(y, -np.array(reverses), color=BLUE, height=0.52, label="endpoint +→− (10)")
        for index, (positive, reverse) in enumerate(zip(positives, reverses)):
            ax.text(positive + 1.5, index, str(positive), va="center", ha="left", fontsize=7.1, color=ORANGE, fontweight="bold")
            if reverse >= 5:
                ax.text(-reverse + 1.2, index, str(reverse), va="center", ha="left", fontsize=6.8, color="white", fontweight="bold")
            else:
                ax.text(-reverse - 1.5, index, str(reverse), va="center", ha="right", fontsize=7.1, color=BLUE, fontweight="bold")
            net_values[MODELS[index]].append(positive - reverse)
        ax.axvline(0, color=CHARCOAL, linewidth=0.7)
        ax.set_yticks(y, MODEL_SHORT)
        ax.invert_yaxis()
        ax.set_xlim(-28, 102)
        ax.set_xticks([-20, 0, 20, 40, 60, 80, 100], ["20", "0", "20", "40", "60", "80", "100"])
        ax.set_title(f"({chr(ord('a') + axis_index)})  {title}", loc="left", pad=2, fontweight="bold")
        ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if axis_index == 1:
            ax.set_xlabel("Matched queries changing endpoint status, count out of 100", labelpad=2)
    fig.text(0.12, 0.985, "■  endpoint +→− (10)", color=BLUE, fontsize=7.0, va="top")
    fig.text(0.32, 0.985, "■  endpoint −→+ (01)", color=ORANGE, fontsize=7.0, va="top")

    for model, short, color in zip(MODELS, MODEL_SHORT, MODEL_COLORS):
        x, y = net_values[model]
        scatter_ax.scatter(x, y, s=42, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        offsets = {
            "Gemini": (-5, 4),
            "GPT-5-Chat": (4, 3),
            "DeepSeek": (-15, -10),
            "Qwen-Plus": (-25, 4),
        }
        dx, dy = offsets[model]
        scatter_ax.annotate(short, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.2, color=color, fontweight="bold")
    scatter_ax.axhline(0, color=GRAY, linewidth=0.7)
    scatter_ax.axvline(0, color=GRAY, linewidth=0.7)
    scatter_ax.set_xlim(-12, 101)
    scatter_ax.set_ylim(-4, 46)
    scatter_ax.set_xlabel("Net D→S shift  (01 − 10)")
    scatter_ax.set_ylabel("Net S→M shift  (01 − 10)")
    scatter_ax.set_title("(c)  Transition signature", loc="left", fontweight="bold", pad=2)
    scatter_ax.grid(color=LIGHT_GRAY, linewidth=0.55)
    scatter_ax.set_axisbelow(True)
    scatter_ax.spines[["top", "right"]].set_visible(False)
    scatter_ax.annotate(
        "41 queries flip\nnet change −3",
        xy=(-3, 38),
        xytext=(22, 27),
        arrowprops={"arrowstyle": "->", "color": BLUE, "linewidth": 0.8},
        fontsize=6.8,
        color=BLUE,
        ha="center",
    )
    scatter_ax.text(97, 3.5, "larger first contrast", ha="right", fontsize=6.4, color=GRAY)
    scatter_ax.text(3, 44.3, "larger second contrast", ha="left", va="top", fontsize=6.4, color=GRAY)

    save_figure(fig, "main_paired_transitions")


def plot_measurement_audit() -> None:
    format_path = ROOT / "src" / "outputs" / "gemini_25" / "format_ablation_fixed" / "gemini_25_format_ablation_fixed_summary.json"
    batch_path = ROOT / "src" / "outputs" / "gpt-5-chat" / "large_batch_sweep" / "gpt-5-chat_large_batch_sweep_fixed_summary.json"
    formats = json.loads(format_path.read_text(encoding="utf-8"))["rows"]
    batches = json.loads(batch_path.read_text(encoding="utf-8"))["rows"]

    fig = plt.figure(figsize=(7.1, 2.18))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.18, 1.55], wspace=0.42)
    ax0, ax1, ax2 = [fig.add_subplot(grid[0, index]) for index in range(3)]

    ax0.barh([0], [98.2], color=AMBER, height=0.36)
    ax0.barh([0], [1.8], left=[98.2], color=CHARCOAL, height=0.36)
    ax0.text(49.1, 0, "824 / 839\nfallback", ha="center", va="center", fontsize=8.2, color="white", fontweight="bold")
    ax0.annotate("15 parsed\nrating ≥ 7", xy=(99.1, 0), xytext=(74, 0.65), arrowprops={"arrowstyle": "-", "color": CHARCOAL}, fontsize=6.8, ha="center")
    ax0.text(49.1, -0.58, "821 / 824 fallback positives\nfollow an empty judge response", ha="center", va="top", fontsize=6.0, color=GRAY)
    ax0.set_xlim(0, 100)
    ax0.set_ylim(-1.0, 1.05)
    ax0.set_yticks([])
    ax0.set_xticks([0, 50, 100])
    ax0.set_xlabel("Positive endpoints (%)")
    ax0.set_title("(a)  Endpoint paths", loc="left", fontweight="bold", fontsize=8.0)
    ax0.spines[["top", "right", "left"]].set_visible(False)

    label_map = {
        "plain": "Plain",
        "bullets": "Bullets",
        "json": "JSON",
        "half_json": "Partial JSON",
        "yaml": "YAML",
        "xml": "XML",
        "markdown_table": "Table",
    }
    for row in formats:
        valid = 100 * row["valid_response_rate"]
        positive = 100 * row["ASR@1"]
        neutral = row["format"] in {"plain", "bullets"}
        color = GRAY if neutral else ORANGE
        marker = "o" if neutral else "s"
        ax1.scatter(valid, positive, s=28, marker=marker, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        offsets = {
            "plain": (-23, 5),
            "bullets": (-31, -12),
            "json": (4, -12),
            "half_json": (-33, 4),
            "yaml": (4, 4),
            "xml": (-20, 5),
            "markdown_table": (-39, -12),
        }
        ax1.annotate(label_map[row["format"]], (valid, positive), xytext=offsets[row["format"]], textcoords="offset points", fontsize=6.3, color=color)
    ax1.plot([0, 100], [0, 100], color=LIGHT_GRAY, linewidth=0.8, linestyle="--", zorder=0)
    ax1.set_xlim(30, 103)
    ax1.set_ylim(-4, 103)
    ax1.set_xlabel("Valid extraction (%)")
    ax1.set_ylabel("Hybrid endpoint + (%)")
    ax1.set_title("(b)  Format audit", loc="left", fontweight="bold")
    ax1.grid(color=LIGHT_GRAY, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(batches))
    sizes = [row["batch_size"] for row in batches]
    positives = [100 * row["ASR@1"] for row in batches]
    valid = [100 * row["valid_response_rate"] for row in batches]
    ax2.axvspan(6.55, 8.45, color="#FFF4DF", zorder=0)
    ax2.plot(x, positives, color=ORANGE, marker="o", linewidth=1.7, markersize=4.2)
    ax2.plot(x, valid, color=BLUE, marker="s", linewidth=1.7, markersize=4.0)
    ax2.set_xticks(x, [str(size) for size in sizes])
    ax2.set_ylim(-2, 105)
    ax2.set_xlabel("Records per prompt")
    ax2.set_ylabel("Queries (%)")
    ax2.set_title("(c)  Batch-size audit", loc="left", fontweight="bold")
    ax2.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(2.4, 91, "valid extraction", color=BLUE, fontsize=6.7, fontweight="bold")
    ax2.text(2.4, 25, "hybrid endpoint +", color=ORANGE, fontsize=6.7, fontweight="bold")
    ax2.annotate("extraction collapse", xy=(7.5, 24), xytext=(5.6, 62), arrowprops={"arrowstyle": "->", "color": GRAY}, fontsize=6.8, color=GRAY)

    save_figure(fig, "main_measurement_audit")


def read_fold_values(result_dir: str) -> dict[str, list[float]]:
    path = ROOT / "src" / "vector_space_viz" / "figs" / "grouped_fold_refusal_readout_layer19_seed42" / result_dir / "fold_auc.csv"
    values: dict[str, list[float]] = {"direct_only": [], "frame_only": [], "direct_plus_frame": []}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["feature_set"]].append(float(row["auc"]))
    return values


def read_pooled_auc(result_dir: str) -> dict[str, float]:
    path = ROOT / "src" / "vector_space_viz" / "figs" / "grouped_fold_refusal_readout_layer19_seed42" / result_dir / "grouped_fold_refusal_readout_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    return {key: float(summary["metrics"][key]["oof_auc"]) for key in ["direct_only", "frame_only", "direct_plus_frame"]}


def plot_open_weight_prediction() -> None:
    behavior = {
        "Qwen2.5-7B": [70.00, 28.50, 20.00],
        "Qwen2.5-0.5B": [62.00, 10.33, 36.67],
        "Qwen3-4B": [74.00, 30.33, 5.00],
    }
    result_dirs = ["qwen2p5_7b", "qwen2p5_0p5b", "qwen3_4b"]
    feature_keys = ["direct_only", "frame_only", "direct_plus_frame"]
    feature_labels = ["Direct", "Frame", "Combined"]

    fig = plt.figure(figsize=(7.1, 5.35))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.08], hspace=0.52, wspace=0.34)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    x = np.arange(3)
    for model, color, marker in zip(OPEN_MODELS, OPEN_COLORS, OPEN_MARKERS):
        values = behavior[model]
        ax0.plot(x, values, color=color, marker=marker, linewidth=1.7, markersize=5.0, label=model)
        for x_value, value in zip(x, values):
            ax0.text(x_value, value + 3.0, f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}", ha="center", fontsize=6.4, color=color, fontweight="bold")
    ax0.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax0.text(1.97, 2.0, "Mistral  0 / 0 / 0", ha="right", va="bottom", fontsize=5.9, color=GRAY)
    ax0.annotate("direction reverses", xy=(2, 36.67), xytext=(1.25, 52), arrowprops={"arrowstyle": "->", "color": GREEN}, fontsize=6.7, color=GREEN)
    ax0.set_xticks(x, ["Direct", "SBQ Single", "SBQ Multi"])
    ax0.set_ylim(-3, 82)
    ax0.set_ylabel("Lexical refusal rate (%)")
    ax0.set_title("(a)  Headline frames", loc="left", fontweight="bold", fontsize=7.8)
    ax0.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55)
    ax0.set_axisbelow(True)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.legend(loc="lower left", bbox_to_anchor=(0.0, 0.04), frameon=False, handlelength=1.4, fontsize=6.1)

    format_values = np.array(
        [
            [40.5, 26.5, 18.5, 15.0, 20.0, 25.0],
            [11.5, 12.5, 7.0, 30.0, 40.0, 40.0],
            [17.0, 40.0, 34.0, 0.0, 15.0, 0.0],
        ]
    )
    image = ax1.imshow(format_values, cmap="YlGnBu", vmin=0, vmax=45, aspect="auto")
    for row in range(format_values.shape[0]):
        for column in range(format_values.shape[1]):
            value = format_values[row, column]
            color = "white" if value >= 27 else CHARCOAL
            ax1.text(column, row, f"{value:g}", ha="center", va="center", fontsize=5.9, color=color, fontweight="bold")
    ax1.axvline(2.5, color="white", linewidth=1.4)
    ax1.set_xticks(np.arange(6), ["S-J", "S-T", "S-X", "M-J", "M-T", "M-X"])
    ax1.set_yticks(np.arange(3), ["7B", "0.5B", "Q3"])
    ax1.tick_params(length=0)
    ax1.set_title("(b)  SBQ format rates", loc="left", fontweight="bold", fontsize=7.8)

    pooled_by_model = [read_pooled_auc(result_dir) for result_dir in result_dirs]
    folds_by_model = [read_fold_values(result_dir) for result_dir in result_dirs]
    feature_x = np.arange(3)
    offsets = [-0.12, 0.0, 0.12]
    for model_index, (model, color, marker, pooled, folds) in enumerate(zip(OPEN_MODELS, OPEN_COLORS, OPEN_MARKERS, pooled_by_model, folds_by_model)):
        y = [pooled[key] for key in feature_keys]
        ax2.plot(feature_x + offsets[model_index], y, color=color, linewidth=1.3, alpha=0.85)
        for feature_index, key in enumerate(feature_keys):
            fold_x = feature_x[feature_index] + offsets[model_index] + np.linspace(-0.025, 0.025, len(folds[key]))
            ax2.scatter(fold_x, folds[key], s=8, color=color, alpha=0.26, linewidth=0)
            ax2.scatter(feature_x[feature_index] + offsets[model_index], pooled[key], s=34, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=4)
        delta = pooled["frame_only"] - pooled["direct_only"]
        ax2.text(1 + offsets[model_index], pooled["frame_only"] + 0.016 + model_index * 0.003, f"+{delta:.3f}", color=color, ha="center", fontsize=6.4, fontweight="bold")
    ax2.axhline(0.5, color=GRAY, linewidth=0.8, linestyle="--")
    ax2.set_xticks(feature_x, feature_labels)
    ax2.set_ylim(0.44, 0.96)
    ax2.set_ylabel("Grouped OOF AUC", labelpad=2)
    ax2.set_title("(c)  Grouped prediction", loc="left", fontweight="bold", fontsize=7.8)
    ax2.tick_params(axis="x", labelsize=6.5)
    ax2.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(1.96, 0.535, "held-out query components\nknown frame families", ha="center", va="bottom", fontsize=5.9, color=GRAY)

    model_x = np.arange(len(OPEN_MODELS))
    for model_index, (model, color, marker, pooled, folds) in enumerate(
        zip(OPEN_MODELS, OPEN_COLORS, OPEN_MARKERS, pooled_by_model, folds_by_model)
    ):
        fold_deltas = np.asarray(folds["frame_only"]) - np.asarray(folds["direct_only"])
        jitter = np.linspace(-0.09, 0.09, len(fold_deltas))
        ax3.scatter(
            model_x[model_index] + jitter,
            fold_deltas,
            s=26,
            color=color,
            marker=marker,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        pooled_delta = pooled["frame_only"] - pooled["direct_only"]
        ax3.plot(
            [model_x[model_index] - 0.20, model_x[model_index] + 0.20],
            [pooled_delta, pooled_delta],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
        )
        positive_folds = int(np.sum(fold_deltas > 0))
        ax3.text(
            model_x[model_index],
            max(fold_deltas.max(), pooled_delta) + 0.025,
            f"{positive_folds}/5 folds",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=color,
            fontweight="bold",
        )
    ax3.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax3.set_xticks(model_x, ["Q2.5-7B", "Q2.5-0.5B", "Q3-4B"])
    ax3.set_ylim(-0.08, 0.43)
    ax3.set_ylabel(r"Fold AUC gain  $AUC_F-AUC_D$")
    ax3.set_title("(d)  Fold-wise frame gains", loc="left", fontweight="bold", fontsize=7.8)
    ax3.grid(axis="y", color=LIGHT_GRAY, linewidth=0.55)
    ax3.set_axisbelow(True)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.text(
        2.08,
        -0.055,
        "bars show pooled OOF gain",
        ha="right",
        va="bottom",
        fontsize=5.9,
        color=GRAY,
    )

    save_figure(fig, "main_open_weight_prediction")


def plot_open_weight_column_panels() -> None:
    behavior = {
        "Qwen2.5-7B": [70.00, 28.50, 20.00],
        "Qwen2.5-0.5B": [62.00, 10.33, 36.67],
        "Qwen3-4B": [74.00, 30.33, 5.00],
    }
    format_values = np.array(
        [
            [40.5, 26.5, 18.5, 15.0, 20.0, 25.0],
            [11.5, 12.5, 7.0, 30.0, 40.0, 40.0],
            [17.0, 40.0, 34.0, 0.0, 15.0, 0.0],
        ]
    )

    fig = plt.figure(figsize=(3.25, 4.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.08, 1.0], hspace=0.52)
    ax0, ax1 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    x = np.arange(3)
    for model, color, marker in zip(OPEN_MODELS, OPEN_COLORS, OPEN_MARKERS):
        values = behavior[model]
        ax0.plot(x, values, color=color, marker=marker, linewidth=1.5, markersize=4.3, label=model)
        for x_value, value in zip(x, values):
            ax0.text(x_value, value + 2.7, f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}", ha="center", fontsize=5.8, color=color, fontweight="bold")
    ax0.annotate("direction reverses", xy=(2, 36.67), xytext=(1.1, 52), arrowprops={"arrowstyle": "->", "color": GREEN}, fontsize=5.9, color=GREEN)
    ax0.axhline(0, color=GRAY, linewidth=0.7, linestyle="--")
    ax0.text(1.97, 1.8, "Mistral  0 / 0 / 0", ha="right", va="bottom", fontsize=5.6, color=GRAY)
    ax0.set_xticks(x, ["Direct", "Single", "Multi"])
    ax0.set_ylim(-3, 82)
    ax0.set_ylabel("Lexical refusal (%)")
    ax0.set_title("(a)  Headline frames", loc="left", fontweight="bold", fontsize=7.7)
    ax0.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    ax0.set_axisbelow(True)
    ax0.spines[["top", "right"]].set_visible(False)
    ax0.legend(loc="lower left", bbox_to_anchor=(0.0, 0.04), frameon=False, handlelength=1.3, fontsize=5.7)

    ax1.imshow(format_values, cmap="YlGnBu", vmin=0, vmax=45, aspect="auto")
    for row in range(format_values.shape[0]):
        for column in range(format_values.shape[1]):
            value = format_values[row, column]
            color = "white" if value >= 27 else CHARCOAL
            ax1.text(column, row, f"{value:g}", ha="center", va="center", fontsize=5.7, color=color, fontweight="bold")
    ax1.axvline(2.5, color="white", linewidth=1.3)
    ax1.set_xticks(np.arange(6), ["S-J", "S-T", "S-X", "M-J", "M-T", "M-X"])
    ax1.set_yticks(np.arange(3), ["7B", "0.5B", "Q3"])
    ax1.tick_params(length=0, labelsize=6.3)
    ax1.set_title("(b)  SBQ format rates", loc="left", fontweight="bold", fontsize=7.7)
    save_figure(fig, "main_open_behavior_formats")

    result_dirs = ["qwen2p5_7b", "qwen2p5_0p5b", "qwen3_4b"]
    feature_keys = ["direct_only", "frame_only", "direct_plus_frame"]
    pooled_by_model = [read_pooled_auc(result_dir) for result_dir in result_dirs]
    folds_by_model = [read_fold_values(result_dir) for result_dir in result_dirs]
    fig, ax = plt.subplots(figsize=(3.25, 2.35))
    feature_x = np.arange(3)
    offsets = [-0.12, 0.0, 0.12]
    for model_index, (color, marker, pooled, folds) in enumerate(zip(OPEN_COLORS, OPEN_MARKERS, pooled_by_model, folds_by_model)):
        y = [pooled[key] for key in feature_keys]
        ax.plot(feature_x + offsets[model_index], y, color=color, linewidth=1.3, alpha=0.85)
        for feature_index, key in enumerate(feature_keys):
            fold_x = feature_x[feature_index] + offsets[model_index] + np.linspace(-0.025, 0.025, len(folds[key]))
            ax.scatter(fold_x, folds[key], s=8, color=color, alpha=0.26, linewidth=0)
            ax.scatter(feature_x[feature_index] + offsets[model_index], pooled[key], s=34, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=4)
        delta = pooled["frame_only"] - pooled["direct_only"]
        ax.text(1 + offsets[model_index], pooled["frame_only"] + 0.016 + model_index * 0.003, f"+{delta:.3f}", color=color, ha="center", fontsize=6.1, fontweight="bold")
    ax.axhline(0.5, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xticks(feature_x, ["Direct", "Frame", "Combined"])
    ax.set_ylim(0.44, 0.96)
    ax.set_ylabel("Grouped OOF AUC")
    ax.set_title("Grouped prediction", loc="left", fontweight="bold", fontsize=7.7)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1.95, 0.535, "held-out query components\nknown frame families", ha="center", va="bottom", fontsize=5.7, color=GRAY)
    save_figure(fig, "main_open_prediction")


def plot_attempt_budget() -> None:
    rates = {
        "Gemini": {"Direct": [5, 8, 9], "SBQ Single": [79, 89, 90], "SBQ Multi": [78, 87, 96]},
        "GPT-5": {"Direct": [34, 47, 50], "SBQ Single": [30, 40, 47], "SBQ Multi": [67, 76, 85]},
        "DeepSeek": {"Direct": [2, 6, 6], "SBQ Single": [96, 99, 99], "SBQ Multi": [91, 100, 100]},
        "Qwen+": {"Direct": [3, 3, 3], "SBQ Single": [54, 58, 60], "SBQ Multi": [89, 100, 100]},
    }
    signatures = {
        "Gemini": "dominant D→S",
        "GPT-5": "dominant S→M",
        "DeepSeek": "dominant D→S",
        "Qwen+": "both contrasts",
    }
    styles = {
        "Direct": (GRAY, "o"),
        "SBQ Single": (ORANGE, "s"),
        "SBQ Multi": (BLUE, "D"),
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.72), sharex=True, sharey=True)
    attempts = np.arange(1, 4)
    for index, (model, conditions) in enumerate(rates.items()):
        ax = axes[index]
        for condition, values in conditions.items():
            color, marker = styles[condition]
            ax.plot(attempts, values, color=color, marker=marker, linewidth=1.5, markersize=3.8, label=condition)
            y_offset = 4 if model == "DeepSeek" and condition == "SBQ Multi" else 0
            y_offset = -4 if model == "DeepSeek" and condition == "SBQ Single" else y_offset
            ax.text(3.07, values[-1] + y_offset, str(values[-1]), color=color, fontsize=6.2, va="center", fontweight="bold")
        ax.set_title(f"({chr(ord('a') + index)})  {model}", loc="left", fontweight="bold", pad=2)
        ax.text(1.05, 7, signatures[model], color=MODEL_COLORS[index], fontsize=6.1, fontweight="bold")
        ax.set_xlim(0.85, 3.32)
        ax.set_ylim(-3, 105)
        ax.set_xticks(attempts)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Hybrid endpoint + (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    save_figure(fig, "main_attempt_budget")


def main() -> None:
    configure_style()
    plot_overview()
    plot_reframing_projection_candidates()
    plot_transition_fingerprints()
    plot_measurement_audit()
    plot_open_weight_prediction()
    plot_open_weight_column_panels()
    plot_attempt_budget()


if __name__ == "__main__":
    main()
