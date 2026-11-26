#!/usr/bin/env python3
"""Render the BatchJail control and batch-size panels used in the paper."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    control_path = (
        root
        / "src"
        / "outputs"
        / "control"
        / "gemini_25"
        / "gemini_25_exp1_control_summary.json"
    )
    sweep_path = (
        root
        / "src"
        / "outputs"
        / "gpt-5-chat"
        / "large_batch_sweep"
        / "gpt-5-chat_large_batch_sweep_fixed_summary.json"
    )
    output_path = root / "AuthorKit27" / "Figures" / "batchjail_controls.png"

    control_rows = load_json(control_path)["rows"]
    sweep_rows = load_json(sweep_path)["rows"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), constrained_layout=True)
    fig.patch.set_facecolor("white")

    labels = ["Direct", "Direct\nbatch", "Plain\nbatch", "Bullet\nbatch", "SBQ\nsingle", "SBQ\nmulti"]
    values = [100 * row["ASR@3"] for row in control_rows]
    colors = ["#6B7280", "#9CA3AF", "#D1D5DB", "#B8C4CE", "#0F766E", "#C2410C"]
    bars = axes[0].bar(np.arange(len(values)), values, color=colors, width=0.72)
    axes[0].set_ylabel("Hybrid ASR@3 (%)")
    axes[0].set_xticks(np.arange(len(labels)), labels)
    axes[0].set_ylim(0, 105)
    axes[0].set_title("(a) Gemini structural controls", pad=5)
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    sizes = [row["batch_size"] for row in sweep_rows]
    success = [100 * row["ASR@1"] for row in sweep_rows]
    valid = [100 * row["valid_response_rate"] for row in sweep_rows]
    x = np.arange(len(sizes))
    axes[1].plot(x, success, marker="o", color="#C2410C", linewidth=1.8, label="Hybrid success")
    axes[1].plot(x, valid, marker="s", color="#2563EB", linewidth=1.8, label="Valid extraction")
    axes[1].set_ylabel("Queries (%)")
    axes[1].set_xticks(x, [str(size) for size in sizes])
    axes[1].set_xlabel("Records per SBQ prompt")
    axes[1].set_ylim(0, 105)
    axes[1].set_title("(b) GPT-5-Chat batch-size sweep", pad=5)
    axes[1].legend(loc="upper left", frameon=False)

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor="white", transparent=False)


if __name__ == "__main__":
    main()
