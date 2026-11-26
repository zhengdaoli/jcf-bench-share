from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot refusal-space probe summaries.")
    parser.add_argument("--probe-transfer", type=Path, default=None, help="probe_transfer.csv.")
    parser.add_argument("--direction-cosines", type=Path, default=None, help="direction_cosines.csv.")
    parser.add_argument("--projection-summary", type=Path, default=None, help="projection_summary.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-point", default="query_span")
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Specific layer to plot. If -1, use max/mean aggregation depending on figure.",
    )
    return parser.parse_args()


def heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    if matrix.empty:
        return
    fig_width = max(5.0, 0.7 * len(matrix.columns) + 2.5)
    fig_height = max(4.0, 0.55 * len(matrix.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix.iat[row_idx, col_idx]
            if pd.isna(value):
                text = "NA"
            else:
                text = f"{value:.2f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_probe_transfer(path: Path, output_dir: Path, capture_point: str, layer_index: int) -> None:
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df[df["capture_point"].astype(str) == capture_point]
    if layer_index >= 0:
        df = df[df["layer_index"] == layer_index]
        pivot = df.pivot_table(index="train_group", columns="test_group", values="auc", aggfunc="mean")
        suffix = f"layer{layer_index}"
    else:
        pivot = df.pivot_table(index="train_group", columns="test_group", values="auc", aggfunc="max")
        suffix = "best_layer"
    heatmap(
        pivot,
        title=f"Cross-mechanism refusal-probe AUC ({capture_point}, {suffix})",
        output_path=output_dir / f"probe_transfer_auc_{capture_point}_{suffix}.png",
        vmin=0.0,
        vmax=1.0,
        cmap="magma",
    )

    if layer_index < 0:
        same = df[df["train_group"] == df["test_group"]]
        if not same.empty:
            plt.figure(figsize=(8, 4.5))
            for group, group_df in same.groupby("train_group"):
                group_df = group_df.sort_values("layer_index")
                plt.plot(group_df["layer_index"], group_df["auc"], marker="o", label=str(group))
            plt.ylim(0.0, 1.0)
            plt.xlabel("hidden_state layer index")
            plt.ylabel("same-mechanism AUC")
            plt.title(f"Layerwise refusal-probe AUC ({capture_point})")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(output_dir / f"layerwise_auc_{capture_point}.png", dpi=220)
            plt.close()


def plot_direction_cosines(path: Path, output_dir: Path, capture_point: str, layer_index: int) -> None:
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df[df["capture_point"].astype(str) == capture_point]
    if layer_index >= 0:
        df = df[df["layer_index"] == layer_index]
        pivot = df.pivot_table(index="group_a", columns="group_b", values="direction_cosine", aggfunc="mean")
        suffix = f"layer{layer_index}"
    else:
        pivot = df.pivot_table(index="group_a", columns="group_b", values="direction_cosine", aggfunc="mean")
        suffix = "mean_layers"
    heatmap(
        pivot,
        title=f"Refusal-direction cosine ({capture_point}, {suffix})",
        output_path=output_dir / f"direction_cosine_{capture_point}_{suffix}.png",
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )


def plot_projection(path: Path, output_dir: Path, capture_point: str, layer_index: int) -> None:
    df = pd.read_csv(path)
    if df.empty:
        return
    df = df[df["capture_point"].astype(str) == capture_point]
    if layer_index >= 0:
        df = df[df["layer_index"] == layer_index]
    if df.empty:
        return
    grouped = (
        df.groupby(["direction_group", "eval_group", "label"], dropna=False)["projection_mean"]
        .mean()
        .reset_index()
    )
    grouped["series"] = grouped["eval_group"].astype(str) + " | y=" + grouped["label"].astype(str)

    for direction_group, part in grouped.groupby("direction_group"):
        part = part.sort_values("series")
        plt.figure(figsize=(max(7, 0.45 * len(part)), 4.5))
        plt.bar(part["series"], part["projection_mean"])
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("mean projection")
        plt.title(f"Projection on {direction_group} direction ({capture_point})")
        plt.tight_layout()
        safe_name = str(direction_group).replace("/", "_").replace(" ", "_")
        plt.savefig(output_dir / f"projection_{capture_point}_{safe_name}.png", dpi=220)
        plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.probe_transfer:
        plot_probe_transfer(args.probe_transfer, args.output_dir, args.capture_point, args.layer_index)
    if args.direction_cosines:
        plot_direction_cosines(args.direction_cosines, args.output_dir, args.capture_point, args.layer_index)
    if args.projection_summary:
        plot_projection(args.projection_summary, args.output_dir, args.capture_point, args.layer_index)
    print({"output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
