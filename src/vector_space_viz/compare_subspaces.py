from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probe_utils import parse_boolish, split_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare refusal directions/subspaces across mechanisms and layers."
    )
    parser.add_argument("--vectors", type=Path, required=True, help="NPZ from export_causal_lm_states.py.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata CSV from export_causal_lm_states.py.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to metadata parent.",
    )
    parser.add_argument(
        "--label-col",
        default="generation_refusal_heuristic",
        help="Binary label used to define positive-negative directions.",
    )
    parser.add_argument("--positive-label", default="true")
    parser.add_argument("--group-col", default="mechanism")
    parser.add_argument("--capture-points", default="query_span,prompt_end")
    parser.add_argument("--layers", default="all", help="Hidden-state layer indices to analyze, or all.")
    parser.add_argument("--min-group", type=int, default=6)
    return parser.parse_args()


def load_vectors(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "vectors" not in payload:
        raise ValueError(f"{path} does not contain a 'vectors' array")
    return np.asarray(payload["vectors"], dtype=np.float32)


def binary_labels(series: pd.Series, positive_label: str) -> pd.Series:
    positive_bool = parse_boolish(positive_label)

    def convert(value: Any) -> float:
        parsed = parse_boolish(value)
        if parsed is not None:
            if positive_bool is None:
                return float(parsed)
            return float(parsed == positive_bool)
        return float(str(value) == positive_label)

    return series.map(convert)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def layer_filter(layer_arg: str) -> set[int] | None:
    if layer_arg == "all":
        return None
    return {int(item) for item in split_csv(layer_arg)}


def compute_directions(
    vectors: np.ndarray,
    meta: pd.DataFrame,
    *,
    label_col: str,
    positive_label: str,
    group_col: str,
    capture_points: set[str],
    min_group: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int, str], np.ndarray]]:
    meta = meta.copy()
    meta["_label"] = binary_labels(meta[label_col], positive_label).astype(int)

    direction_rows: list[dict[str, Any]] = []
    directions: dict[tuple[str, int, str], np.ndarray] = {}

    for capture_point in sorted(capture_points):
        capture_mask = meta["capture_point"].astype(str) == capture_point
        for layer_index in sorted(meta.loc[capture_mask, "layer_index"].unique()):
            layer_mask = capture_mask & (meta["layer_index"] == layer_index)
            for group in sorted(str(value) for value in meta.loc[layer_mask, group_col].dropna().unique()):
                group_mask = layer_mask & (meta[group_col].astype(str) == group)
                idx = np.flatnonzero(group_mask.to_numpy())
                if len(idx) < min_group:
                    continue
                labels = meta.loc[group_mask, "_label"].to_numpy(dtype=int)
                if len(np.unique(labels)) < 2:
                    continue
                pos_idx = idx[labels == 1]
                neg_idx = idx[labels == 0]
                direction = vectors[pos_idx].mean(axis=0) - vectors[neg_idx].mean(axis=0)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm == 0:
                    continue
                key = (capture_point, int(layer_index), group)
                directions[key] = normalize(direction)
                direction_rows.append(
                    {
                        "capture_point": capture_point,
                        "layer_index": int(layer_index),
                        "layer_name": str(meta.loc[idx[0], "layer_name"]),
                        "group": group,
                        "n_total": int(len(idx)),
                        "n_positive": int(len(pos_idx)),
                        "n_negative": int(len(neg_idx)),
                        "direction_norm": direction_norm,
                    }
                )
    return pd.DataFrame(direction_rows), directions


def cosine_rows(directions: dict[tuple[str, int, str], np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = sorted(directions.keys())
    for left in keys:
        for right in keys:
            if left[0] != right[0] or left[1] != right[1]:
                continue
            cosine = float(np.dot(directions[left], directions[right]))
            rows.append(
                {
                    "capture_point": left[0],
                    "layer_index": left[1],
                    "group_a": left[2],
                    "group_b": right[2],
                    "direction_cosine": cosine,
                    "principal_angle_deg": float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))),
                }
            )
    return pd.DataFrame(rows)


def projection_summary(
    vectors: np.ndarray,
    meta: pd.DataFrame,
    directions: dict[tuple[str, int, str], np.ndarray],
    *,
    group_col: str,
    label_col: str,
    positive_label: str,
) -> pd.DataFrame:
    meta = meta.copy()
    meta["_label"] = binary_labels(meta[label_col], positive_label).astype(int)
    rows: list[dict[str, Any]] = []
    for (capture_point, layer_index, direction_group), direction in directions.items():
        mask = (meta["capture_point"].astype(str) == capture_point) & (meta["layer_index"] == layer_index)
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            continue
        scores = vectors[idx] @ direction
        part = meta.iloc[idx].copy()
        part["_projection"] = scores
        for (eval_group, label), group_df in part.groupby([group_col, "_label"], dropna=False):
            rows.append(
                {
                    "capture_point": capture_point,
                    "layer_index": int(layer_index),
                    "direction_group": direction_group,
                    "eval_group": str(eval_group),
                    "label": int(label),
                    "n": int(len(group_df)),
                    "projection_mean": float(group_df["_projection"].mean()),
                    "projection_std": float(group_df["_projection"].std(ddof=0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    vectors = load_vectors(args.vectors)
    meta = pd.read_csv(args.metadata)
    if len(meta) != len(vectors):
        raise ValueError(f"metadata rows ({len(meta)}) != vectors ({len(vectors)})")
    if args.label_col not in meta.columns:
        raise ValueError(f"Missing label column: {args.label_col}")
    if args.group_col not in meta.columns:
        raise ValueError(f"Missing group column: {args.group_col}")

    output_dir = args.output_dir or args.metadata.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_points = set(split_csv(args.capture_points))
    layers = layer_filter(args.layers)
    if layers is not None:
        meta = meta[meta["layer_index"].astype(int).isin(layers)].copy()
        vectors = vectors[meta.index.to_numpy()]
        meta = meta.reset_index(drop=True)

    direction_df, directions = compute_directions(
        vectors,
        meta,
        label_col=args.label_col,
        positive_label=args.positive_label,
        group_col=args.group_col,
        capture_points=capture_points,
        min_group=args.min_group,
    )
    cosine_df = cosine_rows(directions)
    projection_df = projection_summary(
        vectors,
        meta,
        directions,
        group_col=args.group_col,
        label_col=args.label_col,
        positive_label=args.positive_label,
    )

    direction_path = output_dir / "direction_norms.csv"
    cosine_path = output_dir / "direction_cosines.csv"
    projection_path = output_dir / "projection_summary.csv"
    summary_path = output_dir / "subspace_summary.json"
    direction_df.to_csv(direction_path, index=False)
    cosine_df.to_csv(cosine_path, index=False)
    projection_df.to_csv(projection_path, index=False)
    summary = {
        "vectors": str(args.vectors),
        "metadata": str(args.metadata),
        "label_col": args.label_col,
        "positive_label": args.positive_label,
        "group_col": args.group_col,
        "capture_points": sorted(capture_points),
        "layers": sorted(layers) if layers is not None else "all",
        "num_directions": int(len(direction_df)),
        "direction_norms": str(direction_path),
        "direction_cosines": str(cosine_path),
        "projection_summary": str(projection_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
