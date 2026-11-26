from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_utils import parse_boolish, split_csv, write_json


TRANSFER_COLUMNS = [
    "capture_point",
    "layer_index",
    "layer_name",
    "train_group",
    "test_group",
    "n_train",
    "n_test",
    "test_positive_rate",
    "auc",
    "accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit linear refusal probes and cross-mechanism transfer matrices."
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
        help="Binary label column. Alternatives: harmful_label, expected_refusal, prompt_expected_refusal.",
    )
    parser.add_argument(
        "--positive-label",
        default="true",
        help="Positive label value when --label-col is string-like.",
    )
    parser.add_argument(
        "--capture-points",
        default="query_span,prompt_end",
        help="Comma-separated capture points to analyze.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Hidden-state layer indices to analyze, or all.",
    )
    parser.add_argument(
        "--group-col",
        default="mechanism",
        help="Column defining train/test groups for transfer.",
    )
    parser.add_argument(
        "--train-groups",
        default="auto",
        help="Comma-separated train groups, or auto for all groups.",
    )
    parser.add_argument("--min-train", type=int, default=8, help="Minimum train rows per binary probe.")
    parser.add_argument("--test-size", type=float, default=0.35, help="Same-group holdout fraction.")
    parser.add_argument("--seed", type=int, default=42)
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


def auc_or_nan(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def layer_filter(meta: pd.DataFrame, layer_arg: str) -> set[int] | None:
    if layer_arg == "all":
        return None
    return {int(item) for item in split_csv(layer_arg)}


def fit_model(x_train: np.ndarray, y_train: np.ndarray) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    ).fit(x_train, y_train)


def same_group_split(
    indices: np.ndarray,
    labels: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(indices) < 4 or len(np.unique(labels[indices])) < 2:
        return None
    class_counts = pd.Series(labels[indices]).value_counts()
    stratify = labels[indices] if class_counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    if len(np.unique(labels[train_idx])) < 2 or len(test_idx) == 0:
        return None
    return train_idx, test_idx


def run_transfer(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    vectors = load_vectors(args.vectors)
    meta = pd.read_csv(args.metadata)
    if len(meta) != len(vectors):
        raise ValueError(f"metadata rows ({len(meta)}) != vectors ({len(vectors)})")
    if args.label_col not in meta.columns:
        raise ValueError(f"Missing label column: {args.label_col}")
    if args.group_col not in meta.columns:
        raise ValueError(f"Missing group column: {args.group_col}")

    meta = meta.copy()
    meta["_label"] = binary_labels(meta[args.label_col], args.positive_label)

    capture_points = set(split_csv(args.capture_points))
    layers = layer_filter(meta, args.layers)
    if args.train_groups == "auto":
        train_groups = sorted(str(value) for value in meta[args.group_col].dropna().unique())
    else:
        train_groups = split_csv(args.train_groups)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for capture_point in sorted(capture_points):
        capture_mask = meta["capture_point"].astype(str) == capture_point
        for layer_index in sorted(meta.loc[capture_mask, "layer_index"].unique()):
            if layers is not None and int(layer_index) not in layers:
                continue
            layer_mask = capture_mask & (meta["layer_index"] == layer_index)
            layer_indices = np.flatnonzero(layer_mask.to_numpy())
            if len(layer_indices) == 0:
                continue

            for train_group in train_groups:
                train_group_mask = layer_mask & (meta[args.group_col].astype(str) == train_group)
                train_group_indices = np.flatnonzero(train_group_mask.to_numpy())
                y_all = meta["_label"].to_numpy(dtype=int)

                if len(train_group_indices) < args.min_train or len(np.unique(y_all[train_group_indices])) < 2:
                    skipped.append(
                        {
                            "capture_point": capture_point,
                            "layer_index": int(layer_index),
                            "train_group": train_group,
                            "reason": "not_enough_binary_train_rows",
                            "n_train_group": int(len(train_group_indices)),
                        }
                    )
                    continue

                for test_group in sorted(str(value) for value in meta.loc[layer_mask, args.group_col].dropna().unique()):
                    test_group_mask = layer_mask & (meta[args.group_col].astype(str) == test_group)
                    test_group_indices = np.flatnonzero(test_group_mask.to_numpy())
                    if len(test_group_indices) == 0:
                        continue

                    if train_group == test_group:
                        split = same_group_split(
                            train_group_indices,
                            y_all,
                            test_size=args.test_size,
                            seed=args.seed,
                        )
                        if split is None:
                            continue
                        train_idx, test_idx = split
                    else:
                        train_idx = train_group_indices
                        test_idx = test_group_indices

                    if len(np.unique(y_all[train_idx])) < 2:
                        continue
                    model = fit_model(vectors[train_idx], y_all[train_idx])
                    scores = model.predict_proba(vectors[test_idx])[:, 1]
                    preds = (scores >= 0.5).astype(int)
                    rows.append(
                        {
                            "capture_point": capture_point,
                            "layer_index": int(layer_index),
                            "layer_name": str(meta.loc[train_idx[0], "layer_name"]),
                            "train_group": train_group,
                            "test_group": test_group,
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            "test_positive_rate": float(y_all[test_idx].mean()),
                            "auc": auc_or_nan(y_all[test_idx], scores),
                            "accuracy": float(accuracy_score(y_all[test_idx], preds)),
                        }
                    )

    summary = {
        "vectors": str(args.vectors),
        "metadata": str(args.metadata),
        "label_col": args.label_col,
        "positive_label": args.positive_label,
        "group_col": args.group_col,
        "capture_points": sorted(capture_points),
        "layers": sorted(layers) if layers is not None else "all",
        "train_groups": train_groups,
        "num_rows": int(len(rows)),
        "skipped": skipped[:100],
        "num_skipped": int(len(skipped)),
    }
    return pd.DataFrame(rows, columns=TRANSFER_COLUMNS), summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.metadata.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df, summary = run_transfer(args)
    transfer_path = output_dir / "probe_transfer.csv"
    summary_path = output_dir / "probe_summary.json"
    df.to_csv(transfer_path, index=False)
    write_json(summary_path, {**summary, "probe_transfer": str(transfer_path)})
    print(json.dumps({"probe_transfer": str(transfer_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
