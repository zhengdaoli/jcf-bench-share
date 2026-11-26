from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_utils import parse_boolish, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose refusal behavior into direct-refusal, structured-schema, "
            "and batching axes using exported hidden-state vectors."
        )
    )
    parser.add_argument("--vectors", type=Path, required=True, help="NPZ from export_causal_lm_states.py.")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata CSV aligned to vectors.")
    parser.add_argument("--probe-transfer", type=Path, default=None, help="Optional probe_transfer.csv for auto layer choice.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-point", default="query_span")
    parser.add_argument(
        "--layer-index",
        default="auto",
        help="Layer index, or auto to use best direct_single same-mechanism AUC from --probe-transfer.",
    )
    parser.add_argument("--label-col", default="generation_refusal_heuristic")
    parser.add_argument("--positive-label", default="true")
    parser.add_argument("--min-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_vectors(path: Path) -> np.ndarray:
    payload = np.load(path)
    if "vectors" not in payload:
        raise ValueError(f"{path} does not contain a 'vectors' array")
    return np.asarray(payload["vectors"], dtype=np.float32)


def binary_labels(series: pd.Series, positive_label: str) -> pd.Series:
    positive_bool = parse_boolish(positive_label)

    def convert(value: Any) -> int:
        parsed = parse_boolish(value)
        if parsed is not None:
            if positive_bool is None:
                return int(parsed)
            return int(parsed == positive_bool)
        return int(str(value) == positive_label)

    return series.map(convert).astype(int)


def choose_layer(args: argparse.Namespace) -> int:
    if args.layer_index != "auto":
        return int(args.layer_index)
    if args.probe_transfer is None:
        raise ValueError("--layer-index auto requires --probe-transfer")
    transfer = pd.read_csv(args.probe_transfer)
    direct = transfer[
        (transfer["capture_point"].astype(str) == args.capture_point)
        & (transfer["train_group"].astype(str) == "direct_single")
        & (transfer["test_group"].astype(str) == "direct_single")
    ].copy()
    if direct.empty:
        raise ValueError("No direct_single -> direct_single rows found in probe transfer table.")
    direct = direct.sort_values(["auc", "layer_index"], ascending=[False, True])
    return int(direct.iloc[0]["layer_index"])


def normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vector / norm


def make_prompt_table(vectors: np.ndarray, meta: pd.DataFrame, *, capture_point: str, layer_index: int) -> tuple[np.ndarray, pd.DataFrame]:
    mask = (meta["capture_point"].astype(str) == capture_point) & (meta["layer_index"].astype(int) == layer_index)
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        raise ValueError(f"No rows for capture_point={capture_point}, layer_index={layer_index}.")

    rows = meta.iloc[idx].copy()
    rows["_vector_row"] = idx
    grouped_vectors: list[np.ndarray] = []
    grouped_meta: list[dict[str, Any]] = []

    keep_cols = [
        "prompt_id",
        "mechanism",
        "format",
        "field_name",
        "frame_family",
        "context_family",
        "frame_operation",
        "batch_size",
        "prompt_any_harmful",
        "prompt_all_harmful",
        "prompt_expected_refusal",
        "generation_refusal_heuristic",
        "generated_text_len",
    ]
    keep_cols = [col for col in keep_cols if col in rows.columns]

    for prompt_id, group in rows.groupby("prompt_id", sort=False):
        row_idx = group["_vector_row"].to_numpy(dtype=int)
        grouped_vectors.append(vectors[row_idx].mean(axis=0))
        record = {col: group.iloc[0][col] for col in keep_cols if col != "prompt_id"}
        record["prompt_id"] = prompt_id
        record["n_vector_rows"] = int(len(group))
        record["n_records"] = int(group["record_id"].nunique(dropna=True)) if "record_id" in group.columns else int(len(group))
        grouped_meta.append(record)

    prompt_vectors = np.stack(grouped_vectors, axis=0).astype(np.float32)
    prompt_meta = pd.DataFrame(grouped_meta)
    return prompt_vectors, prompt_meta


def define_axis(
    x_centered: np.ndarray,
    mask_pos: np.ndarray,
    mask_neg: np.ndarray,
    *,
    min_class: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    pos_idx = np.flatnonzero(mask_pos)
    neg_idx = np.flatnonzero(mask_neg)
    info = {"n_positive": int(len(pos_idx)), "n_negative": int(len(neg_idx))}
    if len(pos_idx) < min_class or len(neg_idx) < min_class:
        info["status"] = "skipped_min_class"
        return None, info
    direction = x_centered[pos_idx].mean(axis=0) - x_centered[neg_idx].mean(axis=0)
    unit = normalize(direction)
    if unit is None:
        info["status"] = "skipped_zero_norm"
        return None, info
    info["status"] = "ok"
    info["direction_norm"] = float(np.linalg.norm(direction))
    return unit.astype(np.float32), info


def auc_or_nan(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def cv_auc(features: pd.DataFrame, labels: np.ndarray, *, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    class_counts = pd.Series(labels).value_counts()
    n_splits = int(min(5, class_counts.min()))
    if n_splits < 2:
        return float("nan")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model, features.to_numpy(dtype=float), labels, cv=splitter, scoring="roc_auc")
    return float(np.mean(scores))


def format_float(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3f}"


def write_report(
    path: Path,
    *,
    layer_index: int,
    capture_point: str,
    mechanism_summary: pd.DataFrame,
    axis_auc: pd.DataFrame,
    logistic_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Refusal Axis Deep Dive")
    lines.append("")
    lines.append(f"- capture_point: `{capture_point}`")
    lines.append(f"- layer_index: `{layer_index}`")
    lines.append("")
    lines.append("## Mechanism Summary")
    lines.append("")
    lines.append(
        "| mechanism | n | refusal_rate | direct_refusal_auc | direct_refusal_delta | schema_mean | batch_mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in mechanism_summary.iterrows():
        lines.append(
            "| {mechanism} | {n} | {refusal_rate} | {direct_auc} | {direct_delta} | {schema_mean} | {batch_mean} |".format(
                mechanism=row["mechanism"],
                n=int(row["n_prompts"]),
                refusal_rate=format_float(row["refusal_rate"]),
                direct_auc=format_float(row.get("direct_refusal_auc", float("nan"))),
                direct_delta=format_float(row.get("direct_refusal_label_delta", float("nan"))),
                schema_mean=format_float(row.get("schema_score_mean", float("nan"))),
                batch_mean=format_float(row.get("batch_score_mean", float("nan"))),
            )
        )
    lines.append("")
    lines.append("## Axis Readout")
    lines.append("")
    lines.append("| axis | target | auc |")
    lines.append("|---|---|---:|")
    for _, row in axis_auc.iterrows():
        lines.append(f"| {row['axis']} | {row['target']} | {format_float(row['auc'])} |")
    lines.append("")
    lines.append("## Combined Refusal Models")
    lines.append("")
    lines.append("| feature_set | cv_auc |")
    lines.append("|---|---:|")
    for _, row in logistic_summary.iterrows():
        lines.append(f"| {row['feature_set']} | {format_float(row['cv_auc'])} |")
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- If `direct_refusal_auc` drops on `multi_record_sbq`, the direct refusal direction is not the only decision axis under batching.")
    lines.append("- If `schema_score` or `batch_score` is linearly readable and improves combined refusal prediction, the mechanism is likely frame/task competition rather than a scalar safety drop.")
    lines.append("- These are prompt-level summaries. Multi-record rows currently share one generation/refusal label, so per-record causal claims still need judged per-record outputs.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    layer_index = choose_layer(args)
    vectors = load_vectors(args.vectors)
    meta = pd.read_csv(args.metadata)
    if len(meta) != len(vectors):
        raise ValueError(f"metadata rows ({len(meta)}) != vectors ({len(vectors)})")
    if args.label_col not in meta.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    prompt_vectors, prompt_meta = make_prompt_table(
        vectors,
        meta,
        capture_point=args.capture_point,
        layer_index=layer_index,
    )
    prompt_meta["_label"] = binary_labels(prompt_meta[args.label_col], args.positive_label)
    mechanisms = prompt_meta["mechanism"].astype(str)
    field_names = prompt_meta["field_name"].fillna("").astype(str)
    if "frame_family" in prompt_meta.columns:
        frame_families = prompt_meta["frame_family"].fillna("").astype(str)
    else:
        frame_families = pd.Series([""] * len(prompt_meta), index=prompt_meta.index)
    if "context_family" in prompt_meta.columns:
        context_families = prompt_meta["context_family"].fillna("").astype(str)
    else:
        context_families = pd.Series([""] * len(prompt_meta), index=prompt_meta.index)
    x_centered = prompt_vectors - prompt_vectors.mean(axis=0, keepdims=True)

    direct_mask = mechanisms.isin(["direct_single", "direct_batch"])
    schema_mask = (frame_families == "schema_completion") | mechanisms.isin(["single_record_sbq", "multi_record_sbq"])
    task_frame_mask = (frame_families == "semantic_transformation") | mechanisms.isin(
        ["translate_frame", "summarize_frame", "rewrite_frame", "extract_frame"]
    )
    batch_mask = (context_families == "batched") | mechanisms.isin(["direct_batch", "multi_record_sbq"])
    single_context_mask = (~batch_mask) | (context_families == "single")

    axis_specs: dict[str, tuple[np.ndarray, np.ndarray, str]] = {
        "direct_refusal": (
            (mechanisms == "direct_single").to_numpy() & (prompt_meta["_label"].to_numpy() == 1),
            (mechanisms == "direct_single").to_numpy() & (prompt_meta["_label"].to_numpy() == 0),
            "refusal minus non-refusal within direct_single",
        ),
        "all_refusal": (
            prompt_meta["_label"].to_numpy() == 1,
            prompt_meta["_label"].to_numpy() == 0,
            "refusal minus non-refusal over all mechanisms",
        ),
        "schema": (
            schema_mask.to_numpy(),
            direct_mask.to_numpy(),
            "structured SBQ prompts minus direct prompts",
        ),
        "task_frame": (
            task_frame_mask.to_numpy(),
            (mechanisms == "direct_single").to_numpy(),
            "semantic task-frame transformations minus direct_single prompts",
        ),
        "batch": (
            batch_mask.to_numpy(),
            single_context_mask.to_numpy(),
            "batched prompts minus single-record prompts",
        ),
        "multi_vs_single_sbq": (
            (mechanisms == "multi_record_sbq").to_numpy(),
            (mechanisms == "single_record_sbq").to_numpy(),
            "multi-record SBQ minus single-record SBQ",
        ),
        "unsafe_field": (
            (field_names == "unsafe_response").to_numpy(),
            (field_names == "answer").to_numpy(),
            "unsafe_response field name minus answer field name",
        ),
    }

    axes: dict[str, np.ndarray] = {}
    axis_rows: list[dict[str, Any]] = []
    for axis_name, (mask_pos, mask_neg, description) in axis_specs.items():
        direction, info = define_axis(x_centered, mask_pos, mask_neg, min_class=args.min_class)
        axis_rows.append({"axis": axis_name, "description": description, **info})
        if direction is not None:
            axes[axis_name] = direction

    scored = prompt_meta.copy()
    for axis_name, direction in axes.items():
        scored[f"{axis_name}_score"] = x_centered @ direction

    labels = scored["_label"].to_numpy(dtype=int)
    target_defs: dict[str, np.ndarray] = {
        "refusal": labels,
        "schema": schema_mask.to_numpy(dtype=int),
        "task_frame": task_frame_mask.to_numpy(dtype=int),
        "batch": batch_mask.to_numpy(dtype=int),
        "multi_vs_single_sbq": mechanisms.isin(["multi_record_sbq"]).to_numpy(dtype=int),
        "unsafe_field": (field_names == "unsafe_response").to_numpy(dtype=int),
    }
    axis_auc_rows: list[dict[str, Any]] = []
    for axis_name in axes:
        scores = scored[f"{axis_name}_score"].to_numpy(dtype=float)
        for target_name, target in target_defs.items():
            axis_auc_rows.append(
                {
                    "axis": axis_name,
                    "target": target_name,
                    "auc": auc_or_nan(target, scores),
                }
            )

    mechanism_rows: list[dict[str, Any]] = []
    score_cols = [f"{axis_name}_score" for axis_name in axes]
    for mechanism, part in scored.groupby("mechanism", sort=True):
        labels_part = part["_label"].to_numpy(dtype=int)
        row: dict[str, Any] = {
            "mechanism": str(mechanism),
            "n_prompts": int(len(part)),
            "refusal_rate": float(labels_part.mean()),
        }
        for col in score_cols:
            row[f"{col}_mean"] = float(part[col].mean())
            if len(np.unique(labels_part)) >= 2:
                pos_mean = float(part.loc[part["_label"] == 1, col].mean())
                neg_mean = float(part.loc[part["_label"] == 0, col].mean())
                row[f"{col}_label_delta"] = pos_mean - neg_mean
                row[f"{col}_auc"] = auc_or_nan(labels_part, part[col].to_numpy(dtype=float))
            else:
                row[f"{col}_label_delta"] = float("nan")
                row[f"{col}_auc"] = float("nan")
        mechanism_rows.append(row)
    mechanism_summary = pd.DataFrame(mechanism_rows)
    if "direct_refusal_score_auc" in mechanism_summary.columns:
        mechanism_summary = mechanism_summary.rename(
            columns={
                "direct_refusal_score_auc": "direct_refusal_auc",
                "direct_refusal_score_label_delta": "direct_refusal_label_delta",
                "schema_score_score_mean": "schema_score_mean",
                "batch_score_score_mean": "batch_score_mean",
            }
        )
    if "schema_score_mean" not in mechanism_summary.columns and "schema_score_score_mean" in mechanism_summary.columns:
        mechanism_summary["schema_score_mean"] = mechanism_summary["schema_score_score_mean"]
    if "batch_score_mean" not in mechanism_summary.columns and "batch_score_score_mean" in mechanism_summary.columns:
        mechanism_summary["batch_score_mean"] = mechanism_summary["batch_score_score_mean"]

    grouping_cols = ["mechanism", "frame_family", "context_family", "frame_operation", "format", "field_name"]
    available_grouping_cols = [col for col in grouping_cols if col in scored.columns]
    format_summary = (
        scored.groupby(available_grouping_cols, dropna=False)
        .agg(
            n_prompts=("prompt_id", "count"),
            refusal_rate=("_label", "mean"),
            **{f"{col}_mean": (col, "mean") for col in score_cols},
        )
        .reset_index()
    )

    logistic_rows: list[dict[str, Any]] = []
    frame_axis_cols = [
        "schema_score",
        "task_frame_score",
        "batch_score",
        "multi_vs_single_sbq_score",
        "unsafe_field_score",
    ]
    feature_sets = {
        "direct_refusal": ["direct_refusal_score"],
        "direct_refusal+schema": ["direct_refusal_score", "schema_score"],
        "direct_refusal+schema+batch": ["direct_refusal_score", "schema_score", "batch_score"],
        "frame_axes_only": frame_axis_cols,
        "direct_refusal+frame_axes": ["direct_refusal_score", *frame_axis_cols],
    }
    for name, cols in feature_sets.items():
        cols = [col for col in cols if col in scored.columns]
        if not cols:
            continue
        logistic_rows.append({"feature_set": name, "features": ",".join(cols), "cv_auc": cv_auc(scored[cols], labels, seed=args.seed)})
    logistic_summary = pd.DataFrame(logistic_rows)

    axis_defs = pd.DataFrame(axis_rows)
    axis_auc = pd.DataFrame(axis_auc_rows)
    scored_path = args.output_dir / "prompt_axis_scores.csv"
    axis_defs_path = args.output_dir / "axis_definitions.csv"
    axis_auc_path = args.output_dir / "axis_auc_by_target.csv"
    mechanism_path = args.output_dir / "projection_by_mechanism.csv"
    format_path = args.output_dir / "projection_by_format_field.csv"
    logistic_path = args.output_dir / "combined_refusal_models.csv"
    report_path = args.output_dir / "deep_dive_report.md"

    scored.to_csv(scored_path, index=False)
    axis_defs.to_csv(axis_defs_path, index=False)
    axis_auc.to_csv(axis_auc_path, index=False)
    mechanism_summary.to_csv(mechanism_path, index=False)
    format_summary.to_csv(format_path, index=False)
    logistic_summary.to_csv(logistic_path, index=False)
    write_report(
        report_path,
        layer_index=layer_index,
        capture_point=args.capture_point,
        mechanism_summary=mechanism_summary,
        axis_auc=axis_auc[axis_auc["target"].isin(["refusal", "schema", "task_frame", "batch"])].copy(),
        logistic_summary=logistic_summary,
    )

    summary = {
        "vectors": str(args.vectors),
        "metadata": str(args.metadata),
        "capture_point": args.capture_point,
        "layer_index": int(layer_index),
        "num_prompt_units": int(len(scored)),
        "outputs": {
            "prompt_axis_scores": str(scored_path),
            "axis_definitions": str(axis_defs_path),
            "axis_auc_by_target": str(axis_auc_path),
            "projection_by_mechanism": str(mechanism_path),
            "projection_by_format_field": str(format_path),
            "combined_refusal_models": str(logistic_path),
            "report": str(report_path),
        },
    }
    write_json(args.output_dir / "deep_dive_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
