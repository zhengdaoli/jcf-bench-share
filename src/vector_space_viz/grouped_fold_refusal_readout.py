from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .probe_utils import parse_boolish, read_jsonl, write_json
except ImportError:  # Support direct execution from src/vector_space_viz.
    from probe_utils import parse_boolish, read_jsonl, write_json


DIRECT_AXIS = "direct_refusal"
FRAME_AXES = (
    "schema",
    "task_frame",
    "batch",
    "multi_vs_single_sbq",
    "unsafe_field",
)
AXIS_ORDER = (DIRECT_AXIS, *FRAME_AXES)
FEATURE_SETS = {
    "direct_only": (DIRECT_AXIS,),
    "frame_only": FRAME_AXES,
    "direct_plus_frame": AXIS_ORDER,
}
PAIRED_COMPARISONS = (
    ("direct_plus_frame", "direct_only"),
    ("direct_plus_frame", "frame_only"),
    ("frame_only", "direct_only"),
)


@dataclass
class FoldBasis:
    center: np.ndarray
    axes: dict[str, np.ndarray]
    diagnostics: list[dict[str, Any]]


@dataclass
class SplitPlan:
    splitter: str
    splits: list[tuple[np.ndarray, np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free grouped cross-validation for direct-refusal and frame-axis "
            "readouts from previously exported hidden states."
        )
    )
    parser.add_argument("--vectors", type=Path, required=True, help="Existing vectors NPZ.")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV aligned to the NPZ rows.")
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Source prompt JSONL containing the complete records/query IDs for every prompt.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--capture-point",
        default="query_span",
        choices=["query_span"],
        help="This analysis intentionally operates on query-span vectors only.",
    )
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--label-col", default="generation_refusal_heuristic")
    parser.add_argument("--positive-label", default="true")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def _record_query_id(record: dict[str, Any]) -> str:
    value = record.get("query_id")
    if value is None or str(value) == "":
        value = record.get("record_id")
    if value is None or str(value) == "":
        raise ValueError("Prompt record is missing both query_id and record_id.")
    return str(value)


def load_prompt_query_ids(path: Path) -> dict[str, tuple[str, ...]]:
    prompt_query_ids: dict[str, tuple[str, ...]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row.get("prompt_id", ""))
        if not prompt_id:
            raise ValueError(f"Prompt without prompt_id in {path}")
        if prompt_id in prompt_query_ids:
            raise ValueError(f"Duplicate prompt_id in {path}: {prompt_id}")

        query_ids: list[str] = []
        seen: set[str] = set()
        for record in row.get("records", []):
            query_id = _record_query_id(record)
            if query_id not in seen:
                query_ids.append(query_id)
                seen.add(query_id)
        if not query_ids:
            raise ValueError(f"Prompt {prompt_id} has no source query IDs in {path}")
        prompt_query_ids[prompt_id] = tuple(query_ids)

    if not prompt_query_ids:
        raise ValueError(f"No prompts found in {path}")
    return prompt_query_ids


def connected_query_components(
    prompt_query_ids: dict[str, tuple[str, ...]],
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    parent: dict[str, str] = {}
    rank: dict[str, int] = {}

    def add(query_id: str) -> None:
        if query_id not in parent:
            parent[query_id] = query_id
            rank[query_id] = 0

    def find(query_id: str) -> str:
        root = query_id
        while parent[root] != root:
            root = parent[root]
        while parent[query_id] != query_id:
            next_query = parent[query_id]
            parent[query_id] = root
            query_id = next_query
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for query_ids in prompt_query_ids.values():
        for query_id in query_ids:
            add(query_id)
        for query_id in query_ids[1:]:
            union(query_ids[0], query_id)

    members_by_root: dict[str, list[str]] = {}
    for query_id in sorted(parent):
        members_by_root.setdefault(find(query_id), []).append(query_id)
    ordered_components = sorted(tuple(members) for members in members_by_root.values())

    query_to_component: dict[str, str] = {}
    components: dict[str, tuple[str, ...]] = {}
    width = max(4, len(str(max(0, len(ordered_components) - 1))))
    for index, members in enumerate(ordered_components):
        component_id = f"component_{index:0{width}d}"
        components[component_id] = members
        for query_id in members:
            query_to_component[query_id] = component_id

    return query_to_component, components


def load_selected_vector_rows(
    vectors_path: Path,
    metadata: pd.DataFrame,
    *,
    capture_point: str,
    layer_index: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    required = {"prompt_id", "capture_point", "layer_index", "query_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")

    mask = (metadata["capture_point"].astype(str) == capture_point) & (
        metadata["layer_index"].astype(int) == layer_index
    )
    selected_indices = np.flatnonzero(mask.to_numpy())
    if len(selected_indices) == 0:
        raise ValueError(
            f"No vector rows for capture_point={capture_point}, layer_index={layer_index}."
        )

    with np.load(vectors_path, allow_pickle=False) as payload:
        if "vectors" not in payload:
            raise ValueError(f"{vectors_path} does not contain a 'vectors' array")
        stored_vectors = payload["vectors"]
        if len(stored_vectors) != len(metadata):
            raise ValueError(
                f"metadata rows ({len(metadata)}) != vectors ({len(stored_vectors)})"
            )
        vectors = np.asarray(stored_vectors[selected_indices], dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError(f"Expected a 2D vectors array, got shape {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise ValueError("Selected vectors contain NaN or infinite values.")
    rows = metadata.iloc[selected_indices].copy().reset_index(drop=True)
    return vectors, rows


def _assert_prompt_column_consistent(group: pd.DataFrame, column: str, prompt_id: str) -> None:
    if column not in group.columns:
        return
    values = group[column].fillna("").astype(str).unique()
    if len(values) != 1:
        raise ValueError(
            f"Metadata column {column!r} is inconsistent within prompt {prompt_id}: "
            f"{values.tolist()}"
        )


def make_prompt_table(
    selected_vectors: np.ndarray,
    selected_metadata: pd.DataFrame,
    prompt_query_ids: dict[str, tuple[str, ...]],
    *,
    label_col: str = "generation_refusal_heuristic",
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(selected_vectors) != len(selected_metadata):
        raise ValueError("Selected vector and metadata row counts differ.")

    rows = selected_metadata.copy()
    rows["_selected_vector_row"] = np.arange(len(rows), dtype=int)
    keep_cols = [
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
        label_col,
        "generated_text_len",
    ]
    keep_cols = [column for column in keep_cols if column in rows.columns]
    consistency_cols = [
        column
        for column in keep_cols
        if column not in {"generated_text_len"}
    ]

    prompt_vectors: list[np.ndarray] = []
    prompt_records: list[dict[str, Any]] = []
    for raw_prompt_id, group in rows.groupby("prompt_id", sort=False):
        prompt_id = str(raw_prompt_id)
        if prompt_id not in prompt_query_ids:
            raise ValueError(
                f"Metadata prompt {prompt_id} is absent from the source prompt JSONL."
            )
        for column in consistency_cols:
            _assert_prompt_column_consistent(group, column, prompt_id)

        source_query_ids = prompt_query_ids[prompt_id]
        source_query_set = set(source_query_ids)
        vector_query_ids = tuple(
            dict.fromkeys(
                str(value)
                for value in group["query_id"].dropna().tolist()
                if str(value) != ""
            )
        )
        unexpected_query_ids = sorted(set(vector_query_ids) - source_query_set)
        if unexpected_query_ids:
            raise ValueError(
                f"Prompt {prompt_id} has metadata query IDs absent from JSONL records: "
                f"{unexpected_query_ids}"
            )

        vector_rows = group["_selected_vector_row"].to_numpy(dtype=int)
        prompt_vectors.append(
            selected_vectors[vector_rows].mean(axis=0, dtype=np.float64).astype(np.float32)
        )
        record = {column: group.iloc[0][column] for column in keep_cols}
        record.update(
            {
                "prompt_id": prompt_id,
                "n_vector_rows": int(len(group)),
                "n_records": (
                    int(group["record_id"].nunique(dropna=True))
                    if "record_id" in group.columns
                    else int(len(group))
                ),
                "n_source_query_ids": int(len(source_query_ids)),
                "n_vector_query_ids": int(len(vector_query_ids)),
                "_query_ids": source_query_ids,
                "_vector_query_ids": vector_query_ids,
            }
        )
        prompt_records.append(record)

    if not prompt_vectors:
        raise ValueError("No prompt-level vectors could be constructed.")
    return np.stack(prompt_vectors).astype(np.float32), pd.DataFrame(prompt_records)


def assign_prompt_components(
    prompt_metadata: pd.DataFrame,
    query_to_component: dict[str, str],
) -> pd.DataFrame:
    result = prompt_metadata.copy()
    component_ids: list[str] = []
    for prompt_id, query_ids in zip(result["prompt_id"], result["_query_ids"]):
        components = {query_to_component[query_id] for query_id in query_ids}
        if len(components) != 1:
            raise AssertionError(
                f"Prompt {prompt_id} spans multiple query components: {sorted(components)}"
            )
        component_ids.append(next(iter(components)))
    result["component_id"] = component_ids
    return result


def _text_column(metadata: pd.DataFrame, column: str) -> pd.Series:
    if column not in metadata.columns:
        return pd.Series([""] * len(metadata), index=metadata.index, dtype=str)
    return metadata[column].fillna("").astype(str)


def axis_specs(
    metadata: pd.DataFrame,
    labels: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    mechanisms = _text_column(metadata, "mechanism")
    field_names = _text_column(metadata, "field_name")
    frame_families = _text_column(metadata, "frame_family")
    context_families = _text_column(metadata, "context_family")

    direct_mask = mechanisms.isin(["direct_single", "direct_batch"])
    schema_mask = (frame_families == "schema_completion") | mechanisms.isin(
        ["single_record_sbq", "multi_record_sbq"]
    )
    task_frame_mask = (frame_families == "semantic_transformation") | mechanisms.isin(
        ["translate_frame", "summarize_frame", "rewrite_frame", "extract_frame"]
    )
    batch_mask = (context_families == "batched") | mechanisms.isin(
        ["direct_batch", "multi_record_sbq"]
    )
    single_context_mask = (~batch_mask) | (context_families == "single")
    direct_single = (mechanisms == "direct_single").to_numpy()

    return {
        DIRECT_AXIS: (
            direct_single & (labels == 1),
            direct_single & (labels == 0),
            "refusal minus non-refusal within direct_single",
        ),
        "schema": (
            schema_mask.to_numpy(),
            direct_mask.to_numpy(),
            "structured SBQ prompts minus direct prompts",
        ),
        "task_frame": (
            task_frame_mask.to_numpy(),
            direct_single,
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


def fit_fold_axes(
    prompt_vectors: np.ndarray,
    prompt_metadata: pd.DataFrame,
    labels: np.ndarray,
    train_indices: np.ndarray,
    *,
    min_class: int,
) -> FoldBasis:
    if len(prompt_vectors) != len(prompt_metadata) or len(labels) != len(prompt_metadata):
        raise ValueError("Prompt vectors, metadata, and labels must have equal length.")
    if len(train_indices) == 0:
        raise ValueError("Cannot fit fold axes without training prompts.")

    train_vectors = np.asarray(prompt_vectors[train_indices], dtype=np.float64)
    train_metadata = prompt_metadata.iloc[train_indices].reset_index(drop=True)
    train_labels = np.asarray(labels[train_indices], dtype=int)
    center = train_vectors.mean(axis=0)
    train_centered = train_vectors - center

    axes: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    specs = axis_specs(train_metadata, train_labels)
    for axis_name in AXIS_ORDER:
        positive_mask, negative_mask, description = specs[axis_name]
        positive_indices = np.flatnonzero(positive_mask)
        negative_indices = np.flatnonzero(negative_mask)
        if len(positive_indices) < min_class or len(negative_indices) < min_class:
            raise ValueError(
                f"Training fold cannot construct {axis_name}: positive={len(positive_indices)}, "
                f"negative={len(negative_indices)}, min_class={min_class}."
            )
        direction = train_centered[positive_indices].mean(axis=0) - train_centered[
            negative_indices
        ].mean(axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0 or not np.isfinite(direction_norm):
            raise ValueError(
                f"Training fold produced an invalid zero/non-finite {axis_name} direction."
            )
        axes[axis_name] = direction / direction_norm
        diagnostics.append(
            {
                "axis": axis_name,
                "description": description,
                "n_positive_train": int(len(positive_indices)),
                "n_negative_train": int(len(negative_indices)),
                "direction_norm": direction_norm,
            }
        )

    return FoldBasis(center=center, axes=axes, diagnostics=diagnostics)


def project_with_basis(prompt_vectors: np.ndarray, basis: FoldBasis) -> np.ndarray:
    centered = np.asarray(prompt_vectors, dtype=np.float64) - basis.center
    return np.column_stack([centered @ basis.axes[axis_name] for axis_name in AXIS_ORDER])


def query_id_union(prompt_metadata: pd.DataFrame, indices: np.ndarray) -> set[str]:
    query_ids: set[str] = set()
    for query_tuple in prompt_metadata.iloc[indices]["_query_ids"]:
        query_ids.update(query_tuple)
    return query_ids


def assert_query_disjoint(
    prompt_metadata: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[set[str], set[str]]:
    train_query_ids = query_id_union(prompt_metadata, train_indices)
    validation_query_ids = query_id_union(prompt_metadata, validation_indices)
    overlap = train_query_ids & validation_query_ids
    assert not overlap, f"Source query IDs cross train/validation: {sorted(overlap)[:10]}"
    return train_query_ids, validation_query_ids


def _splits_are_usable(
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    groups: np.ndarray,
    fold_validator: Callable[[np.ndarray, np.ndarray], bool] | None,
) -> bool:
    if not splits:
        return False
    validation_indices: list[np.ndarray] = []
    for train_indices, fold_validation_indices in splits:
        if len(np.unique(labels[train_indices])) < 2:
            return False
        if len(np.unique(labels[fold_validation_indices])) < 2:
            return False
        if set(groups[train_indices]) & set(groups[fold_validation_indices]):
            return False
        if fold_validator is not None and not fold_validator(
            train_indices, fold_validation_indices
        ):
            return False
        validation_indices.append(fold_validation_indices)
    observed = np.sort(np.concatenate(validation_indices))
    return np.array_equal(observed, np.arange(len(labels)))


def make_grouped_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    fold_validator: Callable[[np.ndarray, np.ndarray], bool] | None = None,
) -> SplitPlan:
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=str)
    if len(labels) != len(groups):
        raise ValueError("Labels and groups must have equal length.")
    if len(np.unique(labels)) != 2:
        raise ValueError("Grouped refusal readout requires two label classes.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    unique_groups = np.unique(groups)
    max_splits = min(int(n_splits), int(len(unique_groups)))
    if max_splits < 2:
        raise ValueError("At least two connected query components are required.")
    placeholder = np.zeros((len(labels), 1), dtype=np.float32)

    for candidate_splits in range(max_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=candidate_splits,
            shuffle=True,
            random_state=seed,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                splits = list(splitter.split(placeholder, labels, groups))
        except ValueError:
            continue
        if _splits_are_usable(splits, labels, groups, fold_validator):
            return SplitPlan(splitter="StratifiedGroupKFold", splits=splits)

    # GroupKFold is a conservative fallback for distributions the stratified
    # greedy assignment cannot partition into binary validation folds.
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    group_codes = {group: index for index, group in enumerate(shuffled_groups)}
    encoded_groups = np.asarray([group_codes[group] for group in groups], dtype=int)
    for candidate_splits in range(max_splits, 1, -1):
        splitter = GroupKFold(n_splits=candidate_splits)
        try:
            splits = list(splitter.split(placeholder, labels, encoded_groups))
        except ValueError:
            continue
        if _splits_are_usable(splits, labels, groups, fold_validator):
            return SplitPlan(splitter="GroupKFold", splits=splits)

    class_counts = pd.Series(labels).value_counts().sort_index().to_dict()
    raise ValueError(
        "Could not form grouped folds with both label classes in every train and "
        f"validation partition (groups={len(unique_groups)}, class_counts={class_counts})."
    )


def _feature_columns(feature_axes: tuple[str, ...]) -> list[int]:
    return [AXIS_ORDER.index(axis_name) for axis_name in feature_axes]


def _json_float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def evaluate_grouped_readout(
    prompt_vectors: np.ndarray,
    prompt_metadata: pd.DataFrame,
    prompt_query_ids: dict[str, tuple[str, ...]],
    components: dict[str, tuple[str, ...]],
    *,
    output_dir: Path,
    n_splits: int,
    min_class: int,
    seed: int,
    run_arguments: dict[str, Any],
) -> dict[str, Any]:
    labels = prompt_metadata["_label"].to_numpy(dtype=int)
    groups = prompt_metadata["component_id"].astype(str).to_numpy()

    def fold_has_required_axes(
        train_indices: np.ndarray, validation_indices: np.ndarray
    ) -> bool:
        del validation_indices
        try:
            fit_fold_axes(
                prompt_vectors,
                prompt_metadata,
                labels,
                train_indices,
                min_class=min_class,
            )
        except ValueError:
            return False
        return True

    split_plan = make_grouped_splits(
        labels,
        groups,
        n_splits=n_splits,
        seed=seed,
        fold_validator=fold_has_required_axes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_scores = {
        feature_set: np.full(len(prompt_metadata), np.nan, dtype=float)
        for feature_set in FEATURE_SETS
    }
    validation_fold = np.full(len(prompt_metadata), -1, dtype=int)
    fold_auc_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []
    saved_axes: dict[str, np.ndarray] = {}

    for fold_index, (train_indices, validation_indices) in enumerate(split_plan.splits):
        train_query_ids, validation_query_ids = assert_query_disjoint(
            prompt_metadata, train_indices, validation_indices
        )
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])
        assert not train_groups & validation_groups, "Component IDs cross train/validation."

        basis = fit_fold_axes(
            prompt_vectors,
            prompt_metadata,
            labels,
            train_indices,
            min_class=min_class,
        )
        train_projection = project_with_basis(prompt_vectors[train_indices], basis)
        validation_projection = project_with_basis(
            prompt_vectors[validation_indices], basis
        )
        train_labels = labels[train_indices]
        validation_labels = labels[validation_indices]

        saved_axes[f"fold_{fold_index}_center"] = basis.center.astype(np.float32)
        for axis_name in AXIS_ORDER:
            saved_axes[f"fold_{fold_index}_{axis_name}"] = basis.axes[axis_name].astype(
                np.float32
            )
        for diagnostic in basis.diagnostics:
            axis_rows.append(
                {
                    "fold": fold_index,
                    "n_train_prompts": int(len(train_indices)),
                    "n_validation_prompts": int(len(validation_indices)),
                    "n_train_components": int(len(train_groups)),
                    "n_validation_components": int(len(validation_groups)),
                    "n_train_query_ids": int(len(train_query_ids)),
                    "n_validation_query_ids": int(len(validation_query_ids)),
                    **diagnostic,
                }
            )

        for feature_set, feature_axes in FEATURE_SETS.items():
            columns = _feature_columns(feature_axes)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            )
            model.fit(train_projection[:, columns], train_labels)
            probabilities = model.predict_proba(
                validation_projection[:, columns]
            )[:, 1]
            oof_scores[feature_set][validation_indices] = probabilities
            fold_auc_rows.append(
                {
                    "fold": fold_index,
                    "splitter": split_plan.splitter,
                    "feature_set": feature_set,
                    "features": ",".join(feature_axes),
                    "n_train": int(len(train_indices)),
                    "n_validation": int(len(validation_indices)),
                    "validation_positive_rate": float(validation_labels.mean()),
                    "auc": float(roc_auc_score(validation_labels, probabilities)),
                }
            )
        validation_fold[validation_indices] = fold_index

    if (validation_fold < 0).any():
        raise AssertionError("At least one prompt did not receive an OOF fold assignment.")
    for feature_set, scores in oof_scores.items():
        if not np.isfinite(scores).all():
            raise AssertionError(f"Incomplete OOF predictions for {feature_set}.")

    fold_auc = pd.DataFrame(fold_auc_rows)
    paired_rows: list[dict[str, Any]] = []
    for fold_index in sorted(fold_auc["fold"].unique()):
        fold_values = fold_auc[fold_auc["fold"] == fold_index].set_index(
            "feature_set"
        )["auc"]
        for feature_set_a, feature_set_b in PAIRED_COMPARISONS:
            auc_a = float(fold_values[feature_set_a])
            auc_b = float(fold_values[feature_set_b])
            paired_rows.append(
                {
                    "fold": int(fold_index),
                    "feature_set_a": feature_set_a,
                    "feature_set_b": feature_set_b,
                    "auc_a": auc_a,
                    "auc_b": auc_b,
                    "auc_a_minus_b": auc_a - auc_b,
                }
            )
    paired_differences = pd.DataFrame(paired_rows)

    oof_rows: list[dict[str, Any]] = []
    for row_index, row in prompt_metadata.reset_index(drop=True).iterrows():
        source_query_ids = list(row["_query_ids"])
        for feature_set in FEATURE_SETS:
            oof_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "component_id": row["component_id"],
                    "validation_fold": int(validation_fold[row_index]),
                    "label": int(labels[row_index]),
                    "feature_set": feature_set,
                    "probability": float(oof_scores[feature_set][row_index]),
                    "n_source_query_ids": int(len(source_query_ids)),
                    "source_query_ids": json.dumps(source_query_ids, ensure_ascii=False),
                }
            )
    oof_predictions = pd.DataFrame(oof_rows)

    prompt_assignment_rows: list[dict[str, Any]] = []
    component_to_fold: dict[str, int] = {}
    for row_index, row in prompt_metadata.reset_index(drop=True).iterrows():
        component_id = str(row["component_id"])
        fold_index = int(validation_fold[row_index])
        prior_fold = component_to_fold.setdefault(component_id, fold_index)
        assert prior_fold == fold_index, f"Component {component_id} spans OOF folds."
        prompt_assignment_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "mechanism": row.get("mechanism", ""),
                "component_id": component_id,
                "validation_fold": fold_index,
                "n_source_query_ids": int(row["n_source_query_ids"]),
                "source_query_ids": json.dumps(list(row["_query_ids"]), ensure_ascii=False),
                "n_vector_query_ids": int(row["n_vector_query_ids"]),
                "vector_query_ids": json.dumps(
                    list(row["_vector_query_ids"]), ensure_ascii=False
                ),
                "n_vector_rows": int(row["n_vector_rows"]),
            }
        )
    prompt_assignments = pd.DataFrame(prompt_assignment_rows)

    all_prompt_ids_by_query: dict[str, set[str]] = {
        query_id: set() for members in components.values() for query_id in members
    }
    for prompt_id, query_ids in prompt_query_ids.items():
        for query_id in query_ids:
            all_prompt_ids_by_query[query_id].add(prompt_id)
    analyzed_prompt_ids_by_query: dict[str, set[str]] = {
        query_id: set() for query_id in all_prompt_ids_by_query
    }
    for _, row in prompt_metadata.iterrows():
        for query_id in row["_query_ids"]:
            analyzed_prompt_ids_by_query[query_id].add(str(row["prompt_id"]))

    query_assignment_rows: list[dict[str, Any]] = []
    for component_id, members in components.items():
        component_fold = component_to_fold.get(component_id, -1)
        for query_id in members:
            query_assignment_rows.append(
                {
                    "query_id": query_id,
                    "component_id": component_id,
                    "validation_fold": component_fold,
                    "component_size": int(len(members)),
                    "n_source_prompts": int(len(all_prompt_ids_by_query[query_id])),
                    "n_analyzed_prompts": int(
                        len(analyzed_prompt_ids_by_query[query_id])
                    ),
                }
            )
    query_assignments = pd.DataFrame(query_assignment_rows)

    fold_auc_path = output_dir / "fold_auc.csv"
    paired_path = output_dir / "paired_feature_set_differences.csv"
    oof_path = output_dir / "oof_predictions.csv"
    prompt_assignments_path = output_dir / "prompt_group_assignments.csv"
    query_assignments_path = output_dir / "query_group_assignments.csv"
    axis_definitions_path = output_dir / "fold_axis_definitions.csv"
    axes_path = output_dir / "fold_training_axes.npz"
    summary_path = output_dir / "grouped_fold_refusal_readout_summary.json"

    fold_auc.to_csv(fold_auc_path, index=False)
    paired_differences.to_csv(paired_path, index=False)
    oof_predictions.to_csv(oof_path, index=False)
    prompt_assignments.to_csv(prompt_assignments_path, index=False)
    query_assignments.to_csv(query_assignments_path, index=False)
    pd.DataFrame(axis_rows).to_csv(axis_definitions_path, index=False)
    np.savez_compressed(axes_path, **saved_axes)

    overall_metrics: dict[str, dict[str, float | None]] = {}
    for feature_set, scores in oof_scores.items():
        fold_values = fold_auc.loc[
            fold_auc["feature_set"] == feature_set, "auc"
        ].to_numpy(dtype=float)
        overall_metrics[feature_set] = {
            "oof_auc": _json_float(float(roc_auc_score(labels, scores))),
            "mean_fold_auc": _json_float(float(fold_values.mean())),
            "std_fold_auc": _json_float(float(fold_values.std(ddof=1))),
        }

    paired_summary: list[dict[str, Any]] = []
    for feature_set_a, feature_set_b in PAIRED_COMPARISONS:
        differences = paired_differences.loc[
            (paired_differences["feature_set_a"] == feature_set_a)
            & (paired_differences["feature_set_b"] == feature_set_b),
            "auc_a_minus_b",
        ].to_numpy(dtype=float)
        paired_summary.append(
            {
                "feature_set_a": feature_set_a,
                "feature_set_b": feature_set_b,
                "mean_fold_auc_difference": _json_float(float(differences.mean())),
                "std_fold_auc_difference": _json_float(
                    float(differences.std(ddof=1))
                ),
                "oof_auc_difference": _json_float(
                    float(
                        overall_metrics[feature_set_a]["oof_auc"]
                        - overall_metrics[feature_set_b]["oof_auc"]
                    )
                ),
            }
        )

    summary = {
        "run_arguments": run_arguments,
        "random_seed": int(seed),
        "splitter": split_plan.splitter,
        "requested_n_splits": int(n_splits),
        "actual_n_splits": int(len(split_plan.splits)),
        "num_prompt_units": int(len(prompt_metadata)),
        "num_source_prompts": int(len(prompt_query_ids)),
        "num_query_components": int(len(components)),
        "num_source_query_ids": int(sum(len(members) for members in components.values())),
        "label_counts": {
            str(label): int(count)
            for label, count in pd.Series(labels).value_counts().sort_index().items()
        },
        "axes": list(AXIS_ORDER),
        "feature_sets": {
            feature_set: list(feature_axes)
            for feature_set, feature_axes in FEATURE_SETS.items()
        },
        "metrics": overall_metrics,
        "paired_differences": paired_summary,
        "outputs": {
            "fold_auc": str(fold_auc_path),
            "paired_feature_set_differences": str(paired_path),
            "oof_predictions": str(oof_path),
            "prompt_group_assignments": str(prompt_assignments_path),
            "query_group_assignments": str(query_assignments_path),
            "fold_axis_definitions": str(axis_definitions_path),
            "fold_training_axes": str(axes_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)
    return summary


def _serialized_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    if args.min_class < 1:
        raise ValueError("--min-class must be at least 1")

    prompt_query_ids = load_prompt_query_ids(args.prompts)
    query_to_component, components = connected_query_components(prompt_query_ids)
    metadata = pd.read_csv(
        args.metadata,
        dtype={"prompt_id": "string", "query_id": "string", "record_id": "string"},
    )
    if args.label_col not in metadata.columns:
        raise ValueError(f"Missing label column: {args.label_col}")
    selected_vectors, selected_metadata = load_selected_vector_rows(
        args.vectors,
        metadata,
        capture_point=args.capture_point,
        layer_index=args.layer_index,
    )
    prompt_vectors, prompt_metadata = make_prompt_table(
        selected_vectors,
        selected_metadata,
        prompt_query_ids,
        label_col=args.label_col,
    )
    prompt_metadata["_label"] = binary_labels(
        prompt_metadata[args.label_col], args.positive_label
    )
    prompt_metadata = assign_prompt_components(prompt_metadata, query_to_component)

    summary = evaluate_grouped_readout(
        prompt_vectors,
        prompt_metadata,
        prompt_query_ids,
        components,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        min_class=args.min_class,
        seed=args.seed,
        run_arguments=_serialized_arguments(args),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
