#!/usr/bin/env python3
"""Export paired Direct-to-SBQ representation coordinates for paper figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from .grouped_fold_refusal_readout import (
        load_prompt_query_ids,
        load_selected_vector_rows,
        make_prompt_table,
    )
except ImportError:
    from grouped_fold_refusal_readout import (
        load_prompt_query_ids,
        load_selected_vector_rows,
        make_prompt_table,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fold-local axis and deterministic t-SNE coordinates for paired prompts."
    )
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--fold-axes", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-point", default="query_span")
    parser.add_argument("--layer-index", type=int, default=19)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--format", default="json")
    parser.add_argument("--field-name", default="answer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    return parser.parse_args()


def paired_rows(
    prompt_vectors: np.ndarray,
    prompt_meta: pd.DataFrame,
    *,
    format_name: str,
    field_name: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta = prompt_meta.copy()
    meta["_vector_index"] = np.arange(len(meta), dtype=int)
    direct = meta[meta["mechanism"].astype(str).eq("direct_single")].copy()
    framed = meta[
        meta["mechanism"].astype(str).eq("single_record_sbq")
        & meta["format"].fillna("").astype(str).eq(format_name)
        & meta["field_name"].fillna("").astype(str).eq(field_name)
    ].copy()

    for table in (direct, framed):
        table["query_id"] = table["_query_ids"].map(
            lambda values: str(values[0]) if len(values) == 1 else ""
        )
        if (table["query_id"] == "").any():
            raise ValueError("Direct and single-record prompts must each contain one query ID.")

    columns = ["query_id", "prompt_id", "_vector_index", "generation_refusal_heuristic"]
    paired = direct[columns].merge(
        framed[columns], on="query_id", how="inner", suffixes=("_direct", "_framed")
    )
    if len(paired) != 100:
        raise ValueError(f"Expected 100 paired queries, found {len(paired)}.")

    direct_vectors = prompt_vectors[paired["_vector_index_direct"].to_numpy(dtype=int)]
    framed_vectors = prompt_vectors[paired["_vector_index_framed"].to_numpy(dtype=int)]
    paired["direct_label"] = paired["generation_refusal_heuristic_direct"].astype(bool).astype(int)
    paired["framed_label"] = paired["generation_refusal_heuristic_framed"].astype(bool).astype(int)
    paired["transition"] = paired["direct_label"].astype(str) + "→" + paired["framed_label"].astype(str)
    return paired, direct_vectors, framed_vectors


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    selected_vectors, selected_metadata = load_selected_vector_rows(
        args.vectors,
        metadata,
        capture_point=args.capture_point,
        layer_index=args.layer_index,
    )
    prompt_query_ids = load_prompt_query_ids(args.prompts)
    prompt_vectors, prompt_meta = make_prompt_table(
        selected_vectors,
        selected_metadata,
        prompt_query_ids,
    )
    paired, direct_vectors, framed_vectors = paired_rows(
        prompt_vectors,
        prompt_meta,
        format_name=args.format,
        field_name=args.field_name,
    )

    assignments = pd.read_csv(args.assignments)
    fold_by_prompt = assignments.set_index("prompt_id")["validation_fold"].astype(int).to_dict()
    paired["direct_fold"] = paired["prompt_id_direct"].map(fold_by_prompt)
    paired["framed_fold"] = paired["prompt_id_framed"].map(fold_by_prompt)
    if paired[["direct_fold", "framed_fold"]].isna().any().any():
        raise ValueError("Some paired prompts are absent from the fold assignment table.")
    if not (paired["direct_fold"] == paired["framed_fold"]).all():
        raise ValueError("A paired query crosses validation folds.")

    with np.load(args.fold_axes, allow_pickle=False) as axes:
        center = np.asarray(axes[f"fold_{args.fold}_center"], dtype=np.float64)
        refusal_axis = np.asarray(axes[f"fold_{args.fold}_direct_refusal"], dtype=np.float64)
        schema_axis = np.asarray(axes[f"fold_{args.fold}_schema"], dtype=np.float64)

    held_out_mask = paired["direct_fold"].to_numpy(dtype=int) == args.fold
    held_out = paired.loc[held_out_mask].copy().reset_index(drop=True)
    held_direct = direct_vectors[held_out_mask].astype(np.float64) - center
    held_framed = framed_vectors[held_out_mask].astype(np.float64) - center
    held_out["direct_refusal_x"] = held_direct @ schema_axis
    held_out["direct_refusal_y"] = held_direct @ refusal_axis
    held_out["framed_schema_x"] = held_framed @ schema_axis
    held_out["framed_refusal_y"] = held_framed @ refusal_axis
    held_out["schema_shift"] = held_out["framed_schema_x"] - held_out["direct_refusal_x"]
    held_out["refusal_shift"] = held_out["framed_refusal_y"] - held_out["direct_refusal_y"]
    held_out.to_csv(args.output_dir / "paired_fold_axis_coordinates.csv", index=False)

    all_vectors = np.vstack([direct_vectors, framed_vectors]).astype(np.float64)
    pca_components = min(50, len(all_vectors) - 1, all_vectors.shape[1])
    reduced = PCA(n_components=pca_components, random_state=args.seed).fit_transform(all_vectors)
    embedded = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
        max_iter=1500,
    ).fit_transform(reduced)
    tsne = paired[["query_id", "direct_label", "framed_label", "transition"]].copy()
    tsne["direct_x"] = embedded[: len(paired), 0]
    tsne["direct_y"] = embedded[: len(paired), 1]
    tsne["framed_x"] = embedded[len(paired) :, 0]
    tsne["framed_y"] = embedded[len(paired) :, 1]
    tsne.to_csv(args.output_dir / "paired_tsne_coordinates.csv", index=False)

    transition_counts = {
        transition: int(count)
        for transition, count in paired["transition"].value_counts().sort_index().items()
    }
    summary = {
        "model": "Qwen2.5-7B-Instruct",
        "layer_index": args.layer_index,
        "capture_point": args.capture_point,
        "framed_condition": {
            "mechanism": "single_record_sbq",
            "format": args.format,
            "field_name": args.field_name,
        },
        "axis_projection": {
            "validation_fold": args.fold,
            "n_pairs": int(len(held_out)),
            "transition_counts": {
                transition: int(count)
                for transition, count in held_out["transition"].value_counts().sort_index().items()
            },
            "median_schema_shift": float(held_out["schema_shift"].median()),
            "median_refusal_shift": float(held_out["refusal_shift"].median()),
            "positive_schema_shifts": int((held_out["schema_shift"] > 0).sum()),
        },
        "tsne": {
            "n_pairs": int(len(tsne)),
            "pca_components": int(pca_components),
            "perplexity": args.perplexity,
            "seed": args.seed,
            "max_iter": 1500,
            "transition_counts": transition_counts,
        },
    }
    (args.output_dir / "paired_projection_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
