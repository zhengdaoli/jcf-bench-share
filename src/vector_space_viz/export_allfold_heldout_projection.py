#!/usr/bin/env python3
"""Project all 100 matched pairs onto the axes of the fold that held them out.

Every fold's schema / direct-refusal axes are unit-normalised (see
``grouped_fold_refusal_readout.axis_specs``), so per-fold held-out projections
share a comparable scale and can be shown together. This yields 100 fully
out-of-fold pairs (200 points) instead of the single-fold 20.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_fold_refusal_readout import (
    load_prompt_query_ids,
    load_selected_vector_rows,
    make_prompt_table,
)
from export_paired_reframing_projection import paired_rows

OUT = Path(__file__).resolve().parent / "figs/main_figure1"

LAYER_INDEX = 19
CAPTURE_POINT = "query_span"
FORMAT_NAME = "json"
FIELD_NAME = "answer"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Location of an existing local representation run.")
    args = parser.parse_args()
    run = args.run_dir.expanduser()
    model = run / "qwen2p5"
    analysis = model / "grouped_fold_refusal_readout_layer19_seed42"
    metadata = pd.read_csv(model / "qwen2p5_frame_m50_20260511_135500_metadata.csv")
    vectors, selected_meta = load_selected_vector_rows(
        model / "qwen2p5_frame_m50_20260511_135500_vectors.npz",
        metadata,
        capture_point=CAPTURE_POINT,
        layer_index=LAYER_INDEX,
    )
    prompt_query_ids = load_prompt_query_ids(run / "frame_invariance_probe_prompts.jsonl")
    prompt_vectors, prompt_meta = make_prompt_table(vectors, selected_meta, prompt_query_ids)
    paired, direct_vectors, framed_vectors = paired_rows(
        prompt_vectors, prompt_meta, format_name=FORMAT_NAME, field_name=FIELD_NAME
    )

    assignments = pd.read_csv(analysis / "prompt_group_assignments.csv")
    fold_by_prompt = assignments.set_index("prompt_id")["validation_fold"].astype(int).to_dict()
    paired["fold"] = paired["prompt_id_direct"].map(fold_by_prompt).astype(int)

    direct_vectors = direct_vectors.astype(np.float64)
    framed_vectors = framed_vectors.astype(np.float64)

    with np.load(analysis / "fold_training_axes.npz", allow_pickle=False) as axes:
        centers = {k: np.asarray(axes[f"fold_{k}_center"], np.float64) for k in range(5)}
        schema = {k: np.asarray(axes[f"fold_{k}_schema"], np.float64) for k in range(5)}
        refusal = {k: np.asarray(axes[f"fold_{k}_direct_refusal"], np.float64) for k in range(5)}

    rows = []
    for i, (_, r) in enumerate(paired.iterrows()):
        k = int(r["fold"])
        d = direct_vectors[i] - centers[k]
        f = framed_vectors[i] - centers[k]
        rows.append({
            "query_id": r["query_id"],
            "fold": k,
            "direct_label": int(r["direct_label"]),
            "framed_label": int(r["framed_label"]),
            "transition": r["transition"],
            "direct_schema_x": float(d @ schema[k]),
            "direct_refusal_y": float(d @ refusal[k]),
            "framed_schema_x": float(f @ schema[k]),
            "framed_refusal_y": float(f @ refusal[k]),
        })

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "allfold_heldout_axis_coordinates.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} pairs)")
    print(out["transition"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
