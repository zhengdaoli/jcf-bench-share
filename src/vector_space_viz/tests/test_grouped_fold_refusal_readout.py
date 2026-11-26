from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.vector_space_viz import grouped_fold_refusal_readout as readout


class GroupedFoldRefusalReadoutTest(unittest.TestCase):
    @staticmethod
    def _synthetic_prompt_data() -> tuple[
        np.ndarray,
        pd.DataFrame,
        dict[str, tuple[str, ...]],
    ]:
        vectors: list[np.ndarray] = []
        metadata_rows: list[dict[str, object]] = []
        prompt_query_ids: dict[str, tuple[str, ...]] = {}

        def add_prompt(
            prompt_id: str,
            mechanism: str,
            query_ids: tuple[str, ...],
            label: int,
            *,
            frame_family: str,
            context_family: str,
            field_name: str = "",
        ) -> None:
            prompt_query_ids[prompt_id] = query_ids
            is_schema = mechanism in {"single_record_sbq", "multi_record_sbq"}
            is_task_frame = frame_family == "semantic_transformation"
            is_batch = context_family == "batched"
            is_multi = mechanism == "multi_record_sbq"
            field_value = 1.0 if field_name == "unsafe_response" else -1.0
            base = np.asarray(
                [
                    2.0 * label - 1.0,
                    float(is_schema),
                    float(is_task_frame),
                    float(is_batch),
                    float(is_multi),
                    field_value if field_name else 0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )

            # Keep only two vector-side IDs for batched prompts. The complete set
            # of ten source IDs must still drive connected-component grouping.
            vector_query_ids = query_ids[:2] if len(query_ids) > 2 else query_ids
            for position, query_id in enumerate(vector_query_ids):
                vector = base.copy()
                vector[6] = float(int(query_id.split("q")[-1])) / 100.0
                vector[7] = float(position) / 100.0
                vectors.append(vector)
                metadata_rows.append(
                    {
                        "prompt_id": prompt_id,
                        "mechanism": mechanism,
                        "frame_family": frame_family,
                        "context_family": context_family,
                        "frame_operation": mechanism,
                        "format": "json" if is_schema else "direct",
                        "field_name": field_name,
                        "batch_size": len(query_ids),
                        "generation_refusal_heuristic": bool(label),
                        "capture_point": "query_span",
                        "layer_index": 19,
                        "query_id": query_id,
                        "record_id": query_id,
                    }
                )

        for component_index in range(6):
            query_ids = tuple(
                f"c{component_index}_q{query_index}" for query_index in range(10)
            )
            prefix = f"c{component_index}"
            add_prompt(
                f"{prefix}_direct_negative",
                "direct_single",
                (query_ids[0],),
                0,
                frame_family="direct_request",
                context_family="single",
            )
            add_prompt(
                f"{prefix}_direct_positive",
                "direct_single",
                (query_ids[1],),
                1,
                frame_family="direct_request",
                context_family="single",
            )
            add_prompt(
                f"{prefix}_direct_batch",
                "direct_batch",
                query_ids,
                component_index % 2,
                frame_family="direct_request",
                context_family="batched",
            )
            add_prompt(
                f"{prefix}_single_answer",
                "single_record_sbq",
                (query_ids[2],),
                0,
                frame_family="schema_completion",
                context_family="single",
                field_name="answer",
            )
            add_prompt(
                f"{prefix}_single_unsafe",
                "single_record_sbq",
                (query_ids[3],),
                1,
                frame_family="schema_completion",
                context_family="single",
                field_name="unsafe_response",
            )
            add_prompt(
                f"{prefix}_multi_answer",
                "multi_record_sbq",
                query_ids,
                0,
                frame_family="schema_completion",
                context_family="batched",
                field_name="answer",
            )
            add_prompt(
                f"{prefix}_multi_unsafe",
                "multi_record_sbq",
                query_ids,
                1,
                frame_family="schema_completion",
                context_family="batched",
                field_name="unsafe_response",
            )
            add_prompt(
                f"{prefix}_translate",
                "translate_frame",
                (query_ids[4],),
                0,
                frame_family="semantic_transformation",
                context_family="single",
            )
            add_prompt(
                f"{prefix}_summarize",
                "summarize_frame",
                (query_ids[5],),
                1,
                frame_family="semantic_transformation",
                context_family="single",
            )

        return np.stack(vectors), pd.DataFrame(metadata_rows), prompt_query_ids

    def test_group_isolation_and_validation_labels_do_not_change_axes(self) -> None:
        selected_vectors, selected_metadata, prompt_query_ids = (
            self._synthetic_prompt_data()
        )
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        temporary_root = Path(temporary_directory.name)
        vectors_path = temporary_root / "vectors.npz"
        metadata_path = temporary_root / "metadata.csv"
        prompts_path = temporary_root / "prompts.jsonl"
        np.savez_compressed(vectors_path, vectors=selected_vectors)
        selected_metadata.to_csv(metadata_path, index=False)
        prompts_path.write_text(
            "\n".join(
                json.dumps(
                    {
                        "prompt_id": prompt_id,
                        "records": [
                            {"query_id": query_id, "record_id": query_id}
                            for query_id in query_ids
                        ],
                    }
                )
                for prompt_id, query_ids in prompt_query_ids.items()
            )
            + "\n",
            encoding="utf-8",
        )

        prompt_query_ids = readout.load_prompt_query_ids(prompts_path)
        loaded_metadata = pd.read_csv(
            metadata_path,
            dtype={
                "prompt_id": "string",
                "query_id": "string",
                "record_id": "string",
            },
        )
        selected_vectors, selected_metadata = readout.load_selected_vector_rows(
            vectors_path,
            loaded_metadata,
            capture_point="query_span",
            layer_index=19,
        )
        prompt_vectors, prompt_metadata = readout.make_prompt_table(
            selected_vectors,
            selected_metadata,
            prompt_query_ids,
        )
        query_to_component, components = readout.connected_query_components(
            prompt_query_ids
        )
        prompt_metadata["_label"] = readout.binary_labels(
            prompt_metadata["generation_refusal_heuristic"], "true"
        )
        prompt_metadata = readout.assign_prompt_components(
            prompt_metadata, query_to_component
        )

        batch_rows = prompt_metadata[
            prompt_metadata["mechanism"] == "direct_batch"
        ]
        self.assertTrue((batch_rows["n_source_query_ids"] == 10).all())
        self.assertTrue((batch_rows["n_vector_query_ids"] == 2).all())
        self.assertEqual(len(components), 6)

        labels = prompt_metadata["_label"].to_numpy(dtype=int)
        groups = prompt_metadata["component_id"].to_numpy(dtype=str)
        split_plan = readout.make_grouped_splits(
            labels,
            groups,
            n_splits=3,
            seed=17,
        )
        self.assertEqual(split_plan.splitter, "StratifiedGroupKFold")

        for train_indices, validation_indices in split_plan.splits:
            train_query_ids, validation_query_ids = readout.assert_query_disjoint(
                prompt_metadata, train_indices, validation_indices
            )
            self.assertTrue(train_query_ids.isdisjoint(validation_query_ids))

        train_indices, validation_indices = split_plan.splits[0]
        original_basis = readout.fit_fold_axes(
            prompt_vectors,
            prompt_metadata,
            labels,
            train_indices,
            min_class=1,
        )
        changed_labels = labels.copy()
        changed_labels[validation_indices] = 1 - changed_labels[validation_indices]
        changed_basis = readout.fit_fold_axes(
            prompt_vectors,
            prompt_metadata,
            changed_labels,
            train_indices,
            min_class=1,
        )
        np.testing.assert_array_equal(original_basis.center, changed_basis.center)
        for axis_name in readout.AXIS_ORDER:
            np.testing.assert_array_equal(
                original_basis.axes[axis_name], changed_basis.axes[axis_name]
            )

        output_dir = temporary_root / "output"
        summary = readout.evaluate_grouped_readout(
            prompt_vectors,
            prompt_metadata,
            prompt_query_ids,
            components,
            output_dir=output_dir,
            n_splits=3,
            min_class=1,
            seed=17,
            run_arguments={"synthetic": True, "seed": 17},
        )
        self.assertEqual(summary["actual_n_splits"], 3)
        self.assertEqual(summary["random_seed"], 17)
        for output_path in summary["outputs"].values():
            self.assertTrue(Path(output_path).is_file(), output_path)


if __name__ == "__main__":
    unittest.main()
