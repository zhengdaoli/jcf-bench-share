#!/usr/bin/env python3
"""Analyze paired ASR@3 transitions across control framing conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONTROL_DIR = SCRIPT_DIR / "outputs" / "control"
DEFAULT_OUTPUT_DIR = CONTROL_DIR / "paired_analysis"
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "paired_frame_transitions.csv"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "paired_frame_transitions.json"

EXPECTED_QUERY_COUNT = 100
ATTEMPT_INDICES = (0, 1, 2)
CONFIDENCE_LEVEL = 0.95
WILSON_Z = NormalDist().inv_cdf(0.5 + CONFIDENCE_LEVEL / 2.0)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str


@dataclass(frozen=True)
class ConditionSpec:
    key: str
    code: str
    label: str
    filename_token: str
    batch_size: int


@dataclass(frozen=True)
class ValidatedCondition:
    model: ModelSpec
    condition: ConditionSpec
    query_ids: tuple[int | str, ...]
    queries: tuple[str, ...]
    per_sample: tuple[bool, ...]
    extracted_path: Path
    evaluation_path: Path
    extracted_sha256: str
    evaluation_sha256: str


MODELS = (
    ModelSpec("gemini_25", "Gemini"),
    ModelSpec("gpt-5-chat", "GPT-5-Chat"),
    ModelSpec("deepseek", "DeepSeek"),
    ModelSpec("qwen-plus", "Qwen-Plus"),
)

CONDITIONS = (
    ConditionSpec("direct", "D", "Direct", "direct_single", 1),
    ConditionSpec("direct_batch", "DB", "Direct Batch", "direct_batch", 10),
    ConditionSpec("sbq_single", "S", "SBQ Single", "single_record_sbq", 1),
    ConditionSpec("sbq_multi", "M", "SBQ Multi", "multi_record_sbq", 10),
)

CONTRASTS = (
    ("D_to_DB", "direct", "direct_batch"),
    ("D_to_S", "direct", "sbq_single"),
    ("S_to_M", "sbq_single", "sbq_multi"),
)

CSV_FIELDS = (
    "model_key",
    "model",
    "contrast",
    "from_condition",
    "from_condition_name",
    "to_condition",
    "to_condition_name",
    "n_queries",
    "asr_at_3_from",
    "asr_at_3_to",
    "paired_00_count",
    "paired_01_count",
    "paired_10_count",
    "paired_11_count",
    "unsafe_transition_01_rate",
    "unsafe_transition_01_ci95_low",
    "unsafe_transition_01_ci95_high",
    "disagreement_count",
    "disagreement_rate",
    "disagreement_ci95_low",
    "disagreement_ci95_high",
    "paired_marginal_difference",
    "mcnemar_discordant_count",
    "mcnemar_exact_two_sided_p",
    "holm_adjusted_p",
)


class ValidationError(ValueError):
    """Raised when an input cannot support a valid positional pairing."""


def require(condition: bool, path: Path, message: str) -> None:
    if not condition:
        raise ValidationError(f"{display_path(path)}: {message}")


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ValidationError(f"Cannot read {display_path(path)}: {exc}") from exc
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Invalid JSON in {display_path(path)}: {exc}") from exc
    return data, hashlib.sha256(raw).hexdigest()


def condition_paths(model: ModelSpec, condition: ConditionSpec) -> tuple[Path, Path]:
    stem = (
        f"{model.key}_exp1_{condition.filename_token}_"
        f"batch{condition.batch_size}_attack{len(ATTEMPT_INDICES)}"
    )
    model_dir = CONTROL_DIR / model.key
    return model_dir / f"{stem}_extracted.json", model_dir / f"{stem}_evaluation.json"


def require_rate(value: Any, path: Path, name: str) -> float:
    require(type(value) in (int, float), path, f"{name} must be numeric")
    result = float(value)
    require(math.isfinite(result), path, f"{name} must be finite")
    require(0.0 <= result <= 1.0, path, f"{name} must be between 0 and 1")
    return result


def require_close(actual: float, expected: float, path: Path, name: str) -> None:
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12),
        path,
        f"{name}={actual!r} does not match recomputed value {expected!r}",
    )


def validate_condition(model: ModelSpec, condition: ConditionSpec) -> ValidatedCondition:
    extracted_path, evaluation_path = condition_paths(model, condition)
    extracted_root, extracted_sha256 = read_json(extracted_path)
    evaluation_root, evaluation_sha256 = read_json(evaluation_path)

    require(isinstance(extracted_root, dict), extracted_path, "root must be an object")
    require(isinstance(evaluation_root, dict), evaluation_path, "root must be an object")

    extracted_data = extracted_root.get("extracted_data")
    per_sample = evaluation_root.get("per_sample")
    evaluation_items = evaluation_root.get("items")
    require(isinstance(extracted_data, list), extracted_path, "extracted_data must be a list")
    require(isinstance(per_sample, list), evaluation_path, "per_sample must be a list")
    require(isinstance(evaluation_items, list), evaluation_path, "items must be a list")
    require(
        len(extracted_data) == EXPECTED_QUERY_COUNT,
        extracted_path,
        f"expected {EXPECTED_QUERY_COUNT} extracted records, found {len(extracted_data)}",
    )
    require(
        extracted_root.get("total_queries") == EXPECTED_QUERY_COUNT,
        extracted_path,
        f"total_queries must equal {EXPECTED_QUERY_COUNT}",
    )
    require(
        len(per_sample) == EXPECTED_QUERY_COUNT,
        evaluation_path,
        f"expected {EXPECTED_QUERY_COUNT} per_sample values, found {len(per_sample)}",
    )
    require(
        all(type(value) is bool for value in per_sample),
        evaluation_path,
        "every per_sample value must be boolean",
    )
    require(
        evaluation_root.get("completed_samples") == EXPECTED_QUERY_COUNT,
        evaluation_path,
        f"completed_samples must equal {EXPECTED_QUERY_COUNT}",
    )
    require(
        evaluation_root.get("total_samples") == EXPECTED_QUERY_COUNT,
        evaluation_path,
        f"total_samples must equal {EXPECTED_QUERY_COUNT}",
    )
    expected_item_count = EXPECTED_QUERY_COUNT * len(ATTEMPT_INDICES)
    require(
        len(evaluation_items) == expected_item_count,
        evaluation_path,
        f"expected {expected_item_count} evaluation items, found {len(evaluation_items)}",
    )

    query_ids: list[int | str] = []
    queries: list[str] = []
    trial_outcomes: list[tuple[bool, ...]] = []
    for sample_index, extracted_item in enumerate(extracted_data):
        prefix = f"extracted_data[{sample_index}]"
        require(isinstance(extracted_item, dict), extracted_path, f"{prefix} must be an object")

        query = extracted_item.get("user_query")
        ids = extracted_item.get("ids")
        attack_indices = extracted_item.get("attack_indices")
        unsafe_responses = extracted_item.get("unsafe_responses")
        raw_responses = extracted_item.get("raw_responses")
        prompts = extracted_item.get("prompts")
        require(isinstance(query, str), extracted_path, f"{prefix}.user_query must be a string")
        require(isinstance(ids, list), extracted_path, f"{prefix}.ids must be a list")
        require(
            len(ids) == len(ATTEMPT_INDICES),
            extracted_path,
            f"{prefix}.ids must contain exactly three attempt slots",
        )
        require(
            all(type(query_id) in (int, str) for query_id in ids),
            extracted_path,
            f"{prefix}.ids values must be integer or string query IDs",
        )
        require(
            all(query_id == ids[0] for query_id in ids),
            extracted_path,
            f"{prefix}.ids must repeat one query ID across all attempt slots",
        )
        require(
            attack_indices == list(ATTEMPT_INDICES),
            extracted_path,
            f"{prefix}.attack_indices must be {list(ATTEMPT_INDICES)}",
        )
        for field_name, values in (
            ("unsafe_responses", unsafe_responses),
            ("raw_responses", raw_responses),
            ("prompts", prompts),
        ):
            require(isinstance(values, list), extracted_path, f"{prefix}.{field_name} must be a list")
            require(
                len(values) == len(ATTEMPT_INDICES),
                extracted_path,
                f"{prefix}.{field_name} must contain exactly three attempt slots",
            )

        item_start = sample_index * len(ATTEMPT_INDICES)
        sample_evaluations = evaluation_items[item_start : item_start + len(ATTEMPT_INDICES)]
        sample_trials: list[bool] = []
        for slot, evaluation_item in enumerate(sample_evaluations):
            item_index = item_start + slot
            item_prefix = f"items[{item_index}]"
            require(
                isinstance(evaluation_item, dict),
                evaluation_path,
                f"{item_prefix} must be an object",
            )
            require(
                evaluation_item.get("user_query") == query,
                evaluation_path,
                f"{item_prefix}.user_query does not match extracted_data[{sample_index}]",
            )
            require(
                evaluation_item.get("response") == unsafe_responses[slot],
                evaluation_path,
                f"{item_prefix}.response does not match extracted attempt slot {slot}",
            )
            require(
                evaluation_item.get("prompt") == prompts[slot],
                evaluation_path,
                f"{item_prefix}.prompt does not match extracted attempt slot {slot}",
            )
            jailbroken = evaluation_item.get("jailbroken")
            require(
                type(jailbroken) is bool,
                evaluation_path,
                f"{item_prefix}.jailbroken must be boolean",
            )
            sample_trials.append(jailbroken)

        require(
            per_sample[sample_index] == any(sample_trials),
            evaluation_path,
            f"per_sample[{sample_index}] does not equal any of its three attempt judgments",
        )
        query_ids.append(ids[0])
        queries.append(query)
        trial_outcomes.append(tuple(sample_trials))

    require(
        len(set(query_ids)) == EXPECTED_QUERY_COUNT,
        extracted_path,
        "query IDs must be unique across the 100 extracted records",
    )

    recomputed_asr = {
        k: sum(any(trials[:k]) for trials in trial_outcomes) / EXPECTED_QUERY_COUNT
        for k in range(1, len(ATTEMPT_INDICES) + 1)
    }
    for k, expected in recomputed_asr.items():
        reported = require_rate(evaluation_root.get(f"ASR@{k}_trials"), evaluation_path, f"ASR@{k}_trials")
        require_close(reported, expected, evaluation_path, f"ASR@{k}_trials")
    aggregate_asr = require_rate(evaluation_root.get("ASR@1"), evaluation_path, "ASR@1")
    require_close(aggregate_asr, recomputed_asr[3], evaluation_path, "ASR@1 aggregate")
    require_close(
        sum(per_sample) / EXPECTED_QUERY_COUNT,
        recomputed_asr[3],
        evaluation_path,
        "per_sample ASR@3",
    )

    return ValidatedCondition(
        model=model,
        condition=condition,
        query_ids=tuple(query_ids),
        queries=tuple(queries),
        per_sample=tuple(per_sample),
        extracted_path=extracted_path,
        evaluation_path=evaluation_path,
        extracted_sha256=extracted_sha256,
        evaluation_sha256=evaluation_sha256,
    )


def validate_alignment(
    conditions: dict[tuple[str, str], ValidatedCondition],
) -> tuple[tuple[int | str, ...], tuple[str, ...]]:
    reference = conditions[(MODELS[0].key, CONDITIONS[0].key)]
    for model in MODELS:
        for condition in CONDITIONS:
            current = conditions[(model.key, condition.key)]
            require(
                current.query_ids == reference.query_ids,
                current.extracted_path,
                "ordered query IDs do not align with the reference condition",
            )
            require(
                current.queries == reference.queries,
                current.extracted_path,
                "query text does not align by query ID with the reference condition",
            )
    return reference.query_ids, reference.queries


def map_outcomes_by_id(
    condition: ValidatedCondition,
) -> dict[int | str, bool]:
    # Called only after validate_alignment has established safe positional pairing.
    return dict(zip(condition.query_ids, condition.per_sample))


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    proportion = successes / n
    z_squared = WILSON_Z * WILSON_Z
    denominator = 1.0 + z_squared / n
    center = (proportion + z_squared / (2.0 * n)) / denominator
    half_width = (
        WILSON_Z
        * math.sqrt(proportion * (1.0 - proportion) / n + z_squared / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def exact_two_sided_mcnemar_p(count_01: int, count_10: int) -> float:
    discordant = count_01 + count_10
    if discordant == 0:
        return 1.0
    tail_limit = min(count_01, count_10)
    tail_numerator = sum(math.comb(discordant, k) for k in range(tail_limit + 1))
    denominator = 1 << discordant
    return min(1.0, (2.0 * tail_numerator) / denominator)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [0.0] * count
    running_max = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * p_value)
        adjusted[original_index] = min(1.0, running_max)
    return adjusted


def build_contrasts(
    conditions: dict[tuple[str, str], ValidatedCondition],
    aligned_ids: tuple[int | str, ...],
) -> list[dict[str, Any]]:
    condition_specs = {condition.key: condition for condition in CONDITIONS}
    outcomes = {
        key: map_outcomes_by_id(condition)
        for key, condition in conditions.items()
    }
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        for contrast_name, from_key, to_key in CONTRASTS:
            from_condition = condition_specs[from_key]
            to_condition = condition_specs[to_key]
            from_outcomes = outcomes[(model.key, from_key)]
            to_outcomes = outcomes[(model.key, to_key)]
            cells = {"00": 0, "01": 0, "10": 0, "11": 0}
            for query_id in aligned_ids:
                cell = f"{int(from_outcomes[query_id])}{int(to_outcomes[query_id])}"
                cells[cell] += 1

            n = len(aligned_ids)
            asr_from = (cells["10"] + cells["11"]) / n
            asr_to = (cells["01"] + cells["11"]) / n
            transition_low, transition_high = wilson_interval(cells["01"], n)
            disagreement_count = cells["01"] + cells["10"]
            disagreement_low, disagreement_high = wilson_interval(disagreement_count, n)
            mcnemar_p = exact_two_sided_mcnemar_p(cells["01"], cells["10"])
            paired_difference = (cells["01"] - cells["10"]) / n
            row = {
                "model_key": model.key,
                "model": model.label,
                "contrast": contrast_name,
                "from_condition": from_condition.code,
                "from_condition_name": from_condition.label,
                "to_condition": to_condition.code,
                "to_condition_name": to_condition.label,
                "n_queries": n,
                "asr_at_3_from": asr_from,
                "asr_at_3_to": asr_to,
                "paired_00_count": cells["00"],
                "paired_01_count": cells["01"],
                "paired_10_count": cells["10"],
                "paired_11_count": cells["11"],
                "unsafe_transition_01_rate": cells["01"] / n,
                "unsafe_transition_01_ci95_low": transition_low,
                "unsafe_transition_01_ci95_high": transition_high,
                "disagreement_count": disagreement_count,
                "disagreement_rate": disagreement_count / n,
                "disagreement_ci95_low": disagreement_low,
                "disagreement_ci95_high": disagreement_high,
                "paired_marginal_difference": paired_difference,
                "mcnemar_discordant_count": disagreement_count,
                "mcnemar_exact_two_sided_p": mcnemar_p,
                "holm_adjusted_p": None,
            }
            if not math.isclose(
                asr_to - asr_from,
                paired_difference,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("paired marginal difference identity failed")
            rows.append(row)

    adjusted = holm_adjust([row["mcnemar_exact_two_sided_p"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
    return rows


def stable_sequence_hash(values: Sequence[Any]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_json_report(
    conditions: dict[tuple[str, str], ValidatedCondition],
    aligned_ids: tuple[int | str, ...],
    aligned_queries: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    input_files = []
    for model in MODELS:
        for condition in CONDITIONS:
            validated = conditions[(model.key, condition.key)]
            input_files.append(
                {
                    "model_key": model.key,
                    "condition": condition.code,
                    "extracted_file": display_path(validated.extracted_path),
                    "extracted_sha256": validated.extracted_sha256,
                    "evaluation_file": display_path(validated.evaluation_path),
                    "evaluation_sha256": validated.evaluation_sha256,
                }
            )
    return {
        "analysis": "paired_black_box_frame_transitions",
        "schema_version": 1,
        "outcome_definition": {
            "endpoint": "ASR@3",
            "zero": "safe in all three attempt slots",
            "one": "unsafe in at least one of three attempt slots",
            "paired_cell_order": "from_endpoint_then_to_endpoint",
            "unsafe_transition": "01",
            "paired_marginal_difference": "ASR@3(to) - ASR@3(from)",
        },
        "inference": {
            "confidence_level": CONFIDENCE_LEVEL,
            "interval": "Wilson score interval using N=100",
            "mcnemar": "exact two-sided binomial test on 01 and 10 counts",
            "multiple_testing": "Holm adjustment over all 12 contrasts",
            "holm_family_size": len(rows),
        },
        "validation": {
            "model_count": len(MODELS),
            "conditions_per_model": len(CONDITIONS),
            "validated_condition_pairs": len(conditions),
            "aligned_query_count": len(aligned_ids),
            "attempt_indices": list(ATTEMPT_INDICES),
            "evaluation_items_per_condition": len(aligned_ids) * len(ATTEMPT_INDICES),
            "query_id_sequence_sha256": stable_sequence_hash(aligned_ids),
            "query_text_sequence_sha256": stable_sequence_hash(aligned_queries),
            "input_files": input_files,
        },
        "contrasts": rows,
    }


def render_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: format(value, ".15g") if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    return buffer.getvalue()


def atomic_write_text(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute paired ASR@3 frame transitions from fixed control outputs."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"CSV output path (default: {display_path(DEFAULT_CSV_PATH)})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"JSON output path (default: {display_path(DEFAULT_JSON_PATH)})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_csv = args.output_csv.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    if output_csv == output_json:
        raise ValueError("CSV and JSON output paths must differ")

    validated = {
        (model.key, condition.key): validate_condition(model, condition)
        for model in MODELS
        for condition in CONDITIONS
    }
    aligned_ids, aligned_queries = validate_alignment(validated)

    input_paths = {
        path.resolve()
        for condition in validated.values()
        for path in (condition.extracted_path, condition.evaluation_path)
    }
    for output_path in (output_csv, output_json):
        if output_path in input_paths:
            raise ValueError(f"Refusing to overwrite raw input: {display_path(output_path)}")

    rows = build_contrasts(validated, aligned_ids)
    report = build_json_report(validated, aligned_ids, aligned_queries, rows)
    atomic_write_text(output_csv, render_csv(rows))
    atomic_write_text(
        output_json,
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
    )

    print(
        f"Validated {len(validated)} condition pairs: "
        f"{len(aligned_ids)} aligned query IDs x {len(ATTEMPT_INDICES)} attempt slots."
    )
    print(f"Computed {len(rows)} paired contrasts with one Holm family.")
    print(f"CSV: {display_path(output_csv)}")
    print(f"JSON: {display_path(output_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
