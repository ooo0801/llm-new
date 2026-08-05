from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


FAMILIES = (
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
)
SEEDS = {3407, 777}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        default="experiments/prompt-transfer-14b-a1/inputs/accepted16_pairs.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        default="experiments/prompt-transfer-14b-a1/results/tokenizer_preflight.json",
    )
    parser.add_argument(
        "--task",
        default="results/experiment_a_qwen14b_20260805/task_validation.jsonl",
    )
    parser.add_argument(
        "--micro",
        default="results/experiment_a_qwen14b_20260805/micro_scores.jsonl",
    )
    parser.add_argument(
        "--macro",
        default=(
            "results/experiment_a_qwen14b_20260805/macro/"
            "macro_records_test.jsonl"
        ),
    )
    parser.add_argument(
        "--continuity",
        default=(
            "results/experiment_a_qwen14b_20260805/"
            "continuity_validated_prompts.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiment_a_qwen14b_20260805/analysis",
    )
    parser.add_argument("--metric", default="macro_l2_raw")
    parser.add_argument("--family-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--required-retained", type=int, default=12)
    args = parser.parse_args()

    pairs = read_jsonl(resolve(args.pairs))
    if len(pairs) != 16:
        raise ValueError(f"expected 16 pairs, received {len(pairs)}")
    tokenizer = json.loads(resolve(args.tokenizer).read_text(encoding="utf-8"))
    task_rows = read_jsonl(resolve(args.task))
    micro_rows = read_jsonl(resolve(args.micro))
    macro_rows = read_jsonl(resolve(args.macro))

    task_by_id = {str(row["prompt_id"]): row for row in task_rows}
    micro_by_id = {str(row["prompt_id"]): row for row in micro_rows}
    if len(task_by_id) != 32 or len(micro_by_id) != 32:
        raise ValueError(
            f"expected 32 task and micro IDs, received "
            f"{len(task_by_id)} and {len(micro_by_id)}"
        )
    technical_by_source = {
        str(row["prompt_id"]): bool(row["technical_pass"])
        for row in tokenizer["records"]
    }

    macro_values: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in macro_rows:
        prompt_id = str(row["prompt_id"])
        family = str(row["family"])
        if family not in FAMILIES:
            raise ValueError(f"unexpected macro family: {family}")
        if not bool(row.get("valid", True)) or row.get("status", "passed") != "passed":
            raise RuntimeError(
                f"infrastructure-invalid macro record: {prompt_id} {family}"
            )
        value = float(row[args.metric])
        if not finite(value):
            raise RuntimeError(f"non-finite macro value: {prompt_id} {family}")
        macro_values[prompt_id][family].append(row)

    continuity_by_source: dict[str, dict[str, Any]] = {}
    continuity_path = resolve(args.continuity)
    if continuity_path.exists():
        continuity_by_source = {
            str(row["prompt_id"]): row for row in read_jsonl(continuity_path)
        }

    results: list[dict[str, Any]] = []
    family_pass_counts: Counter[str] = Counter()
    for pair in pairs:
        source_id = str(pair["prompt_id"])
        initial_id = f"{source_id}::initial"
        optimized_id = f"{source_id}::optimized"
        if initial_id not in task_by_id or optimized_id not in task_by_id:
            raise ValueError(f"missing task pair: {source_id}")
        if initial_id not in micro_by_id or optimized_id not in micro_by_id:
            raise ValueError(f"missing micro pair: {source_id}")

        initial_micro_row = micro_by_id[initial_id]
        optimized_micro_row = micro_by_id[optimized_id]
        for row in (initial_micro_row, optimized_micro_row):
            if int(row.get("probes", 0)) != 4:
                raise ValueError(f"wrong probe count for {row['prompt_id']}")
            if row.get("complete_parameter_coverage") is not True:
                raise ValueError(f"incomplete micro coverage for {row['prompt_id']}")
        initial_micro = float(initial_micro_row["micro_raw"])
        optimized_micro = float(optimized_micro_row["micro_raw"])
        micro_gain = optimized_micro - initial_micro
        micro_positive = bool(
            finite(initial_micro) and finite(optimized_micro) and micro_gain > 0
        )

        family_details: dict[str, Any] = {}
        initial_family_values = []
        optimized_family_values = []
        family_passes = 0
        for family in FAMILIES:
            initial_records = macro_values[initial_id][family]
            optimized_records = macro_values[optimized_id][family]
            for role, records in (
                ("initial", initial_records),
                ("optimized", optimized_records),
            ):
                seeds = {int(row["seed"]) for row in records}
                if len(records) != 2 or seeds != SEEDS:
                    raise ValueError(
                        f"{source_id} {family} {role}: expected seeds {SEEDS}, "
                        f"received {seeds} across {len(records)} rows"
                    )
            initial = sum(float(row[args.metric]) for row in initial_records) / 2
            optimized = sum(float(row[args.metric]) for row in optimized_records) / 2
            gain = optimized - initial
            relative_gain = gain / max(abs(initial), 1e-12)
            nondegraded = bool(relative_gain >= -args.family_relative_tolerance)
            family_passes += int(nondegraded)
            family_pass_counts[family] += int(nondegraded)
            initial_family_values.append(initial)
            optimized_family_values.append(optimized)
            family_details[family] = {
                "initial": initial,
                "optimized": optimized,
                "gain": gain,
                "relative_gain": relative_gain,
                "nondegraded": nondegraded,
                "seeds": sorted(SEEDS),
            }

        initial_macro = sum(initial_family_values) / len(FAMILIES)
        optimized_macro = sum(optimized_family_values) / len(FAMILIES)
        macro_gain = optimized_macro - initial_macro
        macro_positive = bool(
            finite(initial_macro) and finite(optimized_macro) and macro_gain > 0
        )
        initial_task = bool(task_by_id[initial_id]["task_passed"])
        optimized_task = bool(task_by_id[optimized_id]["task_passed"])
        task_preserved = bool(initial_task and optimized_task)
        technical = technical_by_source.get(source_id) is True
        legacy = bool(
            technical
            and task_preserved
            and micro_positive
            and macro_positive
            and family_passes >= 3
        )
        strict = bool(legacy and family_passes == 5)
        continuity = continuity_by_source.get(source_id)
        hybrid = None
        if continuity is not None:
            hybrid = continuity.get("optimization", {}).get("hybrid_validation")
        results.append(
            {
                "prompt_id": source_id,
                "category": pair["category"],
                "technical_pass": technical,
                "initial_task_passed": initial_task,
                "optimized_task_passed": optimized_task,
                "task_preserved": task_preserved,
                "micro_initial": initial_micro,
                "micro_optimized": optimized_micro,
                "micro_gain": micro_gain,
                "micro_relative_gain": micro_gain / max(abs(initial_micro), 1e-12),
                "micro_positive": micro_positive,
                "macro_initial_equal_family_mean": initial_macro,
                "macro_optimized_equal_family_mean": optimized_macro,
                "macro_gain": macro_gain,
                "macro_relative_gain": macro_gain / max(abs(initial_macro), 1e-12),
                "macro_positive": macro_positive,
                "family_passes": family_passes,
                "families": family_details,
                "legacy_retained": legacy,
                "strict_5of5_retained": strict,
                "v6_fixed_calibration_hybrid": hybrid,
            }
        )

    legacy_count = sum(bool(row["legacy_retained"]) for row in results)
    strict_count = sum(bool(row["strict_5of5_retained"]) for row in results)
    report = {
        "schema_version": "experiment_a_transfer_report_1.0",
        "protocol_commit": "03f4cd7",
        "protocol_clarification_commit": "4870065",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "pairs": len(results),
        "technical_passed": sum(bool(row["technical_pass"]) for row in results),
        "task_preserved": sum(bool(row["task_preserved"]) for row in results),
        "positive_micro_gain": sum(bool(row["micro_positive"]) for row in results),
        "positive_macro_gain": sum(bool(row["macro_positive"]) for row in results),
        "family_pass_counts": dict(family_pass_counts),
        "legacy_retained": legacy_count,
        "strict_5of5_retained": strict_count,
        "required_retained": args.required_retained,
        "h_a1_status": "supported" if legacy_count >= args.required_retained else "refuted",
        "family_relative_tolerance": args.family_relative_tolerance,
        "failed_prompt_ids": [
            row["prompt_id"] for row in results if not row["legacy_retained"]
        ],
        "strict_failed_prompt_ids": [
            row["prompt_id"] for row in results if not row["strict_5of5_retained"]
        ],
        "continuity_hybrid_available": len(continuity_by_source) == 16,
    }
    output_dir = resolve(args.output_dir)
    write_jsonl(output_dir / "per_prompt_transfer_results.jsonl", results)
    atomic_json(output_dir / "FINAL_REPORT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
