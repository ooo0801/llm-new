from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from _bootstrap import ROOT


FAMILIES = ["unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning"]
CATEGORIES = ["code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def expected_variants(manifest: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        result[str(row["family"])].add(str(row["variant_id"]))
    if set(result) != set(FAMILIES):
        raise ValueError(f"manifest families differ from protocol: {sorted(result)}")
    return dict(result)


def evaluate(
    pairs: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]],
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    variants = expected_variants(manifest)
    task_by_id = {str(row["prompt_id"]): row for row in task_rows}
    micro_by_id = {str(row["prompt_id"]): row for row in micro_rows}
    macro_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in macro_rows:
        macro_by_key[(str(row["prompt_id"]), str(row["family"]), str(row["variant_id"]))].append(row)

    errors: list[str] = []
    if len(task_by_id) != len(task_rows):
        errors.append("duplicate task prompt IDs")
    if len(micro_by_id) != len(micro_rows):
        errors.append("duplicate micro prompt IDs")

    decisions: list[dict[str, Any]] = []
    for pair in pairs:
        prompt_id = str(pair["prompt_id"])
        initial_id = f"{prompt_id}::initial"
        optimized_id = f"{prompt_id}::optimized"
        if initial_id not in task_by_id or optimized_id not in task_by_id:
            errors.append(f"{prompt_id}: missing task endpoint")
            continue
        if initial_id not in micro_by_id or optimized_id not in micro_by_id:
            errors.append(f"{prompt_id}: missing micro endpoint")
            continue
        initial_micro_row = micro_by_id[initial_id]
        optimized_micro_row = micro_by_id[optimized_id]
        if initial_micro_row.get("complete_parameter_coverage") is not True or optimized_micro_row.get("complete_parameter_coverage") is not True:
            errors.append(f"{prompt_id}: incomplete micro coverage")
        initial_micro = float(initial_micro_row["micro_per_parameter"])
        optimized_micro = float(optimized_micro_row["micro_per_parameter"])
        if not math.isfinite(initial_micro) or not math.isfinite(optimized_micro):
            errors.append(f"{prompt_id}: nonfinite micro score")

        family_details: dict[str, dict[str, Any]] = {}
        initial_family_means: list[float] = []
        optimized_family_means: list[float] = []
        for family in FAMILIES:
            variant_details: list[dict[str, Any]] = []
            for variant_id in sorted(variants[family]):
                initial_records = macro_by_key.get((initial_id, family, variant_id), [])
                optimized_records = macro_by_key.get((optimized_id, family, variant_id), [])
                if len(initial_records) != 1 or len(optimized_records) != 1:
                    errors.append(f"{prompt_id}/{family}/{variant_id}: expected one record per role")
                    continue
                initial_value = float(initial_records[0]["macro_l2_raw"])
                optimized_value = float(optimized_records[0]["macro_l2_raw"])
                if not math.isfinite(initial_value) or not math.isfinite(optimized_value):
                    errors.append(f"{prompt_id}/{family}/{variant_id}: nonfinite macro score")
                gain = optimized_value - initial_value
                relative_gain = gain / max(abs(initial_value), 1e-12)
                variant_details.append({
                    "variant_id": variant_id,
                    "seed": int(initial_records[0]["seed"]),
                    "initial": initial_value,
                    "optimized": optimized_value,
                    "gain": gain,
                    "relative_gain": relative_gain,
                })
            if len(variant_details) != len(variants[family]):
                continue
            initial_value = mean([row["initial"] for row in variant_details])
            optimized_value = mean([row["optimized"] for row in variant_details])
            relative_values = [row["relative_gain"] for row in variant_details]
            robust_relative_gain = float(median(relative_values))
            family_details[family] = {
                "initial_mean": initial_value,
                "optimized_mean": optimized_value,
                "gain": optimized_value - initial_value,
                "relative_gain": (optimized_value - initial_value) / max(abs(initial_value), 1e-12),
                "robust_median_relative_gain": robust_relative_gain,
                "worst_variant_relative_gain": min(relative_values),
                "nondegraded": bool(robust_relative_gain >= -tolerance),
                "variants": variant_details,
            }
            initial_family_means.append(initial_value)
            optimized_family_means.append(optimized_value)

        if len(family_details) != len(FAMILIES):
            continue
        initial_macro = mean(initial_family_means)
        optimized_macro = mean(optimized_family_means)
        task_preserved = bool(task_by_id[initial_id]["task_passed"] and task_by_id[optimized_id]["task_passed"])
        micro_gain = optimized_micro - initial_micro
        macro_gain = optimized_macro - initial_macro
        micro_relative_gain = micro_gain / max(abs(initial_micro), 1e-12)
        macro_relative_gain = macro_gain / max(abs(initial_macro), 1e-12)
        nondegraded = sum(bool(row["nondegraded"]) for row in family_details.values())
        micro_positive = bool(micro_gain > 0)
        macro_positive = bool(macro_gain > 0)
        development_eligible = bool(task_preserved and micro_positive and macro_positive)
        legacy_retained = bool(development_eligible and nondegraded >= 3)
        strict_retained = bool(development_eligible and nondegraded == 5)
        decisions.append({
            "prompt_id": prompt_id,
            "source_prompt_id": str(pair.get("source_prompt_id", prompt_id)),
            "category": str(pair["category"]),
            "component": str(pair.get("component", "experiment_f1")),
            "initial_prompt": str(pair["initial_prompt"]),
            "optimized_prompt": str(pair["optimized_prompt"]),
            "optimized_prompt_sha256": str(pair["optimized_prompt_sha256"]),
            "task": {"initial_passed": bool(task_by_id[initial_id]["task_passed"]), "optimized_passed": bool(task_by_id[optimized_id]["task_passed"]), "preserved": task_preserved},
            "micro": {"initial": initial_micro, "optimized": optimized_micro, "gain": micro_gain, "relative_gain": micro_relative_gain, "positive": micro_positive},
            "macro": {"initial_equal_family": initial_macro, "optimized_equal_family": optimized_macro, "gain": macro_gain, "relative_gain": macro_relative_gain, "positive": macro_positive},
            "families": family_details,
            "nondegraded_families": nondegraded,
            "structured_nondegraded": bool(family_details["structured_pruning"]["nondegraded"]),
            "worst_family_median_relative_gain": min(float(row["robust_median_relative_gain"]) for row in family_details.values()),
            "development_eligible": development_eligible,
            "legacy_retained": legacy_retained,
            "strict_5of5_retained": strict_retained,
        })
    return decisions, errors


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(bool(row["structured_nondegraded"])),
        -int(row["nondegraded_families"]),
        -float(row["worst_family_median_relative_gain"]),
        -float(row["families"]["structured_pruning"]["robust_median_relative_gain"]),
        -float(row["macro"]["relative_gain"]),
        -float(row["micro"]["relative_gain"]),
        str(row["prompt_id"]),
    )


def select_development(
    pairs: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    selected_size: int = 30,
    quota: int = 3,
) -> tuple[list[dict[str, Any]], list[str]]:
    pair_by_id = {str(row["prompt_id"]): row for row in pairs}
    eligible = sorted((row for row in decisions if row["development_eligible"]), key=ranking_key)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_sources: set[str] = set()
    selected_hashes: set[str] = set()

    def take(row: dict[str, Any]) -> bool:
        prompt_id = str(row["prompt_id"])
        source_id = str(row["source_prompt_id"])
        prompt_hash = str(row["optimized_prompt_sha256"])
        if prompt_id in selected_ids or source_id in selected_sources or prompt_hash in selected_hashes:
            return False
        selected.append(row)
        selected_ids.add(prompt_id)
        selected_sources.add(source_id)
        selected_hashes.add(prompt_hash)
        return True

    failures: list[str] = []
    for category in CATEGORIES:
        before = len(selected)
        for row in eligible:
            if str(row["category"]) == category and take(row) and len(selected) - before == quota:
                break
        if len(selected) - before < quota:
            failures.append(f"{category}: fewer than {quota} eligible unique-source candidates")
    for row in eligible:
        if len(selected) >= selected_size:
            break
        take(row)
    if len(selected) != selected_size:
        failures.append(f"selected {len(selected)} rows instead of {selected_size}")

    frozen: list[dict[str, Any]] = []
    for rank, decision in enumerate(selected, start=1):
        pair = dict(pair_by_id[str(decision["prompt_id"])])
        pair["development_rank"] = rank
        pair["development_selection"] = {
            "structured_nondegraded": decision["structured_nondegraded"],
            "nondegraded_families": decision["nondegraded_families"],
            "worst_family_median_relative_gain": decision["worst_family_median_relative_gain"],
            "structured_median_relative_gain": decision["families"]["structured_pruning"]["robust_median_relative_gain"],
            "macro_relative_gain": decision["macro"]["relative_gain"],
            "micro_relative_gain": decision["micro"]["relative_gain"],
        }
        frozen.append(pair)
    return frozen, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["development", "confirmation"], required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--micro", required=True)
    parser.add_argument("--macro", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family-relative-tolerance", type=float, default=0.01)
    parser.add_argument("--required-legacy", type=int, default=23)
    args = parser.parse_args()

    pairs = read_jsonl(resolve(args.pairs))
    manifest = read_jsonl(resolve(args.manifest))
    task_rows = read_jsonl(resolve(args.task))
    micro_rows = read_jsonl(resolve(args.micro))
    macro_rows = read_jsonl(resolve(args.macro))
    output = resolve(args.output_dir)
    variants = expected_variants(manifest)
    expected_task = len(pairs) * 2
    expected_micro = len(pairs) * 2
    expected_macro = len(pairs) * 2 * sum(len(value) for value in variants.values())
    count_errors = []
    if len(task_rows) != expected_task:
        count_errors.append(f"task count {len(task_rows)} != {expected_task}")
    if len(micro_rows) != expected_micro:
        count_errors.append(f"micro count {len(micro_rows)} != {expected_micro}")
    if len(macro_rows) != expected_macro:
        count_errors.append(f"macro count {len(macro_rows)} != {expected_macro}")
    decisions, errors = evaluate(pairs, manifest, task_rows, micro_rows, macro_rows, args.family_relative_tolerance)
    errors = count_errors + errors
    technical_passed = bool(not errors and len(decisions) == len(pairs))
    write_jsonl(output / "per_prompt_decisions.jsonl", decisions)

    if args.stage == "development":
        frozen, selection_errors = select_development(pairs, decisions)
        selected_categories = Counter(str(row["category"]) for row in frozen)
        gate_passed = bool(technical_passed and not selection_errors and len(frozen) == 30)
        write_jsonl(output / "frozen30_pairs.jsonl", frozen)
        report = {
            "schema_version": "experiment_f1_development_1.0",
            "candidate_rows": len(pairs),
            "endpoint_counts": {"task": len(task_rows), "micro": len(micro_rows), "macro": len(macro_rows)},
            "technical_passed": technical_passed,
            "technical_errors": errors,
            "task_preserved": sum(bool(row["task"]["preserved"]) for row in decisions),
            "micro_positive": sum(bool(row["micro"]["positive"]) for row in decisions),
            "macro_positive": sum(bool(row["macro"]["positive"]) for row in decisions),
            "development_eligible": sum(bool(row["development_eligible"]) for row in decisions),
            "legacy_retained_descriptive": sum(bool(row["legacy_retained"]) for row in decisions),
            "structured_nondegraded": sum(bool(row["structured_nondegraded"]) for row in decisions),
            "selected_rows": len(frozen),
            "selected_by_category": dict(sorted(selected_categories.items())),
            "selection_errors": selection_errors,
            "gate_passed": gate_passed,
        }
    else:
        if len(pairs) != 30:
            errors.append(f"confirmation requires 30 frozen rows, received {len(pairs)}")
            technical_passed = False
        legacy = [row for row in decisions if row["legacy_retained"]]
        strict = [row for row in decisions if row["strict_5of5_retained"]]
        legacy_categories = sorted({str(row["category"]) for row in legacy})
        coverage_passed = legacy_categories == CATEGORIES
        gate_passed = bool(technical_passed and len(legacy) >= args.required_legacy and coverage_passed)
        pair_by_id = {str(row["prompt_id"]): row for row in pairs}
        write_jsonl(output / "confirmed_fingerprint.jsonl", [dict(pair_by_id[row["prompt_id"]], confirmation=row) for row in legacy])
        write_jsonl(output / "strict_5of5_subset.jsonl", [dict(pair_by_id[row["prompt_id"]], confirmation=row) for row in strict])
        report = {
            "schema_version": "experiment_f1_confirmation_1.0",
            "candidate_rows": len(pairs),
            "endpoint_counts": {"task": len(task_rows), "micro": len(micro_rows), "macro": len(macro_rows)},
            "technical_passed": technical_passed,
            "technical_errors": errors,
            "task_preserved": sum(bool(row["task"]["preserved"]) for row in decisions),
            "micro_positive": sum(bool(row["micro"]["positive"]) for row in decisions),
            "macro_positive": sum(bool(row["macro"]["positive"]) for row in decisions),
            "legacy_retained": len(legacy),
            "required_legacy_retained": args.required_legacy,
            "strict_5of5_retained": len(strict),
            "legacy_by_category": dict(sorted(Counter(str(row["category"]) for row in legacy).items())),
            "observed_legacy_categories": legacy_categories,
            "required_categories": CATEGORIES,
            "category_coverage_passed": coverage_passed,
            "family_nondegeneration_counts": {family: sum(bool(row["families"][family]["nondegraded"]) for row in decisions) for family in FAMILIES},
            "hypothesis": "H-F1",
            "hypothesis_supported": gate_passed,
            "gate_passed": gate_passed,
        }
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["gate_passed"] else 1)


if __name__ == "__main__":
    main()
