from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


FAMILIES = ["unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning"]
REQUIRED_CATEGORIES = ["code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"]


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
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="experiments/prompt-union-14b-e1/inputs/final30_pairs.jsonl")
    parser.add_argument("--manifest", default="experiments/prompt-union-14b-e1/inputs/attack_manifest_final_2each.jsonl")
    parser.add_argument("--task", required=True)
    parser.add_argument("--micro", required=True)
    parser.add_argument("--macro", required=True)
    parser.add_argument("--output-dir", default="results/experiment_e_qwen14b_final_20260807/analysis")
    parser.add_argument("--required-legacy", type=int, default=23)
    parser.add_argument("--family-relative-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    pairs = read_jsonl(resolve(args.pairs))
    manifest = read_jsonl(resolve(args.manifest))
    task_rows = read_jsonl(resolve(args.task))
    micro_rows = read_jsonl(resolve(args.micro))
    macro_rows = read_jsonl(resolve(args.macro))
    output = resolve(args.output_dir)
    if len(pairs) != 30 or len(task_rows) != 60 or len(micro_rows) != 60 or len(macro_rows) != 600:
        raise ValueError(f"endpoint count mismatch pairs={len(pairs)} task={len(task_rows)} micro={len(micro_rows)} macro={len(macro_rows)}")

    expected_seeds = {str(row["family"]): set() for row in manifest}
    for row in manifest:
        expected_seeds[str(row["family"])].add(int(row["seed"]))
    if set(expected_seeds) != set(FAMILIES) or any(len(seeds) != 2 for seeds in expected_seeds.values()):
        raise ValueError("manifest must freeze two seeds for every family")

    task_by_id = {str(row["prompt_id"]): row for row in task_rows}
    micro_by_id = {str(row["prompt_id"]): row for row in micro_rows}
    if len(task_by_id) != 60 or len(micro_by_id) != 60:
        raise ValueError("duplicate task or micro ids")
    macro_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in macro_rows:
        macro_grouped[(str(row["prompt_id"]), str(row["family"]))].append(row)

    technical_errors: list[str] = []
    decisions: list[dict[str, Any]] = []
    for pair in pairs:
        prompt_id = str(pair["prompt_id"])
        initial_id, optimized_id = f"{prompt_id}::initial", f"{prompt_id}::optimized"
        if initial_id not in task_by_id or optimized_id not in task_by_id or initial_id not in micro_by_id or optimized_id not in micro_by_id:
            technical_errors.append(f"{prompt_id}: missing task or micro endpoint")
            continue
        initial_micro_row, optimized_micro_row = micro_by_id[initial_id], micro_by_id[optimized_id]
        if initial_micro_row.get("complete_parameter_coverage") is not True or optimized_micro_row.get("complete_parameter_coverage") is not True:
            technical_errors.append(f"{prompt_id}: incomplete micro coverage")
        initial_micro = float(initial_micro_row["micro_per_parameter"])
        optimized_micro = float(optimized_micro_row["micro_per_parameter"])
        if not all(math.isfinite(value) for value in (initial_micro, optimized_micro)):
            technical_errors.append(f"{prompt_id}: nonfinite micro")

        family_details: dict[str, dict[str, Any]] = {}
        initial_family_values: list[float] = []
        optimized_family_values: list[float] = []
        for family in FAMILIES:
            initial_records = macro_grouped.get((initial_id, family), [])
            optimized_records = macro_grouped.get((optimized_id, family), [])
            if len(initial_records) != 2 or len(optimized_records) != 2:
                technical_errors.append(f"{prompt_id}/{family}: expected two records per role")
                continue
            if {int(row["seed"]) for row in initial_records} != expected_seeds[family] or {int(row["seed"]) for row in optimized_records} != expected_seeds[family]:
                technical_errors.append(f"{prompt_id}/{family}: seed mismatch")
            initial_value = mean([float(row["macro_l2_raw"]) for row in initial_records])
            optimized_value = mean([float(row["macro_l2_raw"]) for row in optimized_records])
            if not math.isfinite(initial_value) or not math.isfinite(optimized_value):
                technical_errors.append(f"{prompt_id}/{family}: nonfinite macro")
            tolerance = args.family_relative_tolerance * max(abs(initial_value), 1e-12)
            gain = optimized_value - initial_value
            family_details[family] = {
                "initial": initial_value, "optimized": optimized_value, "gain": gain,
                "relative_gain": gain / max(abs(initial_value), 1e-12),
                "nondegraded": bool(gain >= -tolerance), "relative_tolerance": args.family_relative_tolerance,
            }
            initial_family_values.append(initial_value)
            optimized_family_values.append(optimized_value)

        if len(family_details) != 5:
            continue
        initial_macro, optimized_macro = mean(initial_family_values), mean(optimized_family_values)
        micro_gain = optimized_micro - initial_micro
        macro_gain = optimized_macro - initial_macro
        task_preserved = bool(task_by_id[initial_id]["task_passed"] and task_by_id[optimized_id]["task_passed"])
        micro_positive = bool(micro_gain > 0)
        macro_positive = bool(macro_gain > 0)
        nondegraded = sum(bool(item["nondegraded"]) for item in family_details.values())
        legacy = bool(task_preserved and micro_positive and macro_positive and nondegraded >= 3)
        strict = bool(task_preserved and micro_positive and macro_positive and nondegraded == 5)
        decisions.append({
            "prompt_id": prompt_id, "category": pair["category"], "component": pair["component"],
            "initial_prompt": pair["initial_prompt"], "optimized_prompt": pair["optimized_prompt"],
            "evaluator": pair.get("evaluator"), "expected_answer": pair.get("expected_answer"),
            "expected_contains": pair.get("expected_contains"), "evidence": pair.get("evidence"),
            "task": {"initial_passed": bool(task_by_id[initial_id]["task_passed"]), "optimized_passed": bool(task_by_id[optimized_id]["task_passed"]), "preserved": task_preserved},
            "micro": {"initial_per_parameter": initial_micro, "optimized_per_parameter": optimized_micro, "gain": micro_gain, "relative_gain": micro_gain / max(abs(initial_micro), 1e-12), "positive": micro_positive},
            "macro": {"initial_equal_weight": initial_macro, "optimized_equal_weight": optimized_macro, "gain": macro_gain, "relative_gain": macro_gain / max(abs(initial_macro), 1e-12), "positive": macro_positive},
            "families": family_details, "nondegraded_families": nondegraded,
            "legacy_retained": legacy, "strict_5of5_retained": strict,
            "optimized_prompt_sha256": pair["optimized_prompt_sha256"],
        })

    technical_passed = bool(not technical_errors and len(decisions) == 30)
    legacy_rows = [row for row in decisions if row["legacy_retained"]]
    strict_rows = [row for row in decisions if row["strict_5of5_retained"]]
    legacy_categories = sorted({str(row["category"]) for row in legacy_rows})
    category_coverage_passed = legacy_categories == REQUIRED_CATEGORIES
    scientific_gate_passed = bool(technical_passed and len(legacy_rows) >= args.required_legacy and category_coverage_passed)
    family_pass_counts = {family: sum(bool(row["families"][family]["nondegraded"]) for row in decisions) for family in FAMILIES}
    report = {
        "schema_version": "experiment_e_joint_confirmation_1.0", "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8", "candidate_rows": len(pairs),
        "endpoint_counts": {"task": len(task_rows), "micro": len(micro_rows), "macro": len(macro_rows)},
        "task_preserved": sum(bool(row["task"]["preserved"]) for row in decisions),
        "micro_positive": sum(bool(row["micro"]["positive"]) for row in decisions),
        "macro_positive": sum(bool(row["macro"]["positive"]) for row in decisions),
        "legacy_retained": len(legacy_rows), "required_legacy_retained": args.required_legacy,
        "strict_5of5_retained": len(strict_rows), "strict_threshold": "secondary_descriptive_no_threshold",
        "legacy_by_component": dict(Counter(str(row["component"]) for row in legacy_rows)),
        "strict_by_component": dict(Counter(str(row["component"]) for row in strict_rows)),
        "legacy_by_category": dict(Counter(str(row["category"]) for row in legacy_rows)),
        "required_category_coverage": REQUIRED_CATEGORIES, "observed_legacy_categories": legacy_categories,
        "category_coverage_passed": category_coverage_passed, "family_nondegraded_prompt_counts": family_pass_counts,
        "family_relative_tolerance": args.family_relative_tolerance, "technical_errors": technical_errors,
        "technical_passed": technical_passed, "scientific_gate_passed": scientific_gate_passed,
        "h_e1": {"required_legacy": args.required_legacy, "observed_legacy": len(legacy_rows), "requires_all_categories": True, "status": "supported" if scientific_gate_passed else "refuted" if technical_passed else "technical_no_go"},
        "claim_scope": "single preregistered joint confirmation of the frozen 30-prompt union under new micro and attack seeds",
    }
    write_jsonl(output / "per_prompt_joint_results.jsonl", decisions)
    write_jsonl(output / "final_14b_fingerprint.jsonl", legacy_rows)
    write_jsonl(output / "strict_5of5_subset.jsonl", strict_rows)
    write_json(output / "FINAL_REPORT.json", report)
    lines = [
        "# Experiment E: Final Qwen2.5-14B Fingerprint", "",
        f"- Technical endpoints: task {len(task_rows)}/60, micro {len(micro_rows)}/60, macro {len(macro_rows)}/600",
        f"- Task preserved: {report['task_preserved']}/30", f"- Micro positive: {report['micro_positive']}/30",
        f"- Macro positive: {report['macro_positive']}/30", f"- Legacy retained: {len(legacy_rows)}/30 (required {args.required_legacy})",
        f"- Strict 5/5 retained: {len(strict_rows)}/30", f"- All eight categories represented: {category_coverage_passed}",
        f"- H-E1: {report['h_e1']['status']}", "", "No candidate, seed, threshold, evaluator, or attack configuration was changed after endpoints were observed.", "",
        "## Final retained prompt IDs", "",
    ]
    lines.extend(f"- `{row['prompt_id']}` ({row['category']}; {row['component']})" for row in legacy_rows)
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not technical_passed:
        raise SystemExit(2)
    if not scientific_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
