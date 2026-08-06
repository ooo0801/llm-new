from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


FAMILIES = ("unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="experiments/prompt-transfer-14b-b1/inputs/survivor_pairs.jsonl")
    parser.add_argument("--cohort", default="experiments/prompt-transfer-14b-b1/inputs/cohort_manifest.json")
    parser.add_argument("--tokenizer", default="experiments/prompt-transfer-14b-a1/results/tokenizer_preflight.json")
    parser.add_argument("--task", default="results/experiment_b_qwen14b_20260806/task_validation.jsonl")
    parser.add_argument("--micro", default="results/experiment_b_qwen14b_20260806/micro_scores.jsonl")
    parser.add_argument("--macro", default="results/experiment_b_qwen14b_20260806/macro/macro_records_validation.jsonl")
    parser.add_argument("--output-dir", default="results/experiment_b_qwen14b_20260806/analysis")
    args = parser.parse_args()

    pairs = read_jsonl(resolve(args.pairs))
    cohort = json.loads(resolve(args.cohort).read_text(encoding="utf-8"))
    tokenizer = json.loads(resolve(args.tokenizer).read_text(encoding="utf-8"))
    tasks = {str(row["prompt_id"]): row for row in read_jsonl(resolve(args.task))}
    micros = {str(row["prompt_id"]): row for row in read_jsonl(resolve(args.micro))}
    macros: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in read_jsonl(resolve(args.macro)):
        if not bool(row.get("valid", True)) or row.get("status", "passed") != "passed":
            raise RuntimeError(f"Invalid macro record: {row.get('prompt_id')} {row.get('family')}")
        macros[str(row["prompt_id"])][str(row["family"])].append(row)

    seeds = set(map(int, cohort["attack_seeds"]))
    strict_source = set(map(str, cohort["strict_prompt_ids"]))
    technical = {str(row["prompt_id"]): bool(row["technical_pass"]) for row in tokenizer["records"]}
    if len(pairs) != 8 or len(tasks) != 16 or len(micros) != 16:
        raise ValueError("Experiment B requires 8 pairs and 16 task/micro endpoints")

    results: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for pair in pairs:
        prompt_id = str(pair["prompt_id"])
        initial_id, optimized_id = f"{prompt_id}::initial", f"{prompt_id}::optimized"
        initial_micro, optimized_micro = micros[initial_id], micros[optimized_id]
        for row in (initial_micro, optimized_micro):
            if row.get("complete_parameter_coverage") is not True or int(row["probes"]) != 4:
                raise ValueError(f"Invalid micro audit for {row['prompt_id']}")
            if int(row["seed"]) != int(cohort["micro_seed"]):
                raise ValueError(f"Wrong micro seed for {row['prompt_id']}")
        micro_i, micro_o = float(initial_micro["micro_raw"]), float(optimized_micro["micro_raw"])
        family_details: dict[str, Any] = {}
        macro_i_values, macro_o_values = [], []
        family_passes = 0
        for family in FAMILIES:
            i_rows, o_rows = macros[initial_id][family], macros[optimized_id][family]
            if len(i_rows) != 2 or len(o_rows) != 2 or {int(row["seed"]) for row in i_rows} != seeds or {int(row["seed"]) for row in o_rows} != seeds:
                raise ValueError(f"Wrong macro replication records for {prompt_id} {family}")
            value_i = sum(float(row["macro_l2_raw"]) for row in i_rows) / 2
            value_o = sum(float(row["macro_l2_raw"]) for row in o_rows) / 2
            relative_gain = (value_o - value_i) / max(abs(value_i), 1e-12)
            nondegraded = relative_gain >= -0.01
            family_passes += int(nondegraded)
            family_counts[family] += int(nondegraded)
            macro_i_values.append(value_i)
            macro_o_values.append(value_o)
            family_details[family] = {"initial": value_i, "optimized": value_o, "relative_gain": relative_gain, "nondegraded": nondegraded, "seeds": sorted(seeds)}
        macro_i = sum(macro_i_values) / 5
        macro_o = sum(macro_o_values) / 5
        task_preserved = bool(tasks[initial_id]["task_passed"] and tasks[optimized_id]["task_passed"])
        micro_positive = bool(math.isfinite(micro_i) and math.isfinite(micro_o) and micro_o > micro_i)
        macro_positive = bool(math.isfinite(macro_i) and math.isfinite(macro_o) and macro_o > macro_i)
        legacy = bool(technical.get(prompt_id) is True and task_preserved and micro_positive and macro_positive and family_passes >= 3)
        strict = bool(legacy and family_passes == 5)
        results.append({
            "prompt_id": prompt_id,
            "category": pair["category"],
            "experiment_a_strict_source": prompt_id in strict_source,
            "technical_pass": technical.get(prompt_id) is True,
            "task_preserved": task_preserved,
            "micro_initial": micro_i,
            "micro_optimized": micro_o,
            "micro_relative_gain": (micro_o - micro_i) / max(abs(micro_i), 1e-12),
            "micro_positive": micro_positive,
            "macro_initial_equal_family_mean": macro_i,
            "macro_optimized_equal_family_mean": macro_o,
            "macro_relative_gain": (macro_o - macro_i) / max(abs(macro_i), 1e-12),
            "macro_positive": macro_positive,
            "family_passes": family_passes,
            "families": family_details,
            "legacy_retained": legacy,
            "strict_5of5_retained": strict,
        })

    hb1_count = sum(row["strict_5of5_retained"] for row in results if row["experiment_a_strict_source"])
    hb2_count = sum(row["legacy_retained"] for row in results)
    report = {
        "schema_version": "experiment_b_replication_report_1.0",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "pairs": 8,
        "attack_seeds": sorted(seeds),
        "micro_seed": int(cohort["micro_seed"]),
        "task_preserved": sum(row["task_preserved"] for row in results),
        "positive_micro_gain": sum(row["micro_positive"] for row in results),
        "positive_macro_gain": sum(row["macro_positive"] for row in results),
        "family_pass_counts": dict(family_counts),
        "legacy_replicated": hb2_count,
        "strict_replicated_all_legacy": sum(row["strict_5of5_retained"] for row in results),
        "strict_source_replicated_strict": hb1_count,
        "h_b1": {"required": 3, "observed": hb1_count, "status": "supported" if hb1_count >= 3 else "refuted"},
        "h_b2": {"required": 6, "observed": hb2_count, "status": "supported" if hb2_count >= 6 else "refuted"},
        "replicated_legacy_prompt_ids": [row["prompt_id"] for row in results if row["legacy_retained"]],
        "replicated_strict_prompt_ids": [row["prompt_id"] for row in results if row["strict_5of5_retained"]],
        "deterministic_family_limitation": ["unstructured_pruning", "quantization"],
    }
    output = resolve(args.output_dir)
    write_jsonl(output / "per_prompt_replication_results.jsonl", results)
    write_json(output / "FINAL_REPORT.json", report)
    lines = ["# Experiment B Final Report", "", f"- H-B1 strict replication: {hb1_count}/4 ({report['h_b1']['status']})", f"- H-B2 legacy replication: {hb2_count}/8 ({report['h_b2']['status']})", f"- Task preserved: {report['task_preserved']}/8", f"- Positive micro: {report['positive_micro_gain']}/8", f"- Positive macro: {report['positive_macro_gain']}/8", "", "Global-magnitude pruning and NF4 are deterministic fresh-materialization repeats; the other three attack families use independent seeds.", "", "## Prompt results", "", "| Prompt | A strict | Task | Micro | Macro | Families | Legacy | Strict |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['prompt_id']} | {int(row['experiment_a_strict_source'])} | {int(row['task_preserved'])} | {int(row['micro_positive'])} | {int(row['macro_positive'])} | {row['family_passes']}/5 | {int(row['legacy_retained'])} | {int(row['strict_5of5_retained'])} |")
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
