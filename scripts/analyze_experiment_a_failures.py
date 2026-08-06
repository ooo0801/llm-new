from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


GATES = ("task", "micro", "macro", "legacy_family", "strict_family")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc error attribution for frozen Experiment A outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.input)
    rows = read_jsonl(source)
    if len(rows) != 16 or len({row["prompt_id"] for row in rows}) != 16:
        raise ValueError("Experiment A attribution requires exactly 16 unique prompts")

    matrix: list[dict] = []
    family_pass_counts: Counter[str] = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    independent_fail_counts: Counter[str] = Counter()

    for row in rows:
        family_failures = [name for name, detail in row["families"].items() if not detail["nondegraded"]]
        family_relative_gains = {name: float(detail["relative_gain"]) for name, detail in row["families"].items()}
        for name in row["families"]:
            family_pass_counts[name] += int(row["families"][name]["nondegraded"])

        gate_pass = {
            "task": bool(row["task_preserved"]),
            "micro": bool(row["micro_positive"]),
            "macro": bool(row["macro_positive"]),
            "legacy_family": int(row["family_passes"]) >= 3,
            "strict_family": int(row["family_passes"]) == 5,
        }
        failed_legacy_gates = [name for name in GATES[:4] if not gate_pass[name]]
        failed_strict_gates = [name for name in ("task", "micro", "macro", "strict_family") if not gate_pass[name]]
        for name in failed_legacy_gates:
            independent_fail_counts[name] += 1

        if not row["technical_pass"]:
            primary = "technical"
        elif not gate_pass["task"]:
            primary = "task"
        elif not gate_pass["micro"]:
            primary = "micro"
        elif not gate_pass["macro"]:
            primary = "macro"
        elif not gate_pass["legacy_family"]:
            primary = "legacy_family"
        elif not gate_pass["strict_family"]:
            primary = "strict_family_only"
        else:
            primary = "strict_retained"

        category = str(row["category"])
        category_counts[category]["total"] += 1
        category_counts[category]["task"] += int(gate_pass["task"])
        category_counts[category]["micro"] += int(gate_pass["micro"])
        category_counts[category]["macro"] += int(gate_pass["macro"])
        category_counts[category]["legacy"] += int(row["legacy_retained"])
        category_counts[category]["strict"] += int(row["strict_5of5_retained"])

        matrix.append(
            {
                "prompt_id": row["prompt_id"],
                "category": category,
                "technical_pass": bool(row["technical_pass"]),
                "initial_task_passed": bool(row["initial_task_passed"]),
                "optimized_task_passed": bool(row["optimized_task_passed"]),
                "task_preserved": gate_pass["task"],
                "micro_positive": gate_pass["micro"],
                "micro_relative_gain": float(row["micro_relative_gain"]),
                "macro_positive": gate_pass["macro"],
                "macro_relative_gain": float(row["macro_relative_gain"]),
                "family_passes": int(row["family_passes"]),
                "family_failures": family_failures,
                "family_relative_gains": family_relative_gains,
                "legacy_retained": bool(row["legacy_retained"]),
                "strict_retained": bool(row["strict_5of5_retained"]),
                "failed_legacy_gates": failed_legacy_gates,
                "failed_strict_gates": failed_strict_gates,
                "primary_sequential_outcome": primary,
            }
        )

    waterfall = {
        "technical": sum(row["technical_pass"] for row in matrix),
        "after_task": sum(row["technical_pass"] and row["task_preserved"] for row in matrix),
        "after_micro": sum(row["technical_pass"] and row["task_preserved"] and row["micro_positive"] for row in matrix),
        "after_macro": sum(row["technical_pass"] and row["task_preserved"] and row["micro_positive"] and row["macro_positive"] for row in matrix),
        "after_legacy_family": sum(row["legacy_retained"] for row in matrix),
        "after_strict_family": sum(row["strict_retained"] for row in matrix),
    }
    sequential_outcomes = Counter(row["primary_sequential_outcome"] for row in matrix)
    single_gate_bottlenecks = {}
    for gate in GATES[:4]:
        other = [name for name in GATES[:4] if name != gate]
        single_gate_bottlenecks[gate] = [
            row["prompt_id"]
            for row in matrix
            if not ({
                "task": row["task_preserved"],
                "micro": row["micro_positive"],
                "macro": row["macro_positive"],
                "legacy_family": row["family_passes"] >= 3,
            }[gate])
            and all({
                "task": row["task_preserved"],
                "micro": row["micro_positive"],
                "macro": row["macro_positive"],
                "legacy_family": row["family_passes"] >= 3,
            }[name] for name in other)
        ]

    report = {
        "schema_version": "experiment_a_failure_attribution_1.0",
        "analysis_type": "post_hoc_descriptive_existing_outputs_only",
        "source": str(source),
        "prompt_count": 16,
        "waterfall": waterfall,
        "independent_legacy_gate_fail_counts": dict(independent_fail_counts),
        "sequential_primary_outcomes": dict(sequential_outcomes),
        "family_pass_counts": dict(family_pass_counts),
        "single_gate_bottleneck_prompt_ids": single_gate_bottlenecks,
        "category_counts": {key: dict(value) for key, value in sorted(category_counts.items())},
        "legacy_retained_prompt_ids": [row["prompt_id"] for row in matrix if row["legacy_retained"]],
        "strict_retained_prompt_ids": [row["prompt_id"] for row in matrix if row["strict_retained"]],
        "interpretation": {
            "confirmatory_status_unchanged": True,
            "statement": "This attribution explains the frozen Experiment A result and does not alter any gate or rescore any model endpoint.",
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "failure_attribution.json", report)
    with (output_dir / "failure_matrix.jsonl").open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "failure_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "prompt_id", "category", "task_preserved", "micro_positive", "micro_relative_gain",
            "macro_positive", "macro_relative_gain", "family_passes", "family_failures",
            "legacy_retained", "strict_retained", "failed_legacy_gates", "primary_sequential_outcome",
        ])
        writer.writeheader()
        for row in matrix:
            flat = {key: row[key] for key in writer.fieldnames}
            flat["family_failures"] = ";".join(row["family_failures"])
            flat["failed_legacy_gates"] = ";".join(row["failed_legacy_gates"])
            writer.writerow(flat)

    lines = [
        "# Experiment A Failure Attribution",
        "",
        "This is a post-hoc descriptive analysis of frozen Experiment A outputs. No model endpoint was rerun and no gate was changed.",
        "",
        "## Gate waterfall",
        "",
        "| Stage | Remaining |",
        "|---|---:|",
    ]
    for name, value in waterfall.items():
        lines.append(f"| {name} | {value}/16 |")
    lines += ["", "## Independent failures", ""]
    for name in GATES[:4]:
        lines.append(f"- {name}: {independent_fail_counts[name]}/16")
    lines += ["", "## Five-family pass counts", ""]
    for name, value in family_pass_counts.items():
        lines.append(f"- {name}: {value}/16")
    lines += ["", "## Prompt-level matrix", "", "| Prompt | Category | Task | Micro | Macro | Families | Legacy | Strict | Failed families |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in matrix:
        lines.append(
            f"| {row['prompt_id']} | {row['category']} | {int(row['task_preserved'])} | "
            f"{int(row['micro_positive'])} | {int(row['macro_positive'])} | {row['family_passes']}/5 | "
            f"{int(row['legacy_retained'])} | {int(row['strict_retained'])} | {', '.join(row['family_failures']) or '-'} |"
        )
    (output_dir / "FAILURE_ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
