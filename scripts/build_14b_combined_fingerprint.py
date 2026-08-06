from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-results", default="results/experiment_b_qwen14b_20260806/analysis/per_prompt_replication_results.jsonl")
    parser.add_argument("--b-pairs", default="experiments/prompt-transfer-14b-b1/inputs/survivor_pairs.jsonl")
    parser.add_argument("--c-results", default="results/experiment_c_qwen14b_20260806/analysis/accepted14b_complement.jsonl")
    parser.add_argument("--output-dir", default="results/experiment_d_combined14b_20260806")
    args = parser.parse_args()

    b_results = read_jsonl(resolve(args.b_results))
    b_pairs = {str(row["prompt_id"]): row for row in read_jsonl(resolve(args.b_pairs))}
    c_rows = read_jsonl(resolve(args.c_results))
    output = resolve(args.output_dir)

    combined: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    used_hashes: set[str] = set()
    for result in b_results:
        if not result["strict_5of5_retained"]:
            continue
        prompt_id = str(result["prompt_id"])
        pair = b_pairs[prompt_id]
        prompt = str(pair["optimized_prompt"])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        combined.append({
            "prompt_id": prompt_id,
            "category": pair["category"],
            "prompt": prompt,
            "evaluator": pair["evaluator"],
            "expected_answer": pair.get("expected_answer"),
            "expected_contains": pair.get("expected_contains"),
            "component": "cross_model_strict_core",
            "evidence": ["experiment_a_strict_5of5", "experiment_b_independent_seed_strict_5of5"],
            "prompt_sha256": digest,
        })
        used_sources.add(prompt_id)
        used_hashes.add(digest)

    excluded_overlap: list[str] = []
    for row in c_rows:
        prompt_id = str(row["prompt_id"])
        optimization = row["optimization"]
        prompt = str(optimization["optimized_prompt"])
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt_id in used_sources or digest in used_hashes:
            excluded_overlap.append(prompt_id)
            continue
        combined.append({
            "prompt_id": prompt_id,
            "category": row.get("category", "unknown"),
            "prompt": prompt,
            "evaluator": row.get("evaluator"),
            "expected_answer": row.get("expected_answer"),
            "expected_contains": row.get("expected_contains"),
            "component": "qwen14b_model_specific_complement",
            "evidence": ["experiment_c_development_selected", "experiment_c_heldout_test_accepted"],
            "prompt_sha256": digest,
        })
        used_sources.add(prompt_id)
        used_hashes.add(digest)

    report = {
        "schema_version": "experiment_d_combined_14b_fingerprint_1.0",
        "cross_model_strict_core": sum(row["component"] == "cross_model_strict_core" for row in combined),
        "qwen14b_specific_complement": sum(row["component"] == "qwen14b_model_specific_complement" for row in combined),
        "combined_total": len(combined),
        "excluded_source_or_text_overlaps": excluded_overlap,
        "claim_scope": "descriptive union of independently validated components; no new joint-union endpoint was run",
    }
    write_jsonl(output / "combined_fingerprint.jsonl", combined)
    output.mkdir(parents=True, exist_ok=True)
    (output / "FINAL_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Combined Qwen2.5-14B Fingerprint", "", f"- Cross-model strict core: {report['cross_model_strict_core']}", f"- 14B-specific held-out complement: {report['qwen14b_specific_complement']}", f"- Combined total: {report['combined_total']}", "", "This is a provenance-preserving union, not a new claim that the union was jointly re-evaluated under one additional attack set."]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
