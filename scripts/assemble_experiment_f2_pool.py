from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in rows
    )
    path.write_bytes(payload.encode("utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    path.write_bytes(serialized.encode("utf-8"))


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_new(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    prompt_id = str(row["prompt_id"])
    provenance = row.get("candidate_provenance") or {}
    source_id = str(provenance.get("source_prompt_id", ""))
    initial = str(row["initial_prompt"]).strip()
    optimized = str(row["optimized_prompt"]).strip()
    category = str(row.get("category", ""))
    if category not in {"logic", "summary"}:
        raise ValueError(f"{prompt_id}: F2 expansion candidate category must be logic or summary")
    if not source_id or not initial or not optimized or initial == optimized:
        raise ValueError(f"{prompt_id}: invalid source identity or prompt pair")
    result = dict(row)
    result.update(
        {
            "source_index": source_index,
            "id": prompt_id,
            "prompt_id": prompt_id,
            "source_prompt_id": source_id,
            "component": "f2_targeted_expansion",
            "evidence": ["f2_construction_training_only_portfolio"],
            "accepted": True,
            "initial_prompt_sha256": text_sha256(initial),
            "optimized_prompt_sha256": text_sha256(optimized),
            "source_record_sha256": canonical_sha256(row),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the frozen F2 development candidate pool")
    parser.add_argument("--base", default="experiments/prompt-robust-14b-f1/inputs/candidate_pairs64.jsonl")
    parser.add_argument("--new-portfolio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-new-sources-per-category", type=int, default=3)
    args = parser.parse_args()

    base = read_jsonl(resolve(args.base))
    new_raw = read_jsonl(resolve(args.new_portfolio))
    if len(base) != 64:
        raise ValueError(f"expected 64 frozen F1 base candidates, received {len(base)}")
    new_rows = [normalize_new(row, len(base) + index) for index, row in enumerate(new_raw)]
    unique_new_sources = Counter()
    for category in ("logic", "summary"):
        unique_new_sources[category] = len({row["source_prompt_id"] for row in new_rows if row["category"] == category})

    failures = [
        f"{category}: {unique_new_sources[category]} new candidate sources < {args.minimum_new_sources_per_category}"
        for category in ("logic", "summary")
        if unique_new_sources[category] < args.minimum_new_sources_per_category
    ]
    merged = [dict(row) for row in base] + new_rows
    ids = [str(row["prompt_id"]) for row in merged]
    hashes = [str(row.get("optimized_prompt_sha256") or text_sha256(str(row["optimized_prompt"]).strip())) for row in merged]
    if len(ids) != len(set(ids)):
        raise ValueError("assembled F2 pool has duplicate prompt IDs")
    if len(hashes) != len(set(hashes)):
        raise ValueError("assembled F2 pool has duplicate optimized prompt text")
    categories = Counter(str(row["category"]) for row in merged)
    required = {"code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"}
    if set(categories) != required:
        raise ValueError(f"assembled categories differ from protocol: {sorted(categories)}")

    output = resolve(args.output)
    write_jsonl(output, merged)
    report = {
        "schema_version": "experiment_f2_pool_assembly_1.0",
        "base_candidates": len(base),
        "new_portfolio_candidates": len(new_rows),
        "new_unique_sources_by_category": dict(sorted(unique_new_sources.items())),
        "minimum_new_sources_per_category": args.minimum_new_sources_per_category,
        "assembled_candidates": len(merged),
        "assembled_unique_sources": len({str(row["source_prompt_id"]) for row in merged}),
        "category_counts": dict(sorted(categories.items())),
        "construction_gate_passed": not failures,
        "construction_failures": failures,
        "test_or_confirmation_data_used": False,
    }
    write_json(resolve(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["construction_gate_passed"] else 1)


if __name__ == "__main__":
    main()
