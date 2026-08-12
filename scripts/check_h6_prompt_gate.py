from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


CORE_CATEGORIES = {"code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(row: dict[str, Any]) -> dict[str, Any]:
    optimization = row.get("optimization", {})
    prompt_id = str(row.get("prompt_id", row.get("id")))
    prompt = str(optimization.get("optimized_prompt", row.get("optimized_prompt", row.get("prompt", "")))).strip()
    accepted = bool(optimization.get("accepted", row.get("accepted", False)))
    result = dict(row)
    result["prompt_id"] = prompt_id
    result["optimized_prompt"] = prompt
    result["optimized_prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    result["h6_confirmed"] = accepted
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-confirmed", type=int, default=19)
    args = parser.parse_args()
    rows = [normalized(row) for row in read_jsonl(args.input)]
    confirmed = [row for row in rows if row["h6_confirmed"]]
    categories = Counter(str(row.get("category", "unknown")) for row in confirmed)
    observed_core = CORE_CATEGORIES & set(categories)
    errors = []
    if len({row["prompt_id"] for row in confirmed}) != len(confirmed):
        errors.append("duplicate confirmed prompt IDs")
    if len({row["optimized_prompt_sha256"] for row in confirmed}) != len(confirmed):
        errors.append("duplicate confirmed prompt texts")
    missing = sorted(CORE_CATEGORIES - observed_core)
    passed = not errors and len(confirmed) >= args.minimum_confirmed and not missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in confirmed).encode("utf-8"))
    report = {
        "schema_version": "experiment_h6_prompt_gate_1.0",
        "input_rows": len(rows),
        "confirmed_rows": len(confirmed),
        "minimum_confirmed": args.minimum_confirmed,
        "confirmed_by_category": dict(sorted(categories.items())),
        "required_core_categories": sorted(CORE_CATEGORIES),
        "missing_core_categories": missing,
        "technical_errors": errors,
        "hypothesis": "H6-P",
        "hypothesis_supported": passed,
        "candidate_source_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
