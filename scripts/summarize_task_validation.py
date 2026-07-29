from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    role_counts: Counter[str] = Counter()
    role_passed: Counter[str] = Counter()
    pairs: dict[str, dict[str, bool]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        role = str(row.get("candidate_role", "unknown"))
        passed = bool(row.get("task_passed"))
        role_counts[role] += 1
        role_passed[role] += int(passed)
        source_id = str(row.get("source_prompt_id"))
        pairs.setdefault(source_id, {})[role] = passed
        if not passed:
            failures.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "source_prompt_id": source_id,
                    "candidate_role": role,
                    "category": row.get("category"),
                    "evaluator": row.get("evaluator"),
                    "evaluation_rule": row.get("evaluation_rule"),
                    "expected_answer": row.get("expected_answer"),
                    "expected_contains": row.get("expected_contains"),
                    "generated_text": row.get("generated_text"),
                }
            )

    pair_passed = sum(
        roles.get("initial") is True
        and roles.get("optimized") is True
        for roles in pairs.values()
    )
    payload = {
        "records": len(rows),
        "passed": sum(bool(row.get("task_passed")) for row in rows),
        "role_records": dict(role_counts),
        "role_passed": dict(role_passed),
        "source_pairs": len(pairs),
        "both_initial_and_optimized_passed": pair_passed,
        "failures": failures,
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
