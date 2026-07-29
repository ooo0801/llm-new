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
    reason_counts: Counter[str] = Counter()
    family_nondegraded: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for row in rows:
        optimization = row["optimization"]
        strict = optimization["strict_validation"]
        hybrid = optimization["hybrid_validation"]
        reasons: list[str] = []
        if not optimization.get("proxy_accepted"):
            reasons.append("proxy_rejected")
        if not optimization.get("hybrid_base_accepted"):
            reasons.append("heldout_hybrid_gain_not_positive")
        if not strict["initial_task_passed"]:
            reasons.append("initial_task_failed")
        if not strict["optimized_task_passed"]:
            reasons.append("optimized_task_failed")
        if not strict["proxy_constraints_passed"]:
            reasons.append("proxy_constraints_failed")
        if not strict["discrete_prompt_changed"]:
            reasons.append("no_discrete_change")
        if (
            int(strict["nondegraded_families"])
            < int(strict["minimum_nondegraded_families"])
        ):
            reasons.append("fewer_than_3_nondegraded_families")
        if not strict["finite_objective"]:
            reasons.append("nonfinite_objective")
        if not reasons and not optimization["accepted"]:
            reasons.append("unknown")
        reason_counts.update(reasons)
        for family, family_result in strict["families"].items():
            if family_result["nondegraded"]:
                family_nondegraded[family] += 1
        details.append(
            {
                "prompt_id": row.get("prompt_id", row.get("id")),
                "category": row.get("category"),
                "accepted": bool(optimization["accepted"]),
                "reasons": reasons,
                "objective_gain": hybrid["objective_gain"],
                "initial_task_passed": strict[
                    "initial_task_passed"
                ],
                "optimized_task_passed": strict[
                    "optimized_task_passed"
                ],
                "nondegraded_families": strict[
                    "nondegraded_families"
                ],
                "family_relative_gains": {
                    family: result["relative_gain"]
                    for family, result in strict["families"].items()
                },
            }
        )

    payload = {
        "records": len(rows),
        "strict_accepted": sum(
            bool(row["optimization"]["accepted"]) for row in rows
        ),
        "rejection_reason_counts": dict(reason_counts),
        "family_nondegraded_prompt_counts": dict(family_nondegraded),
        "details": details,
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
