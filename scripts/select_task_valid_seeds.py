from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--task-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    seeds = read_jsonl(args.seeds)
    task = {
        str(row["prompt_id"]): bool(row["task_passed"])
        for row in read_jsonl(args.task_results)
    }
    valid = [
        row for row in seeds if task.get(str(row["id"])) is True
    ]
    if len(valid) < args.count:
        raise ValueError(
            f"Only {len(valid)} task-valid seeds for count={args.count}"
        )

    grouped: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    category_order: list[str] = []
    for row in valid:
        category = str(row.get("category", "unknown"))
        if category not in grouped:
            category_order.append(category)
        grouped[category].append(row)

    selected: list[dict[str, Any]] = []
    while len(selected) < args.count:
        progress = False
        for category in category_order:
            if grouped[category]:
                selected.append(grouped[category].popleft())
                progress = True
                if len(selected) >= args.count:
                    break
        if not progress:
            break
    if len(selected) != args.count:
        raise RuntimeError("Could not assemble requested task-valid set")

    write_jsonl(args.output, selected)
    report = {
        "source_seeds": len(seeds),
        "task_valid_seeds": len(valid),
        "task_invalid_seeds": len(seeds) - len(valid),
        "selected": len(selected),
        "selection_method": (
            "deterministic_category_round_robin_in_original_seed_order"
        ),
        "selected_category_counts": dict(
            Counter(str(row.get("category", "unknown")) for row in selected)
        ),
        "selected_prompt_ids": [str(row["id"]) for row in selected],
        "all_selected_task_valid": all(
            task[str(row["id"])] for row in selected
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
