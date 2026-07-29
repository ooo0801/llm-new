from __future__ import annotations

import argparse
import json
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
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["micro", "macro"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mapping = read_jsonl(args.mapping)
    scores = read_jsonl(args.input)
    if args.kind == "micro":
        by_unique: dict[str, list[dict[str, Any]]] = {}
        for row in scores:
            by_unique.setdefault(str(row["prompt_id"]), []).append(row)
    else:
        by_unique = {}
        for row in scores:
            by_unique.setdefault(str(row["prompt_id"]), []).append(row)

    expanded: list[dict[str, Any]] = []
    for target in mapping:
        unique_id = str(target["unique_prompt_id"])
        if unique_id not in by_unique:
            raise ValueError(f"Missing {args.kind} score for {unique_id}")
        for source in by_unique[unique_id]:
            row = dict(source)
            row["prompt_id"] = target["candidate_prompt_id"]
            row["source_prompt_id"] = target["source_prompt_id"]
            row["candidate_role"] = target["candidate_role"]
            row["category"] = target["category"]
            row["deduplicated_from_prompt_id"] = unique_id
            expanded.append(row)

    expected_multiplier = 1 if args.kind == "micro" else len(scores) // len(by_unique)
    expected = len(mapping) * expected_multiplier
    if len(expanded) != expected:
        raise RuntimeError(f"Expanded {len(expanded)} rows, expected {expected}")
    write_jsonl(args.output, expanded)
    print(
        json.dumps(
            {
                "kind": args.kind,
                "unique_input_rows": len(scores),
                "expanded_rows": len(expanded),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
