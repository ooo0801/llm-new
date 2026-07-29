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
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def rename_row(
    row: dict[str, Any],
    *,
    source_id: str,
    target_id: str,
) -> dict[str, Any]:
    output = dict(row)
    for key in ("id", "prompt_id"):
        value = output.get(key)
        if isinstance(value, str) and value.startswith(source_id):
            output[key] = target_id + value[len(source_id) :]
    if "source_prompt_id" in output:
        output["source_prompt_id"] = target_id
    return output


def replace_pair_records(
    base: list[dict[str, Any]],
    replacement: list[dict[str, Any]],
    *,
    base_id: str,
    replacement_id: str,
) -> list[dict[str, Any]]:
    kept = [
        row
        for row in base
        if not str(row.get("prompt_id", "")).startswith(base_id + "::")
    ]
    renamed = [
        rename_row(
            row,
            source_id=replacement_id,
            target_id=base_id,
        )
        for row in replacement
    ]
    return [*kept, *renamed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--replacement-dir", required=True, type=Path)
    parser.add_argument("--base-id", required=True)
    parser.add_argument("--replacement-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    base_proxy = read_jsonl(args.base_dir / "proxy_results.jsonl")
    replacement_proxy = read_jsonl(
        args.replacement_dir / "proxy_results.jsonl"
    )
    if len(replacement_proxy) != 1:
        raise ValueError("Expected exactly one replacement proxy row")
    renamed_proxy = rename_row(
        replacement_proxy[0],
        source_id=args.replacement_id,
        target_id=args.base_id,
    )
    proxy_output: list[dict[str, Any]] = []
    replacements = 0
    for row in base_proxy:
        if str(row.get("id")) == args.base_id:
            proxy_output.append(renamed_proxy)
            replacements += 1
        else:
            proxy_output.append(row)
    if replacements != 1:
        raise ValueError("Base proxy id was not found uniquely")

    specifications = (
        ("micro_scores_24.jsonl", "micro_scores_2.jsonl", 24),
        ("macro_records_24x10.jsonl", "macro_records_2x10.jsonl", 240),
        ("task_validation_24.jsonl", "task_validation_2.jsonl", 24),
    )
    output_rows: dict[str, list[dict[str, Any]]] = {}
    for base_name, replacement_name, expected in specifications:
        rows = replace_pair_records(
            read_jsonl(args.base_dir / base_name),
            read_jsonl(args.replacement_dir / replacement_name),
            base_id=args.base_id,
            replacement_id=args.replacement_id,
        )
        if len(rows) != expected:
            raise ValueError(
                f"{base_name}: expected {expected} rows, found {len(rows)}"
            )
        output_rows[base_name] = rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "proxy_results.jsonl", proxy_output)
    for name, rows in output_rows.items():
        write_jsonl(args.output_dir / name, rows)
    report = {
        "version": 1,
        "method": "validation_development_candidate_replacement",
        "base_prompt_count": len(base_proxy),
        "base_id": args.base_id,
        "replacement_id": args.replacement_id,
        "replacement_proxy_accepted": bool(
            renamed_proxy["optimization"]["proxy_accepted"]
        ),
        "replacement_optimized_prompt": renamed_proxy["optimization"][
            "optimized_prompt"
        ],
        "output_counts": {
            "proxy": len(proxy_output),
            **{name: len(rows) for name, rows in output_rows.items()},
        },
    }
    (args.output_dir / "replacement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
