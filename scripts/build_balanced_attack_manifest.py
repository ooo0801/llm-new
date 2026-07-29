from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FAMILIES = (
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def configuration_key(row: dict[str, Any]) -> str:
    return json.dumps(
        row["configuration"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-family", type=int, default=2)
    args = parser.parse_args()
    if args.per_family < 1:
        raise ValueError("per-family must be positive")

    rows = read_jsonl(args.input)
    selected: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    for family in FAMILIES:
        family_rows = [
            row
            for row in rows
            if row.get("family") == family and row.get("split") == args.split
        ]
        groups: dict[str, list[dict[str, Any]]] = {}
        group_order: list[str] = []
        for row in family_rows:
            key = configuration_key(row)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(row)
        chosen_group = next(
            (groups[key] for key in group_order if len(groups[key]) >= args.per_family),
            None,
        )
        if chosen_group is None:
            raise ValueError(
                f"{family}: no configuration has {args.per_family} variants"
            )
        chosen = chosen_group[: args.per_family]
        selected.extend(chosen)
        selection[family] = {
            "configuration": chosen[0]["configuration"],
            "variant_ids": [str(row["variant_id"]) for row in chosen],
            "seeds": [int(row["seed"]) for row in chosen],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "source_manifest": str(args.input),
        "split": args.split,
        "families": len(FAMILIES),
        "per_family": args.per_family,
        "selected_variants": len(selected),
        "policy": (
            "first_manifest_configuration_with_required_seed_replicates"
        ),
        "selection": selection,
        "outcome_metrics_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
