from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


EXPECTED_COUNT = 16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    parser.add_argument(
        "--source",
        default=(
            "reproducibility/v6_20260729/"
            "final_test_validated_prompts.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/prompt-transfer-14b-a1/inputs",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = ROOT / source
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    source_rows = read_jsonl(source)
    selected = [
        (source_index, row)
        for source_index, row in enumerate(source_rows)
        if row.get("optimization", {}).get("accepted") is True
    ]
    if len(source_rows) != 22:
        raise ValueError(f"expected 22 V6 rows, received {len(source_rows)}")
    if len(selected) != EXPECTED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COUNT} accepted V6 rows, received {len(selected)}"
        )

    pairs: list[dict[str, Any]] = []
    ids: set[str] = set()
    for source_index, row in selected:
        optimization = row["optimization"]
        prompt_id = str(row["prompt_id"])
        if prompt_id in ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        ids.add(prompt_id)
        initial = str(optimization["initial_prompt"])
        optimized = str(optimization["optimized_prompt"])
        if not initial.strip() or not optimized.strip() or initial == optimized:
            raise ValueError(f"invalid frozen pair: {prompt_id}")
        pairs.append(
            {
                "source_index": source_index,
                "id": prompt_id,
                "prompt_id": prompt_id,
                "category": row["category"],
                "initial_prompt": initial,
                "optimized_prompt": optimized,
                "evaluator": row["evaluator"],
                "expected_answer": row.get("expected_answer"),
                "expected_contains": row.get("expected_contains"),
                "accepted": True,
                "proxy_objective_gain": float(
                    optimization["proxy_objective_gain"]
                ),
                "edit_count": int(optimization["edit_count"]),
                "edit_ratio": float(optimization["edit_ratio"]),
                "initial_ppl": float(optimization["initial_ppl"]),
                "final_ppl": float(optimization["final_ppl"]),
                "ppl_ratio": float(optimization["ppl_ratio"]),
                "committed_rounds": int(optimization["committed_rounds"]),
                "rounds_completed": int(optimization["rounds_run"]),
                "source_v6_nondegraded_families": int(
                    optimization["strict_validation"]["nondegraded_families"]
                ),
                "source_v6_record_sha256": hashlib.sha256(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )

    pair_path = output_dir / "accepted16_pairs.jsonl"
    write_jsonl(pair_path, pairs)
    manifest = {
        "schema_version": "experiment_a_frozen_prompt_pairs_1.0",
        "source_path": str(source.relative_to(ROOT)),
        "source_sha256": sha256(source),
        "source_rows": len(source_rows),
        "selection_rule": "optimization.accepted == true",
        "accepted_rows": len(pairs),
        "order_rule": "source_file_order",
        "initial_field": "optimization.initial_prompt",
        "optimized_field": "optimization.optimized_prompt",
        "categories": dict(Counter(row["category"] for row in pairs)),
        "prompt_ids": [row["prompt_id"] for row in pairs],
        "accepted16_path": str(pair_path.relative_to(ROOT)),
        "accepted16_sha256": sha256(pair_path),
    }
    manifest_path = output_dir / "accepted16_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
