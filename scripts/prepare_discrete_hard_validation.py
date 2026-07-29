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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    proxy_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    unique_rows: list[dict[str, Any]] = []
    expansion_rows: list[dict[str, Any]] = []
    prompt_to_unique: dict[str, str] = {}

    for row in rows:
        source_id = str(row["prompt_id"])
        initial = str(row["initial_prompt"]).strip()
        optimized = str(row["optimized_prompt"]).strip()
        changed = initial != optimized
        proxy_accepted = bool(row.get("accepted"))
        if proxy_accepted and not changed:
            raise ValueError(
                f"{source_id}: proxy accepted without a discrete change"
            )
        if proxy_accepted and float(
            row.get("proxy_objective_gain", 0.0)
        ) <= 0:
            raise ValueError(
                f"{source_id}: proxy accepted without positive gain"
            )
        proxy_rows.append(
            {
                "id": source_id,
                "prompt_id": source_id,
                "category": row.get("category", "unknown"),
                "prompt": initial,
                "evaluator": row.get("evaluator"),
                "expected_answer": row.get("expected_answer"),
                "expected_contains": row.get("expected_contains"),
                "optimization": {
                    "initial_prompt": initial,
                    "optimized_prompt": optimized,
                    "proxy_accepted": proxy_accepted,
                    "accepted": None,
                    "acceptance_stage": (
                        "proxy_search_pending_hard_validation"
                    ),
                    "rounds_run": int(
                        row.get("rounds_completed", 0)
                    ),
                    "committed_rounds": int(
                        row.get("committed_rounds", 0)
                    ),
                    "edit_count": int(row.get("edit_count", 0)),
                    "edit_ratio": float(row.get("edit_ratio", 0.0)),
                    "initial_ppl": float(
                        row.get("initial_ppl", 0.0)
                    ),
                    "final_ppl": float(row.get("final_ppl", 0.0)),
                    "ppl_ratio": float(row.get("ppl_ratio", 0.0)),
                    "proxy_objective_gain": float(
                        row.get("proxy_objective_gain", 0.0)
                    ),
                    "inner_objective": (
                        "discrete_joint_micro_five_family_macro_v2"
                    ),
                    "discrete_change_verified": changed,
                },
            }
        )
        for role, prompt in (("initial", initial), ("optimized", optimized)):
            candidate_id = f"{source_id}::{role}"
            if prompt not in prompt_to_unique:
                unique_id = f"unique::{len(unique_rows):04d}"
                prompt_to_unique[prompt] = unique_id
                unique_rows.append(
                    {
                        "id": unique_id,
                        "prompt_id": unique_id,
                        "prompt": prompt,
                        "category": row.get("category", "unknown"),
                    }
                )
            unique_id = prompt_to_unique[prompt]
            candidate_rows.append(
                {
                    "id": candidate_id,
                    "prompt_id": candidate_id,
                    "source_prompt_id": source_id,
                    "candidate_role": role,
                    "category": row.get("category", "unknown"),
                    "prompt": prompt,
                    "evaluator": row.get("evaluator"),
                    "expected_answer": row.get("expected_answer"),
                    "expected_contains": row.get("expected_contains"),
                    "language": row.get("language"),
                }
            )
            expansion_rows.append(
                {
                    "candidate_prompt_id": candidate_id,
                    "source_prompt_id": source_id,
                    "candidate_role": role,
                    "category": row.get("category", "unknown"),
                    "unique_prompt_id": unique_id,
                }
            )

    output_dir = args.output_dir
    write_jsonl(output_dir / "proxy_results.jsonl", proxy_rows)
    write_jsonl(
        output_dir / "validation_candidates.jsonl",
        candidate_rows,
    )
    write_jsonl(
        output_dir / "validation_unique_prompts.jsonl",
        unique_rows,
    )
    write_jsonl(
        output_dir / "validation_expansion_map.jsonl",
        expansion_rows,
    )
    report = {
        "source_prompts": len(rows),
        "proxy_accepted": sum(
            bool(row["optimization"]["proxy_accepted"])
            for row in proxy_rows
        ),
        "candidate_pairs": len(candidate_rows),
        "unique_prompt_strings": len(unique_rows),
        "deduplicated_evaluations_saved_per_variant": (
            len(candidate_rows) - len(unique_rows)
        ),
        "strict_proxy_semantics_verified": True,
        "exact_deduplication": True,
    }
    (output_dir / "validation_preparation_summary.json").write_text(
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
