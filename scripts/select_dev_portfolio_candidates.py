from __future__ import annotations

import argparse
import json
import math
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
    parser.add_argument("--source-results", required=True, type=Path)
    parser.add_argument("--portfolio-results", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--strict-validation", required=True, type=Path)
    parser.add_argument("--output-all", required=True, type=Path)
    parser.add_argument("--output-accepted", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--required-accepted", type=int, default=12)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_results)
    portfolio_rows = {
        str(row["prompt_id"]): row for row in read_jsonl(args.portfolio_results)
    }
    strict_rows = {
        str(row.get("id", row["prompt_id"])): row
        for row in read_jsonl(args.strict_validation)
    }
    group_by_source: dict[str, list[str]] = {}
    for row in read_jsonl(args.mapping):
        group_by_source.setdefault(str(row["source_prompt_id"]), []).append(
            str(row["portfolio_prompt_id"])
        )

    all_outputs: list[dict[str, Any]] = []
    accepted_outputs: list[dict[str, Any]] = []
    chosen_details: list[dict[str, Any]] = []

    for source in source_rows:
        source_id = str(source["prompt_id"])
        accepted_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for portfolio_id in group_by_source.get(source_id, []):
            strict = strict_rows.get(portfolio_id)
            portfolio = portfolio_rows.get(portfolio_id)
            if strict is None or portfolio is None:
                raise ValueError(f"{portfolio_id}: missing portfolio or validation row")
            optimization = strict["optimization"]
            if optimization.get("accepted") is not True:
                continue
            gain = float(
                optimization["hybrid_validation"]["objective_gain"]
            )
            if not math.isfinite(gain):
                continue
            accepted_candidates.append((strict, portfolio))

        accepted_candidates.sort(
            key=lambda pair: (
                -float(
                    pair[0]["optimization"]["hybrid_validation"][
                        "objective_gain"
                    ]
                ),
                -int(
                    pair[0]["optimization"]["strict_validation"][
                        "nondegraded_families"
                    ]
                ),
                float(pair[0]["optimization"]["ppl_ratio"]),
                -float(pair[0]["optimization"]["proxy_objective_gain"]),
                str(pair[0]["prompt_id"]),
            )
        )

        if accepted_candidates:
            strict, portfolio = accepted_candidates[0]
            output = {
                **source,
                "prompt_id": source_id,
                "optimized_prompt": str(
                    strict["optimization"]["optimized_prompt"]
                ),
                "accepted": True,
                "acceptance_stage": "frozen_after_development_portfolio_rerank",
                "requires_hard_validation": True,
                "final_ppl": float(strict["optimization"]["final_ppl"]),
                "ppl_ratio": float(strict["optimization"]["ppl_ratio"]),
                "edit_ratio": float(strict["optimization"]["edit_ratio"]),
                "edit_count": int(strict["optimization"]["edit_count"]),
                "proxy_objective_gain": float(
                    strict["optimization"]["proxy_objective_gain"]
                ),
                "final_task_passed": True,
                "history": [],
                "development_selection": {
                    "portfolio_prompt_id": str(strict["prompt_id"]),
                    "candidate_provenance": portfolio.get(
                        "candidate_provenance", {}
                    ),
                    "fixed_hybrid_validation": strict["optimization"][
                        "hybrid_validation"
                    ],
                    "strict_validation": strict["optimization"][
                        "strict_validation"
                    ],
                },
            }
            accepted_outputs.append(output)
            chosen_details.append(
                {
                    "source_prompt_id": source_id,
                    "portfolio_prompt_id": str(strict["prompt_id"]),
                    "objective_gain": float(
                        strict["optimization"]["hybrid_validation"][
                            "objective_gain"
                        ]
                    ),
                    "nondegraded_families": int(
                        strict["optimization"]["strict_validation"][
                            "nondegraded_families"
                        ]
                    ),
                }
            )
        else:
            output = {
                **source,
                "prompt_id": source_id,
                "optimized_prompt": str(source["initial_prompt"]),
                "accepted": False,
                "acceptance_stage": "rejected_by_development_portfolio_rerank",
                "final_ppl": float(source["initial_ppl"]),
                "ppl_ratio": 1.0,
                "edit_ratio": 0.0,
                "edit_count": 0,
                "proxy_objective_gain": 0.0,
                "final_task_passed": bool(source["initial_task_passed"]),
                "history": [],
                "development_selection": None,
            }
        all_outputs.append(output)

    if len(all_outputs) != len(source_rows):
        raise AssertionError("one frozen output per source prompt is required")
    write_jsonl(args.output_all, all_outputs)
    write_jsonl(args.output_accepted, accepted_outputs)
    report = {
        "source_prompts": len(source_rows),
        "portfolio_candidates": len(portfolio_rows),
        "development_accepted": len(accepted_outputs),
        "required_accepted": args.required_accepted,
        "gate_passed": len(accepted_outputs) >= args.required_accepted,
        "selection_rule": (
            "strict_accept_then_max_fixed_hybrid_gain_then_family_count"
        ),
        "test_data_used": False,
        "chosen": chosen_details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
