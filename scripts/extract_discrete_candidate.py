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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--candidate-index", required=True, type=int)
    parser.add_argument("--candidate-tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.input)
        if str(row["prompt_id"]) == args.prompt_id
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one row for {args.prompt_id}, found {len(rows)}"
        )
    source = rows[0]
    rounds = [
        item
        for item in source["history"]
        if int(item["round"]) == args.round
    ]
    if len(rounds) != 1:
        raise ValueError(f"Round {args.round} was not found uniquely")
    candidates = rounds[0]["reranked_candidates"]
    index = args.candidate_index - 1
    if index < 0 or index >= len(candidates):
        raise ValueError("candidate-index is out of range")
    candidate = candidates[index]
    task = candidate.get("task_validation", {})
    if task.get("task_passed") is not True:
        raise ValueError("Selected candidate did not pass task validation")
    if candidate.get("constraints_passed") is not True:
        raise ValueError("Selected candidate did not pass proxy constraints")
    if float(candidate.get("proxy_gain", 0.0)) <= 0:
        raise ValueError("Selected candidate has no positive proxy gain")

    total_tokens = int(source["active_user_tokens"]) + int(
        source.get("preserved_overflow_tokens", 0)
    )
    edit_ratio = float(candidate["edit_ratio"])
    edit_count = max(1, int(round(edit_ratio * total_tokens)))
    output = {
        **{
            key: source.get(key)
            for key in (
                "category",
                "evaluator",
                "expected_answer",
                "expected_contains",
                "language",
                "source",
            )
        },
        "prompt_id": f"{args.prompt_id}__{args.candidate_tag}",
        "initial_prompt": str(source["initial_prompt"]),
        "optimized_prompt": str(candidate["decoded_prompt"]),
        "accepted": True,
        "acceptance_stage": "portfolio_candidate_pending_dev_rerank",
        "requires_hard_validation": True,
        "initial_ppl": float(source["initial_ppl"]),
        "final_ppl": float(candidate["ppl"]),
        "ppl_ratio": float(candidate["ppl_ratio"]),
        "edit_ratio": edit_ratio,
        "edit_count": edit_count,
        "rounds_requested": int(source["rounds_requested"]),
        "rounds_completed": int(args.round),
        "committed_rounds": 1,
        "proxy_objective_gain": float(candidate["proxy_gain"]),
        "active_user_tokens": int(source["active_user_tokens"]),
        "preserved_overflow_tokens": int(
            source.get("preserved_overflow_tokens", 0)
        ),
        "optimization_max_length": int(
            source["optimization_max_length"]
        ),
        "initial_task_passed": bool(source["initial_task_passed"]),
        "final_task_passed": True,
        "candidate_provenance": {
            "source_prompt_id": args.prompt_id,
            "source_result": str(args.input),
            "round": int(args.round),
            "candidate_index_one_based": int(args.candidate_index),
            "candidate_tag": args.candidate_tag,
            "training_nondegraded_families": int(
                candidate["training_nondegraded_families"]
            ),
            "task_validation": task,
        },
    }
    if not math.isfinite(output["ppl_ratio"]):
        raise ValueError("Candidate PPL ratio is not finite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
