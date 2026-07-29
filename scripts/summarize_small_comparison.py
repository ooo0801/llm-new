from __future__ import annotations

import argparse
from collections import Counter
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_joint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    histories = [step for row in rows for step in row.get("history", [])]
    filter_totals: Counter[str] = Counter()
    committed_blocks: Counter[str] = Counter()
    for step in histories:
        filter_totals.update(step.get("candidate_filter_stats", {}))
        if step.get("committed"):
            committed_blocks[str(step.get("block_type", "unknown"))] += 1
    return {
        "records": len(rows),
        "accepted": sum(bool(row.get("accepted")) for row in rows),
        "complete_requested_rounds": sum(
            int(row.get("rounds_completed", 0))
            == int(row.get("rounds_requested", 0))
            for row in rows
        ),
        "failed": sum(row.get("failure") is not None for row in rows),
        "changed_discrete_prompts": sum(
            row.get("optimized_prompt") != row.get("initial_prompt") for row in rows
        ),
        "positive_proxy_gain_prompts": sum(
            float(row.get("proxy_objective_gain", 0.0)) > 0
            for row in rows
        ),
        "total_committed_rounds": sum(
            int(row.get("committed_rounds", 0)) for row in rows
        ),
        "mean_edit_ratio": mean([float(row["edit_ratio"]) for row in rows]),
        "mean_ppl_ratio": mean([float(row["ppl_ratio"]) for row in rows]),
        "mean_proxy_objective_gain": mean(
            [float(row.get("proxy_objective_gain", 0.0)) for row in rows]
        ),
        "rounds": len(histories),
        "committed_blocks": dict(committed_blocks),
        "candidate_filter_totals": dict(filter_totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True, type=Path)
    parser.add_argument("--old-nf4", required=True, type=Path)
    parser.add_argument("--macro-only", required=True, type=Path)
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    seeds = read_jsonl(args.seeds)[: args.count]
    seed_ids = [str(row["id"]) for row in seeds]
    old_index = {
        str(row["id"]): row for row in read_jsonl(args.old_nf4)
    }
    macro_index = {
        str(row["prompt_id"]): row for row in read_jsonl(args.macro_only)
    }
    joint_index = {
        str(row["prompt_id"]): row for row in read_jsonl(args.joint)
    }
    old = [old_index[prompt_id] for prompt_id in seed_ids]
    macro = [macro_index[prompt_id] for prompt_id in seed_ids]
    joint = [joint_index[prompt_id] for prompt_id in seed_ids]
    old_optimization = [row["optimization"] for row in old]

    summary = {
        "comparison_prompt_count": args.count,
        "same_prompt_ids_across_arms": bool(
            len(old_index) >= args.count
            and len(macro_index) == args.count
            and len(joint_index) == args.count
            and all(
                prompt_id in old_index
                and prompt_id in macro_index
                and prompt_id in joint_index
                for prompt_id in seed_ids
            )
        ),
        "arms": {
            "original": {
                "records": len(seeds),
                "changed_discrete_prompts": 0,
            },
            "old_nf4": {
                "records": len(old),
                "proxy_accepted": sum(
                    bool(row.get("proxy_accepted"))
                    for row in old_optimization
                ),
                "changed_discrete_prompts": sum(
                    row.get("optimized_prompt") != row.get("initial_prompt")
                    for row in old_optimization
                ),
                "mean_edit_ratio": mean(
                    [float(row["edit_ratio"]) for row in old_optimization]
                ),
                "mean_ppl_ratio": mean(
                    [float(row["ppl_ratio"]) for row in old_optimization]
                ),
                "mean_proxy_score_gain": mean(
                    [float(row["score_gain"]) for row in old_optimization]
                ),
            },
            "macro_only_a0": summarize_joint(macro),
            "joint_calibrated_a": summarize_joint(joint),
        },
        "joint_vs_macro_same_discrete_outputs": sum(
            left.get("optimized_prompt") == right.get("optimized_prompt")
            for left, right in zip(macro, joint, strict=True)
        ),
        "interpretation": (
            "V2 acceptance requires a committed discrete token change, positive exact "
            "proxy gain, PPL ratio <= 2, edit ratio <= 0.25, digit preservation, and "
            "tokenizer round-trip validity. Final effectiveness remains subject to "
            "held-out full-micro plus five-family hard validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
