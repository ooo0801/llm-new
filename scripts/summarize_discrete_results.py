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
    parser.add_argument("results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    path = Path(args.results).resolve()
    rows = read_jsonl(path)
    blocks: Counter[str] = Counter()
    committed_blocks: Counter[str] = Counter()
    filter_totals: Counter[str] = Counter()
    prompts: list[dict[str, Any]] = []
    for row in rows:
        rounds: list[dict[str, Any]] = []
        for trace in row.get("history", []):
            block = str(trace.get("block_type", "unknown"))
            blocks[block] += 1
            if trace.get("committed"):
                committed_blocks[block] += 1
            filter_totals.update(trace.get("candidate_filter_stats", {}))
            rounds.append(
                {
                    "round": trace.get("round"),
                    "block_type": block,
                    "state_before": trace.get("state_token_ids_before"),
                    "state_after": trace.get("state_token_ids_after"),
                    "committed": bool(trace.get("committed")),
                    "committed_position": trace.get("committed_position"),
                    "source_token": trace.get("committed_source_token"),
                    "candidate_token": trace.get(
                        "committed_candidate_token"
                    ),
                    "proxy_gain": trace.get("committed_proxy_gain"),
                    "candidate_filter_stats": trace.get(
                        "candidate_filter_stats"
                    ),
                }
            )
        prompts.append(
            {
                "prompt_id": row.get("prompt_id"),
                "accepted": bool(row.get("accepted")),
                "initial_prompt": row.get("initial_prompt"),
                "optimized_prompt": row.get("optimized_prompt"),
                "edit_count": int(row.get("edit_count", 0)),
                "committed_rounds": int(
                    row.get("committed_rounds", 0)
                ),
                "proxy_objective_gain": row.get(
                    "proxy_objective_gain"
                ),
                "initial_ppl": row.get("initial_ppl"),
                "final_ppl": row.get("final_ppl"),
                "ppl_ratio": row.get("ppl_ratio"),
                "rounds": rounds,
            }
        )

    state_links_valid = True
    for prompt in prompts:
        rounds = prompt["rounds"]
        for previous, current in zip(rounds, rounds[1:]):
            if previous["state_after"] != current["state_before"]:
                state_links_valid = False

    payload = {
        "results": str(path),
        "prompt_count": len(rows),
        "accepted": sum(bool(row.get("accepted")) for row in rows),
        "changed": sum(
            row.get("optimized_prompt") != row.get("initial_prompt")
            for row in rows
        ),
        "total_committed_rounds": sum(
            int(row.get("committed_rounds", 0)) for row in rows
        ),
        "round_blocks": dict(blocks),
        "committed_blocks": dict(committed_blocks),
        "candidate_filter_totals": dict(filter_totals),
        "state_links_valid": state_links_valid,
        "prompts": prompts,
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
