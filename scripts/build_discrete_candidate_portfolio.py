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


def candidate_is_eligible(
    candidate: dict[str, Any], minimum_families: int
) -> bool:
    task = candidate.get("task_validation") or {}
    try:
        proxy_gain = float(candidate.get("proxy_gain", 0.0))
        ppl_ratio = float(candidate.get("ppl_ratio", math.inf))
        families = int(candidate.get("training_nondegraded_families", 0))
    except (TypeError, ValueError):
        return False
    return bool(
        candidate.get("constraints_passed") is True
        and task.get("task_passed") is True
        and math.isfinite(proxy_gain)
        and proxy_gain > 0.0
        and math.isfinite(ppl_ratio)
        and families >= minimum_families
        and str(candidate.get("decoded_prompt", "")).strip()
    )


def result_row(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    round_number: int,
    candidate_index: int,
    candidate_tag: str,
) -> dict[str, Any]:
    initial = str(source["initial_prompt"]).strip()
    optimized = str(candidate["decoded_prompt"]).strip()
    if not optimized or optimized == initial:
        raise ValueError(f"{source['prompt_id']}: candidate has no discrete change")
    active_tokens = int(source["active_user_tokens"])
    overflow_tokens = int(source.get("preserved_overflow_tokens", 0))
    total_tokens = active_tokens + overflow_tokens
    edit_ratio = float(candidate["edit_ratio"])
    edit_count = max(1, int(round(edit_ratio * total_tokens)))
    source_prompt_id = str(source["prompt_id"])
    portfolio_prompt_id = f"{source_prompt_id}__{candidate_tag}"
    return {
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
        "prompt_id": portfolio_prompt_id,
        "initial_prompt": initial,
        "optimized_prompt": optimized,
        "accepted": True,
        "acceptance_stage": "portfolio_candidate_pending_dev_rerank",
        "requires_hard_validation": True,
        "initial_ppl": float(source["initial_ppl"]),
        "final_ppl": float(candidate["ppl"]),
        "ppl_ratio": float(candidate["ppl_ratio"]),
        "edit_ratio": edit_ratio,
        "edit_count": edit_count,
        "rounds_requested": int(source["rounds_requested"]),
        "rounds_completed": round_number,
        "committed_rounds": 1,
        "proxy_objective_gain": float(candidate["proxy_gain"]),
        "active_user_tokens": active_tokens,
        "preserved_overflow_tokens": overflow_tokens,
        "optimization_max_length": int(source["optimization_max_length"]),
        "initial_task_passed": bool(source["initial_task_passed"]),
        "final_task_passed": True,
        "candidate_provenance": {
            "source_prompt_id": source_prompt_id,
            "round": round_number,
            "candidate_index_one_based": candidate_index,
            "candidate_tag": candidate_tag,
            "training_nondegraded_families": int(
                candidate["training_nondegraded_families"]
            ),
            "selected_by_inner_optimizer": (
                optimized == str(source.get("optimized_prompt", "")).strip()
            ),
            "task_validation": candidate.get("task_validation") or {},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maximum-candidates-per-prompt", type=int, default=2)
    parser.add_argument("--minimum-nondegraded-families", type=int, default=3)
    args = parser.parse_args()
    if args.maximum_candidates_per_prompt < 1:
        raise ValueError("maximum-candidates-per-prompt must be positive")

    outputs: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    source_rows = read_jsonl(args.input)
    per_source_counts: dict[str, int] = {}
    eligible_before_dedup = 0
    skipped_non_round_history_entries = 0

    for source in source_rows:
        source_prompt_id = str(source["prompt_id"])
        candidates_by_prompt: dict[str, tuple[dict[str, Any], int, int]] = {}
        for history in source.get("history", []):
            # The optimizer also records initial task validation in the
            # history list.  It is provenance, not an optimization round,
            # and therefore has neither ``round`` nor candidate records.
            if "round" not in history:
                skipped_non_round_history_entries += 1
                continue
            round_number = int(history["round"])
            for candidate_index, candidate in enumerate(
                history.get("reranked_candidates", []), start=1
            ):
                if not candidate_is_eligible(
                    candidate, args.minimum_nondegraded_families
                ):
                    continue
                optimized = str(candidate["decoded_prompt"]).strip()
                if optimized == str(source["initial_prompt"]).strip():
                    continue
                eligible_before_dedup += 1
                existing = candidates_by_prompt.get(optimized)
                current_key = (
                    int(candidate["training_nondegraded_families"]),
                    float(candidate["proxy_gain"]),
                    -float(candidate["ppl_ratio"]),
                    -round_number,
                    -candidate_index,
                )
                existing_key = None
                if existing is not None:
                    old, old_round, old_index = existing
                    existing_key = (
                        int(old["training_nondegraded_families"]),
                        float(old["proxy_gain"]),
                        -float(old["ppl_ratio"]),
                        -old_round,
                        -old_index,
                    )
                if existing_key is None or current_key > existing_key:
                    candidates_by_prompt[optimized] = (
                        candidate,
                        round_number,
                        candidate_index,
                    )

        unique_candidates = list(candidates_by_prompt.values())
        final_prompt = str(source.get("optimized_prompt", "")).strip()
        chosen: list[tuple[dict[str, Any], int, int]] = []

        # Keep the optimizer's committed final candidate when it is valid.  The
        # second slot is deliberately reserved for a lower-ranked alternative:
        # development validation showed that the maximum proxy candidate can
        # transfer worse than an alternative with a smaller proxy gain.
        if bool(source.get("accepted")) and final_prompt:
            final_match = candidates_by_prompt.get(final_prompt)
            if final_match is not None:
                chosen.append(final_match)

        alternatives = sorted(
            (
                item
                for item in unique_candidates
                if not chosen
                or str(item[0]["decoded_prompt"]).strip()
                != str(chosen[0][0]["decoded_prompt"]).strip()
            ),
            key=lambda item: (
                -int(item[0]["training_nondegraded_families"]),
                -float(item[0]["proxy_gain"]),
                float(item[0]["ppl_ratio"]),
                item[1],
                item[2],
            ),
        )
        chosen.extend(
            alternatives[
                : max(0, args.maximum_candidates_per_prompt - len(chosen))
            ]
        )

        per_source_counts[source_prompt_id] = len(chosen)
        for ordinal, (candidate, round_number, candidate_index) in enumerate(
            chosen, start=1
        ):
            tag = f"portfolio_{ordinal:02d}_r{round_number}_c{candidate_index}"
            output = result_row(
                source,
                candidate,
                round_number=round_number,
                candidate_index=candidate_index,
                candidate_tag=tag,
            )
            outputs.append(output)
            mapping.append(
                {
                    "portfolio_prompt_id": output["prompt_id"],
                    "source_prompt_id": source_prompt_id,
                    "candidate_ordinal": ordinal,
                    "selected_by_inner_optimizer": bool(
                        output["candidate_provenance"][
                            "selected_by_inner_optimizer"
                        ]
                    ),
                    "round": round_number,
                    "candidate_index_one_based": candidate_index,
                }
            )

    write_jsonl(args.output, outputs)
    write_jsonl(args.mapping, mapping)
    report = {
        "source_prompts": len(source_rows),
        "portfolio_candidates": len(outputs),
        "source_prompts_with_candidates": sum(
            count > 0 for count in per_source_counts.values()
        ),
        "source_prompts_without_candidates": sum(
            count == 0 for count in per_source_counts.values()
        ),
        "eligible_candidates_before_exact_prompt_deduplication": (
            eligible_before_dedup
        ),
        "skipped_non_round_history_entries": (
            skipped_non_round_history_entries
        ),
        "maximum_candidates_per_prompt": args.maximum_candidates_per_prompt,
        "minimum_nondegraded_families": args.minimum_nondegraded_families,
        "candidate_counts_by_source": per_source_counts,
        "selection_policy": (
            "committed_final_then_best_distinct_task_valid_alternative"
        ),
        "test_data_used": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
