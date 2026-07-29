from __future__ import annotations

import json
import sys
from pathlib import Path

import build_discrete_candidate_portfolio as build
import select_dev_portfolio_candidates as select


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def candidate(prompt: str, gain: float, index: int) -> dict:
    return {
        "decoded_prompt": prompt,
        "constraints_passed": True,
        "task_validation": {"task_passed": True},
        "proxy_gain": gain,
        "ppl": 2.0 + index,
        "ppl_ratio": 1.0 + index / 10,
        "edit_ratio": 0.1,
        "training_nondegraded_families": 3,
    }


def source_row() -> dict:
    final = candidate("final candidate", 10.0, 1)
    alternative = candidate("transfer candidate", 0.2, 2)
    return {
        "prompt_id": "p1",
        "category": "knowledge",
        "initial_prompt": "initial prompt",
        "optimized_prompt": "final candidate",
        "accepted": True,
        "initial_ppl": 2.0,
        "active_user_tokens": 10,
        "preserved_overflow_tokens": 0,
        "rounds_requested": 3,
        "optimization_max_length": 128,
        "initial_task_passed": True,
        "history": [
            {
                "stage": "initial_task_validation",
                "task_passed": True,
                "evaluation_rule": "contains",
                "generated_text": "ok",
            },
            {
                "round": 1,
                "reranked_candidates": [final, alternative],
            }
        ],
    }


def strict_row(portfolio: dict, *, gain: float, accepted: bool) -> dict:
    return {
        "id": portfolio["prompt_id"],
        "prompt_id": portfolio["prompt_id"],
        "optimization": {
            "accepted": accepted,
            "initial_prompt": portfolio["initial_prompt"],
            "optimized_prompt": portfolio["optimized_prompt"],
            "final_ppl": portfolio["final_ppl"],
            "ppl_ratio": portfolio["ppl_ratio"],
            "edit_ratio": portfolio["edit_ratio"],
            "edit_count": portfolio["edit_count"],
            "proxy_objective_gain": portfolio["proxy_objective_gain"],
            "hybrid_validation": {"objective_gain": gain},
            "strict_validation": {"nondegraded_families": 5},
        },
    }


def test_portfolio_keeps_final_and_lower_proxy_transfer_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    source_path = tmp_path / "source.jsonl"
    portfolio_path = tmp_path / "portfolio.jsonl"
    mapping_path = tmp_path / "mapping.jsonl"
    build_report = tmp_path / "build_report.json"
    write_jsonl(source_path, [source_row()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            "--input",
            str(source_path),
            "--output",
            str(portfolio_path),
            "--mapping",
            str(mapping_path),
            "--report",
            str(build_report),
            "--maximum-candidates-per-prompt",
            "2",
        ],
    )
    build.main()
    portfolio = build.read_jsonl(portfolio_path)
    report = json.loads(build_report.read_text(encoding="utf-8"))
    assert report["skipped_non_round_history_entries"] == 1
    assert [row["optimized_prompt"] for row in portfolio] == [
        "final candidate",
        "transfer candidate",
    ]

    strict_path = tmp_path / "strict.jsonl"
    write_jsonl(
        strict_path,
        [
            strict_row(portfolio[0], gain=-0.1, accepted=False),
            strict_row(portfolio[1], gain=0.9, accepted=True),
        ],
    )
    all_path = tmp_path / "all.jsonl"
    accepted_path = tmp_path / "accepted.jsonl"
    select_report = tmp_path / "select_report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--source-results",
            str(source_path),
            "--portfolio-results",
            str(portfolio_path),
            "--mapping",
            str(mapping_path),
            "--strict-validation",
            str(strict_path),
            "--output-all",
            str(all_path),
            "--output-accepted",
            str(accepted_path),
            "--report",
            str(select_report),
            "--required-accepted",
            "1",
        ],
    )
    select.main()
    accepted = select.read_jsonl(accepted_path)
    assert len(accepted) == 1
    assert accepted[0]["optimized_prompt"] == "transfer candidate"
    assert accepted[0]["development_selection"][
        "fixed_hybrid_validation"
    ]["objective_gain"] == 0.9
