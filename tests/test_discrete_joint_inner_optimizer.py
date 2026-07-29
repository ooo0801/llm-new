from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_integrity.discrete_joint_inner_optimizer import (
    aggregate_family_trace_scores,
    family_anchor_nondegradation_checks,
    family_nondegradation_checks,
    hotflip_top_candidates,
    merge_restart_candidate_pools,
    resolve_final_task_status,
    select_position_diverse_candidates,
    token_surface_class,
    token_surfaces_compatible,
)
from llm_integrity.inner_variant_sampler import FAMILIES


def test_hotflip_uses_objective_direction_not_nearest_neighbor() -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    gradient = torch.tensor([[1.0, 0.0]])

    candidates = hotflip_top_candidates(
        embeddings,
        gradient,
        current_ids=[0],
        editable_positions=[0],
        candidates_per_position=2,
    )

    assert candidates[0].candidate_token_id == 3
    assert candidates[0].linear_gain == pytest.approx(2.0)
    assert candidates[1].candidate_token_id == 1
    assert candidates[1].linear_gain == pytest.approx(1.0)


def test_hotflip_excludes_forbidden_and_current_tokens() -> None:
    embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    gradient = torch.tensor([[1.0, 0.0]])

    candidates = hotflip_top_candidates(
        embeddings,
        gradient,
        current_ids=[0],
        editable_positions=[0],
        candidates_per_position=2,
        forbidden_token_ids={3},
    )

    assert [item.candidate_token_id for item in candidates] == [1, 2]
    assert all(item.candidate_token_id != 0 for item in candidates)


def test_rerank_shortlist_covers_positions_before_duplicates() -> None:
    candidates = [
        {"position": 1, "candidate_token_id": 101, "linear_gain": 9.0},
        {"position": 1, "candidate_token_id": 102, "linear_gain": 8.0},
        {"position": 2, "candidate_token_id": 201, "linear_gain": 7.0},
        {"position": 3, "candidate_token_id": 301, "linear_gain": 6.0},
        {"position": 2, "candidate_token_id": 202, "linear_gain": 5.0},
    ]

    selected = select_position_diverse_candidates(candidates, limit=4)

    assert [
        (item["position"], item["candidate_token_id"])
        for item in selected
    ] == [(1, 101), (2, 201), (3, 301), (1, 102)]


def test_rerank_shortlist_requires_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        select_position_diverse_candidates([], limit=0)


def test_restart_candidate_merge_deduplicates_and_keeps_provenance() -> None:
    first = {
        "token_ids": [1, 20, 3],
        "position": 1,
        "candidate_token_id": 20,
        "linear_gain": 2.0,
    }
    duplicate = {**first, "linear_gain": 3.0}
    other = {
        "token_ids": [10, 2, 3],
        "position": 0,
        "candidate_token_id": 10,
        "linear_gain": 1.0,
    }

    merged = merge_restart_candidate_pools([[first], [duplicate, other]])

    assert len(merged) == 2
    assert merged[0]["linear_gain"] == pytest.approx(3.0)
    assert merged[0]["search_restarts"] == [1, 2]
    assert merged[0]["restart_linear_gains"] == {"1": 2.0, "2": 3.0}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" 前", "cjk"),
        (" square", "latin"),
        ("。", "punctuation_symbol"),
        ("  ", "whitespace"),
        ("5", "digit_mixed"),
        ("条item", "mixed_cjk_latin"),
    ],
)
def test_token_surface_class(text: str, expected: str) -> None:
    assert token_surface_class(text) == expected


def test_surface_filter_rejects_cross_script_replacements() -> None:
    assert token_surfaces_compatible("前", "把")
    assert token_surfaces_compatible(" code", "bat")
    assert token_surfaces_compatible("。", "！")
    assert not token_surfaces_compatible("前", " c")
    assert not token_surfaces_compatible("代码", "bat")
    assert not token_surfaces_compatible("。", "ing")
    assert not token_surfaces_compatible("5", "6")


def test_family_nondegradation_requires_familywise_comparison() -> None:
    baseline = {
        "unstructured_pruning": 10.0,
        "structured_pruning": 10.0,
        "quantization": 10.0,
        "gaussian_noise": 10.0,
        "finetuning": 10.0,
    }
    candidate = {
        "unstructured_pruning": 11.0,
        "structured_pruning": 9.95,
        "quantization": 10.0,
        "gaussian_noise": 8.0,
        "finetuning": 7.0,
    }

    count, checks = family_nondegradation_checks(
        baseline,
        candidate,
        relative_tolerance=0.01,
    )

    assert count == 3
    assert checks["structured_pruning"]["nondegraded"]
    assert not checks["gaussian_noise"]["nondegraded"]


def test_family_nondegradation_rejects_missing_families() -> None:
    with pytest.raises(ValueError, match="all five families"):
        family_nondegradation_checks(
            {"quantization": 1.0},
            {"quantization": 1.0},
            relative_tolerance=0.01,
        )


def test_anchor_gate_requires_every_variant_inside_family() -> None:
    baseline = {
        family: [
            {"variant_id": f"{family}-a", "normalized": 1.0},
            {"variant_id": f"{family}-b", "normalized": 2.0},
        ]
        for family in FAMILIES
    }
    candidate = {
        family: [
            {"variant_id": f"{family}-a", "normalized": 1.1},
            {"variant_id": f"{family}-b", "normalized": 2.1},
        ]
        for family in FAMILIES
    }
    candidate["quantization"][1]["normalized"] = 1.0

    count, checks = family_anchor_nondegradation_checks(
        baseline,
        candidate,
        relative_tolerance=0.01,
    )

    assert count == 4
    assert not checks["quantization"]["nondegraded"]
    assert checks["quantization"]["aggregation"] == "all_anchor_variants"


def test_rejected_search_reports_rolled_back_initial_task_status() -> None:
    status = resolve_final_task_status(
        require_task_preservation=True,
        initial_task_passed=True,
        accepted=False,
        history=[],
    )

    assert status is True


def test_accepted_search_requires_every_committed_task_check() -> None:
    status = resolve_final_task_status(
        require_task_preservation=True,
        initial_task_passed=True,
        accepted=True,
        history=[
            {
                "committed": True,
                "committed_task_validation": {"task_passed": True},
            },
            {
                "committed": True,
                "committed_task_validation": {"task_passed": False},
            },
        ],
    )

    assert status is False


def test_task_status_is_not_claimed_when_guard_is_disabled() -> None:
    status = resolve_final_task_status(
        require_task_preservation=False,
        initial_task_passed=True,
        accepted=True,
        history=[],
    )

    assert status is None


def test_family_trace_scores_are_averaged_within_family() -> None:
    traces = [
        {
            "family": family,
            "macro_score_normalized": float(index + offset),
        }
        for index, family in enumerate(
            (
                "unstructured_pruning",
                "structured_pruning",
                "quantization",
                "gaussian_noise",
                "finetuning",
            ),
            start=1,
        )
        for offset in (0, 2)
    ]

    scores = aggregate_family_trace_scores(
        traces,
        score_key="macro_score_normalized",
    )

    assert scores["unstructured_pruning"] == pytest.approx(2.0)
    assert scores["finetuning"] == pytest.approx(6.0)
