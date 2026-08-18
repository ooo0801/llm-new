from __future__ import annotations

import pytest

from llm_integrity.h8_sampling import (
    H8_SMOKE_RESPONSE_LIMIT,
    build_smoke_plan,
    completion_stop_metadata,
    exercise_retry_contract,
    run_with_same_seed_retry,
    sha256_text,
)


def entries() -> list[dict]:
    result = []
    for index in range(12):
        prompt = f"prompt {index}"
        result.append(
            {
                "prompt_id": f"p{index}",
                "prompt": prompt,
                "metadata": {"prompt_sha256": sha256_text(prompt)},
            }
        )
    return result


def test_smoke_plan_is_exactly_24_batch1_unique_seeds() -> None:
    plan = build_smoke_plan(
        entries(),
        1000,
        mode="smoke_only",
        data_role="smoke_only_not_calibration",
        batch_size=1,
        formal_calibration_authorized=False,
    )
    assert len(plan) == H8_SMOKE_RESPONSE_LIMIT
    assert len({request.seed for request in plan}) == 24
    assert all(request.response_id.startswith("h8-smoke-") for request in plan)


@pytest.mark.parametrize("seed_base", [-1, 2**32])
def test_smoke_plan_rejects_generation_seed_outside_uint32(seed_base: int) -> None:
    with pytest.raises(ValueError, match="uint32"):
        build_smoke_plan(
            entries(),
            seed_base,
            mode="smoke_only",
            data_role="smoke_only_not_calibration",
            batch_size=1,
            formal_calibration_authorized=False,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "formal_calibration"},
        {"data_role": "mmd_precalibration_fit_only"},
        {"batch_size": 2},
        {"formal_calibration_authorized": True},
    ],
)
def test_smoke_guard_rejects_scope_expansion(overrides: dict) -> None:
    settings = {
        "mode": "smoke_only",
        "data_role": "smoke_only_not_calibration",
        "batch_size": 1,
        "formal_calibration_authorized": False,
    }
    settings.update(overrides)
    with pytest.raises((PermissionError, ValueError)):
        build_smoke_plan(entries(), 1000, **settings)


def test_first_token_eos_is_a_valid_empty_completion() -> None:
    metadata = completion_stop_metadata([151645], [151645, 151643], 64)
    assert metadata["stop_reason"] == "eos"
    assert metadata["legal_first_token_eos"] is True
    assert metadata["response_token_count_including_eos"] == 1


def test_length_and_unknown_stop_reasons() -> None:
    assert completion_stop_metadata([1, 2, 3], [99], 3)["stop_reason"] == "length"
    assert (
        completion_stop_metadata([1, 2], [99], 3)["stop_reason"]
        == "generation_stopped_without_eos_or_length"
    )


def test_same_seed_retry_contract() -> None:
    report = exercise_retry_contract()
    assert report["status"] == "PASS"
    assert report["observed_seeds"] == [998877, 998877]


def test_valid_empty_result_is_not_retried() -> None:
    calls = []

    def attempt(seed: int, attempt_index: int) -> dict:
        calls.append((seed, attempt_index))
        return {"raw_response": "", "legal_first_token_eos": True}

    result, records = run_with_same_seed_retry(attempt, 42, max_attempts=2)
    assert calls == [(42, 0)]
    assert result["raw_response"] == ""
    assert len(records) == 1


def test_retry_exhaustion_never_changes_seed() -> None:
    calls = []

    def attempt(seed: int, attempt_index: int) -> dict:
        calls.append(seed)
        raise OSError("technical")

    with pytest.raises(RuntimeError, match="same-seed attempts"):
        run_with_same_seed_retry(attempt, 77, max_attempts=2)
    assert calls == [77, 77]
