from __future__ import annotations

import pytest

from llm_integrity.paper_hybrid_sensitivity import fit_robust_calibration
from llm_integrity.paper_hybrid_validation import validate_proxy_candidates


def proxy(prompt_id: str) -> dict:
    return {
        "id": prompt_id,
        "optimization": {"accepted": True},
    }


def micro(prompt_id: str, initial: float, optimized: float) -> list[dict]:
    return [
        {"prompt_id": f"{prompt_id}::initial", "micro_raw": initial},
        {"prompt_id": f"{prompt_id}::optimized", "micro_raw": optimized},
    ]


def macro(prompt_id: str, initial: float, optimized: float) -> list[dict]:
    return [
        {
            "prompt_id": f"{prompt_id}::initial",
            "family": "family",
            "family_weight": 1.0,
            "macro_l2_raw": initial,
        },
        {
            "prompt_id": f"{prompt_id}::optimized",
            "family": "family",
            "family_weight": 1.0,
            "macro_l2_raw": optimized,
        },
    ]


def test_fixed_calibration_makes_score_batch_independent() -> None:
    micro_calibration = fit_robust_calibration([10.0, 20.0, 30.0])
    macro_calibration = fit_robust_calibration([100.0, 200.0, 300.0])
    common = {
        "micro_calibration": micro_calibration,
        "macro_calibration": macro_calibration,
    }
    single, single_report = validate_proxy_candidates(
        [proxy("target")],
        micro("target", 20.0, 22.0),
        macro("target", 200.0, 210.0),
        **common,
    )
    batch, batch_report = validate_proxy_candidates(
        [proxy("target"), proxy("unrelated")],
        micro("target", 20.0, 22.0)
        + micro("unrelated", 1.0, 1000.0),
        macro("target", 200.0, 210.0)
        + macro("unrelated", 1.0, 1000.0),
        **common,
    )

    target_single = single[0]["optimization"]["hybrid_validation"]
    target_batch = batch[0]["optimization"]["hybrid_validation"]
    assert target_single == target_batch
    assert single_report["calibration_source"] == (
        "provided_fixed_calibration"
    )
    assert batch_report["calibration_source"] == (
        "provided_fixed_calibration"
    )


def test_calibrations_must_be_provided_together() -> None:
    calibration = fit_robust_calibration([10.0, 20.0])
    with pytest.raises(ValueError, match="must be provided together"):
        validate_proxy_candidates(
            [proxy("target")],
            micro("target", 10.0, 11.0),
            macro("target", 100.0, 110.0),
            micro_calibration=calibration,
        )
