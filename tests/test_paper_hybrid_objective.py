import numpy as np
import torch

from llm_integrity.paper_hybrid_sensitivity import (
    RobustCalibration,
    combine_hybrid_scalar,
)
from llm_integrity.paper_hybrid_validation import (
    validate_proxy_candidates,
)
from llm_integrity.paper_prompt_objective import (
    RobustHybridPromptObjective,
)


def calibration() -> RobustCalibration:
    return RobustCalibration(center=0.0, scale=1.0, sample_count=4)


def test_adaptive_alpha_tracks_robust_sensitivity_ratio():
    micro_dominant = combine_hybrid_scalar(
        micro_raw=20.0,
        macro_raw=1.0,
        micro_calibration=calibration(),
        macro_calibration=calibration(),
        weighting_mode="adaptive",
        hybrid_beta=1.0,
    )
    macro_dominant = combine_hybrid_scalar(
        micro_raw=1.0,
        macro_raw=20.0,
        micro_calibration=calibration(),
        macro_calibration=calibration(),
        weighting_mode="adaptive",
        hybrid_beta=1.0,
    )
    assert micro_dominant.micro_weight > 0.5
    assert macro_dominant.micro_weight < 0.5
    assert np.isclose(
        micro_dominant.micro_weight,
        1.0 - macro_dominant.micro_weight,
    )


def test_soft_adaptive_objective_retains_gradient_chain():
    objective = RobustHybridPromptObjective(
        micro_calibration=calibration(),
        macro_calibration=calibration(),
        weighting_mode="adaptive",
        hybrid_beta=1.0,
    )
    micro = torch.tensor(2.0, requires_grad=True)
    macro = torch.tensor(3.0, requires_grad=True)
    result = objective.score_soft(micro_raw=micro, macro_raw=macro)
    result.objective.backward()
    assert micro.grad is not None
    assert macro.grad is not None
    assert torch.isfinite(micro.grad)
    assert torch.isfinite(macro.grad)


def test_outer_hard_validation_controls_final_acceptance():
    proxy_rows = [
        {
            "id": "p1",
            "optimization": {
                "accepted": True,
                "proxy_accepted": True,
            },
        },
        {
            "id": "p2",
            "optimization": {
                "accepted": True,
                "proxy_accepted": True,
            },
        },
    ]
    micro_records = [
        {"prompt_id": "p1::initial", "micro_raw": 1.0},
        {"prompt_id": "p1::optimized", "micro_raw": 8.0},
        {"prompt_id": "p2::initial", "micro_raw": 2.0},
        {"prompt_id": "p2::optimized", "micro_raw": 0.2},
    ]
    macro_records = []
    macro_values = {
        "p1::initial": 1.0,
        "p1::optimized": 8.0,
        "p2::initial": 2.0,
        "p2::optimized": 0.2,
    }
    for prompt_id, value in macro_values.items():
        for family in ("pruning", "quantization"):
            macro_records.append(
                {
                    "prompt_id": prompt_id,
                    "family": family,
                    "family_weight": 0.5,
                    "macro_l2_raw": value,
                }
            )

    outputs, report = validate_proxy_candidates(
        proxy_rows,
        micro_records,
        macro_records,
        weighting_mode="adaptive",
        hybrid_beta=1.0,
    )
    assert outputs[0]["optimization"]["accepted"] is True
    assert outputs[1]["optimization"]["accepted"] is False
    assert report["accepted"] == 1
    assert all(
        row["optimization"]["acceptance_stage"]
        == "full_hybrid_hard_validation"
        for row in outputs
    )
