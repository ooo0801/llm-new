import copy
import json
from pathlib import Path

import pytest
from run_stage1_r2_small import small_summary
from llm_integrity.stage1_r2 import schedule


def test_small_scope_is_exact_prefix_and_caps_additional_sampling():
    root = Path(__file__).resolve().parents[1]
    large = json.loads((root / "configs/stage1_r2_independent_null.json").read_text())
    small = copy.deepcopy(large)
    small["sampling"]["validation_units_per_prompt"] = 5
    prompts = [{"id": str(i)} for i in range(12)]
    old = schedule(large, prompts, "validation")
    new = schedule(small, prompts, "validation")
    assert new == old[:2880]
    assert len(new) - 808 == 2072


def test_small_summary_cannot_claim_original_gate_or_zero_FPR():
    prompts = [{"id": str(i)} for i in range(12)]
    rows = [{"prompt_id": p["id"], "unit": i, "raw": {"detected": False}, "h8": {"detected": False}}
            for p in prompts for i in range(5)]
    result = small_summary(rows, prompts, {})
    assert result["status"] == "EXPLORATORY_SMALL_CHECK_COMPLETE"
    assert result["original_large_sample_acceptance_applied"] is False
    assert result["formal_FPR_calibration_pass"] is None
    assert result["per_prompt"][0]["pointwise_CP95_interval"][1] > .5
    rows[0]["raw"]["detected"] = None
    assert small_summary(rows, prompts, {})["technical_status"] == "FAIL"
    with pytest.raises(ValueError):
        small_summary(rows[:-1], prompts, {})
