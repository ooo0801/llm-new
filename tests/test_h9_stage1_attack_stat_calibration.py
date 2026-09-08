from __future__ import annotations

import numpy as np

from run_h9_stage1_attack_stat_calibration import (
    bigram_js,
    calibrate_composite,
    evaluate_task,
    observed_composite,
    prefix10_tv,
)


def test_raw_distances_are_zero_for_identical_groups() -> None:
    rows = ["甲乙丙", "甲乙丁"]
    assert prefix10_tv(rows, rows) == 0.0
    assert bigram_js(rows, rows) == 0.0


def test_prefix_tv_detects_disjoint_prefixes() -> None:
    assert prefix10_tv(["甲甲"], ["乙乙"]) > 0.0


def test_composite_excludes_degenerate_component_and_calibrates_max() -> None:
    fit = [
        {"variable_a": float(index), "variable_b": float(2 * index), "constant": 1.0}
        for index in range(1, 30)
    ]
    evaluation = [
        {"variable_a": float(index), "variable_b": float(2 * index), "constant": 1.0}
        for index in range(2, 31)
    ]
    calibration = calibrate_composite(fit, evaluation, epsilon=1e-12, alpha=0.05)
    assert calibration["evaluable"] is True
    assert calibration["excluded_degenerate_components"] == ["constant"]
    result = observed_composite(
        calibration,
        {"variable_a": 100.0, "variable_b": 10.0, "constant": 999999.0},
    )
    assert result["z"] == result["component_z"]["variable_a"]
    assert "constant" not in result["component_z"]


def test_all_degenerate_composite_is_explicitly_unevaluable() -> None:
    rows = [{"a": 1.0, "b": 2.0} for _ in range(20)]
    calibration = calibrate_composite(rows, rows, epsilon=1e-12, alpha=0.05)
    assert calibration["evaluable"] is False
    assert observed_composite(calibration, {"a": 10.0, "b": 20.0})["z"] is None


def test_task_evaluators_cover_frozen_prompt_types() -> None:
    assert evaluate_task("1,2,3,4,5,6", {"evaluator": "exact", "expected_answer": "1,2,3,4,5,6"})
    assert evaluate_task("答案是360。", {"evaluator": "numeric", "expected_answer": "360"})
    assert evaluate_task(
        '{"name":"x","level":1,"enabled":true}',
        {"evaluator": "json", "expected_contains": ["name", "level", "enabled"]},
    )

