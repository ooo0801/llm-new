from __future__ import annotations

import copy

import numpy as np
import pytest

from llm_integrity.h8_d2a import (
    ATTACK_FAMILIES,
    ATTACK_ROLE,
    INTACT_TARGET_ROLE,
    SAMPLE_STRUCTURES,
    TOP_R_VALUES,
    configuration_manifest_payload,
)
from llm_integrity.h8_d2c import (
    evaluate_all_top_r,
    result_records_for_unit,
    validate_and_select,
)
from llm_integrity.h8_precalibration import canonical_sha256, h8_rbf_kernel


def _parameters() -> dict[str, object]:
    return {
        "effective_scale": 1.0,
        "mapping": "nondegenerate_centered",
        "null_median_m": 0.0,
    }


def _toy_inputs() -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    kernels: dict[str, np.ndarray] = {}
    parameters: dict[str, dict[str, object]] = {}
    for index in range(12):
        values = np.random.default_rng(100 + index).normal(size=(7, 4))
        kernels[f"p{index:02d}"] = h8_rbf_kernel(values, values, 1.2)
        parameters[f"p{index:02d}"] = _parameters()
    return kernels, parameters


def test_multi_top_r_global_permutation_is_deterministic_and_nested() -> None:
    kernels, parameters = _toy_inputs()
    first = evaluate_all_top_r(
        kernels,
        parameters,
        n_reference=4,
        n_target=3,
        permutation_stream_seed=991,
        permutations=39,
    )
    second = evaluate_all_top_r(
        kernels,
        parameters,
        n_reference=4,
        n_target=3,
        permutation_stream_seed=991,
        permutations=39,
    )
    assert first.observed_scores == second.observed_scores
    assert first.derived_seed_set_sha256 == second.derived_seed_set_sha256
    for r in TOP_R_VALUES:
        assert np.array_equal(first.permutation_top_r[r], second.permutation_top_r[r])
        assert 0.0 < first.global_p_values[r] <= 1.0
    assert first.observed_top_r[2] <= first.observed_top_r[3] <= first.observed_top_r[4]
    assert np.all(first.permutation_top_r[2] <= first.permutation_top_r[3])
    assert np.all(first.permutation_top_r[3] <= first.permutation_top_r[4])


def test_multi_top_r_all_zero_scores_produce_p_one() -> None:
    kernels = {f"p{i:02d}": np.ones((6, 6), dtype=np.float64) for i in range(12)}
    parameters = {f"p{i:02d}": _parameters() for i in range(12)}
    result = evaluate_all_top_r(
        kernels,
        parameters,
        n_reference=4,
        n_target=2,
        permutation_stream_seed=7,
        permutations=19,
    )
    assert result.observed_scores == (0.0,) * 12
    assert all(result.global_p_values[r] == 1.0 for r in TOP_R_VALUES)
    assert not any(result.detected.values())


def test_result_records_reject_non_development_roles() -> None:
    kernels, parameters = _toy_inputs()
    result = evaluate_all_top_r(
        kernels,
        parameters,
        n_reference=4,
        n_target=3,
        permutation_stream_seed=2,
        permutations=5,
    )
    with pytest.raises(ValueError, match="development intact/attack"):
        result_records_for_unit(
            result,
            data_role="detector_final_heldout_only",
            evaluation_unit_id="forbidden",
            sample_structure="r40_q10",
            attack_family=None,
            permutations=5,
        )


def _selection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for structure in SAMPLE_STRUCTURES:
        for r in TOP_R_VALUES:
            config = f"{structure}_top{r}"
            for unit in range(5):
                rows.append(
                    {
                        "data_role": INTACT_TARGET_ROLE,
                        "evaluation_unit_id": f"intact_{unit}",
                        "attack_family": None,
                        "configuration_id": config,
                        "detected": False,
                        "global_p_value": 1.0,
                        "measurement_or_score_refit_performed": False,
                        "final_or_heldout_data_read": False,
                    }
                )
            for family in ATTACK_FAMILIES:
                for endpoint in range(2):
                    rows.append(
                        {
                            "data_role": ATTACK_ROLE,
                            "evaluation_unit_id": f"{family}_{endpoint}",
                            "attack_family": family,
                            "configuration_id": config,
                            "detected": config == "r40_q10_top3",
                            "global_p_value": 0.01 if config == "r40_q10_top3" else 1.0,
                            "measurement_or_score_refit_performed": False,
                            "final_or_heldout_data_read": False,
                        }
                    )
    return rows


def test_selection_is_bound_to_frozen_rule_and_chooses_lexicographic_winner() -> None:
    rule = configuration_manifest_payload()["selection_rule"]
    result = validate_and_select(
        _selection_rows(),
        frozen_selection_rule=rule,
        expected_selection_rule_sha256=canonical_sha256(rule),
    )
    assert result["selected_configuration_id"] == "r40_q10_top3"
    assert result["formal_fpr_or_tpr_estimation_performed"] is False


def test_selection_rejects_rule_hash_tampering_and_forbidden_data_flags() -> None:
    rule = configuration_manifest_payload()["selection_rule"]
    with pytest.raises(ValueError, match="rule SHA256"):
        validate_and_select(
            _selection_rows(),
            frozen_selection_rule=rule,
            expected_selection_rule_sha256="0" * 64,
        )
    rows = _selection_rows()
    rows[0]["final_or_heldout_data_read"] = True
    with pytest.raises(ValueError, match="boundaries"):
        validate_and_select(
            rows,
            frozen_selection_rule=rule,
            expected_selection_rule_sha256=canonical_sha256(rule),
        )


def test_selection_rejects_result_count_drift() -> None:
    rule = copy.deepcopy(configuration_manifest_payload()["selection_rule"])
    with pytest.raises(ValueError, match="156"):
        validate_and_select(
            _selection_rows()[:-1],
            frozen_selection_rule=rule,
            expected_selection_rule_sha256=canonical_sha256(rule),
        )
