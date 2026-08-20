from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from llm_integrity.h8_d1c import load_frozen_score_calibration
from llm_integrity.h8_precalibration import h8_rbf_kernel
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement
from llm_integrity.h8_d2a import (
    ATTACK_FAMILIES,
    ATTACK_ROLE,
    INTACT_TARGET_ROLE,
    SAMPLE_STRUCTURES,
    TOP_R_VALUES,
    build_generation_schedules,
    configuration_id,
    configuration_manifest_payload,
    empirical_global_p_value,
    evaluate_development_unit,
    global_permutation_test,
    intact_sanity_summary,
    make_nested_subset_payload,
    max_score_diagnostic,
    select_development_configuration,
    top_r_sum,
    validate_nested_subset_payload,
)


ROOT = Path(__file__).resolve().parents[1]
MMD_ARCHIVE = ROOT / "reproducibility/h8_qwen32b_mmd_precalibration_20260818/mmd_measurement_frozen_v1"
SCORE_ARCHIVE = ROOT / "reproducibility/h8_qwen32b_score_calibration_20260820/score_calibration_frozen_v1"
MMD_SHA = "df406d671dcd0867a679a095eaf52db5111590129024cb012908026a4558a063"
SCORE_SHA = "b57d9d5062b13e9e02624e69bbcf4eb585e716084acf73ab96cf00a733802028"
MMD_COMMIT = "72a72df6917d73f4dbe169b19191b29e06730362"
SCORE_SCHEMA_SHA = "502589bb4abb3dbf38bd10e9aeab37a1f97d1621d2bd8ca28eda94848b5b7000"


def dummy_entries() -> list[dict[str, object]]:
    return [
        {
            "prompt_id": f"prompt_{index:02d}",
            "metadata": {"prompt_sha256": f"{index:064x}"},
        }
        for index in range(12)
    ]


def dummy_attacks() -> list[dict[str, str]]:
    return [
        {"family": family, "attack_instance_id": f"h8d2_dev_{family}_{index}"}
        for family in ATTACK_FAMILIES
        for index in range(2)
    ]


def simple_score_params() -> dict[str, object]:
    return {
        "mapping": "nondegenerate_centered",
        "null_median_m": 0.0,
        "effective_scale": 1.0,
    }


def test_top_r_sort_sum_zero_and_single_extreme() -> None:
    scores = [0.0] * 12
    assert top_r_sum(scores, 2) == 0.0
    assert top_r_sum(scores, 3) == 0.0
    assert top_r_sum(scores, 4) == 0.0
    scores[-1] = 100.0
    assert top_r_sum(scores, 2) == 100.0
    assert top_r_sum(scores, 3) == 100.0
    assert top_r_sum(scores, 4) == 100.0
    assert max_score_diagnostic(scores) == 100.0
    with pytest.raises(ValueError, match="only permits"):
        top_r_sum(scores, 1)


def test_configuration_manifest_contains_only_twelve_top_r_candidates() -> None:
    payload = configuration_manifest_payload()
    assert payload["configuration_count"] == 12
    assert {row["top_r"] for row in payload["configurations"]} == set(TOP_R_VALUES)
    assert {row["sample_structure"] for row in payload["configurations"]} == set(SAMPLE_STRUCTURES)
    assert payload["energy_implemented"] is False
    assert all(row["local_p_values_used"] is False for row in payload["configurations"])


def test_global_permutation_is_deterministic_preserves_sizes_and_has_nonzero_p() -> None:
    kernels = {}
    parameters = {}
    for index in range(12):
        rng = np.random.default_rng(100 + index)
        values = rng.normal(size=(7, 3))
        kernels[f"p{index:02d}"] = h8_rbf_kernel(values, values, 1.0)
        parameters[f"p{index:02d}"] = simple_score_params()
    first = global_permutation_test(
        kernels,
        parameters,
        n_reference=4,
        n_target=3,
        top_r=3,
        permutation_root_seed=12345,
        permutations=39,
        alpha=0.05,
    )
    second = global_permutation_test(
        kernels,
        parameters,
        n_reference=4,
        n_target=3,
        top_r=3,
        permutation_root_seed=12345,
        permutations=39,
        alpha=0.05,
    )
    assert first == second
    assert len(first.observed_scores) == 12
    assert len(first.permutation_top_r) == 39
    assert 0.0 < first.global_p_value <= 1.0
    assert first.global_p_value == empirical_global_p_value(first.observed_top_r, first.permutation_top_r)


def test_development_evaluation_interface_rejects_final_role() -> None:
    with pytest.raises(ValueError, match="development intact or attack"):
        evaluate_development_unit(
            {},
            {},
            data_role="detector_final_heldout_only",
            evaluation_unit_id="final",
            structure="r40_q10",
            top_r=2,
            permutation_root_seed=1,
        )


def test_nested_subset_membership_is_frozen_and_valid() -> None:
    schedules = build_generation_schedules(
        dummy_entries(),
        dummy_attacks(),
        reference_root_seed=11,
        intact_target_root_seed=22,
        attack_root_seed=33,
    )
    payload = make_nested_subset_payload(schedules, subset_root_seed=44)
    validate_nested_subset_payload(payload)
    for row in payload["reference_membership_by_prompt"].values():
        assert set(row["R40"]) < set(row["R60"])
    for unit in payload["target_evaluation_units"]:
        for row in unit["members_by_prompt"].values():
            assert set(row["Q10"]) < set(row["Q20"])
    assert make_nested_subset_payload(schedules, subset_root_seed=44) == payload


def development_attack_records() -> list[dict[str, object]]:
    rows = []
    for structure in SAMPLE_STRUCTURES:
        for top_r in TOP_R_VALUES:
            for family in ATTACK_FAMILIES:
                for endpoint in range(2):
                    rows.append(
                        {
                            "data_role": ATTACK_ROLE,
                            "configuration_id": configuration_id(structure, top_r),
                            "attack_family": family,
                            "attack_instance_id": f"{family}_{endpoint}",
                            "detected": True,
                        }
                    )
    return rows


def test_selection_rule_uses_all_families_and_frozen_tie_break() -> None:
    result = select_development_configuration(development_attack_records())
    assert result["selected_configuration_id"] == "r40_q10_top2"
    assert result["final_or_heldout_data_read"] is False


@pytest.mark.parametrize(
    "role",
    ["detector_final_attack_only", "detector_development_final_attack_only", "final_heldout"],
)
def test_selection_rejects_final_or_heldout_data(role: str) -> None:
    rows = development_attack_records()
    rows[0] = {**rows[0], "data_role": role}
    with pytest.raises(ValueError, match="development attack data only"):
        select_development_configuration(rows)


def test_intact_sanity_is_diagnostic_not_selection() -> None:
    rows = [
        {
            "data_role": INTACT_TARGET_ROLE,
            "configuration_id": configuration_id(structure, top_r),
            "detected": False,
        }
        for structure in SAMPLE_STRUCTURES
        for top_r in TOP_R_VALUES
        for _ in range(5)
    ]
    result = intact_sanity_summary(rows)
    assert result["used_for_configuration_selection"] is False
    assert all(value["false_positive_rate"] == 0.0 for value in result["by_configuration"].values())


def test_tampered_mmd_or_score_frozen_hash_fails_closed(tmp_path: Path) -> None:
    binding = load_frozen_mmd_measurement(
        MMD_ARCHIVE,
        expected_manifest_sha256=MMD_SHA,
        expected_mmd_implementation_commit=MMD_COMMIT,
    )
    load_frozen_score_calibration(
        SCORE_ARCHIVE,
        expected_manifest_sha256=SCORE_SHA,
        binding=binding,
        expected_score_schema_sha256=SCORE_SCHEMA_SHA,
    )
    mmd_copy = tmp_path / "mmd"
    shutil.copytree(MMD_ARCHIVE, mmd_copy)
    (mmd_copy / "MMD_FROZEN_MANIFEST.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_frozen_mmd_measurement(
            mmd_copy,
            expected_manifest_sha256=MMD_SHA,
            expected_mmd_implementation_commit=MMD_COMMIT,
        )
    score_copy = tmp_path / "score"
    shutil.copytree(SCORE_ARCHIVE, score_copy)
    (score_copy / "SCORE_CALIBRATION_PARAMETERS.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_frozen_score_calibration(
            score_copy,
            expected_manifest_sha256=SCORE_SHA,
            binding=binding,
            expected_score_schema_sha256=SCORE_SCHEMA_SHA,
        )
