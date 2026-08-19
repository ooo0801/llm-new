from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_integrity.h8_precalibration import canonical_sha256
from llm_integrity.h8_score_calibration import (
    SAMPLE_STRUCTURES,
    fit_score_calibration_parameters,
    load_frozen_mmd_measurement,
    load_score_calibration_artifact,
    save_score_calibration_artifact,
    score_schema_sha256,
    score_value,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = (
    ROOT
    / "reproducibility"
    / "h8_qwen32b_mmd_precalibration_20260818"
    / "mmd_measurement_frozen_v1"
)
MANIFEST_SHA = "df406d671dcd0867a679a095eaf52db5111590129024cb012908026a4558a063"
MMD_COMMIT = "72a72df6917d73f4dbe169b19191b29e06730362"


@pytest.fixture(scope="module")
def binding():
    return load_frozen_mmd_measurement(
        ARCHIVE,
        expected_manifest_sha256=MANIFEST_SHA,
        expected_mmd_implementation_commit=MMD_COMMIT,
    )


def synthetic_nulls(binding, scale_multiplier_by_structure: dict[str, float] | None = None):
    multipliers = scale_multiplier_by_structure or {structure: 1.0 for structure in SAMPLE_STRUCTURES}
    output = {}
    for structure in SAMPLE_STRUCTURES:
        prompt_values = {}
        for prompt_index, prompt_id in enumerate(binding.prompt_ids):
            if binding.feature_degenerate_by_prompt[prompt_id]:
                values = np.zeros(20, dtype=np.float64)
            else:
                values = (
                    np.linspace(-1.0, 1.0, 20, dtype=np.float64)
                    * multipliers[structure]
                    + prompt_index * 0.01
                )
            prompt_values[prompt_id] = values.tolist()
        output[structure] = prompt_values
    return output


def test_frozen_mmd_loader_validates_all_bindings(binding) -> None:
    assert binding.manifest_sha256 == MANIFEST_SHA
    assert binding.mmd_implementation_commit == MMD_COMMIT
    assert len(binding.feature_schema_payload_sha256) == 64
    assert len(binding.prompt_ids) == 12
    assert sum(binding.feature_degenerate_by_prompt.values()) == 6
    assert set(binding.scaler_payload_sha256_by_prompt) == set(binding.bandwidth_payload_sha256_by_prompt)


def test_modified_frozen_mmd_file_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "frozen"
    shutil.copytree(ARCHIVE, copied)
    implementation = copied / "implementation" / "h8_m0ij.py"
    implementation.write_text(implementation.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen MMD file hash mismatch"):
        load_frozen_mmd_measurement(
            copied,
            expected_manifest_sha256=MANIFEST_SHA,
            expected_mmd_implementation_commit=MMD_COMMIT,
        )


def test_one_sided_nondegenerate_toy_mapping() -> None:
    parameters = {
        "mapping": "nondegenerate_centered",
        "null_median_m": 2.0,
        "effective_scale": 0.5,
    }
    assert score_value(2.0, parameters) == 0.0
    assert score_value(-4.0, parameters) == 0.0
    assert 0.0 < score_value(2.25, parameters) < score_value(2.5, parameters) < score_value(3.0, parameters)


def test_degenerate_toy_mapping_and_finite_scores() -> None:
    parameters = {
        "mapping": "degenerate_global_scale",
        "null_median_m": 0.0,
        "effective_scale": 0.25,
    }
    values = [score_value(value, parameters) for value in (-1.0, 0.0, 0.1, 1.0)]
    assert values[:2] == [0.0, 0.0]
    assert values[2] > 0.0
    assert np.isfinite(values).all()


def test_fit_preserves_four_independent_structures_and_fallback(binding) -> None:
    multipliers = {"r40_q10": 1.0, "r40_q20": 2.0, "r60_q10": 3.0, "r60_q20": 4.0}
    payload = fit_score_calibration_parameters(
        synthetic_nulls(binding, multipliers),
        binding,
        data_role="development_dry_run",
    )
    assert set(payload["structures"]) == set(SAMPLE_STRUCTURES)
    assert payload["eligible_to_freeze"] is False
    scales = [payload["structures"][structure]["a_global"] for structure in SAMPLE_STRUCTURES]
    assert scales == sorted(scales)
    assert len(set(scales)) == 4
    for structure in SAMPLE_STRUCTURES:
        for prompt_id, parameters in payload["structures"][structure]["prompt_parameters"].items():
            if binding.feature_degenerate_by_prompt[prompt_id]:
                assert parameters["mapping"] == "degenerate_global_scale"
                assert parameters["own_robust_scale_a"] == 0.0
                assert parameters["effective_scale"] == payload["structures"][structure]["a_global"]
            else:
                assert parameters["mapping"] == "nondegenerate_centered"
                assert parameters["effective_scale"] == parameters["own_robust_scale_a"]


@pytest.mark.parametrize("role", ["target", "attack", "heldout", "formal_reference"])
def test_forbidden_roles_cannot_fit_parameters(binding, role: str) -> None:
    with pytest.raises(ValueError, match="cannot fit score parameters"):
        fit_score_calibration_parameters(synthetic_nulls(binding), binding, data_role=role)


def test_nondegenerate_zero_scale_fails_instead_of_using_epsilon(binding) -> None:
    nulls = synthetic_nulls(binding)
    nondegenerate = next(
        prompt_id for prompt_id in binding.prompt_ids if not binding.feature_degenerate_by_prompt[prompt_id]
    )
    nulls["r40_q10"][nondegenerate] = [0.0] * 20
    with pytest.raises(ValueError, match="Nondegenerate prompt scale is numerically zero"):
        fit_score_calibration_parameters(nulls, binding, data_role="development_dry_run")


def test_structurally_degenerate_nonzero_scale_fails_closed(binding) -> None:
    nulls = synthetic_nulls(binding)
    degenerate = next(prompt_id for prompt_id in binding.prompt_ids if binding.feature_degenerate_by_prompt[prompt_id])
    nulls["r60_q20"][degenerate] = np.linspace(0.0, 1.0, 20).tolist()
    with pytest.raises(ValueError, match="Structurally degenerate prompt has nonzero robust scale"):
        fit_score_calibration_parameters(nulls, binding, data_role="development_dry_run")


def test_canonical_serialization_hash_and_fail_closed_loading(binding, tmp_path: Path) -> None:
    payload = fit_score_calibration_parameters(
        synthetic_nulls(binding), binding, data_role="development_dry_run"
    )
    path = tmp_path / "score.json"
    payload_sha = save_score_calibration_artifact(path, payload)
    loaded = load_score_calibration_artifact(
        path,
        binding=binding,
        expected_payload_sha256=payload_sha,
        expected_score_schema_sha256=score_schema_sha256(1e-12),
    )
    assert canonical_sha256(loaded) == payload_sha
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["sample_size_selection_performed"] = True
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        load_score_calibration_artifact(
            path,
            binding=binding,
            expected_payload_sha256=payload_sha,
            expected_score_schema_sha256=score_schema_sha256(1e-12),
        )


def test_validly_rehashed_but_wrong_mmd_binding_is_rejected(binding, tmp_path: Path) -> None:
    payload = fit_score_calibration_parameters(
        synthetic_nulls(binding), binding, data_role="development_dry_run"
    )
    payload["global_bandwidth_payload_sha256"] = "0" * 64
    path = tmp_path / "wrong-binding.json"
    payload_sha = save_score_calibration_artifact(path, payload)
    with pytest.raises(ValueError, match="different global bandwidth"):
        load_score_calibration_artifact(
            path,
            binding=binding,
            expected_payload_sha256=payload_sha,
            expected_score_schema_sha256=score_schema_sha256(1e-12),
        )


def test_raw_negative_values_are_not_mutated_by_fit(binding) -> None:
    nulls = synthetic_nulls(binding)
    prompt_id = next(prompt_id for prompt_id in binding.prompt_ids if not binding.feature_degenerate_by_prompt[prompt_id])
    before = list(nulls["r40_q10"][prompt_id])
    assert any(value < 0.0 for value in before)
    fit_score_calibration_parameters(nulls, binding, data_role="development_dry_run")
    assert nulls["r40_q10"][prompt_id] == before


def test_d1a_config_and_runner_forbid_sampling_and_selection() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "h8_d1a_score_calibration_preflight.yaml").read_text(encoding="utf-8")
    )
    for key in (
        "model_sampling_authorized",
        "formal_reference_access_authorized",
        "heldout_access_authorized",
        "attack_access_authorized",
    ):
        assert config[key] is False
    assert config["score_calibration"]["sample_size_selection_performed"] is False
    assert config["score_calibration"]["top_r_selection_performed"] is False
    assert config["score_calibration"]["final_parameter_fit_performed"] is False
    assert config["development_dry_run"]["eligible_to_freeze"] is False
    source = (ROOT / "scripts" / "run_h8_d1a_score_preflight.py").read_text(encoding="utf-8")
    assert "load_model(" not in source
    assert ".generate(" not in source
    assert "formal_reference_read_or_generated\": False" in source
    assert "final_score_parameter_fit_performed\": False" in source
