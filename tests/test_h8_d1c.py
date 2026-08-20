from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from llm_integrity.h8_d1c import (
    AUDIT_ROLE,
    FIT_ROLE,
    STRUCTURE_SIZES,
    audit_fit_parameters,
    deterministic_uint64,
    load_frozen_score_calibration,
    make_split_manifest,
    null_distribution_from_kernel,
    trial_split,
    validate_split_manifest,
)
from llm_integrity.h8_precalibration import h8_mmd2_unbiased_unequal, h8_rbf_kernel
from llm_integrity.h8_score_calibration import (
    SAMPLE_STRUCTURES,
    FrozenMMDMeasurementBinding,
    fit_score_calibration_parameters,
    save_score_calibration_artifact,
    score_schema_sha256,
)


PROMPTS = tuple(f"prompt_{index:02d}" for index in range(12))


def binding(degenerate: set[str] | None = None) -> FrozenMMDMeasurementBinding:
    degenerate = degenerate or set()
    hashes = {prompt: f"s-{prompt}" for prompt in PROMPTS}
    return FrozenMMDMeasurementBinding(
        manifest_sha256="mmd",
        feature_schema_sha256="schema",
        feature_schema_payload_sha256="schema-payload",
        global_exclusion_mask_payload_sha256="mask",
        global_bandwidth_payload_sha256="global-bw",
        mmd_implementation_commit="commit",
        scaler_payload_sha256_by_prompt=hashes,
        bandwidth_payload_sha256_by_prompt={prompt: f"b-{prompt}" for prompt in PROMPTS},
        feature_degenerate_by_prompt={prompt: prompt in degenerate for prompt in PROMPTS},
    )


def synthetic_nulls(scale: float = 1.0) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for s_index, structure in enumerate(SAMPLE_STRUCTURES):
        output[structure] = {}
        for p_index, prompt in enumerate(PROMPTS):
            rng = np.random.Generator(np.random.PCG64(1000 + 100 * s_index + p_index))
            output[structure][prompt] = rng.normal(0.0, scale, 1000)
    return output


def test_split_manifest_is_reproducible_unique_and_role_independent() -> None:
    manifest = make_split_manifest(PROMPTS, fit_root_seed=11, audit_root_seed=22)
    streams = validate_split_manifest(manifest, PROMPTS)
    assert len(streams) == 96
    assert manifest["payload"]["trial_seed_count"] == 96_000
    assert manifest["payload"]["stream_seed_unique"] is True
    assert manifest["payload"]["trial_seed_unique"] is True
    assert streams[(FIT_ROLE, "r40_q10", PROMPTS[0])] != streams[(AUDIT_ROLE, "r40_q10", PROMPTS[0])]


def test_split_manifest_detects_mutation() -> None:
    manifest = make_split_manifest(PROMPTS, fit_root_seed=11, audit_root_seed=22)
    manifest["payload"]["streams"][0]["stream_seed_uint64"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_split_manifest(manifest, PROMPTS)


@pytest.mark.parametrize("structure", SAMPLE_STRUCTURES)
def test_trial_split_is_unique_disjoint_and_correct_size(structure: str) -> None:
    n_reference, n_target = STRUCTURE_SIZES[structure]
    reference, target = trial_split(100, n_reference, n_target, 1234)
    assert len(reference) == n_reference
    assert len(target) == n_target
    assert len(set(reference.tolist())) == n_reference
    assert len(set(target.tolist())) == n_target
    assert not set(reference.tolist()) & set(target.tolist())


def test_kernel_null_preserves_negative_and_matches_direct_mmd() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(100, 4))
    kernel = h8_rbf_kernel(values, values, 1.3)
    null = null_distribution_from_kernel(
        kernel, stream_seed=42, n_reference=40, n_target=10, trials=1000
    )
    reference, target = trial_split(100, 40, 10, deterministic_uint64(42, "trial", 0))
    direct = h8_mmd2_unbiased_unequal(values[reference], values[target], 1.3)
    assert null.dtype == np.float64
    assert null[0] == pytest.approx(direct, abs=1e-15)
    assert np.any(null < 0.0)


def test_fit_and_independent_audit_pass_without_rewriting_fit() -> None:
    frozen = binding()
    fit_nulls = synthetic_nulls(1.0)
    audit_nulls = synthetic_nulls(1.1)
    fit = fit_score_calibration_parameters(fit_nulls, frozen, data_role=FIT_ROLE)
    before = repr(fit)
    result = audit_fit_parameters(fit, audit_nulls, frozen)
    assert result.status == "PASS"
    assert not result.failures
    assert repr(fit) == before


def test_audit_fails_scale_gate_without_refitting() -> None:
    frozen = binding()
    fit = fit_score_calibration_parameters(synthetic_nulls(1.0), frozen, data_role=FIT_ROLE)
    audit_nulls = synthetic_nulls(3.0)
    result = audit_fit_parameters(fit, audit_nulls, frozen)
    assert result.status == "FAIL"
    assert result.failures


def test_structural_degenerate_zero_invariant_and_score_zero() -> None:
    degenerate = {PROMPTS[0]}
    frozen = binding(degenerate)
    fit_nulls = synthetic_nulls(1.0)
    audit_nulls = synthetic_nulls(1.0)
    for structure in SAMPLE_STRUCTURES:
        fit_nulls[structure][PROMPTS[0]] = np.zeros(1000)
        audit_nulls[structure][PROMPTS[0]] = np.zeros(1000)
    fit = fit_score_calibration_parameters(fit_nulls, frozen, data_role=FIT_ROLE)
    result = audit_fit_parameters(fit, audit_nulls, frozen)
    assert result.status == "PASS"
    for structure in SAMPLE_STRUCTURES:
        row = result.structures[structure]["prompt_audits"][PROMPTS[0]]
        assert row["zero_invariant_pass"] is True
        assert row["max_abs_score"] == 0.0


def test_structural_degenerate_nonzero_audit_fails() -> None:
    frozen = binding({PROMPTS[0]})
    fit_nulls = synthetic_nulls(1.0)
    audit_nulls = synthetic_nulls(1.0)
    for structure in SAMPLE_STRUCTURES:
        fit_nulls[structure][PROMPTS[0]] = np.zeros(1000)
        audit_nulls[structure][PROMPTS[0]] = np.full(1000, 2e-12)
    fit = fit_score_calibration_parameters(fit_nulls, frozen, data_role=FIT_ROLE)
    result = audit_fit_parameters(fit, audit_nulls, frozen)
    assert result.status == "FAIL"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_score_loader_is_fail_closed(tmp_path: Path) -> None:
    frozen = binding()
    payload = fit_score_calibration_parameters(synthetic_nulls(), frozen, data_role=FIT_ROLE)
    payload.update(
        {
            "artifact_status": "frozen",
            "audit_data_used_for_parameter_fit": False,
            "d1b1_responses_sha256": "responses",
        }
    )
    parameters = tmp_path / "SCORE_CALIBRATION_PARAMETERS.json"
    payload_sha = save_score_calibration_artifact(parameters, payload)
    manifest = {
        "measurement_layer": "frozen",
        "score_layer": "frozen",
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "audit_data_used_for_parameter_fit": False,
        "new_model_responses": 0,
        "formal_reference_responses": 0,
        "heldout_responses": 0,
        "attack_responses": 0,
        "mmd_manifest_sha256": frozen.manifest_sha256,
        "score_schema_sha256": score_schema_sha256(1e-12),
        "d1b1_responses_sha256": "responses",
        "score_parameter_payload_sha256": payload_sha,
        "files": {parameters.name: file_sha(parameters)},
    }
    manifest_path = tmp_path / "SCORE_CALIBRATION_FROZEN_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_sha = file_sha(manifest_path)
    loaded = load_frozen_score_calibration(
        tmp_path,
        expected_manifest_sha256=manifest_sha,
        binding=frozen,
        expected_score_schema_sha256=score_schema_sha256(1e-12),
    )
    assert loaded["artifact_status"] == "frozen"
    parameters.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_frozen_score_calibration(
            tmp_path,
            expected_manifest_sha256=manifest_sha,
            binding=frozen,
            expected_score_schema_sha256=score_schema_sha256(1e-12),
        )
