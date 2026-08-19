from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_integrity.h8_m0ij import (
    deterministic_uint64,
    frozen_split,
    mmd2_biased_from_kernel,
    mmd2_unbiased_from_kernel,
    numerical_summary,
    permutation_sanity,
)
from llm_integrity.h8_precalibration import (
    h8_mmd2_biased,
    h8_mmd2_unbiased_unequal,
    h8_rbf_kernel,
    load_h8_artifact,
    save_h8_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_unequal_size_kernel_implementation_and_symmetry() -> None:
    x = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float64)
    y = np.asarray([[1.0], [3.0]], dtype=np.float64)
    pooled = np.vstack([x, y])
    kernel = h8_rbf_kernel(pooled, pooled, 1.1)
    indexed = mmd2_unbiased_from_kernel(kernel, [0, 1, 2], [3, 4])
    direct = h8_mmd2_unbiased_unequal(x, y, 1.1)
    assert indexed == pytest.approx(direct, abs=1e-15)
    assert direct == pytest.approx(h8_mmd2_unbiased_unequal(y, x, 1.1), abs=1e-15)
    assert mmd2_biased_from_kernel(kernel, [0, 1, 2], [3, 4]) == pytest.approx(
        h8_mmd2_biased(x, y, 1.1), abs=1e-15
    )


def test_negative_unbiased_value_is_preserved() -> None:
    x = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float64)
    y = np.asarray([[1.0], [3.0]], dtype=np.float64)
    value = h8_mmd2_unbiased_unequal(x, y, 1.1)
    assert value < 0.0


def test_degenerate_invariant_and_permutation_p_one() -> None:
    kernel = np.ones((100, 100), dtype=np.float64)
    ref, target = frozen_split(100, 40, 20, 7)
    assert abs(mmd2_unbiased_from_kernel(kernel, ref, target)) <= 1e-12
    assert abs(mmd2_biased_from_kernel(kernel, ref, target)) <= 1e-12
    result = permutation_sanity(kernel, ref, target, permutation_seed=11, permutations=99)
    assert result.observed == pytest.approx(0.0, abs=1e-12)
    assert max(map(abs, result.permutation_values)) <= 1e-12
    assert result.p_value == 1.0
    assert result.observed_evaluation_count == 1


def test_permutation_is_reproducible_and_valid() -> None:
    values = np.arange(100, dtype=np.float64)[:, None]
    kernel = h8_rbf_kernel(values, values, 12.0)
    ref, target = frozen_split(100, 60, 10, 42)
    first = permutation_sanity(kernel, ref, target, permutation_seed=123, permutations=99)
    second = permutation_sanity(kernel, ref, target, permutation_seed=123, permutations=99)
    assert first == second
    assert first.reproducible and first.group_size_correct
    assert 0.0 < first.p_value <= 1.0
    assert len(first.permutation_values) == 99


def test_seed_and_split_are_deterministic_and_disjoint() -> None:
    seed = deterministic_uint64(1, "prompt", "r40_q10", 0)
    assert seed == deterministic_uint64(1, "prompt", "r40_q10", 0)
    ref, target = frozen_split(100, 40, 10, seed)
    assert len(ref) == 40 and len(target) == 10
    assert set(ref).isdisjoint(target)
    assert len(set(ref) | set(target)) == 50


def test_numerical_summary_preserves_negative_fraction_and_nonfinite_counts() -> None:
    summary = numerical_summary([-2.0, -1.0, 1.0, np.nan, np.inf])
    assert summary["negative_fraction"] == pytest.approx(2 / 3)
    assert summary["nan_count"] == 1
    assert summary["inf_count"] == 1
    assert summary["mad"] == 1.0


def test_candidate_artifact_loading_fails_closed_on_tamper(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload_sha = save_h8_artifact(path, "h8_prompt_bandwidth_candidate", {"sigma": 1.0})
    assert load_h8_artifact(path, "h8_prompt_bandwidth_candidate", payload_sha)["sigma"] == 1.0
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["sigma"] = 2.0
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_h8_artifact(path, "h8_prompt_bandwidth_candidate", payload_sha)


def test_m0ij_config_forbids_sampling_and_selection() -> None:
    config = yaml.safe_load(
        (ROOT / "configs" / "h8_m0ij_mmd_numerical_validation.yaml").read_text(encoding="utf-8")
    )
    for key in (
        "model_sampling_authorized",
        "formal_reference_access_authorized",
        "heldout_access_authorized",
        "attack_access_authorized",
        "parameter_modification_authorized",
    ):
        assert config[key] is False
    assert config["numerical_validation"]["sample_size_selection_performed"] is False
    assert config["numerical_validation"]["clamp_negative_unbiased"] is False
    assert config["permutation_sanity"]["B_precheck"] == 999
    assert all(config["forbidden_operations"].values())


def test_candidate_global_mask_is_bound_to_every_scaler() -> None:
    candidate = ROOT / "results" / "h8_qwen32b_mmd_precalibration" / "m0gh"
    mask = load_h8_artifact(
        candidate / "global_exclusion_mask_candidate.json",
        "h8_global_exclusion_mask_candidate",
        "d3537537f4ff92609c008a58cecf4fc5ba5788ccc66d3d0c3aff44d9088dd464",
    )["mask"]
    assert len(mask) == 528
    for path in (candidate / "scaler_candidates").glob("*.json"):
        scaler = load_h8_artifact(path, "h8_family_balanced_scaler_candidate")
        assert scaler["global_exclusion_mask"] == mask


def test_runner_has_no_model_generation_or_refit_entrypoint() -> None:
    source = (ROOT / "scripts" / "run_h8_m0ij_validation.py").read_text(encoding="utf-8")
    assert "load_model(" not in source
    assert ".generate(" not in source
    assert "fit_family_balanced_scaler" not in source
    assert "h8_median_positive_pairwise_distance" not in source
    assert "sample_size_selection_performed\": False" in source
    assert 'item.get(f"{key}_file_sha256") or item.get(f"{key}_sha256")' in source
