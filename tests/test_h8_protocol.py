from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "h8_qwen32b_mmd_precalibration.yaml"
FINGERPRINT = (
    ROOT
    / "reproducibility"
    / "fingerprint_h6_qwen32b_mcc_20260812"
    / "components"
    / "fingerprints"
    / "global_mcc_32b_h6.json"
)


def test_h8_pre_sampling_scope_is_fail_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["formal_calibration_authorized"] is False
    assert config["calibration"]["sampling_authorized"] is False
    assert config["calibration"]["data_role"] == "mmd_precalibration_fit_only"
    assert config["calibration"]["may_enter_reference"] is False
    assert config["calibration"]["may_enter_heldout"] is False
    assert config["smoke"]["hard_response_limit"] == 24
    assert config["generation"]["batch_size"] == 1
    assert config["generation"]["response_level_unique_seed"] is True


def test_h8_statistics_and_feature_conventions_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["features"]["dtype"] == "float64"
    assert config["features"]["ddof"] == 0
    assert config["features"]["primary_scaling"] == "per_prompt_standard_then_family_balanced"
    assert config["mmd"]["bandwidth_convention"] == "median_positive_pairwise_euclidean_distance"
    assert config["mmd"]["legacy_h4_bandwidth_forbidden"] is True
    assert config["mmd"]["negative_unbiased_values_allowed"] is True
    assert set(config["bandwidth_stability"]["tests"]) == {
        "fixed_full_scaler",
        "refit_subsample_scaler",
    }


def test_h6_fingerprint_hash_and_revision_match() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert hashlib.sha256(FINGERPRINT.read_bytes()).hexdigest() == config["fingerprint"]["source_sha256"]
    assert config["model"]["revision"] == "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    assert config["generation"]["eos_token_ids"] == [151645, 151643]

