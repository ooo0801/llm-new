from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_fingerprint_14b_mmd",
    ROOT / "scripts/summarize_fingerprint_14b_mmd.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CONFIG = yaml.safe_load((ROOT / "configs/fingerprint_g_qwen14b_mcc.yaml").read_text(encoding="utf-8"))


def valid_row() -> tuple[dict, dict]:
    attack = CONFIG["attacks"][0]
    prompts = [f"p{index}" for index in range(12)]
    seeds = list(range(2026080900, 2026080910))
    records = [
        {"prompt_id": prompt_id, "repetition": repetition, "seed": seeds[repetition], "response": "ok"}
        for repetition in range(10)
        for prompt_id in prompts
    ]
    primary = {
        "statistic": 0.01,
        "p_value": 0.5,
        "reject": False,
        "alpha": 0.05,
        "method": "prompt_stratified_mmd",
        "effect_size": 0.1,
        "diagnostics": {
            "permutations": 999,
            "strata": 12,
            "samples_per_group": {prompt_id: 10 for prompt_id in prompts},
            "bandwidths": {prompt_id: 1.0 for prompt_id in prompts},
        },
    }
    row = {
        "variant_id": attack["variant_id"],
        "family": attack["family"],
        "ground_truth_modified": False,
        "predicted_modified": False,
        "correct": True,
        "primary_test": primary,
        "secondary_tests": {"pooled_mmd": {"method": "pooled_mmd", "statistic": 0.0, "p_value": 1.0}},
        "prompts": 12,
        "repetitions": 10,
        "queries": 120,
        "target_seeds": seeds,
        "response_records": records,
        "variant_realization": {
            "variant_id": attack["variant_id"],
            "family": attack["family"],
            "isolated_base_reload": True,
        },
    }
    return row, attack


def test_complete_registered_verification_passes_technical_validation() -> None:
    row, attack = valid_row()
    assert MODULE.validate_verification(row, attack, CONFIG) == []


def test_wrong_target_seeds_and_nonfinite_statistic_are_rejected() -> None:
    row, attack = valid_row()
    row["target_seeds"][0] = -1
    row["primary_test"]["statistic"] = float("nan")
    errors = MODULE.validate_verification(row, attack, CONFIG)
    assert "target seed mismatch" in errors
    assert "nonfinite primary statistic" in errors


def test_missing_response_and_realization_structure_are_rejected() -> None:
    row, attack = valid_row()
    row["response_records"].pop()
    row["variant_realization"]["isolated_base_reload"] = False
    errors = MODULE.validate_verification(row, attack, CONFIG)
    assert "response record count mismatch" in errors
    assert "variant was not independently reloaded" in errors
