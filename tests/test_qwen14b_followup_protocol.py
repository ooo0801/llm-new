from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/fingerprint_g_qwen14b_mcc.yaml").read_text(encoding="utf-8"))


def test_g1_uses_frozen_7b_component_and_universe_gates() -> None:
    activations = CONFIG["activations"]
    assert activations["repetitions"] == 2
    assert activations["stable_component_frequency"] == 1.0
    assert activations["attention_entropy_fraction"] == 0.70
    assert activations["ffn_quantile"] == 0.95
    assert activations["residual_threshold"] == 0.50
    assert activations["max_ffn_units_per_layer"] == 128
    assert activations["minimum_jaccard"] == 0.999

    calibration = CONFIG["calibration"]
    assert calibration["task_families"] == 14
    assert calibration["build_per_family"] == 20
    assert calibration["audit_per_family"] == 4
    assert calibration["last_batches_for_saturation"] == 3
    assert calibration["max_last_batch_relative_gain"] == 0.02
    assert calibration["max_audit_novelty_rate"] == 0.08


def test_h1_generation_and_stratified_mmd_match_frozen_protocol() -> None:
    generation = CONFIG["generation"]
    assert generation["do_sample"] is True
    assert generation["temperature"] == 0.7
    assert generation["top_p"] == 0.9
    assert generation["top_k"] == 50
    assert generation["system_prompt"] is None

    fingerprint = CONFIG["fingerprint"]
    statistics = CONFIG["statistics"]
    assert fingerprint["selected_size"] == 12
    assert fingerprint["reference_repetitions"] == 10
    assert fingerprint["reference_seed_base"] == 2026080800
    assert statistics["method"] == "prompt_stratified_mmd"
    assert statistics["repetitions"] == 10
    assert statistics["target_seed_base"] == 2026080900
    assert statistics["permutations"] == 999
    assert statistics["alpha"] == 0.05


def test_h1_has_one_intact_and_eleven_registered_modified_states() -> None:
    counts = Counter(str(row["family"]) for row in CONFIG["attacks"])
    assert len(CONFIG["attacks"]) == 12
    assert counts == {
        "intact": 1,
        "unstructured_pruning": 2,
        "structured_pruning": 2,
        "gaussian_noise": 3,
        "quantization": 2,
        "finetuning": 2,
    }
    assert len({str(row["variant_id"]) for row in CONFIG["attacks"]}) == 12
