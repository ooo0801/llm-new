from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_g3_freezes_exact_f2_legacy_pool_and_mcc12() -> None:
    config = yaml.safe_load((ROOT / "configs/fingerprint_g3_qwen14b_mcc.yaml").read_text(encoding="utf-8"))
    source = ROOT / config["data"]["candidate_source"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert config["protocol_labels"] == {
        "construction": "G3",
        "detection": "H3",
        "candidate_source": "experiment_f2_legacy_retained_recovery",
    }
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["data"]["candidate_source_sha256"]
    assert len(rows) == config["fingerprint"]["minimum_candidate_size"] == 19
    assert config["fingerprint"]["selected_size"] == 12
    assert len({row["prompt_id"] for row in rows}) == 19
    assert len({row["optimized_prompt_sha256"] for row in rows}) == 19
    assert Counter(row["category"] for row in rows) == {
        "code": 1,
        "instruction": 4,
        "knowledge": 2,
        "logic": 4,
        "reasoning": 3,
        "safety": 2,
        "structured": 1,
        "summary": 2,
    }


def test_g3_preserves_component_and_universe_gates() -> None:
    g2 = yaml.safe_load((ROOT / "configs/fingerprint_g2_qwen14b_mcc.yaml").read_text(encoding="utf-8"))
    g3 = yaml.safe_load((ROOT / "configs/fingerprint_g3_qwen14b_mcc.yaml").read_text(encoding="utf-8"))
    assert g3["activations"] == g2["activations"]
    assert g3["calibration"] == g2["calibration"]
    assert g3["fingerprint"]["primary_selection"] == "global_unweighted_mcc"
    assert g3["fingerprint"]["ablation_weights"] == g2["fingerprint"]["ablation_weights"]


def test_h3_freezes_new_independent_sampling_and_attack_seeds() -> None:
    g2 = yaml.safe_load((ROOT / "configs/fingerprint_g2_qwen14b_mcc.yaml").read_text(encoding="utf-8"))
    g3 = yaml.safe_load((ROOT / "configs/fingerprint_g3_qwen14b_mcc.yaml").read_text(encoding="utf-8"))
    assert g3["fingerprint"]["reference_repetitions"] == 10
    assert g3["statistics"]["repetitions"] == 10
    assert g3["statistics"]["permutations"] == 999
    assert g3["statistics"]["alpha"] == 0.05
    assert len(g3["attacks"]) == 12
    assert Counter(row["family"] for row in g3["attacks"]) == {
        "intact": 1,
        "unstructured_pruning": 2,
        "structured_pruning": 2,
        "gaussian_noise": 3,
        "quantization": 2,
        "finetuning": 2,
    }
    g2_seeds = {row["seed"] for row in g2["attacks"]}
    g3_seeds = {row["seed"] for row in g3["attacks"]}
    assert g2_seeds.isdisjoint(g3_seeds)
    assert g3["fingerprint"]["reference_seed_base"] != g2["fingerprint"]["reference_seed_base"]
    assert g3["statistics"]["target_seed_base"] != g2["statistics"]["target_seed_base"]
