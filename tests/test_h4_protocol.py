from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_h4_freezes_g3_mcc12_and_common_random_numbers() -> None:
    config = yaml.safe_load((ROOT / "configs/fingerprint_h4_qwen14b_paired.yaml").read_text(encoding="utf-8"))
    source = ROOT / config["data"]["source_fingerprint"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["data"]["source_fingerprint_sha256"]
    assert config["fingerprint"]["selected_size"] == 12
    reference = [config["fingerprint"]["reference_seed_base"] + index for index in range(10)]
    target = [config["statistics"]["target_seed_base"] + index for index in range(10)]
    assert reference == target == list(range(2026081600, 2026081610))
    assert config["statistics"]["method"] == "paired_block_sign_flip"
    assert config["statistics"]["standardization"] == "pooled_symmetric"
    assert config["statistics"]["exact_sign_flips"] is True


def test_h4_freezes_three_intact_and_eleven_new_modified_states() -> None:
    config = yaml.safe_load((ROOT / "configs/fingerprint_h4_qwen14b_paired.yaml").read_text(encoding="utf-8"))
    assert len(config["attacks"]) == 14
    assert Counter(row["family"] for row in config["attacks"]) == {
        "intact": 3,
        "unstructured_pruning": 2,
        "structured_pruning": 2,
        "gaussian_noise": 3,
        "quantization": 2,
        "finetuning": 2,
    }
    assert config["statistics"]["expected_intact_states"] == 3
    assert config["statistics"]["expected_modified_states"] == 11
    assert len({row["seed"] for row in config["attacks"]}) == 14
    assert all(str(row["variant_id"]).endswith("_h4") for row in config["attacks"])
