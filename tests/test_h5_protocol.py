from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_h5_freezes_new_common_seeds_and_block_mismatch_gate() -> None:
    config = yaml.safe_load((ROOT / "configs/fingerprint_h5_qwen14b_mismatch.yaml").read_text(encoding="utf-8"))
    source = ROOT / config["data"]["source_fingerprint"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config["data"]["source_fingerprint_sha256"]
    reference = [config["fingerprint"]["reference_seed_base"] + index for index in range(10)]
    target = [config["statistics"]["target_seed_base"] + index for index in range(10)]
    assert reference == target == list(range(2026081700, 2026081710))
    assert config["statistics"]["method"] == "paired_block_mismatch_binomial"
    assert config["statistics"]["null_block_mismatch_rate"] == 0.1
    assert config["statistics"]["alpha"] == 0.05 / 3
    assert config["statistics"]["required_family_detected"] == {"gaussian_noise": 3}


def test_h5_freezes_three_intact_and_eleven_new_modified_states() -> None:
    config = yaml.safe_load((ROOT / "configs/fingerprint_h5_qwen14b_mismatch.yaml").read_text(encoding="utf-8"))
    assert len(config["attacks"]) == 14
    assert Counter(row["family"] for row in config["attacks"]) == {
        "intact": 3,
        "unstructured_pruning": 2,
        "structured_pruning": 2,
        "gaussian_noise": 3,
        "quantization": 2,
        "finetuning": 2,
    }
    assert len({row["seed"] for row in config["attacks"]}) == 14
    assert all(str(row["variant_id"]).endswith("_h5") for row in config["attacks"])
