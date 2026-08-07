from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f2_expansion_is_balanced_unique_and_semantically_guarded() -> None:
    module = load_script("build_experiment_f2_inputs")
    rows = module.build_sources()
    assert len(rows) == 32
    assert Counter(row["category"] for row in rows) == {"logic": 16, "summary": 16}
    assert len({row["id"] for row in rows}) == 32
    assert len({row["prompt"] for row in rows}) == 32
    summaries = [row for row in rows if row["category"] == "summary"]
    assert all(row["evaluator"] == "length_and_contains" for row in summaries)
    assert all(len(row["expected_contains"]) == 2 for row in summaries)
    assert all(all(word in row["prompt"] for word in row["expected_contains"]) for row in summaries)


def test_f2_manifests_are_disjoint_and_have_frozen_shapes() -> None:
    module = load_script("build_experiment_f2_inputs")
    development = module.build_manifest("validation", module.DEVELOPMENT_SEEDS)
    confirmation = module.build_manifest("test", module.CONFIRMATION_SEEDS)
    assert len(development) == 15
    assert len(confirmation) == 10
    assert Counter(row["family"] for row in development) == {family: 3 for family in module.FAMILIES}
    assert Counter(row["family"] for row in confirmation) == {family: 2 for family in module.FAMILIES}
    assert {row["variant_id"] for row in development}.isdisjoint(row["variant_id"] for row in confirmation)
    assert {row["seed"] for row in development}.isdisjoint(row["seed"] for row in confirmation)
    confirm_structured = [row for row in confirmation if row["family"] == "structured_pruning"]
    assert all(row["configuration"]["ratio"] == 0.06 for row in confirm_structured)


def test_f2_frozen_input_manifest_matches_generated_files() -> None:
    inputs = ROOT / "experiments/prompt-robust-14b-f2/inputs"
    design = json.loads((inputs / "design_manifest.json").read_text(encoding="utf-8"))
    assert design["expansion"]["category_counts"] == {"logic": 16, "summary": 16}
    assert design["construction"]["minimum_unique_candidate_sources_per_expanded_category"] == 3
    assert design["selection"]["rows"] == 30
    assert design["selection"]["minimum_per_category"] == 3
    assert design["confirmation_gate"]["required_legacy_retained"] == 23


def test_g2_h2_config_preserves_final_fingerprint_gates() -> None:
    config = yaml.safe_load((ROOT / "configs/fingerprint_g2_qwen14b_mcc.yaml").read_text(encoding="utf-8"))
    assert config["protocol_labels"] == {
        "construction": "G2",
        "detection": "H2",
        "candidate_source": "experiment_f2_independent_confirmation",
    }
    assert config["fingerprint"]["minimum_candidate_size"] == 23
    assert config["fingerprint"]["selected_size"] == 12
    assert config["fingerprint"]["reference_repetitions"] == 10
    assert config["statistics"]["repetitions"] == 10
    assert config["statistics"]["permutations"] == 999
    assert config["statistics"]["alpha"] == 0.05
    assert len(config["attacks"]) == 12
    assert Counter(row["family"] for row in config["attacks"]) == {
        "intact": 1,
        "unstructured_pruning": 2,
        "structured_pruning": 2,
        "gaussian_noise": 3,
        "quantization": 2,
        "finetuning": 2,
    }
