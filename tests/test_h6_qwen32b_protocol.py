from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
CORE = {"code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_h6_model_and_scientific_gates_are_frozen() -> None:
    configs = [
        yaml.safe_load((ROOT / "configs/experiment_h6_qwen32b_inner.yaml").read_text(encoding="utf-8")),
        yaml.safe_load((ROOT / "configs/experiment_h6_qwen32b_development.yaml").read_text(encoding="utf-8")),
        yaml.safe_load((ROOT / "configs/experiment_h6_qwen32b_confirmation.yaml").read_text(encoding="utf-8")),
        yaml.safe_load((ROOT / "configs/fingerprint_h6_qwen32b_mcc.yaml").read_text(encoding="utf-8")),
    ]
    for config in configs:
        assert config["model"]["name"] == "Qwen/Qwen2.5-32B-Instruct"
        assert config["model"]["revision"] == REVISION
        assert config["model"]["dtype"] == "bfloat16"
        assert config["model"]["device_map"] == "balanced"
        assert config["model"]["max_memory"] == {0: "30GiB", 1: "30GiB", 2: "30GiB"}
        assert config["model"]["local_files_only"] is True
        assert config["model"]["eager_attention"] is True
    mcc = configs[-1]
    assert mcc["fingerprint"]["minimum_candidate_size"] == 19
    assert mcc["fingerprint"]["selected_size"] == 12
    assert set(mcc["fingerprint"]["required_categories"]) == CORE
    assert mcc["activations"]["minimum_jaccard"] == 0.999
    assert mcc["calibration"]["max_last_batch_relative_gain"] == 0.02
    assert mcc["calibration"]["max_audit_novelty_rate"] == 0.08


def test_h6_inputs_are_32b_specific_and_split_isolated() -> None:
    inputs = ROOT / "experiments/prompt-reconstruction-32b-h6/inputs"
    sources = read_jsonl(inputs / "construction_prompts_k60.jsonl")
    assert len(sources) == 60
    assert len({row["id"] for row in sources}) == 60
    assert len({row["prompt"] for row in sources}) == 60
    assert all("parent_14b_source_id" in row for row in sources)
    assert all(row["id"] != row["parent_14b_source_id"] for row in sources)
    assert CORE <= set(Counter(row["category"] for row in sources))
    manifests = {
        name: read_jsonl(inputs / name)
        for name in [
            "attack_manifest_train_executable.jsonl",
            "attack_manifest_development_2each.jsonl",
            "attack_manifest_confirmation_2each.jsonl",
        ]
    }
    all_ids: set[str] = set()
    all_seeds: set[int] = set()
    for rows in manifests.values():
        assert Counter(row["family"] for row in rows) == {
            "unstructured_pruning": 2,
            "structured_pruning": 2,
            "quantization": 2,
            "gaussian_noise": 2,
            "finetuning": 2,
        }
        ids = {row["variant_id"] for row in rows}
        seeds = {int(row["seed"]) for row in rows}
        assert all_ids.isdisjoint(ids)
        assert all_seeds.isdisjoint(seeds)
        all_ids |= ids
        all_seeds |= seeds


def test_h6_inner_partition_fits_frozen_training_variants() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiment_h6_qwen32b_inner.yaml").read_text(
            encoding="utf-8"
        )
    )
    settings = config["discrete_joint_inner"]
    train = read_jsonl(
        ROOT
        / "experiments/prompt-reconstruction-32b-h6/inputs/"
        "attack_manifest_train_executable.jsonl"
    )
    counts = Counter(row["family"] for row in train)
    required = (
        int(settings["variants_per_family"])
        + int(settings["anchor_variants_per_family"])
    )
    assert required == 2
    assert all(count == required for count in counts.values())
    assert settings["sequential_model_execution"] is True


def test_h6_runner_stops_before_verification() -> None:
    runner = (ROOT / "scripts/run_experiment_h6_qwen32b.sh").read_text(encoding="utf-8")
    assert "run_h6_engineering_smoke.py" not in runner
    assert "engineering_smoke.json" in runner
    assert "check_h6_prompt_gate.py" in runner
    assert "extract_fingerprint_v2_activations.py" in runner
    assert "build_fingerprint_h6_artifact.py" in runner
    assert "run_fingerprint_v2_verification.py" not in runner
    assert "summarize_fingerprint" not in runner


def test_h6_lora_variants_are_process_isolated() -> None:
    runner = (ROOT / "scripts/run_experiment_h6_qwen32b.sh").read_text(encoding="utf-8")
    assert "train_lora_variant()" in runner
    assert '--variant-id "$variant_id"' in runner
    expected = {
        "h6_train_finetuning_6a35e421e2bd",
        "h6_train_finetuning_98965877bebd",
        "h6_validation_finetuning_c0a68aedce5f",
        "h6_validation_finetuning_c8ad5995aa5e",
        "h6_test_finetuning_d4d9c8a29d07",
        "h6_test_finetuning_cd2bbaed31af",
    }
    assert all(runner.count(variant_id) == 1 for variant_id in expected)


def test_h6_resume_chain_preserves_parent_gates() -> None:
    runner = (ROOT / "scripts/resume_h6_after_download.sh").read_text(encoding="utf-8")
    assert 'r.get("status") == "complete"' in runner
    assert 'len(r.get("weight_shards", [])) == 17' in runner
    assert "run_h6_preflight.sh" in runner
    assert 'r.get("passed") is True' in runner
    assert "run_experiment_h6_qwen32b.sh" in runner


def test_h6_post_construction_resume_is_guarded() -> None:
    runner = (
        ROOT / "scripts/resume_h6_after_construction.sh"
    ).read_text(encoding="utf-8")
    assert 'r.get("requested_prompts") == 60' in runner
    assert 'r.get("technically_complete") == 60' in runner
    assert 'r.get("technical_passed") is True' in runner
    assert "summarize_discrete_results.py" in runner
    assert "run_experiment_h6_qwen32b.sh" in runner


def test_h6_construction_recovery_defines_post_optimizer_context() -> None:
    runner = (
        ROOT / "scripts/recover_h6_pipeline_from_construction.sh"
    ).read_text(encoding="utf-8")
    for variable in (
        "DEV_CONFIG",
        "CONFIRM_CONFIG",
        "MCC_CONFIG",
        "DEV",
        "CONFIRM",
        "ADAPTERS",
    ):
        assert f"{variable}=" in runner
    assert "train_lora_variant()" in runner
