from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


OUT = ROOT / "experiments/prompt-union-14b-e1/inputs"
COMBINED = ROOT / "reproducibility/experiment_d_combined14b_20260806/combined_fingerprint.jsonl"
B_PAIRS = ROOT / "experiments/prompt-transfer-14b-b1/inputs/survivor_pairs.jsonl"
C_ACCEPTED = ROOT / "reproducibility/experiment_c_qwen14b_20260806/accepted14b_complement.jsonl"
ATTACK_DATA = ROOT / "data/attack_train_lora.jsonl"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
MICRO_SEED = 2026080701
ATTACK_SEEDS = [2026080702, 2026080703]
BOOTSTRAP_SEED = 2026080704


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_sha256(row: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def fixed_configuration(family: str) -> dict[str, Any]:
    configurations = {
        "unstructured_pruning": {
            "type": "unstructured_pruning", "ratio": 0.30,
            "method": "global_magnitude", "target_scope": "attention",
        },
        "structured_pruning": {
            "type": "structured_pruning", "ratio": 0.05,
            "structure": "ffn_channels", "selection": "random",
            "layer_scope": "random_layer_subset", "implementation": "mask",
        },
        "quantization": {
            "type": "quantization", "method": "nf4",
            "compute_dtype": "bfloat16", "double_quant": True,
            "target_scope": "full_model",
        },
        "gaussian_noise": {
            "type": "gaussian_noise", "std_ratio": 0.0005,
            "target_scope": "ffn", "scale_rule": "parameter_tensor_std",
        },
        "finetuning": {
            "type": "finetuning", "method": "lora", "rank": 16,
            "alpha": 32, "dropout": 0.05, "learning_rate": 1e-5,
            "steps": 50, "target_scope": "attention_ffn",
            "data_source": "isolated_attack_training_data",
        },
    }
    return configurations[family]


def main() -> None:
    combined = read_jsonl(COMBINED)
    b_pairs = {str(row["prompt_id"]): row for row in read_jsonl(B_PAIRS)}
    c_rows = {str(row["prompt_id"]): row for row in read_jsonl(C_ACCEPTED)}
    if len(combined) != 30:
        raise ValueError(f"expected 30 combined rows, received {len(combined)}")

    pairs: list[dict[str, Any]] = []
    for index, combined_row in enumerate(combined):
        prompt_id = str(combined_row["prompt_id"])
        component = str(combined_row["component"])
        if component == "cross_model_strict_core":
            source = b_pairs[prompt_id]
            optimization = source
            source_path = B_PAIRS
        elif component == "qwen14b_model_specific_complement":
            source = c_rows[prompt_id]
            optimization = source["optimization"]
            source_path = C_ACCEPTED
        else:
            raise ValueError(f"{prompt_id}: unknown component {component}")

        initial = str(optimization["initial_prompt"]).strip()
        optimized = str(optimization["optimized_prompt"]).strip()
        if not initial or not optimized or initial == optimized:
            raise ValueError(f"{prompt_id}: invalid frozen pair")
        if optimized != str(combined_row["prompt"]):
            raise ValueError(f"{prompt_id}: combined optimized text mismatch")
        if sha256_text(optimized) != str(combined_row["prompt_sha256"]):
            raise ValueError(f"{prompt_id}: combined text hash mismatch")

        pair = {
            "source_index": index,
            "id": prompt_id,
            "prompt_id": prompt_id,
            "category": str(combined_row["category"]),
            "initial_prompt": initial,
            "optimized_prompt": optimized,
            "evaluator": combined_row.get("evaluator"),
            "expected_answer": combined_row.get("expected_answer"),
            "expected_contains": combined_row.get("expected_contains"),
            "accepted": True,
            "proxy_objective_gain": float(optimization.get("proxy_objective_gain", 1.0)),
            "edit_count": int(optimization.get("edit_count", 1)),
            "edit_ratio": float(optimization.get("edit_ratio", 0.0)),
            "initial_ppl": float(optimization.get("initial_ppl", 0.0)),
            "final_ppl": float(optimization.get("final_ppl", 0.0)),
            "ppl_ratio": float(optimization.get("ppl_ratio", 1.0)),
            "committed_rounds": int(optimization.get("committed_rounds", 0)),
            "rounds_completed": int(optimization.get("rounds_completed", optimization.get("rounds_run", 0))),
            "component": component,
            "evidence": list(combined_row["evidence"]),
            "optimized_prompt_sha256": sha256_text(optimized),
            "initial_prompt_sha256": sha256_text(initial),
            "source_evidence_path": str(source_path.relative_to(ROOT)),
            "source_record_sha256": canonical_sha256(source),
            "combined_record_sha256": canonical_sha256(combined_row),
        }
        if pair["proxy_objective_gain"] <= 0 or pair["edit_count"] < 1:
            raise ValueError(f"{prompt_id}: frozen source constraints are inconsistent")
        pairs.append(pair)

    if len({row["prompt_id"] for row in pairs}) != 30:
        raise ValueError("duplicate prompt ids")
    if len({row["optimized_prompt_sha256"] for row in pairs}) != 30:
        raise ValueError("duplicate optimized prompt texts")
    expected_components = {"cross_model_strict_core": 4, "qwen14b_model_specific_complement": 26}
    observed_components = {key: sum(row["component"] == key for row in pairs) for key in expected_components}
    if observed_components != expected_components:
        raise ValueError(f"component mismatch: {observed_components}")

    families = ["unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning"]
    manifest: list[dict[str, Any]] = []
    for family in families:
        for seed in ATTACK_SEEDS:
            configuration = fixed_configuration(family)
            identity = {"split": "test", "family": family, "seed": seed, "configuration": configuration}
            manifest.append({
                "configuration": configuration,
                "family": family,
                "seed": seed,
                "split": "test",
                "variant_id": f"final_{family}_{canonical_sha256(identity)[:12]}",
                "weight": 0.2,
            })

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "final30_pairs.jsonl", pairs)
    write_jsonl(OUT / "smoke_pair.jsonl", [pairs[0]])
    write_jsonl(OUT / "attack_manifest_final_2each.jsonl", manifest)
    design = {
        "schema_version": "experiment_e_final_union_design_1.0",
        "model": {"id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "device_map": "balanced_3gpu_no_cpu_disk_offload"},
        "frozen_candidates": {
            "rows": 30,
            "components": expected_components,
            "categories": sorted({row["category"] for row in pairs}),
            "source": str(COMBINED.relative_to(ROOT)),
            "source_sha256": sha256_file(COMBINED),
            "pairs_sha256": sha256_file(OUT / "final30_pairs.jsonl"),
        },
        "seeds": {"micro": MICRO_SEED, "attack": ATTACK_SEEDS, "bootstrap": BOOTSTRAP_SEED},
        "attack_manifest": {
            "rows": len(manifest), "families": families, "variants_per_family": 2,
            "sha256": sha256_file(OUT / "attack_manifest_final_2each.jsonl"),
            "lora_training_data": str(ATTACK_DATA.relative_to(ROOT)),
            "lora_training_data_sha256": sha256_file(ATTACK_DATA),
        },
        "endpoints": {"task": 60, "micro": 60, "macro": 600},
        "gates": {
            "task": "initial_and_optimized_pass",
            "micro": "optimized_micro_per_parameter_gt_initial",
            "macro": "optimized_equal_weight_five_family_macro_gt_initial",
            "family_relative_tolerance": 0.01,
            "legacy_minimum_nondegraded_families": 3,
            "strict_minimum_nondegraded_families": 5,
            "required_legacy_retained": 23,
            "required_category_coverage": sorted({row["category"] for row in pairs}),
            "strict_retained": "secondary_descriptive_no_threshold",
        },
        "leakage_guard": "No Experiment E result may alter candidates, thresholds, seeds, attack configurations, or task evaluators.",
        "prior_seed_collision_check": [42, 777, 3407, 1685839144, 2435199792, 4072939906],
    }
    used = set(design["prior_seed_collision_check"])
    if MICRO_SEED in used or any(seed in used for seed in ATTACK_SEEDS):
        raise ValueError("Experiment E seed collides with a recorded prior seed")
    write_json(OUT / "design_manifest.json", design)
    print(json.dumps(design, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
