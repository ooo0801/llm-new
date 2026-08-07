from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


DEFAULT_SOURCE = ROOT / "results/experiment_c_qwen14b_20260806/03_development/portfolio_candidates.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments/prompt-robust-14b-f1/inputs"
SOURCE_EVIDENCE_PATH = "results/experiment_c_qwen14b_20260806/03_development/portfolio_candidates.jsonl"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
FAMILIES = ["unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning"]
DEVELOPMENT_SEEDS = [2026080711, 2026080712, 2026080713]
CONFIRMATION_SEEDS = [2026080721, 2026080722]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def base_configuration(family: str) -> dict[str, Any]:
    return {
        "unstructured_pruning": {"type": family, "ratio": 0.30, "method": "global_magnitude", "target_scope": "attention"},
        "quantization": {"type": family, "method": "nf4", "compute_dtype": "bfloat16", "double_quant": True, "target_scope": "full_model"},
        "gaussian_noise": {"type": family, "std_ratio": 0.0005, "target_scope": "ffn", "scale_rule": "parameter_tensor_std"},
        "finetuning": {"type": family, "method": "lora", "rank": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 1e-5, "steps": 50, "target_scope": "attention_ffn", "data_source": "isolated_attack_training_data"},
    }[family]


def structured_configuration(split: str, index: int) -> dict[str, Any]:
    development = [
        {"structure": "ffn_channels", "selection": "random", "layer_scope": "random_layer_subset"},
        {"structure": "ffn_channels", "selection": "magnitude", "layer_scope": "all_layers"},
        {"structure": "attention_heads", "selection": "random", "layer_scope": "random_layer_subset"},
    ]
    confirmation = [
        {"structure": "ffn_channels", "selection": "magnitude", "layer_scope": "random_layer_subset"},
        {"structure": "attention_heads", "selection": "magnitude", "layer_scope": "all_layers"},
    ]
    value = dict((development if split == "validation" else confirmation)[index])
    value.update({"type": "structured_pruning", "ratio": 0.05, "implementation": "mask"})
    return value


def build_manifest(split: str, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prefix = "f1_dev" if split == "validation" else "f1_confirm"
    for family in FAMILIES:
        for index, seed in enumerate(seeds):
            config = structured_configuration(split, index) if family == "structured_pruning" else base_configuration(family)
            identity = {"split": split, "family": family, "seed": seed, "configuration": config}
            rows.append({
                "configuration": config,
                "family": family,
                "seed": seed,
                "split": split,
                "variant_id": f"{prefix}_{family}_{canonical_sha256(identity)[:12]}",
                "weight": 0.2,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output_dir).resolve()
    rows = read_jsonl(source)
    if len(rows) != 64:
        raise ValueError(f"expected 64 Experiment C portfolio candidates, received {len(rows)}")

    pairs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prompt_id = str(row["prompt_id"])
        source_id = str(row["candidate_provenance"]["source_prompt_id"])
        initial = str(row["initial_prompt"]).strip()
        optimized = str(row["optimized_prompt"]).strip()
        if not initial or not optimized or initial == optimized:
            raise ValueError(f"{prompt_id}: invalid prompt pair")
        pair = dict(row)
        pair.update({
            "source_index": index,
            "id": prompt_id,
            "prompt_id": prompt_id,
            "source_prompt_id": source_id,
            "component": "experiment_c_full_portfolio",
            "evidence": ["experiment_c_pre_development_portfolio"],
            "accepted": True,
            "initial_prompt_sha256": text_sha256(initial),
            "optimized_prompt_sha256": text_sha256(optimized),
            "source_record_sha256": canonical_sha256(row),
        })
        pairs.append(pair)

    if len({row["prompt_id"] for row in pairs}) != 64:
        raise ValueError("duplicate candidate prompt IDs")
    if len({row["optimized_prompt_sha256"] for row in pairs}) != 64:
        raise ValueError("duplicate optimized prompt text")
    source_count = len({row["source_prompt_id"] for row in pairs})
    if source_count != 38:
        raise ValueError(f"expected 38 source prompts, received {source_count}")
    category_counts = Counter(str(row["category"]) for row in pairs)
    required_categories = ["code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"]
    if sorted(category_counts) != required_categories:
        raise ValueError(f"category mismatch: {category_counts}")

    development = build_manifest("validation", DEVELOPMENT_SEEDS)
    confirmation = build_manifest("test", CONFIRMATION_SEEDS)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "candidate_pairs64.jsonl", pairs)
    write_jsonl(output / "smoke_pair.jsonl", [pairs[0]])
    write_jsonl(output / "attack_manifest_development_3each.jsonl", development)
    write_jsonl(output / "attack_manifest_confirmation_2each.jsonl", confirmation)
    design = {
        "schema_version": "experiment_f1_design_1.0",
        "model": {"id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "placement": "balanced_3gpu_no_cpu_disk_offload"},
        "candidate_source": SOURCE_EVIDENCE_PATH,
        "candidate_source_sha256": file_sha256(source),
        "candidate_rows": len(pairs),
        "source_prompts": source_count,
        "category_counts": dict(sorted(category_counts.items())),
        "development": {"micro_seed": 2026080710, "bootstrap_seed": 2026080714, "attack_seeds": DEVELOPMENT_SEEDS, "variants_per_family": 3},
        "confirmation": {"micro_seed": 2026080720, "bootstrap_seed": 2026080723, "attack_seeds": CONFIRMATION_SEEDS, "variants_per_family": 2},
        "selection": {"rows": 30, "minimum_per_category": 3, "maximum_per_source_prompt": 1, "ranking": ["structured_nondegraded", "nondegraded_family_count", "worst_family_median_relative_gain", "structured_median_relative_gain", "macro_relative_gain", "micro_relative_gain", "prompt_id"]},
        "confirmation_gate": {"required_legacy_retained": 23, "required_categories": required_categories, "family_relative_tolerance": 0.01, "minimum_nondegraded_families": 3},
        "inputs_sha256": {
            "pairs": file_sha256(output / "candidate_pairs64.jsonl"),
            "development_manifest": file_sha256(output / "attack_manifest_development_3each.jsonl"),
            "confirmation_manifest": file_sha256(output / "attack_manifest_confirmation_2each.jsonl"),
        },
        "leakage_guard": "Experiment C held-out decisions and Experiment E outcomes are not inputs to F1 ranking; confirmation endpoints cannot alter selection or gates.",
    }
    write_json(output / "design_manifest.json", design)
    print(json.dumps(design, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
