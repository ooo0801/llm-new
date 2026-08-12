from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from llm_integrity.io import stable_id


MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
SOURCE = ROOT / "experiments/prompt-construction-14b-c1/inputs/construction_prompts_k60.jsonl"
SOURCE_SHA256 = "f6c55d4897e37c65e6779e24e1cbfc80e63a7d790ada15ac7a0a2b96282e7a4c"
OUTPUT = ROOT / "experiments/prompt-reconstruction-32b-h6/inputs"
FAMILIES = [
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
]
CORE_CATEGORIES = [
    "code",
    "instruction",
    "knowledge",
    "logic",
    "reasoning",
    "safety",
    "structured",
    "summary",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode("utf-8")
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def remap_sources() -> list[dict[str, Any]]:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("shared clean source SHA-256 differs from H6 protocol")
    rows = read_jsonl(SOURCE)
    if len(rows) != 60:
        raise ValueError(f"H6 requires exactly 60 shared source tasks, received {len(rows)}")
    result = []
    for row in rows:
        prompt = str(row["prompt"])
        parent = str(row.get("prompt_id", row.get("id")))
        prompt_id = stable_id(f"h6|32b|{parent}|{prompt}", f"{row['category']}_h6")
        result.append(
            {
                **{key: value for key, value in row.items() if key not in {"id", "prompt_id", "pool_selection", "split"}},
                "id": prompt_id,
                "prompt_id": prompt_id,
                "parent_14b_source_id": parent,
                "source": "h6_shared_clean_task_text_32b_retokenized",
                "split": "construction",
            }
        )
    if len({row["id"] for row in result}) != 60 or len({row["prompt"] for row in result}) != 60:
        raise ValueError("H6 source IDs and texts must be unique")
    counts = Counter(str(row["category"]) for row in result)
    missing = [category for category in CORE_CATEGORIES if counts[category] == 0]
    if missing:
        raise ValueError(f"H6 shared source set misses core categories: {missing}")
    return result


def configuration(family: str, split: str, index: int) -> dict[str, Any]:
    if family == "unstructured_pruning":
        values = [
            {"ratio": 0.30, "method": "global_magnitude", "target_scope": "attention"},
            {"ratio": 0.30, "method": "layerwise_magnitude", "target_scope": "attention_ffn"},
        ]
    elif family == "structured_pruning":
        values = [
            {"ratio": 0.05, "structure": "ffn_channels", "selection": "magnitude", "layer_scope": "random_layer_subset", "implementation": "mask"},
            {"ratio": 0.05, "structure": "attention_heads", "selection": "magnitude", "layer_scope": "all_layers", "implementation": "mask"},
        ]
    elif family == "quantization":
        values = [
            {"method": "int8", "compute_dtype": "bfloat16", "double_quant": False, "target_scope": "full_model"},
            {"method": "nf4", "compute_dtype": "bfloat16", "double_quant": True, "target_scope": "full_model"},
        ]
    elif family == "gaussian_noise":
        values = [
            {"std_ratio": 0.0005, "target_scope": "ffn", "scale_rule": "parameter_tensor_std"},
            {"std_ratio": 0.001, "target_scope": "all_transformer_layers", "scale_rule": "parameter_tensor_std"},
        ]
    elif family == "finetuning":
        values = [
            {"method": "lora", "rank": 16, "alpha": 32, "dropout": 0.05, "learning_rate": 1e-5, "steps": 50, "target_scope": "attention_ffn", "data_source": "isolated_attack_training_data"},
            {"method": "lora", "rank": 8, "alpha": 16, "dropout": 0.05, "learning_rate": 2e-5, "steps": 50, "target_scope": "attention_ffn", "data_source": "isolated_attack_training_data"},
        ]
    else:
        raise ValueError(family)
    value = dict(values[index])
    value["type"] = family
    return value


def manifest(split: str, seed_base: int) -> list[dict[str, Any]]:
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for index in range(2):
            seed = seed_base + family_index * 2 + index
            config = configuration(family, split, index)
            identity = {"split": split, "family": family, "seed": seed, "configuration": config}
            rows.append(
                {
                    "variant_id": f"h6_{split}_{family}_{canonical_sha256(identity)[:12]}",
                    "family": family,
                    "split": split,
                    "seed": seed,
                    "weight": 0.2,
                    "configuration": config,
                }
            )
    return rows


def train_manifest() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    executable = manifest("train", 2026082000)
    lora = [row for row in executable if row["family"] == "finetuning"]
    return executable, lora


def main() -> None:
    sources = remap_sources()
    train, train_lora = train_manifest()
    development = manifest("validation", 2026082021)
    confirmation = manifest("test", 2026082031)
    write_jsonl(OUTPUT / "construction_prompts_k60.jsonl", sources)
    write_jsonl(OUTPUT / "calibration_prompts_k12.jsonl", sources[:12])
    write_jsonl(OUTPUT / "attack_manifest_train_executable.jsonl", train)
    write_jsonl(OUTPUT / "attack_manifest_train_lora_registered.jsonl", train_lora)
    write_jsonl(OUTPUT / "attack_manifest_development_2each.jsonl", development)
    write_jsonl(OUTPUT / "attack_manifest_confirmation_2each.jsonl", confirmation)
    all_ids = {row["variant_id"] for rows in (train, development, confirmation) for row in rows}
    if len(all_ids) != len(train) + len(development) + len(confirmation):
        raise ValueError("H6 attack IDs overlap across splits")
    design = {
        "schema_version": "experiment_h6_design_1.0",
        "model": {"id": MODEL_ID, "revision": REVISION, "dtype": "bfloat16", "placement": "balanced_3gpu_no_cpu_disk_offload"},
        "source": {"parent_path": str(SOURCE.relative_to(ROOT).as_posix()), "parent_sha256": SOURCE_SHA256, "rows": len(sources), "category_counts": dict(sorted(Counter(row["category"] for row in sources).items())), "core_categories": CORE_CATEGORIES},
        "construction": {"rounds": 3, "maximum_candidates_per_source": 2, "minimum_nondegraded_families": 3, "family_relative_tolerance": 0.01},
        "development_gate": {"minimum_frozen": 30, "all_core_categories_required": True, "maximum_per_source": 1},
        "confirmation_gate": {"minimum_confirmed": 19, "all_core_categories_required": True},
        "component_gate": {"selected_size": 12, "minimum_jaccard": 0.999, "maximum_last_batch_relative_gain": 0.02, "maximum_audit_novelty_rate": 0.08},
        "input_sha256": {name: sha256(OUTPUT / name) for name in ["construction_prompts_k60.jsonl", "calibration_prompts_k12.jsonl", "attack_manifest_train_executable.jsonl", "attack_manifest_train_lora_registered.jsonl", "attack_manifest_development_2each.jsonl", "attack_manifest_confirmation_2each.jsonl"]},
        "leakage_guard": "H6 uses shared clean texts only; all 32B tokenization, calibration, gradients, attacks, adapters, confirmation endpoints, activations and components are newly generated.",
    }
    write_json(OUTPUT / "design_manifest.json", design)
    print(json.dumps(design, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
