from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


FAMILIES = ("unstructured_pruning", "structured_pruning", "quantization", "gaussian_noise", "finetuning")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable(row: dict[str, Any]) -> bool:
    if row.get("family") != "quantization":
        return True
    config = row.get("configuration", {})
    return str(config.get("target_scope", "")).lower() == "full_model" and str(config.get("compute_dtype", "")).lower() == "bfloat16"


def balanced(rows: list[dict[str, Any]], split: str, per_family: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for family in FAMILIES:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        order: list[str] = []
        for row in rows:
            if row.get("split") != split or row.get("family") != family or not executable(row):
                continue
            key = json.dumps(row["configuration"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if key not in groups:
                order.append(key)
            groups[key].append(row)
        chosen = next((groups[key] for key in order if len(groups[key]) >= per_family), None)
        if chosen is None:
            raise ValueError(f"{split}/{family}: no executable configuration has {per_family} seeds")
        cohort = chosen[:per_family]
        selected.extend(cohort)
        report[family] = {"configuration": cohort[0]["configuration"], "variant_ids": [str(row["variant_id"]) for row in cohort], "seeds": [int(row["seed"]) for row in cohort]}
    return selected, report


def exact_frozen_balanced(rows: list[dict[str, Any]], split: str, per_family: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [row for row in rows if row.get("split") == split and executable(row)]
    report: dict[str, Any] = {}
    for family in FAMILIES:
        family_rows = [row for row in selected if row.get("family") == family]
        if len(family_rows) != per_family:
            raise ValueError(f"{split}/{family}: expected exactly {per_family} frozen executable rows, received {len(family_rows)}")
        report[family] = {
            "configurations": [row["configuration"] for row in family_rows],
            "variant_ids": [str(row["variant_id"]) for row in family_rows],
            "seeds": [int(row["seed"]) for row in family_rows],
        }
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-pool", default="data/candidate.jsonl")
    parser.add_argument("--prompt-sample", default="results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl")
    parser.add_argument("--train-manifest", default="results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_train.jsonl")
    parser.add_argument("--train-registry", default="results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_train.json")
    parser.add_argument("--validation-manifest", default="results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_validation_balanced_2each.jsonl")
    parser.add_argument("--test-manifest", default="results/paper_aligned_qwen_7b/joint_discrete_v2_20260728/13_v6_formal_60/03_frozen_test_evaluation/00_design/attack_manifest_test_balanced_2each.jsonl")
    parser.add_argument("--output-dir", default="experiments/prompt-construction-14b-c1/inputs")
    args = parser.parse_args()

    candidate_path, sample_path, train_path, train_registry_path, validation_path, test_path = map(resolve, (args.candidate_pool, args.prompt_sample, args.train_manifest, args.train_registry, args.validation_manifest, args.test_manifest))
    candidates, prompts = read_jsonl(candidate_path), read_jsonl(sample_path)
    if len(candidates) != 336 or len(prompts) != 60:
        raise ValueError(f"Expected 336 candidates and 60 frozen samples, received {len(candidates)} and {len(prompts)}")
    candidate_ids = {str(row.get("id", row.get("prompt_id"))) for row in candidates}
    prompt_ids = [str(row.get("id", row.get("prompt_id"))) for row in prompts]
    if len(set(prompt_ids)) != 60 or not set(prompt_ids) <= candidate_ids:
        raise ValueError("The frozen 60-prompt sample is not a unique subset of the 336-row candidate pool")

    train_source = [row for row in read_jsonl(train_path) if row.get("split") == "train" and executable(row)]
    registered_train_lora = set(json.loads(train_registry_path.read_text(encoding="utf-8")))
    train_lora = [row for row in train_source if row.get("family") == "finetuning" and str(row.get("variant_id")) in registered_train_lora]
    if len(train_lora) < 3:
        raise ValueError(f"The original executable V6 train registry exposes only {len(train_lora)} LoRA variants; at least 3 are required")
    train_report = {
        family: {
            "manifest_rows": sum(row.get("family") == family for row in train_source),
            "registered_lora_rows": sum(row.get("family") == family for row in train_lora),
        }
        for family in FAMILIES
    }
    validation, validation_report = exact_frozen_balanced(read_jsonl(validation_path), "validation", 2)
    test, test_report = exact_frozen_balanced(read_jsonl(test_path), "test", 2)
    split_ids = [{str(row["variant_id"]) for row in cohort} for cohort in (train_source, validation, test)]
    if any(split_ids[i] & split_ids[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Attack variant IDs overlap across train/validation/test")

    output = resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "construction_prompts_k60.jsonl": prompts,
        "calibration_prompts_k12.jsonl": prompts[:12],
        "attack_manifest_train_executable.jsonl": train_source,
        "attack_manifest_train_lora_registered.jsonl": train_lora,
        "attack_manifest_validation_2each.jsonl": validation,
        "attack_manifest_test_2each.jsonl": test,
    }
    for name, rows in files.items():
        write_jsonl(output / name, rows)

    payload = {
        "schema_version": "experiment_c_frozen_design_1.0",
        "candidate_pool": {"path": str(candidate_path.relative_to(ROOT)), "rows": 336, "sha256": sha256(candidate_path)},
        "prompt_sample": {"source_path": str(sample_path.relative_to(ROOT)), "rows": 60, "sha256": sha256(output / "construction_prompts_k60.jsonl"), "selection": "existing V6 stratified_seeded size=60 seed=42"},
        "calibration_prompt_ids": prompt_ids[:12],
        "attack_selection_policy": "all executable original train rows with original registered LoRA subset; first executable same-configuration seed replicates for validation/test; no outcomes used",
        "attack_splits": {"train": train_report, "validation": validation_report, "test": test_report},
        "files": {name: {"rows": len(rows), "sha256": sha256(output / name)} for name, rows in files.items()},
        "lora_variant_ids": {"train": [row["variant_id"] for row in train_lora], "validation": [row["variant_id"] for row in validation if row["family"] == "finetuning"], "test": [row["variant_id"] for row in test if row["family"] == "finetuning"]},
    }
    (output / "design_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
