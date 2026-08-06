from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


ATTACK_SEEDS = (2435199792, 4072939906)
MICRO_SEED = 1685839144
FAMILIES = (
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
)
STOCHASTIC_FAMILIES = {"structured_pruning", "gaussian_noise", "finetuning"}


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


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-results", default="results/experiment_a_qwen14b_20260805/analysis/per_prompt_transfer_results.jsonl")
    parser.add_argument("--pairs", default="experiments/prompt-transfer-14b-a1/inputs/accepted16_pairs.jsonl")
    parser.add_argument("--expanded", default="experiments/prompt-transfer-14b-a1/inputs/prepared/validation_candidates.jsonl")
    parser.add_argument("--attack-manifest", default="reproducibility/v6_20260729/test_attack_manifest.jsonl")
    parser.add_argument("--output-dir", default="experiments/prompt-transfer-14b-b1/inputs")
    args = parser.parse_args()

    a_path, pair_path, expanded_path, attack_path = map(resolve, (args.a_results, args.pairs, args.expanded, args.attack_manifest))
    output_dir = resolve(args.output_dir)
    a_rows = read_jsonl(a_path)
    pair_rows = read_jsonl(pair_path)
    expanded_rows = read_jsonl(expanded_path)
    attack_rows = read_jsonl(attack_path)

    legacy_ids = [str(row["prompt_id"]) for row in a_rows if row["legacy_retained"]]
    strict_ids = [str(row["prompt_id"]) for row in a_rows if row["strict_5of5_retained"]]
    if len(legacy_ids) != 8 or len(strict_ids) != 4 or not set(strict_ids) <= set(legacy_ids):
        raise ValueError("Frozen Experiment A cohorts do not match 8 legacy / 4 strict")

    pair_by_id = {str(row["prompt_id"]): row for row in pair_rows}
    selected_pairs = [pair_by_id[prompt_id] for prompt_id in legacy_ids]
    selected_expanded = [row for row in expanded_rows if str(row["source_prompt_id"]) in set(legacy_ids)]
    if len(selected_expanded) != 16:
        raise ValueError(f"Expected 16 expanded endpoints, received {len(selected_expanded)}")

    templates: dict[str, dict[str, Any]] = {}
    for row in attack_rows:
        family = str(row["family"])
        templates.setdefault(family, row)
    if set(templates) != set(FAMILIES):
        raise ValueError(f"Attack template families mismatch: {sorted(templates)}")

    manifests: list[dict[str, Any]] = []
    for family in FAMILIES:
        for seed in ATTACK_SEEDS:
            row = json.loads(json.dumps(templates[family]))
            row["split"] = "validation"
            row["seed"] = seed
            row["variant_id"] = f"b1_{family}_{seed}"
            row["replication_independence"] = (
                "independent_seed_and_fresh_materialization"
                if family in STOCHASTIC_FAMILIES
                else "fresh_materialization_deterministic_configuration"
            )
            manifests.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_pairs_path = output_dir / "survivor_pairs.jsonl"
    selected_expanded_path = output_dir / "survivor_prompts.jsonl"
    manifest_path = output_dir / "attack_manifest_validation.jsonl"
    write_jsonl(selected_pairs_path, selected_pairs)
    write_jsonl(selected_expanded_path, selected_expanded)
    write_jsonl(manifest_path, manifests)

    payload = {
        "schema_version": "experiment_b_frozen_inputs_1.0",
        "source_result": str(a_path.relative_to(ROOT)),
        "source_result_sha256": sha256(a_path),
        "selection_rules": {
            "legacy": "Experiment A legacy_retained == true",
            "strict": "Experiment A strict_5of5_retained == true",
        },
        "legacy_prompt_ids": legacy_ids,
        "strict_prompt_ids": strict_ids,
        "attack_seeds": list(ATTACK_SEEDS),
        "micro_seed": MICRO_SEED,
        "seed_derivation": "first three big-endian uint32 words of SHA256(experiment-b-survivor-replication-v1)",
        "files": {
            "pairs": {"path": str(selected_pairs_path.relative_to(ROOT)), "sha256": sha256(selected_pairs_path), "rows": 8},
            "prompts": {"path": str(selected_expanded_path.relative_to(ROOT)), "sha256": sha256(selected_expanded_path), "rows": 16},
            "attacks": {"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path), "rows": 10},
        },
    }
    manifest = output_dir / "cohort_manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
