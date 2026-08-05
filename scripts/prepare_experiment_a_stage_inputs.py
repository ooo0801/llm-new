from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


FAMILIES = (
    "unstructured_pruning",
    "structured_pruning",
    "quantization",
    "gaussian_noise",
    "finetuning",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        default="experiments/prompt-transfer-14b-a1/inputs/accepted16_pairs.jsonl",
    )
    parser.add_argument(
        "--manifest",
        default="reproducibility/v6_20260729/test_attack_manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/prompt-transfer-14b-a1/inputs",
    )
    args = parser.parse_args()
    pair_path = ROOT / args.pairs
    manifest_path = ROOT / args.manifest
    output_dir = ROOT / args.output_dir

    pairs = read_jsonl(pair_path)
    if len(pairs) != 16:
        raise ValueError(f"expected 16 pairs, received {len(pairs)}")
    first = pairs[0]
    source_id = str(first["prompt_id"])
    stage_pair = []
    for role in ("initial", "optimized"):
        stage_pair.append(
            {
                "id": f"{source_id}::{role}",
                "prompt_id": f"{source_id}::{role}",
                "source_prompt_id": source_id,
                "candidate_role": role,
                "category": first["category"],
                "prompt": first[f"{role}_prompt"],
                "evaluator": first["evaluator"],
                "expected_answer": first.get("expected_answer"),
                "expected_contains": first.get("expected_contains"),
            }
        )
    stage_pair_path = output_dir / "stage_one_pair.jsonl"
    write_jsonl(stage_pair_path, stage_pair)

    frozen = read_jsonl(manifest_path)
    engineering = []
    source_variant_ids = {}
    for family in FAMILIES:
        selected = [
            row
            for row in frozen
            if row["family"] == family and int(row["seed"]) == 3407
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one seed-3407 row for {family}, received {len(selected)}"
            )
        row = deepcopy(selected[0])
        source_variant_ids[family] = row["variant_id"]
        row["split"] = "test"
        row["variant_id"] = f"engineering_{family}_seed3407"
        if family == "finetuning":
            row["configuration"]["steps"] = 1
            row["variant_id"] = "engineering_finetuning_one_step_seed3407"
        engineering.append(row)
    engineering_path = output_dir / "engineering_five_family_seed3407.jsonl"
    write_jsonl(engineering_path, engineering)

    report = {
        "schema_version": "experiment_a_stage_inputs_1.0",
        "stage_one_source_prompt_id": source_id,
        "stage_one_pair_path": str(stage_pair_path.relative_to(ROOT)),
        "stage_one_pair_sha256": sha256(stage_pair_path),
        "frozen_manifest_path": str(manifest_path.relative_to(ROOT)),
        "frozen_manifest_sha256": sha256(manifest_path),
        "engineering_manifest_path": str(engineering_path.relative_to(ROOT)),
        "engineering_manifest_sha256": sha256(engineering_path),
        "engineering_seed": 3407,
        "families": list(FAMILIES),
        "source_variant_ids": source_variant_ids,
        "engineering_only": True,
        "lora_steps": 1,
        "confirmatory_lora_steps_unchanged": 50,
    }
    report_path = output_dir / "stage_inputs_manifest.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
