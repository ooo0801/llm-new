from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from llm_integrity.inner_variant_sampler import (
    FAMILIES,
    StratifiedVariantSampler,
    load_registry,
    read_jsonl,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = root / "configs/paper_aligned_qwen_7b_joint_inner.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompts = read_jsonl(
        root
        / "results/paper_aligned_qwen_7b/prompt_optimization/"
        "seeds_stratified_k60.jsonl"
    )
    manifests = read_jsonl(
        root
        / "results/paper_aligned_qwen_7b/manifests_truthful_quant/"
        "attack_manifest_train.jsonl"
    )
    registry = load_registry(
        root
        / "results/paper_aligned_qwen_7b/finetuning_adapters/"
        "adapter_registry_train.json"
    )
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights=config["joint_inner"]["macro"]["families"],
        adapter_registry=registry,
        seed=int(config["seed"]),
    )

    schedule_audits: list[dict[str, Any]] = []
    all_valid = True
    for row in prompts:
        prompt_id = str(row["id"])
        schedule = sampler.balanced_schedule(prompt_id)
        family_counts = Counter(item["sample"].family for item in schedule)
        block_counts = Counter(item["block_type"] for item in schedule)
        probabilities_valid = all(
            item["sample"].q_family > 0
            and item["sample"].q_variant_given_family > 0
            and item["sample"].q_joint > 0
            and math.isfinite(item["sample"].importance_correction)
            for item in schedule
        )
        valid = (
            len(schedule) == 15
            and family_counts == Counter({family: 3 for family in FAMILIES})
            and block_counts
            == Counter({"q_proj": 5, "v_proj": 5, "down_proj": 5})
            and probabilities_valid
        )
        all_valid = all_valid and valid
        schedule_audits.append(
            {
                "prompt_id": prompt_id,
                "valid": valid,
                "family_counts": dict(family_counts),
                "block_counts": dict(block_counts),
                "family_order": sampler.family_order(prompt_id),
                "variants": [
                    {
                        "family": item["sample"].family,
                        "variant_id": item["sample"].variant_id,
                        "q_family": item["sample"].q_family,
                        "q_variant_given_family": (
                            item["sample"].q_variant_given_family
                        ),
                        "q_joint": item["sample"].q_joint,
                        "importance_correction": (
                            item["sample"].importance_correction
                        ),
                    }
                    for item in schedule[::3]
                ],
            }
        )

    code_files = [
        config_path,
        root / "src/llm_integrity/inner_micro_proxy.py",
        root / "src/llm_integrity/inner_variant_sampler.py",
        root / "src/llm_integrity/joint_inner_optimizer.py",
        root / "scripts/calibrate_inner_micro.py",
        root / "scripts/calibrate_inner_macro.py",
        root / "scripts/run_joint_inner_optimization.py",
    ]
    payload = {
        "passed": all_valid and len(schedule_audits) == 60,
        "prompt_schedules_audited": len(schedule_audits),
        "valid_schedules": sum(row["valid"] for row in schedule_audits),
        "families": list(FAMILIES),
        "steps_per_prompt": 15,
        "steps_per_family": 3,
        "code_sha256": {
            str(path.relative_to(root)): sha256(path) for path in code_files
        },
        "schedules": schedule_audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "passed",
                    "prompt_schedules_audited",
                    "valid_schedules",
                    "steps_per_prompt",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
