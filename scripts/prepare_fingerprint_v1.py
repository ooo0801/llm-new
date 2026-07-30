from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_json, write_jsonl


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(serialized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the accepted V6 prompts as fingerprint V1")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v1_qwen_7b.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    source_path = project_path(config["data"]["strict16_source"])
    target_path = project_path(config["data"]["strict16_manifest"])
    attack_path = project_path(config["data"]["attack_manifest"])
    rows = read_jsonl(source_path)
    accepted = [row for row in rows if bool(row.get("optimization", {}).get("accepted"))]
    if len(rows) != 22 or len(accepted) != 16:
        raise ValueError(f"Expected the frozen V6 evidence to contain 22 rows and 16 accepts; got {len(rows)} and {len(accepted)}")

    frozen = []
    for row in accepted:
        optimization = dict(row.get("optimization", {}))
        prompt_id = str(row.get("prompt_id", row.get("id", "")))
        initial_prompt = str(optimization.get("initial_prompt", ""))
        optimized_prompt = str(optimization.get("optimized_prompt", ""))
        if not prompt_id or not initial_prompt or not optimized_prompt:
            raise ValueError(f"Incomplete V6 provenance for {prompt_id!r}")
        if optimized_prompt == initial_prompt or optimized_prompt == str(row.get("prompt", "")):
            raise ValueError(f"V1 must freeze the discrete optimized prompt for {prompt_id!r}")
        if not bool(optimization.get("discrete_change_verified")):
            raise ValueError(f"V6 did not verify a discrete change for {prompt_id!r}")
        validation = dict(optimization.get("hybrid_validation", {}))
        strict = dict(optimization.get("strict_validation", {}))
        frozen.append(
            {
                "id": prompt_id,
                "prompt_id": prompt_id,
                "category": row.get("category"),
                "prompt": optimized_prompt,
                "initial_prompt": initial_prompt,
                "evaluator": row.get("evaluator"),
                "expected_answer": row.get("expected_answer"),
                "expected_contains": row.get("expected_contains"),
                "prompt_sha256": sha256_text(optimized_prompt),
                "source_record_sha256": canonical_sha256(row),
                "v6_hybrid_gain": float(validation.get("objective_gain", 0.0)),
                "v6_nondegraded_families": int(strict.get("nondegraded_families", 0)),
                "v6_acceptance_stage": optimization.get("acceptance_stage"),
                "source_evidence": str(config["data"]["strict16_source"]),
            }
        )
    if len({row["id"] for row in frozen}) != 16:
        raise ValueError("The strict V1 prompt IDs are not unique")
    write_jsonl(target_path, frozen)

    attacks = list(config.get("attacks", []))
    if not attacks or len({str(row["variant_id"]) for row in attacks}) != len(attacks):
        raise ValueError("Attack manifest must contain unique variants")
    write_jsonl(attack_path, attacks)

    report = {
        "schema_version": "fingerprint_v1_protocol_1.0",
        "protocol_frozen_on": "2026-07-30",
        "version_boundary": "Close the full downstream experiment on the existing 16 V6 strict accepts; category expansion is a future version.",
        "source_rows": len(rows),
        "strict16_rows": len(frozen),
        "category_counts": dict(sorted(Counter(str(row["category"]) for row in frozen).items())),
        "optimized_prompt_enforced": True,
        "selected_size": int(config["fingerprint"]["selected_size"]),
        "reference_repetitions": int(config["fingerprint"]["reference_repetitions"]),
        "target_repetitions": int(config["statistics"]["repetitions"]),
        "primary_test": str(config["statistics"]["method"]),
        "pooled_mmd_is_ablation": bool(config["statistics"].get("report_pooled_baseline")),
        "attack_variants": [str(row["variant_id"]) for row in attacks],
        "strict16_sha256": canonical_sha256(frozen),
        "attack_manifest_sha256": canonical_sha256(attacks),
        "config_path": str(Path(args.config).as_posix()),
        "config_sha256": sha256_text(Path(args.config).read_text(encoding="utf-8")),
    }
    output = project_path(config["reproducibility_dir"]) / "protocol.json"
    write_json(output, report)
    print(json.dumps(report | {"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
