from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from _bootstrap import ROOT, project_path
from prepare_fingerprint_v2 import TASK_FAMILIES, calibration_prompt, canonical_sha256, normalize_text, trigram_jaccard

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, stable_id, write_json, write_jsonl


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Qwen2.5-14B component-universe and MCC inputs")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_g_qwen14b_mcc.yaml"))
    args = parser.parse_args()
    config_path = resolve_config(args.config)
    config = load_config(config_path)
    labels = config.get("protocol_labels", {})
    construction_label = str(labels.get("construction", "G1"))
    candidate_source_label = str(
        labels.get("candidate_source", "experiment_f1_independent_confirmation")
    )
    source_path = project_path(config["data"]["candidate_source"])
    source_rows = read_jsonl(source_path)
    minimum_candidates = int(config["fingerprint"].get("minimum_candidate_size", 23))
    if len(source_rows) < minimum_candidates:
        raise ValueError(f"confirmed fingerprint source has {len(source_rows)} rows; at least {minimum_candidates} required")

    candidates = []
    for row in source_rows:
        prompt_id = str(row["prompt_id"])
        prompt = str(row["optimized_prompt"]).strip()
        if not prompt:
            raise ValueError(f"{prompt_id}: empty optimized prompt")
        expected_hash = str(row["optimized_prompt_sha256"])
        observed_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if observed_hash != expected_hash:
            raise ValueError(f"{prompt_id}: optimized prompt hash mismatch")
        candidates.append({
            "id": prompt_id,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "category": str(row["category"]),
            "language": row.get("language", "unknown"),
            "source": candidate_source_label,
            "prompt_sha256": observed_hash,
            "evaluator": row.get("evaluator"),
            "expected_answer": row.get("expected_answer"),
            "expected_contains": row.get("expected_contains"),
            "parent_source_prompt_id": row.get("source_prompt_id"),
        })
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate confirmed candidate IDs")
    if len({row["prompt_sha256"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate confirmed candidate texts")

    build_per_family = int(config["calibration"]["build_per_family"])
    audit_per_family = int(config["calibration"]["audit_per_family"])
    calibration = []
    for index in range(build_per_family + audit_per_family):
        for family in TASK_FAMILIES:
            prompt, language, length_bucket = calibration_prompt(family, index)
            split = "build" if index < build_per_family else "audit"
            prompt_id = stable_id(
                f"fingerprint_14b_{construction_label.lower()}|{family}|{prompt}",
                f"cal14b_{family}",
            )
            calibration.append({
                "id": prompt_id,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "category": family,
                "language": language,
                "length_bucket": length_bucket,
                "calibration_split": split,
                "round": index,
                "source": f"fingerprint_14b_{construction_label.lower()}_independent_calibration",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })

    build = [row for row in calibration if row["calibration_split"] == "build"]
    audit = [row for row in calibration if row["calibration_split"] == "audit"]
    if len(build) != len(TASK_FAMILIES) * build_per_family or len(audit) != len(TASK_FAMILIES) * audit_per_family:
        raise RuntimeError("calibration allocation mismatch")
    if len({row["id"] for row in calibration}) != len(calibration):
        raise ValueError("duplicate calibration IDs")

    threshold = float(config["calibration"]["near_duplicate_trigram_jaccard"])
    closest = {"score": 0.0, "calibration_id": None, "candidate_id": None}
    for cal in calibration:
        for candidate in candidates:
            score = trigram_jaccard(str(cal["prompt"]), str(candidate["prompt"]))
            if score > float(closest["score"]):
                closest = {"score": score, "calibration_id": cal["id"], "candidate_id": candidate["id"]}
    if float(closest["score"]) >= threshold:
        raise ValueError(f"calibration/candidate near-duplicate gate failed: {closest}")
    normalized = [normalize_text(str(row["prompt"])) for row in calibration]
    if len(normalized) != len(set(normalized)):
        raise ValueError("normalized calibration prompts are not unique")

    candidate_target = project_path(config["data"]["candidate_manifest"])
    build_target = project_path(config["data"]["calibration_build_manifest"])
    audit_target = project_path(config["data"]["calibration_audit_manifest"])
    write_jsonl(candidate_target, candidates)
    write_jsonl(build_target, build)
    write_jsonl(audit_target, audit)
    attacks = list(config.get("attacks", []))
    if attacks:
        write_jsonl(project_path(config["data"]["attack_manifest"]), attacks)
    release_dir = project_path(config["reproducibility_dir"])
    protocol = {
        "schema_version": f"fingerprint_14b_{construction_label.lower()}_mcc_protocol_1.0",
        "protocol_frozen_on": "2026-08-07",
        "hypothesis": f"H-{construction_label}",
        "candidate_source": str(source_path.relative_to(ROOT).as_posix()),
        "candidate_rows": len(candidates),
        "candidate_categories": dict(sorted(Counter(row["category"] for row in candidates).items())),
        "candidate_source_sha256": file_sha256(source_path),
        "candidate_manifest_sha256": canonical_sha256(candidates),
        "calibration_build_prompts": len(build),
        "calibration_audit_prompts": len(audit),
        "calibration_families": list(TASK_FAMILIES),
        "calibration_build_sha256": canonical_sha256(build),
        "calibration_audit_sha256": canonical_sha256(audit),
        "closest_candidate_trigram_jaccard": closest,
        "component_thresholds": {
            "attention_entropy_fraction": config["activations"]["attention_entropy_fraction"],
            "ffn_quantile": config["activations"]["ffn_quantile"],
            "residual_threshold": config["activations"]["residual_threshold"],
            "max_ffn_units_per_layer": config["activations"]["max_ffn_units_per_layer"],
        },
        "selection": {"method": config["fingerprint"]["primary_selection"], "selected_size": config["fingerprint"]["selected_size"]},
        "attack_manifest_sha256": canonical_sha256(attacks) if attacks else None,
        "leakage_guard": "Thresholds and saturation/audit gates are the frozen 7B values tested without adjustment on 14B; a failure cannot be repaired inside this protocol.",
        "config_path": str(config_path.relative_to(ROOT).as_posix()),
        "config_sha256": file_sha256(config_path),
    }
    write_json(release_dir / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))


def resolve_config(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_path(path)


if __name__ == "__main__":
    main()
