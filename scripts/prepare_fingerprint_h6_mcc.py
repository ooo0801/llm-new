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


CORE_CATEGORIES = {"code", "instruction", "knowledge", "logic", "reasoning", "safety", "structured", "summary"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_h6_qwen32b_mcc.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_path(config_path)
    config = load_config(config_path)
    source_path = project_path(config["data"]["candidate_source"])
    source_rows = read_jsonl(source_path)
    minimum = int(config["fingerprint"]["minimum_candidate_size"])
    if len(source_rows) < minimum:
        raise ValueError(f"H6 confirmed pool has {len(source_rows)} rows; {minimum} required")
    categories = {str(row["category"]) for row in source_rows}
    missing = sorted(CORE_CATEGORIES - categories)
    if missing:
        raise ValueError(f"H6 confirmed pool misses core categories: {missing}")
    candidates = []
    for row in source_rows:
        prompt_id = str(row["prompt_id"])
        prompt = str(row["optimized_prompt"]).strip()
        observed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if observed != str(row["optimized_prompt_sha256"]):
            raise ValueError(f"{prompt_id}: optimized prompt SHA-256 mismatch")
        candidates.append(
            {
                "id": prompt_id,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "category": str(row["category"]),
                "language": row.get("language", "unknown"),
                "source": "experiment_h6_independent_confirmation",
                "prompt_sha256": observed,
                "evaluator": row.get("evaluator"),
                "expected_answer": row.get("expected_answer"),
                "expected_contains": row.get("expected_contains"),
                "parent_source_prompt_id": row.get("parent_14b_source_id", row.get("source_prompt_id")),
            }
        )
    if len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate H6 candidate IDs")
    if len({row["prompt_sha256"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate H6 candidate texts")

    calibration = []
    build_count = int(config["calibration"]["build_per_family"])
    audit_count = int(config["calibration"]["audit_per_family"])
    for index in range(build_count + audit_count):
        for family in TASK_FAMILIES:
            prompt, language, length_bucket = calibration_prompt(family, index)
            split = "build" if index < build_count else "audit"
            prompt_id = stable_id(f"fingerprint_h6_32b|{family}|{prompt}", f"cal32b_{family}")
            calibration.append(
                {
                    "id": prompt_id,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "category": family,
                    "language": language,
                    "length_bucket": length_bucket,
                    "calibration_split": split,
                    "round": index,
                    "source": "fingerprint_h6_32b_independent_calibration",
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                }
            )
    build = [row for row in calibration if row["calibration_split"] == "build"]
    audit = [row for row in calibration if row["calibration_split"] == "audit"]
    threshold = float(config["calibration"]["near_duplicate_trigram_jaccard"])
    closest = {"score": 0.0, "calibration_id": None, "candidate_id": None}
    for cal in calibration:
        for candidate in candidates:
            score = trigram_jaccard(str(cal["prompt"]), str(candidate["prompt"]))
            if score > float(closest["score"]):
                closest = {"score": score, "calibration_id": cal["id"], "candidate_id": candidate["id"]}
    if float(closest["score"]) >= threshold:
        raise ValueError(f"H6 calibration/candidate near-duplicate gate failed: {closest}")
    normalized = [normalize_text(str(row["prompt"])) for row in calibration]
    if len(normalized) != len(set(normalized)):
        raise ValueError("H6 calibration prompts are not unique after normalization")
    write_jsonl(project_path(config["data"]["candidate_manifest"]), candidates)
    write_jsonl(project_path(config["data"]["calibration_build_manifest"]), build)
    write_jsonl(project_path(config["data"]["calibration_audit_manifest"]), audit)
    protocol = {
        "schema_version": "fingerprint_h6_qwen32b_mcc_protocol_1.0",
        "hypothesis": "H6-G",
        "model": {"name": config["model"]["name"], "revision": config["model"]["revision"]},
        "candidate_source": str(source_path.relative_to(ROOT).as_posix()),
        "candidate_source_sha256": file_sha256(source_path),
        "candidate_rows": len(candidates),
        "candidate_categories": dict(sorted(Counter(row["category"] for row in candidates).items())),
        "candidate_manifest_sha256": canonical_sha256(candidates),
        "calibration_build_prompts": len(build),
        "calibration_audit_prompts": len(audit),
        "calibration_build_sha256": canonical_sha256(build),
        "calibration_audit_sha256": canonical_sha256(audit),
        "closest_candidate_trigram_jaccard": closest,
        "component_thresholds": dict(config["activations"]),
        "selection": {"method": config["fingerprint"]["primary_selection"], "selected_size": config["fingerprint"]["selected_size"]},
        "config_sha256": file_sha256(config_path),
        "leakage_guard": "H6-G uses only the independently confirmed H6-P pool; 14B components and confirmation outcomes are not reused.",
    }
    write_json(project_path(config["reproducibility_dir"]) / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
