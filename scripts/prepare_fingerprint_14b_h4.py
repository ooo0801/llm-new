from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import write_json, write_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_h4_qwen14b_paired.yaml"))
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    source = project_path(config["data"]["source_fingerprint"])
    if sha256(source) != config["data"]["source_fingerprint_sha256"]:
        raise ValueError("H4 source fingerprint hash mismatch")
    fingerprint = ModelFingerprint.load(source)
    expected_size = int(config["fingerprint"]["selected_size"])
    if len(fingerprint.entries) != expected_size or len({entry.prompt_id for entry in fingerprint.entries}) != expected_size:
        raise ValueError("H4 source is not the frozen unique MCC12")
    if fingerprint.model_revision != config["model"]["revision"]:
        raise ValueError("H4 source model revision mismatch")
    attacks = list(config["attacks"])
    if len({row["variant_id"] for row in attacks}) != len(attacks):
        raise ValueError("duplicate H4 variant IDs")
    write_jsonl(project_path(config["data"]["attack_manifest"]), attacks)
    protocol = {
        "schema_version": "fingerprint_14b_h4_protocol_1.0",
        "hypothesis": "H-H4",
        "source_fingerprint": config["data"]["source_fingerprint"],
        "source_fingerprint_sha256": sha256(source),
        "selected_prompt_ids": [entry.prompt_id for entry in fingerprint.entries],
        "reference_seeds": [config["fingerprint"]["reference_seed_base"] + index for index in range(config["fingerprint"]["reference_repetitions"])],
        "target_seeds": [config["statistics"]["target_seed_base"] + index for index in range(config["statistics"]["repetitions"])],
        "primary_method": config["statistics"]["method"],
        "standardization": config["statistics"]["standardization"],
        "exact_sign_patterns": 1 << int(config["statistics"]["repetitions"]),
        "attack_counts": dict(sorted(Counter(row["family"] for row in attacks).items())),
        "attack_manifest_sha256": canonical_sha256(attacks),
        "config_sha256": sha256(config_path),
        "leakage_guard": "H3 remains No-Go; no H4 endpoint may change prompts, seeds, states, features, alpha, or paired block decision rule.",
    }
    if protocol["reference_seeds"] != protocol["target_seeds"]:
        raise ValueError("H4 requires common random numbers")
    write_json(project_path(config["reproducibility_dir"]) / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
