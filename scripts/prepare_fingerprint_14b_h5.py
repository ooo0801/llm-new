from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import write_json, write_jsonl


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(rows):
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_h5_qwen14b_mismatch.yaml"))
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    source = project_path(config["data"]["source_fingerprint"])
    if sha256(source) != config["data"]["source_fingerprint_sha256"]:
        raise ValueError("H5 source fingerprint hash mismatch")
    fingerprint = ModelFingerprint.load(source)
    expected_size = int(config["fingerprint"]["selected_size"])
    if len(fingerprint.entries) != expected_size or len({entry.prompt_id for entry in fingerprint.entries}) != expected_size:
        raise ValueError("H5 source is not the frozen unique MCC12")
    if fingerprint.model_revision != config["model"]["revision"]:
        raise ValueError("H5 source model revision mismatch")
    attacks = list(config["attacks"])
    if len({row["variant_id"] for row in attacks}) != len(attacks):
        raise ValueError("duplicate H5 variant IDs")
    write_jsonl(project_path(config["data"]["attack_manifest"]), attacks)
    repetitions = int(config["statistics"]["repetitions"])
    protocol = {
        "schema_version": "fingerprint_14b_h5_protocol_1.0",
        "hypothesis": "H-H5",
        "source_fingerprint": config["data"]["source_fingerprint"],
        "source_fingerprint_sha256": sha256(source),
        "selected_prompt_ids": [entry.prompt_id for entry in fingerprint.entries],
        "reference_seeds": [config["fingerprint"]["reference_seed_base"] + index for index in range(repetitions)],
        "target_seeds": [config["statistics"]["target_seed_base"] + index for index in range(repetitions)],
        "primary_method": config["statistics"]["method"],
        "null_block_mismatch_rate": config["statistics"]["null_block_mismatch_rate"],
        "alpha": config["statistics"]["alpha"],
        "minimum_reject_blocks": 4,
        "required_modified": 9,
        "required_family_detected": config["statistics"]["required_family_detected"],
        "attack_counts": dict(sorted(Counter(row["family"] for row in attacks).items())),
        "attack_manifest_sha256": canonical_sha256(attacks),
        "config_sha256": sha256(config_path),
        "leakage_guard": "H3/H4 remain No-Go; no H5 endpoint may change prompts, seeds, states, null rate, alpha, four-block boundary, or family gates.",
    }
    if protocol["reference_seeds"] != protocol["target_seeds"]:
        raise ValueError("H5 requires common random numbers")
    write_json(project_path(config["reproducibility_dir"]) / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
