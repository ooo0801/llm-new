#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from llm_integrity.h8_d2a import (
    configuration_manifest_payload,
    load_canonical_envelope as load_d2a_envelope,
)
from llm_integrity.h8_d2b0 import (
    audit_fresh_attack_provenance,
    audit_lora_training_isolation,
    build_attack_smoke_manifest,
    file_sha256,
    save_canonical_envelope,
)
from llm_integrity.h8_m0f_sampling import discover_repository_seeds


D2A_DIR = ROOT / "reproducibility/h8_qwen32b_detector_development_20260820/d2a"
OUTPUT_DIR = ROOT / "reproducibility/h8_qwen32b_detector_development_20260820/d2b0_authorization"
FINGERPRINT = ROOT / "reproducibility/fingerprint_h6_qwen32b_mcc_20260812/components/fingerprints/global_mcc_32b_h6.json"
CONFIRMED = ROOT / "reproducibility/fingerprint_h6_qwen32b_mcc_20260812/confirmation/confirmed_prompt_pool.jsonl"
TRAINING_DATA = ROOT / "data/h8_d2_fresh_isolated_attack_training_data_v1.jsonl"
H6_MANIFESTS = (
    ROOT / "experiments/prompt-reconstruction-32b-h6/inputs/attack_manifest_train_executable.jsonl",
    ROOT / "experiments/prompt-reconstruction-32b-h6/inputs/attack_manifest_development_2each.jsonl",
    ROOT / "experiments/prompt-reconstruction-32b-h6/inputs/attack_manifest_confirmation_2each.jsonl",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise ValueError("D2-B0 authorization artifacts must be prepared from a clean implementation commit")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_d2a_artifact(index: dict[str, Any], key: str, artifact_type: str) -> dict[str, Any]:
    row = index[key]
    return load_d2a_envelope(
        D2A_DIR / row["path"],
        expected_artifact_type=artifact_type,
        expected_file_sha256=row["file_sha256"],
        expected_payload_sha256=row["payload_sha256"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty D2-B0 authorization archive")
    implementation_commit = clean_commit()
    index = json.loads((D2A_DIR / "D2A_ARTIFACT_SHA256_INDEX.json").read_text(encoding="utf-8"))
    attack_instances = load_d2a_artifact(
        index, "attack_instance_manifest", "h8_d2a_fresh_attack_instance_manifest"
    )["instances"]
    reference = load_d2a_artifact(index, "reference_manifest", "h8_d2a_reference_generation_manifest")
    intact = load_d2a_artifact(index, "intact_target_manifest", "h8_d2a_intact_target_generation_manifest")
    attack = load_d2a_artifact(index, "attack_generation_manifest", "h8_d2a_attack_generation_manifest")
    nested = load_d2a_artifact(index, "nested_subset_manifest", "h8_d2a_nested_subset_manifest")
    permutation = load_d2a_artifact(
        index, "permutation_seed_manifest", "h8_d2a_global_permutation_seed_manifest"
    )
    fingerprint = json.loads(FINGERPRINT.read_text(encoding="utf-8"))
    confirmed = read_jsonl(CONFIRMED)
    training_rows = read_jsonl(TRAINING_DATA)
    lora_audit = audit_lora_training_isolation(training_rows, fingerprint, confirmed)
    freshness = audit_fresh_attack_provenance(attack_instances, H6_MANIFESTS)
    if freshness["status"] != "PASS" or lora_audit["status"] != "PASS":
        raise RuntimeError("D2-B0 freshness or LoRA isolation preflight failed")
    formal_seeds = {
        int(row["generation_seed"])
        for payload in (reference, intact, attack)
        for row in payload["schedule"]
    }
    repository_seeds, seed_sources = discover_repository_seeds(ROOT)
    smoke = build_attack_smoke_manifest(
        attack_instances,
        fingerprint["entries"],
        root_seed=2026082201,
        forbidden_seeds=repository_seeds,
        formal_development_seeds=formal_seeds,
    )
    config_payload = configuration_manifest_payload()
    config_payload.update(
        {
            "revision": "D2-B0-count-based-selection-v1",
            "supersedes_d2a_configuration_manifest_file_sha256": index["configuration_manifest"]["file_sha256"],
            "development_intact_unit_count": 5,
            "development_attack_endpoints_per_family": 2,
            "development_count_interpretation": (
                "configuration-selection counts only; not formal FPR/TPR estimates"
            ),
            "future_fpr_tpr_evaluation": "fresh_heldout_confirmation_only",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, str]] = {}
    artifacts["revised_configuration_manifest"] = save_canonical_envelope(
        output_dir / "D2B0_REVISED_CONFIGURATION_MANIFEST.json",
        "h8_d2b0_revised_configuration_manifest",
        config_payload,
    )
    artifacts["freshness_preflight"] = save_canonical_envelope(
        output_dir / "D2B0_FRESH_ATTACK_PROVENANCE_PREFLIGHT.json",
        "h8_d2b0_fresh_attack_provenance_preflight",
        freshness,
    )
    artifacts["lora_training_isolation"] = save_canonical_envelope(
        output_dir / "D2B0_LORA_TRAINING_DATA_ISOLATION_AUDIT.json",
        "h8_d2b0_lora_training_isolation_audit",
        {
            **lora_audit,
            "training_data_path": str(TRAINING_DATA.relative_to(ROOT)),
            "training_data_sha256": file_sha256(TRAINING_DATA),
        },
    )
    artifacts["attack_smoke_manifest"] = save_canonical_envelope(
        output_dir / "D2B0_ATTACK_SMOKE_GENERATION_MANIFEST.json",
        "h8_d2b0_attack_smoke_generation_manifest",
        smoke,
    )
    report = {
        "schema_version": "h8-d2b0-pre-smoke-authorization-report-1.0",
        "phase": "H8_D2B0_DEVELOPMENT_SAMPLING_AUTHORIZATION_PREFLIGHT",
        "status": "PASS_READY_FOR_EIGHT_SMOKE_ONLY",
        "implementation_commit": implementation_commit,
        "formal_development_sampling_authorized": False,
        "attack_smoke_authorized": True,
        "formal_development_responses": 0,
        "attack_smoke_responses": 0,
        "development_comparison_performed": False,
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "revised_configuration_count": config_payload["configuration_count"],
        "selection_rule": config_payload["selection_rule"],
        "freshness_status": freshness["status"],
        "h6_explicit_variant_id_count": freshness["h6_explicit_variant_id_count"],
        "id_level_overlap_status": freshness["id_level_overlap_status"],
        "lora_training_isolation_status": lora_audit["status"],
        "smoke_request_count": smoke["request_count"],
        "smoke_unique_seed_count": smoke["unique_generation_seed_count"],
        "smoke_formal_seed_overlap_count": smoke["formal_development_seed_overlap_count"],
        "d2a_nested_subset_file_sha256": index["nested_subset_manifest"]["file_sha256"],
        "d2a_permutation_seed_file_sha256": index["permutation_seed_manifest"]["file_sha256"],
        "d2a_nested_invariants": nested["nested_invariants"],
        "d2a_permutation_derived_seed_set_sha256": permutation["derived_seed_set_sha256"],
        "repository_seed_source_count": len(seed_sources),
        "artifacts": artifacts,
        "created_at_utc": now(),
        "next_gate": "COMMIT_AUTHORIZATION_ARTIFACTS_THEN_RUN_EXACTLY_EIGHT_ATTACK_SMOKE_RESPONSES",
    }
    atomic_json(output_dir / "H8_D2B0_PRE_SMOKE_AUTHORIZATION_REPORT.json", report)
    atomic_json(
        output_dir / "D2B0_PRE_SMOKE_SHA256_INDEX.json",
        {
            **artifacts,
            "pre_smoke_report": {
                "file_sha256": file_sha256(output_dir / "H8_D2B0_PRE_SMOKE_AUTHORIZATION_REPORT.json")
            },
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
