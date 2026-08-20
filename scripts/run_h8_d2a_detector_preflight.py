#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d1c import load_frozen_score_calibration
from llm_integrity.h8_d2a import (
    ATTACK_FAMILIES,
    SAMPLE_STRUCTURES,
    TOP_R_VALUES,
    build_generation_schedules,
    configuration_manifest_payload,
    generation_manifest_payload,
    make_nested_subset_payload,
    make_permutation_seed_payload,
    save_canonical_envelope,
)
from llm_integrity.h8_m0f_sampling import discover_repository_seeds
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement


DEFAULT_CONFIG = ROOT / "configs" / "h8_d2a_detector_development_preflight.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes repository")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Frozen {label} SHA256 mismatch")


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
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree must be clean for D2-A preflight")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def collect_identifier_values(path: Path) -> set[str]:
    identifiers: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"variant_id", "attack_instance_id", "endpoint_id"} and isinstance(item, str):
                    identifiers.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for candidate in path.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml"}:
            continue
        try:
            if candidate.suffix.lower() == ".jsonl":
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    if line:
                        visit(json.loads(line))
            elif candidate.suffix.lower() in {".yaml", ".yml"}:
                visit(yaml.safe_load(candidate.read_text(encoding="utf-8")))
            else:
                visit(json.loads(candidate.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
            continue
    return identifiers


def load_context(config_path: Path) -> tuple[dict[str, Any], Any, dict[str, Any], dict[str, Any], str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("phase") != "H8_D2A_DETECTOR_DEVELOPMENT_PREFLIGHT"
        or config.get("authorization_status") != "user_approved_preflight_only"
        or config.get("model_sampling_authorized") is not False
        or config.get("development_comparison_authorized") is not False
    ):
        raise PermissionError("D2-A authorization boundary mismatch")
    if not all(value is True for value in config["forbidden_operations"].values()):
        raise PermissionError("Every D2-A forbidden operation must remain true")
    detector = config["detector"]
    if (
        list(detector["sample_structures"]) != list(SAMPLE_STRUCTURES)
        or list(detector["primary_top_r"]) != list(TOP_R_VALUES)
        or int(detector["development_permutations"]) != 999
        or float(detector["alpha"]) != 0.05
        or detector["energy_implemented"] is not False
    ):
        raise ValueError("D2-A detector constants changed")
    frozen = config["frozen_inputs"]
    require_hash(resolve(frozen["fingerprint"]), frozen["fingerprint_sha256"], "H6 fingerprint")
    require_hash(resolve(frozen["h8_generation_config"]), frozen["h8_generation_config_sha256"], "H8 config")
    require_hash(resolve(frozen["protocol"]), frozen["protocol_sha256"], "D2-A protocol")
    binding = load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=frozen["mmd_manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )
    score = load_frozen_score_calibration(
        resolve(frozen["score_archive"]),
        expected_manifest_sha256=frozen["score_manifest_sha256"],
        binding=binding,
        expected_score_schema_sha256=frozen["score_schema_sha256"],
    )
    if score.get("artifact_status") != "frozen" or score.get("sample_size_selection_performed") is not False:
        raise ValueError("Frozen score layer state mismatch")
    if score.get("audit_data_used_for_parameter_fit") is not False:
        raise ValueError("Frozen score layer indicates leakage")
    if score.get("structures", {}).keys() != set(SAMPLE_STRUCTURES):
        raise ValueError("Frozen score layer does not retain all structures")
    fingerprint = json.loads(resolve(frozen["fingerprint"]).read_text(encoding="utf-8"))
    if len(fingerprint.get("entries", [])) != 12:
        raise ValueError("H6 fingerprint no longer contains MCC12")
    commit = clean_commit()
    return config, binding, score, fingerprint, commit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config, binding, score, fingerprint, commit = load_context(config_path)
    output = config["output"]
    output_dir = resolve(output["directory"])
    if output_dir.exists():
        raise FileExistsError("D2-A output already exists; refusing overwrite")
    output_dir.mkdir(parents=True)

    test = subprocess.run(
        ["python", "-m", "pytest", "tests/test_h8_d2a.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError(f"D2-A CPU unit tests failed:\n{test.stdout}\n{test.stderr}")

    repository_seeds, seed_sources = discover_repository_seeds(ROOT)
    attacks = [dict(row) for row in config["fresh_attack_instances"]]
    h6_ids = collect_identifier_values(resolve(config["frozen_inputs"]["h6_archive"]))
    fresh_ids = {row["attack_instance_id"] for row in attacks}
    overlap = fresh_ids & h6_ids
    if overlap:
        raise ValueError(f"Fresh D2 attack IDs overlap H6 archive IDs: {sorted(overlap)}")
    schedules = build_generation_schedules(
        fingerprint["entries"],
        attacks,
        reference_root_seed=int(config["seed_roots"]["reference_generation_uint32"]),
        intact_target_root_seed=int(config["seed_roots"]["intact_target_generation_uint32"]),
        attack_root_seed=int(config["seed_roots"]["attack_generation_uint32"]),
        forbidden_generation_seeds=repository_seeds,
    )
    all_generation_seeds = {row.generation_seed for rows in schedules.values() for row in rows}
    if all_generation_seeds & repository_seeds:
        raise ValueError("D2-A proposed generation seeds overlap tracked repository seeds")
    h8_generation = yaml.safe_load(resolve(config["frozen_inputs"]["h8_generation_config"]).read_text(encoding="utf-8"))
    frozen_identity = {
        "preflight_commit": commit,
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": config["frozen_inputs"]["protocol_sha256"],
        "mmd_manifest_sha256": binding.manifest_sha256,
        "score_manifest_sha256": config["frozen_inputs"]["score_manifest_sha256"],
        "score_parameter_payload_sha256": config["frozen_inputs"]["score_parameter_payload_sha256"],
        "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
        "fingerprint_sha256": config["frozen_inputs"]["fingerprint_sha256"],
        "model_name": h8_generation["model"]["name"],
        "model_revision": h8_generation["model"]["revision"],
        "generation_config_sha256": hashlib.sha256(
            json.dumps(h8_generation["generation"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    infos: dict[str, dict[str, str]] = {}
    for schedule_name, output_key, kind, artifact_type in (
        ("reference", "reference_manifest", "development_reference_720", "h8_d2a_reference_generation_manifest"),
        ("intact_target", "intact_target_manifest", "development_intact_target_1200", "h8_d2a_intact_target_generation_manifest"),
        ("attack", "attack_generation_manifest", "development_attack_1920", "h8_d2a_attack_generation_manifest"),
    ):
        payload = generation_manifest_payload(
            schedules[schedule_name],
            manifest_kind=kind,
            frozen_identity=frozen_identity,
            repository_seed_source_count=len(seed_sources),
        )
        infos[output_key] = save_canonical_envelope(
            output_dir / output[output_key], artifact_type, payload
        )
    attack_payload = {
        "schema_version": "h8-d2a-detector-development-1.0",
        "status": "proposed_not_materialized",
        "fresh_endpoint_contract": "materialize_after_separate_authorization_never_reuse_h6_endpoint",
        "h6_archived_identifier_count": len(h6_ids),
        "h6_identifier_overlap_count": len(overlap),
        "families": list(ATTACK_FAMILIES),
        "endpoints_per_family": 2,
        "instances": attacks,
    }
    infos["attack_instance_manifest"] = save_canonical_envelope(
        output_dir / output["attack_instance_manifest"], "h8_d2a_fresh_attack_instance_manifest", attack_payload
    )
    nested = make_nested_subset_payload(
        schedules, subset_root_seed=int(config["seed_roots"]["nested_subset_uint64"])
    )
    infos["nested_subset_manifest"] = save_canonical_envelope(
        output_dir / output["nested_subset_manifest"], "h8_d2a_nested_subset_manifest", nested
    )
    permutation = make_permutation_seed_payload(
        nested, permutation_root_seed=int(config["seed_roots"]["global_permutation_uint64"])
    )
    infos["permutation_seed_manifest"] = save_canonical_envelope(
        output_dir / output["permutation_seed_manifest"], "h8_d2a_global_permutation_seed_manifest", permutation
    )
    configs = configuration_manifest_payload()
    infos["configuration_manifest"] = save_canonical_envelope(
        output_dir / output["configuration_manifest"], "h8_d2a_configuration_manifest", configs
    )
    index = {
        key: {
            "path": output[key],
            **info,
        }
        for key, info in sorted(infos.items())
    }
    atomic_json(output_dir / output["artifact_index"], index)
    report = {
        "schema_version": "h8-d2a-preflight-report-1.0",
        "phase": config["phase"],
        "status": "PASS",
        "preflight_commit": commit,
        "measurement_layer": "frozen",
        "score_layer": "frozen",
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "new_model_responses": 0,
        "development_comparison_performed": False,
        "formal_reference_responses": 0,
        "final_heldout_responses": 0,
        "final_attack_or_confirmation_responses": 0,
        "proposed_reference_responses": len(schedules["reference"]),
        "proposed_intact_target_responses": len(schedules["intact_target"]),
        "proposed_attack_responses": len(schedules["attack"]),
        "proposed_total_responses": sum(len(rows) for rows in schedules.values()),
        "unique_proposed_generation_seeds": len(all_generation_seeds),
        "repository_generation_seed_overlap_count": len(all_generation_seeds & repository_seeds),
        "configuration_count": configs["configuration_count"],
        "primary_top_r": list(TOP_R_VALUES),
        "max_diagnostic_only": True,
        "energy_implemented": False,
        "local_p_values_used_for_aggregation": False,
        "development_permutations": 999,
        "alpha": 0.05,
        "nested_subset_invariants": nested["nested_invariants"],
        "intact_evaluation_units": nested["intact_evaluation_unit_count"],
        "fresh_attack_evaluation_units": nested["attack_evaluation_unit_count"],
        "fresh_attack_family_counts": dict(sorted(Counter(row["family"] for row in attacks).items())),
        "fresh_attack_h6_identifier_overlap_count": len(overlap),
        "permutation_stream_count": permutation["stream_count"],
        "derived_permutation_seed_count": permutation["derived_seed_count"],
        "derived_permutation_seed_unique": permutation["derived_seed_unique"],
        "mmd_manifest_sha256": binding.manifest_sha256,
        "score_manifest_sha256": config["frozen_inputs"]["score_manifest_sha256"],
        "score_parameter_payload_sha256": config["frozen_inputs"]["score_parameter_payload_sha256"],
        "unit_tests": {
            "command": "python -m pytest tests/test_h8_d2a.py -q",
            "status": "PASS",
            "stdout": test.stdout.strip(),
        },
        "artifacts": index,
        "created_at_utc": now(),
        "next_gate": "STOP_WAIT_FOR_EXPLICIT_D2_DEVELOPMENT_SAMPLING_AUTHORIZATION",
    }
    atomic_json(output_dir / output["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

