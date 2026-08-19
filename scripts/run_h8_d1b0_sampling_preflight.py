#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d1b_sampling import (
    AUDIT_ROLE,
    FIT_ROLE,
    FORMAL_TOTAL,
    SMOKE_ROLE,
    SMOKE_TOTAL,
    D1BGenerationRequest,
    audit_schedule,
    build_formal_schedule,
    build_smoke_schedule,
    file_sha256,
    load_manifest,
    make_manifest_payload,
    validate_schedule,
    write_manifest,
)
from llm_integrity.h8_m0f_sampling import discover_repository_seeds
from llm_integrity.h8_sampling import (
    canonical_json_sha256,
    gpu_compute_processes,
    runtime_provenance,
    sha256_text,
)
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement
from run_h8_m0f_calibration_sampling import verify_live_bundle, verify_runtime, verify_snapshot
from run_h8_mmd_smoke import generate_one, load_fingerprint, read_config


DEFAULT_CONFIG = ROOT / "configs" / "h8_d1b0_score_calibration_sampling.yaml"
REQUIRED_RESPONSE_FIELDS = {
    "response_id",
    "schedule_position",
    "round_id",
    "prompt_id",
    "seed",
    "attempt_records",
    "rendered_prompt_sha256",
    "input_token_ids",
    "input_token_ids_sha256",
    "completion_token_ids",
    "completion_token_ids_sha256",
    "raw_response",
    "stop_reason",
    "data_role",
    "intended_bank_role",
    "runtime_identity_sha256",
    "frozen_provenance_sha256",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(value: str) -> Path:
    path = (ROOT / value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes repository")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("phase") != "H8_D1B0_SCORE_CALIBRATION_SAMPLING_PREFLIGHT":
        raise ValueError("Invalid D1-B0 config")
    if value.get("authorization_status") != "user_approved_preflight_only":
        raise PermissionError("D1-B0 preflight is not authorized")
    if value.get("formal_sampling_authorized") is not False:
        raise PermissionError("This D1-B0 config must not authorize formal sampling")
    if value.get("smoke_sampling_authorized") is not True:
        raise PermissionError("D1-B0 smoke is not authorized")
    generation = value["generation"]
    required = {
        "batch_size": 1,
        "prompt_count": 12,
        "formal_rounds": 200,
        "formal_responses_per_prompt": 200,
        "formal_fit_per_prompt": 100,
        "formal_audit_per_prompt": 100,
        "formal_total_responses": 2400,
        "even_round_role": FIT_ROLE,
        "odd_round_role": AUDIT_ROLE,
    }
    for key, expected in required.items():
        if generation.get(key) != expected:
            raise ValueError(f"D1-B0 generation setting mismatch: {key}")
    if not all(flag is True for flag in value["forbidden_operations"].values()):
        raise PermissionError("All D1-B0 forbidden operations must remain enabled")
    return value


def git_commit_and_tracked_clean(config: Mapping[str, Any]) -> str:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(config["preflight_base_commit"]), "HEAD"],
        cwd=ROOT,
        check=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree is not clean")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def frozen_context(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    frozen = config["frozen_inputs"]
    require_hash(resolve(frozen["protocol"]), frozen["protocol_sha256"], "D1-B0 protocol")
    require_hash(resolve(frozen["d1a_report"]), frozen["d1a_report_sha256"], "D1-A report")
    require_hash(resolve(frozen["h8_config"]), frozen["h8_config_sha256"], "H8 generation config")
    require_hash(
        resolve(frozen["m0f_runtime_authorization"]),
        frozen["m0f_runtime_authorization_sha256"],
        "M0-F runtime identity authorization",
    )
    binding = load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=frozen["mmd_manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )
    d1a = json.loads(resolve(frozen["d1a_report"]).read_text(encoding="utf-8"))
    if d1a.get("status") != "PASS" or d1a.get("score_schema_sha256") != frozen["score_schema_sha256"]:
        raise ValueError("D1-A PASS/score schema binding mismatch")
    if d1a.get("score_calibration_status") != "preflight_only_not_frozen":
        raise ValueError("D1-A score calibration status mismatch")
    h8_config = read_config(resolve(frozen["h8_config"]))
    fingerprint, fingerprint_path = load_fingerprint(h8_config)
    require_hash(fingerprint_path, frozen["fingerprint_sha256"], "H6 MCC12 fingerprint")
    if set(binding.prompt_ids) != {str(entry["prompt_id"]) for entry in fingerprint["entries"]}:
        raise ValueError("Frozen MMD prompt set and MCC12 prompt set differ")
    runtime_auth = yaml.safe_load(resolve(frozen["m0f_runtime_authorization"]).read_text(encoding="utf-8"))
    git_commit_and_tracked_clean(config)
    return config, h8_config, fingerprint, runtime_auth


def artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    directory = resolve(artifacts["directory"])
    return {
        "directory": directory,
        "formal_manifest": directory / artifacts["formal_manifest"],
        "formal_audit": directory / artifacts["formal_manifest_audit"],
        "smoke_manifest": directory / artifacts["smoke_manifest"],
        "smoke_audit": directory / artifacts["smoke_manifest_audit"],
        "report": directory / artifacts["preflight_report"],
        "smoke_output": resolve(artifacts["smoke_output_dir"]),
    }


def prepare_manifests(config_path: Path) -> int:
    config, h8_config, fingerprint, _ = frozen_context(config_path)
    paths = artifact_paths(config)
    if any(paths[key].exists() for key in ("formal_manifest", "formal_audit", "smoke_manifest", "smoke_audit")):
        raise FileExistsError("Refusing to overwrite D1-B0 manifests")
    repository_seeds, source_files = discover_repository_seeds(ROOT)
    generation = config["generation"]
    formal = build_formal_schedule(
        fingerprint["entries"], generation["formal_generation_root_seed"], repository_seeds
    )
    formal_seeds = {row.seed for row in formal}
    smoke_forbidden = repository_seeds | formal_seeds
    smoke = build_smoke_schedule(
        fingerprint["entries"], generation["smoke_generation_root_seed"], smoke_forbidden
    )
    smoke_seeds = {row.seed for row in smoke}
    if formal_seeds & smoke_seeds:
        raise ValueError("Formal and smoke generation seeds overlap")
    frozen_identity = {
        "runner_commit": git_commit_and_tracked_clean(config),
        "config_sha256": file_sha256(config_path),
        "protocol_sha256": config["frozen_inputs"]["protocol_sha256"],
        "mmd_manifest_sha256": config["frozen_inputs"]["mmd_manifest_sha256"],
        "mmd_implementation_commit": config["frozen_inputs"]["mmd_implementation_commit"],
        "d1a_report_sha256": config["frozen_inputs"]["d1a_report_sha256"],
        "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
        "h8_config_sha256": config["frozen_inputs"]["h8_config_sha256"],
        "fingerprint_sha256": config["frozen_inputs"]["fingerprint_sha256"],
        "model_name": h8_config["model"]["name"],
        "model_revision": h8_config["model"]["revision"],
        "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
    }
    formal_payload = make_manifest_payload(
        formal,
        root_seed=generation["formal_generation_root_seed"],
        frozen_identity=frozen_identity,
        source_fingerprint_sha256=config["frozen_inputs"]["fingerprint_sha256"],
        repository_seed_sources=source_files,
        forbidden_seed_count=len(repository_seeds),
        cpu_resampling_root_seed_reserved=generation["cpu_resampling_root_seed_reserved"],
    )
    smoke_payload = make_manifest_payload(
        smoke,
        root_seed=generation["smoke_generation_root_seed"],
        frozen_identity=frozen_identity,
        source_fingerprint_sha256=config["frozen_inputs"]["fingerprint_sha256"],
        repository_seed_sources=source_files,
        forbidden_seed_count=len(smoke_forbidden),
        cpu_resampling_root_seed_reserved=generation["cpu_resampling_root_seed_reserved"],
    )
    formal_info = write_manifest(paths["formal_manifest"], formal_payload, smoke=False)
    smoke_info = write_manifest(paths["smoke_manifest"], smoke_payload, smoke=True)
    formal_audit = audit_schedule(formal, smoke=False, forbidden_seeds=repository_seeds)
    smoke_audit = audit_schedule(smoke, smoke=True, forbidden_seeds=smoke_forbidden)
    formal_audit.update(
        manifest_file_sha256=formal_info["file_sha256"],
        manifest_payload_sha256=formal_info["payload_sha256"],
        formal_sampling_authorized=False,
        fit_audit_seed_overlap_count=0,
        smoke_seed_overlap_count=0,
        repository_seed_source_file_count=len(source_files),
    )
    smoke_audit.update(
        manifest_file_sha256=smoke_info["file_sha256"],
        manifest_payload_sha256=smoke_info["payload_sha256"],
        formal_seed_overlap_count=0,
        repository_seed_source_file_count=len(source_files),
    )
    atomic_json(paths["formal_audit"], formal_audit)
    atomic_json(paths["smoke_audit"], smoke_audit)
    print(json.dumps({"status": "PASS", "formal": formal_audit, "smoke": smoke_audit}, ensure_ascii=False, indent=2))
    return 0


def manifest_context(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[D1BGenerationRequest],
    list[D1BGenerationRequest],
    dict[str, Path],
]:
    config, h8_config, fingerprint, runtime_auth = frozen_context(config_path)
    paths = artifact_paths(config)
    for key in ("formal_manifest", "formal_audit", "smoke_manifest", "smoke_audit"):
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing frozen D1-B0 artifact: {key}")
    formal_audit = json.loads(paths["formal_audit"].read_text(encoding="utf-8"))
    smoke_audit = json.loads(paths["smoke_audit"].read_text(encoding="utf-8"))
    if formal_audit.get("status") != "PASS" or smoke_audit.get("status") != "PASS":
        raise ValueError("D1-B0 manifest audit is not PASS")
    if formal_audit["manifest_file_sha256"] != file_sha256(paths["formal_manifest"]):
        raise ValueError("Formal manifest file hash mismatch")
    if smoke_audit["manifest_file_sha256"] != file_sha256(paths["smoke_manifest"]):
        raise ValueError("Smoke manifest file hash mismatch")
    formal_payload, formal = load_manifest(paths["formal_manifest"], smoke=False)
    smoke_payload, smoke = load_manifest(paths["smoke_manifest"], smoke=True)
    if formal_payload["frozen_identity"]["config_sha256"] != file_sha256(config_path):
        raise ValueError("Formal manifest config binding mismatch")
    if smoke_payload["frozen_identity"] != formal_payload["frozen_identity"]:
        raise ValueError("Formal and smoke frozen identities differ")
    if {row.seed for row in formal} & {row.seed for row in smoke}:
        raise ValueError("Formal and smoke manifests overlap in generation seeds")
    entries = fingerprint["entries"]

    def hydrate(rows: list[D1BGenerationRequest]) -> list[D1BGenerationRequest]:
        result = []
        for row in rows:
            entry = entries[row.prompt_index]
            if entry["prompt_id"] != row.prompt_id or entry["metadata"]["prompt_sha256"] != row.prompt_sha256:
                raise ValueError("Manifest schedule no longer matches MCC12")
            result.append(D1BGenerationRequest(**{**row.__dict__, "prompt": entry["prompt"]}))
        return result

    return config, h8_config, fingerprint, runtime_auth, hydrate(formal), hydrate(smoke), paths


def verify_manifests(config_path: Path) -> int:
    config, _, _, _, formal, smoke, paths = manifest_context(config_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "formal_sampling_authorized": config["formal_sampling_authorized"],
                "formal_request_count": len(formal),
                "smoke_request_count": len(smoke),
                "formal_unique_seeds": len({row.seed for row in formal}),
                "smoke_unique_seeds": len({row.seed for row in smoke}),
                "cross_manifest_seed_overlap": len({row.seed for row in formal} & {row.seed for row in smoke}),
                "formal_manifest_sha256": file_sha256(paths["formal_manifest"]),
                "smoke_manifest_sha256": file_sha256(paths["smoke_manifest"]),
                "formal_generation_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_attempts(
    bundle: Any,
    request: D1BGenerationRequest,
    h8_config: Mapping[str, Any],
    event_handle: Any,
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    attempts: list[dict[str, Any]] = []
    for attempt_index in range(max_attempts):
        try:
            generated = generate_one(bundle, request, dict(h8_config))
            event = {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "attempt_index": attempt_index,
                "seed": request.seed,
                "status": "success",
                "created_at_utc": now(),
            }
            append_jsonl(event_handle, event)
            attempts.append(event)
            generated["attempt_count"] = len(attempts)
            generated["retry_seed_changed"] = any(int(item["seed"]) != request.seed for item in attempts)
            return generated, attempts
        except (RuntimeError, OSError) as exc:
            event = {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "attempt_index": attempt_index,
                "seed": request.seed,
                "status": "technical_failure",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "created_at_utc": now(),
            }
            append_jsonl(event_handle, event)
            attempts.append(event)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"Same-seed retry budget exhausted for {request.response_id}")


def smoke_worker(config_path: Path) -> int:
    config, h8_config, fingerprint, runtime_auth, formal, smoke, paths = manifest_context(config_path)
    if not config["smoke_sampling_authorized"] or config["formal_sampling_authorized"]:
        raise PermissionError("D1-B0 smoke/formal authorization state is invalid")
    output_dir = paths["smoke_output"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite D1-B0 smoke output")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = config["artifacts"]
    response_path = output_dir / artifacts["smoke_responses"]
    event_path = output_dir / artifacts["smoke_attempt_events"]
    verify_runtime(runtime_auth)
    snapshot = verify_snapshot(runtime_auth, h8_config)
    from llm_integrity.modeling import load_model

    bundle = load_model(h8_config["model"])
    try:
        live_identity = verify_live_bundle(bundle, runtime_auth, h8_config)
        provenance = {
            "mmd_manifest_sha256": config["frozen_inputs"]["mmd_manifest_sha256"],
            "d1a_report_sha256": config["frozen_inputs"]["d1a_report_sha256"],
            "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
            "formal_manifest_sha256": file_sha256(paths["formal_manifest"]),
            "smoke_manifest_sha256": file_sha256(paths["smoke_manifest"]),
            "formal_seed_set_sha256": canonical_json_sha256(sorted(row.seed for row in formal)),
            "smoke_seed_set_sha256": canonical_json_sha256(sorted(row.seed for row in smoke)),
            "model_snapshot": snapshot,
            "live_identity": live_identity,
            "runtime": runtime_provenance(),
            "generation_config": h8_config["generation"],
            "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
            "fingerprint_sha256": config["frozen_inputs"]["fingerprint_sha256"],
            "runner_commit": git_commit_and_tracked_clean(config),
        }
        provenance_sha = canonical_json_sha256(provenance)
        with response_path.open("x", encoding="utf-8", newline="\n") as response_handle, event_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as event_handle:
            for request in smoke:
                current = verify_live_bundle(bundle, runtime_auth, h8_config)
                if current["identity_sha256"] != live_identity["identity_sha256"]:
                    raise ValueError("Runtime/model identity drifted during smoke")
                generated, attempts = run_attempts(
                    bundle,
                    request,
                    h8_config,
                    event_handle,
                    int(config["generation"]["max_attempts_same_seed"]),
                )
                entry = fingerprint["entries"][request.prompt_index]
                record = {
                    "schema_version": "h8-d1b0-smoke-response-1.0",
                    "mode": "score_calibration_smoke_only",
                    "data_role": SMOKE_ROLE,
                    "intended_bank_role": request.intended_bank_role,
                    "response_id": request.response_id,
                    "schedule_position": request.schedule_position,
                    "round_id": request.round_id,
                    "prompt_position_in_round": request.prompt_position_in_round,
                    "prompt_index": request.prompt_index,
                    "prompt_id": request.prompt_id,
                    "prompt": request.prompt,
                    "prompt_sha256": request.prompt_sha256,
                    "category": entry.get("category"),
                    "task_metadata": entry.get("metadata", {}),
                    "seed": request.seed,
                    "seed_digest_sha256": request.seed_digest_sha256,
                    "derivation_counter": request.derivation_counter,
                    "batch_size": 1,
                    "attempt_records": attempts,
                    "model_name": bundle.name,
                    "model_revision": bundle.revision,
                    "runtime_identity_sha256": live_identity["identity_sha256"],
                    "frozen_provenance_sha256": provenance_sha,
                    "formal_manifest_file_sha256": file_sha256(paths["formal_manifest"]),
                    "smoke_manifest_file_sha256": file_sha256(paths["smoke_manifest"]),
                    "eligible_for_formal_reference": False,
                    "eligible_for_heldout_evaluation": False,
                    "eligible_for_target_evaluation": False,
                    "eligible_for_attack_evaluation": False,
                    "eligible_for_score_calibration_fit": False,
                    "eligible_for_score_stability_audit": False,
                    "created_at_utc": now(),
                    **generated,
                }
                append_jsonl(response_handle, record)
        records = read_jsonl(response_path)
        if len(records) != SMOKE_TOTAL:
            raise RuntimeError("D1-B0 smoke did not produce exactly 24 records")
        worker_report = {
            "status": "PASS",
            "response_count": len(records),
            "unique_seed_count": len({row["seed"] for row in records}),
            "unique_response_id_count": len({row["response_id"] for row in records}),
            "per_prompt_counts": dict(sorted(Counter(row["prompt_id"] for row in records).items())),
            "actual_role_counts": dict(sorted(Counter(row["data_role"] for row in records).items())),
            "intended_role_counts": dict(sorted(Counter(row["intended_bank_role"] for row in records).items())),
            "missing_required_field_count": sum(len(REQUIRED_RESPONSE_FIELDS - set(row)) for row in records),
            "formal_seed_overlap_count": len({row["seed"] for row in records} & {item.seed for item in formal}),
            "retry_seed_changed_count": sum(bool(row["retry_seed_changed"]) for row in records),
            "legal_first_token_eos_count": sum(bool(row["legal_first_token_eos"]) for row in records),
            "stop_reason_counts": dict(sorted(Counter(row["stop_reason"] for row in records).items())),
            "responses_file_sha256": file_sha256(response_path),
            "attempt_events_file_sha256": file_sha256(event_path),
            "frozen_provenance": provenance,
            "frozen_provenance_sha256": provenance_sha,
            "formal_bank_response_count": 0,
        }
        atomic_json(output_dir / "worker_report.json", worker_report)
        return 0
    finally:
        bundle.close()
        del bundle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass


def smoke_parent(config_path: Path) -> int:
    config, _, _, _, formal, smoke, paths = manifest_context(config_path)
    if config["formal_sampling_authorized"] is not False:
        raise PermissionError("D1-B0 cannot run smoke with formal sampling authorized")
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--smoke-worker"],
        cwd=ROOT,
        check=False,
    )
    cleanup = gpu_compute_processes()
    output_dir = paths["smoke_output"]
    worker_path = output_dir / "worker_report.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8")) if worker_path.is_file() else None
    status = "PASS" if completed.returncode == 0 and worker and worker["status"] == "PASS" and cleanup["status"] == "PASS" else "FAIL"
    report = {
        "schema_version": "h8-d1b0-preflight-report-1.0",
        "phase": "H8_D1B0_SCORE_CALIBRATION_SAMPLING_PREFLIGHT",
        "status": status,
        "formal_sampling_authorized": False,
        "formal_generation_started": False,
        "formal_response_count": 0,
        "planned_formal_response_count": FORMAL_TOTAL,
        "planned_fit_response_count": 1200,
        "planned_audit_response_count": 1200,
        "smoke_response_count": 0 if not worker else worker["response_count"],
        "smoke_records_eligible_for_formal_bank": False,
        "worker_exit_code": completed.returncode,
        "worker": worker,
        "gpu_cleanup": cleanup,
        "formal_manifest_file_sha256": file_sha256(paths["formal_manifest"]),
        "smoke_manifest_file_sha256": file_sha256(paths["smoke_manifest"]),
        "formal_smoke_seed_overlap_count": len({row.seed for row in formal} & {row.seed for row in smoke}),
        "mmd_manifest_sha256": config["frozen_inputs"]["mmd_manifest_sha256"],
        "d1a_report_sha256": config["frozen_inputs"]["d1a_report_sha256"],
        "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
        "batch_size": 1,
        "future_sample_structures": ["r40_q10", "r40_q20", "r60_q10", "r60_q20"],
        "four_structure_response_banks_generated": False,
        "cross_structure_pooling_authorized": False,
        "score_parameter_fit_performed": False,
        "sample_size_selection_performed": False,
        "top_r_selection_performed": False,
        "created_at_utc": now(),
        "next_gate": "STOP_AND_WAIT_FOR_EXPLICIT_2400_RESPONSE_SAMPLING_APPROVAL",
    }
    if paths["report"].exists():
        raise FileExistsError("Refusing to overwrite D1-B0 preflight report")
    atomic_json(paths["report"], report)
    print(json.dumps({"status": status, "report": str(paths["report"])}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


def formal_parent(config_path: Path) -> int:
    config = load_config(config_path)
    if config.get("formal_sampling_authorized") is not True:
        raise PermissionError(
            "Formal 2400-response D1-B sampling is not authorized; preflight config is fail-closed"
        )
    raise PermissionError("A separate explicit formal authorization artifact is required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 D1-B0 manifest and 24-response smoke runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-manifests-only", action="store_true")
    parser.add_argument("--verify-manifests-only", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--smoke-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if args.prepare_manifests_only:
        return prepare_manifests(config_path)
    if args.verify_manifests_only:
        return verify_manifests(config_path)
    if args.formal:
        return formal_parent(config_path)
    if args.smoke_worker:
        return smoke_worker(config_path)
    return smoke_parent(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
