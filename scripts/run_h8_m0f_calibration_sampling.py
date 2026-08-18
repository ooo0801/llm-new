#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_m0f_sampling import (
    M0F_DATA_ROLE,
    M0F_TOTAL_RESPONSES,
    CalibrationRequest,
    audit_calibration_records,
    build_calibration_schedule,
    discover_repository_seeds,
    file_sha256,
    load_manifest,
    make_manifest_payload,
    validate_calibration_schedule,
    verify_file_hash,
    write_manifest,
)
from llm_integrity.h8_precalibration import build_h8_feature_schema, canonical_sha256
from llm_integrity.h8_sampling import (
    canonical_json_sha256,
    gpu_compute_processes,
    runtime_provenance,
    sha256_bytes,
    sha256_text,
)
from run_h8_mmd_smoke import (
    cached_snapshot_provenance,
    generate_one,
    load_fingerprint,
    read_config,
)


DEFAULT_AUTH = ROOT / "configs" / "h8_m0f_calibration_sampling.yaml"
REQUIRED_RECORD_FIELDS = (
    "response_id",
    "schedule_position",
    "prompt_id",
    "replicate_id",
    "seed",
    "attempt_count",
    "attempt_records",
    "rendered_prompt_sha256",
    "input_token_ids",
    "input_token_ids_sha256",
    "completion_token_ids",
    "completion_token_ids_sha256",
    "raw_response",
    "raw_response_sha256",
    "response_token_count_including_eos",
    "stop_reason",
    "model_revision",
    "runtime_identity_sha256",
    "data_role",
    "eligible_for_formal_reference",
    "eligible_for_heldout_evaluation",
    "eligible_for_attack_evaluation",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_authorization(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("phase") != "H8_M0F_CALIBRATION_SAMPLING":
        raise ValueError("Invalid H8 M0-F authorization config")
    if value.get("authorization_status") != "user_approved":
        raise PermissionError("H8 M0-F has not been explicitly authorized")
    sampling = value["sampling"]
    expected = {
        "data_role": M0F_DATA_ROLE,
        "eligible_for_formal_reference": False,
        "eligible_for_heldout_evaluation": False,
        "eligible_for_attack_evaluation": False,
        "prompt_count": 12,
        "replicate_rounds": 100,
        "responses_per_prompt": 100,
        "total_responses": 1200,
        "batch_size": 1,
    }
    for key, expected_value in expected.items():
        if sampling.get(key) != expected_value:
            raise ValueError(f"Frozen M0-F setting mismatch: {key}")
    forbidden = value["forbidden_operations"]
    if not forbidden or not all(flag is True for flag in forbidden.values()):
        raise PermissionError("Every post-sampling/attack operation must remain forbidden")
    return value


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_base_files(auth: Mapping[str, Any]) -> None:
    frozen = auth["frozen_inputs"]
    verify_file_hash(resolve(frozen["h8_config"]), frozen["h8_config_sha256"], "H8 config")
    verify_file_hash(resolve(frozen["protocol"]), frozen["protocol_sha256"], "H8 protocol")
    verify_file_hash(
        resolve(frozen["preflight_archive"]),
        frozen["preflight_archive_sha256"],
        "H8 preflight archive",
    )
    verify_file_hash(
        resolve(frozen["preflight_report"]),
        frozen["preflight_report_sha256"],
        "H8 full preflight report",
    )
    schema = build_h8_feature_schema(512)
    if schema.sha256 != frozen["feature_schema_sha256"]:
        raise ValueError("Frozen feature schema hash mismatch")


def verify_git_lineage(auth: Mapping[str, Any]) -> str:
    base = str(auth["preflight_base_commit"])
    subprocess.run(["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT, check=True)
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if status:
        raise ValueError(f"Tracked working tree is not clean: {status}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def runtime_identity_for_comparison() -> dict[str, Any]:
    raw = runtime_provenance()
    return {
        "python": str(raw["python"]).split()[0],
        "numpy": raw["numpy"],
        "torch": raw["torch"],
        "transformers": raw["transformers"],
        "torch_cuda": str(raw["torch_cuda"]),
        "cudnn": raw["cudnn"],
        "gpu_names": raw["gpu_names"],
    }


def verify_runtime(auth: Mapping[str, Any]) -> dict[str, Any]:
    actual = runtime_identity_for_comparison()
    expected = auth["runtime_identity"]
    for key in ("python", "numpy", "torch", "transformers", "torch_cuda", "cudnn", "gpu_names"):
        if actual[key] != expected[key]:
            raise ValueError(f"Runtime identity mismatch for {key}: {actual[key]!r} != {expected[key]!r}")
    return actual


def verify_snapshot(auth: Mapping[str, Any], h8_config: Mapping[str, Any]) -> dict[str, Any]:
    expected = auth["runtime_identity"]
    snapshot = cached_snapshot_provenance(
        h8_config["model"]["name"], h8_config["model"]["revision"]
    )
    if snapshot.get("resolution_error") or not snapshot.get("resolved_snapshot"):
        raise ValueError(f"Model snapshot provenance failed: {snapshot.get('resolution_error')}")
    if snapshot.get("snapshot_listing_sha256") != expected["model_snapshot_listing_sha256"]:
        raise ValueError("Model snapshot listing hash mismatch")
    actual_small = {row["name"]: row["sha256"] for row in snapshot["small_files"]}
    if actual_small != expected["small_file_sha256"]:
        raise ValueError("Model/tokenizer small-file hash mismatch")
    snapshot_path = Path(snapshot["resolved_snapshot"])
    if len(list(snapshot_path.glob("model-*.safetensors"))) != 17:
        raise ValueError("Expected exactly 17 Qwen32B weight shards")
    return snapshot


def load_frozen_context(auth_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    auth = load_authorization(auth_path)
    verify_base_files(auth)
    verify_git_lineage(auth)
    h8_config_path = resolve(auth["frozen_inputs"]["h8_config"])
    h8_config = read_config(h8_config_path)
    fingerprint, fingerprint_path = load_fingerprint(h8_config)
    if file_sha256(fingerprint_path) != auth["frozen_inputs"]["fingerprint_sha256"]:
        raise ValueError("H6 MCC12 fingerprint hash mismatch")
    verify_runtime(auth)
    verify_snapshot(auth, h8_config)
    return auth, h8_config, fingerprint, fingerprint_path


def smoke_seed_set(auth: Mapping[str, Any]) -> set[int]:
    sampling = auth["sampling"]
    return set(range(int(sampling["smoke_seed_min"]), int(sampling["smoke_seed_max"]) + 1))


def prepare_manifest(auth_path: Path) -> int:
    auth, h8_config, fingerprint, fingerprint_path = load_frozen_context(auth_path)
    manifest_path = resolve(auth["artifacts"]["manifest"])
    audit_path = resolve(auth["artifacts"]["manifest_audit"])
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError("Refusing to overwrite a frozen M0-F manifest/audit")
    repository_seeds, source_files = discover_repository_seeds(ROOT)
    forbidden = repository_seeds | smoke_seed_set(auth)
    schedule = build_calibration_schedule(
        fingerprint["entries"], int(auth["sampling"]["root_seed"]), forbidden
    )
    final_seeds = {request.seed for request in schedule}
    overlap = final_seeds & forbidden
    if overlap:
        raise ValueError(f"Generated M0-F seeds overlap frozen history: {sorted(overlap)[:10]}")
    frozen_identity = {
        "preflight_base_commit": auth["preflight_base_commit"],
        "authorization_sha256": file_sha256(auth_path),
        "h8_config_sha256": auth["frozen_inputs"]["h8_config_sha256"],
        "protocol_sha256": auth["frozen_inputs"]["protocol_sha256"],
        "preflight_archive_sha256": auth["frozen_inputs"]["preflight_archive_sha256"],
        "preflight_report_sha256": auth["frozen_inputs"]["preflight_report_sha256"],
        "feature_schema_sha256": auth["frozen_inputs"]["feature_schema_sha256"],
        "runtime_identity": auth["runtime_identity"],
    }
    payload = make_manifest_payload(
        schedule,
        int(auth["sampling"]["root_seed"]),
        file_sha256(fingerprint_path),
        frozen_identity,
        source_files,
        len(forbidden),
    )
    manifest_info = write_manifest(manifest_path, payload)
    audit = {
        "schema_version": "h8-m0f-calibration-seed-manifest-audit-1.0",
        "status": "PASS",
        "created_at_utc": now(),
        "manifest": manifest_info,
        "total_responses": len(schedule),
        "prompt_count": len({request.prompt_id for request in schedule}),
        "per_prompt_counts": dict(sorted(Counter(request.prompt_id for request in schedule).items())),
        "replicate_rounds": len({request.replicate_id for request in schedule}),
        "unique_seed_count": len(final_seeds),
        "seed_min": min(final_seeds),
        "seed_max": max(final_seeds),
        "uint32_range_pass": all(0 <= seed <= 2**32 - 1 for seed in final_seeds),
        "smoke_overlap_count": len(final_seeds & smoke_seed_set(auth)),
        "repository_overlap_count": len(final_seeds & repository_seeds),
        "unique_response_id_count": len({request.response_id for request in schedule}),
        "schedule_positions_contiguous": [request.schedule_position for request in schedule]
        == list(range(M0F_TOTAL_RESPONSES)),
        "seed_set_sha256": canonical_json_sha256(sorted(final_seeds)),
        "schedule_sha256": canonical_sha256([request.as_dict() for request in schedule]),
        "repository_seed_source_file_count": len(source_files),
        "forbidden_seed_count": len(forbidden),
        "generation_started": False,
    }
    atomic_json(audit_path, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


def manifest_context(
    auth_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[CalibrationRequest], Path]:
    auth, h8_config, fingerprint, _ = load_frozen_context(auth_path)
    manifest_path = resolve(auth["artifacts"]["manifest"])
    audit_path = resolve(auth["artifacts"]["manifest_audit"])
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("Frozen M0-F manifest and audit must exist before sampling")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("generation_started") is not False:
        raise ValueError("M0-F manifest audit did not pass before generation")
    if audit["manifest"]["file_sha256"] != file_sha256(manifest_path):
        raise ValueError("M0-F manifest file hash mismatch")
    payload, schedule = load_manifest(manifest_path)
    if payload["frozen_identity"]["authorization_sha256"] != file_sha256(auth_path):
        raise ValueError("M0-F manifest was built from a different authorization config")
    entries = fingerprint["entries"]
    hydrated: list[CalibrationRequest] = []
    for request in schedule:
        entry = entries[request.prompt_index]
        if entry["prompt_id"] != request.prompt_id or entry["metadata"]["prompt_sha256"] != request.prompt_sha256:
            raise ValueError("Manifest schedule no longer matches H6 MCC12")
        hydrated.append(
            CalibrationRequest(
                **{
                    **request.__dict__,
                    "prompt": entry["prompt"],
                }
            )
        )
    validate_calibration_schedule(hydrated)
    return auth, h8_config, fingerprint, hydrated, manifest_path


def verify_completed_prefix(
    records: list[dict[str, Any]], schedule: list[CalibrationRequest]
) -> None:
    if len(records) > len(schedule):
        raise ValueError("Existing response file is longer than the frozen schedule")
    for index, record in enumerate(records):
        request = schedule[index]
        if (
            record.get("response_id") != request.response_id
            or int(record.get("schedule_position", -1)) != index
            or int(record.get("seed", -1)) != request.seed
            or record.get("data_role") != M0F_DATA_ROLE
        ):
            raise ValueError(f"Existing response prefix mismatch at schedule position {index}")


def verify_live_bundle(
    bundle: Any,
    auth: Mapping[str, Any],
    h8_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected = auth["runtime_identity"]
    tokenizer = bundle.tokenizer
    actual = {
        "model_name": bundle.name,
        "model_revision": bundle.revision,
        "model_dtype": str(next(bundle.model.parameters()).dtype),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        "frozen_eos_token_ids": [int(value) for value in h8_config["generation"]["eos_token_ids"]],
        "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
        "runtime": verify_runtime(auth),
    }
    required = {
        "model_name": h8_config["model"]["name"],
        "model_revision": h8_config["model"]["revision"],
        "model_dtype": "torch.bfloat16",
        "tokenizer_class": expected["tokenizer_class"],
        "tokenizer_pad_token_id": expected["tokenizer_pad_token_id"],
        "tokenizer_eos_token_id": expected["tokenizer_eos_token_id"],
        "chat_template_sha256": expected["chat_template_sha256"],
        "frozen_eos_token_ids": expected["frozen_eos_token_ids"],
        "generation_config_sha256": expected["generation_config_sha256"],
    }
    for key, value in required.items():
        if actual[key] != value:
            raise ValueError(f"Live model/tokenizer identity mismatch for {key}")
    device_map = dict(getattr(bundle.model, "hf_device_map", {}))
    mapped = {str(value) for value in device_map.values()}
    if mapped & {"cpu", "disk"}:
        raise ValueError("CPU/disk model offload is forbidden")
    actual["hf_device_map_sha256"] = canonical_json_sha256(device_map)
    actual["identity_sha256"] = canonical_json_sha256(actual)
    return actual


def run_attempts(
    bundle: Any,
    request: CalibrationRequest,
    h8_config: Mapping[str, Any],
    event_handle: Any,
    previous_events: list[dict[str, Any]],
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    attempt_records = list(previous_events)
    if any(record.get("status") == "success" for record in attempt_records):
        raise ValueError("A success attempt event exists without its corresponding response record")
    remaining_attempts = max_attempts - len(attempt_records)
    if remaining_attempts <= 0:
        raise RuntimeError(
            f"Frozen same-seed attempt budget already exhausted for {request.response_id}"
        )
    run_attempts_used = 0
    for _ in range(remaining_attempts):
        attempt_index = len(attempt_records)
        run_attempts_used += 1
        try:
            result = generate_one(bundle, request, dict(h8_config))
            event = {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "attempt_index": attempt_index,
                "seed": request.seed,
                "status": "success",
                "created_at_utc": now(),
            }
            append_jsonl(event_handle, event)
            attempt_records.append(event)
            result["attempt_count"] = len(attempt_records)
            result["attempts_in_current_process"] = run_attempts_used
            result["retry_seed_changed"] = any(
                int(record["seed"]) != request.seed for record in attempt_records
            )
            return result, attempt_records
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
            attempt_records.append(event)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(
        f"Technical generation failed after {max_attempts} same-seed attempts for {request.response_id}"
    )


def worker(auth_path: Path) -> int:
    auth, h8_config, fingerprint, schedule, manifest_path = manifest_context(auth_path)
    output_dir = resolve(auth["artifacts"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / auth["artifacts"]["responses"]
    events_path = output_dir / auth["artifacts"]["attempt_events"]
    records = read_jsonl(responses_path)
    verify_completed_prefix(records, schedule)
    events = read_jsonl(events_path)
    events_by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_response[str(event["response_id"])].append(event)
    verify_base_files(auth)
    verify_runtime(auth)
    snapshot = verify_snapshot(auth, h8_config)

    from llm_integrity.modeling import load_model

    bundle = load_model(h8_config["model"])
    try:
        live_identity = verify_live_bundle(bundle, auth, h8_config)
        shared_provenance = {
            "live_identity": live_identity,
            "snapshot": snapshot,
            "manifest_file_sha256": file_sha256(manifest_path),
            "authorization_sha256": file_sha256(auth_path),
            "git_commit": verify_git_lineage(auth),
        }
        provenance_sha = canonical_json_sha256(shared_provenance)
        response_mode = "a" if responses_path.exists() else "x"
        event_mode = "a" if events_path.exists() else "x"
        with responses_path.open(response_mode, encoding="utf-8", newline="\n") as response_handle, events_path.open(
            event_mode, encoding="utf-8", newline="\n"
        ) as event_handle:
            for request in schedule[len(records) :]:
                verify_base_files(auth)
                current_live = verify_live_bundle(bundle, auth, h8_config)
                if current_live["identity_sha256"] != live_identity["identity_sha256"]:
                    raise ValueError("Live runtime/model identity drifted before the next response")
                if request.prompt_position_in_round == 0:
                    current_snapshot = verify_snapshot(auth, h8_config)
                    if current_snapshot["snapshot_listing_sha256"] != snapshot["snapshot_listing_sha256"]:
                        raise ValueError("Snapshot identity drifted before the next replicate round")
                generated, attempt_records = run_attempts(
                    bundle,
                    request,
                    h8_config,
                    event_handle,
                    events_by_response[request.response_id],
                    int(auth["sampling"]["max_attempts_same_seed"]),
                )
                entry = fingerprint["entries"][request.prompt_index]
                record = {
                    "schema_version": "h8-m0f-calibration-response-1.0",
                    "response_id": request.response_id,
                    "schedule_position": request.schedule_position,
                    "replicate_id": request.replicate_id,
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
                    "attempt_records": attempt_records,
                    "model_name": bundle.name,
                    "model_revision": bundle.revision,
                    "runtime_identity_sha256": live_identity["identity_sha256"],
                    "shared_provenance_sha256": provenance_sha,
                    "manifest_file_sha256": file_sha256(manifest_path),
                    "data_role": M0F_DATA_ROLE,
                    "eligible_for_formal_reference": False,
                    "eligible_for_heldout_evaluation": False,
                    "eligible_for_attack_evaluation": False,
                    "created_at_utc": now(),
                    **generated,
                }
                append_jsonl(response_handle, record)
                records.append(record)
                events_by_response[request.response_id] = attempt_records
                if len(records) % 12 == 0:
                    progress = {
                        "status": "running",
                        "completed_responses": len(records),
                        "completed_replicate_rounds": len(records) // 12,
                        "last_response_id": request.response_id,
                        "updated_at_utc": now(),
                        "formal_reference_responses": 0,
                        "attack_responses": 0,
                        "heldout_responses": 0,
                    }
                    atomic_json(output_dir / "sampling_progress.json", progress)
        if len(records) != M0F_TOTAL_RESPONSES:
            raise RuntimeError("M0-F worker ended without exactly 1200 responses")
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


def parent(auth_path: Path) -> int:
    auth, _, _, schedule, manifest_path = manifest_context(auth_path)
    output_dir = resolve(auth["artifacts"]["output_dir"])
    command = [sys.executable, str(Path(__file__).resolve()), "--auth", str(auth_path), "--worker"]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    gpu_cleanup = gpu_compute_processes()
    responses_path = output_dir / auth["artifacts"]["responses"]
    events_path = output_dir / auth["artifacts"]["attempt_events"]
    records = read_jsonl(responses_path)
    events = read_jsonl(events_path)
    report_path = output_dir / auth["artifacts"]["report"]
    if completed.returncode != 0:
        report = {
            "schema_version": "h8-m0f-calibration-sampling-report-1.0",
            "status": "NO_GO_TECHNICAL_FAILURE",
            "worker_exit_code": completed.returncode,
            "completed_fit_only_responses": len(records),
            "mmd_precalibration_fit_only_responses": len(records),
            "formal_reference_responses": 0,
            "attack_responses": 0,
            "heldout_responses": 0,
            "attempt_event_count": len(events),
            "gpu_cleanup_after_worker_exit": gpu_cleanup,
            "manifest_file_sha256": file_sha256(manifest_path),
            "created_at_utc": now(),
            "next_gate": "STOP_AND_REVIEW_THEN_RESUME_ONLY_WITH_FROZEN_PREFIX_AND_SAME_SEEDS",
        }
        atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    audit = audit_calibration_records(records, schedule, REQUIRED_RECORD_FIELDS)
    technical_failures = [event for event in events if event.get("status") == "technical_failure"]
    success_events = [event for event in events if event.get("status") == "success"]
    status = "PASS" if audit["status"] == "PASS" and gpu_cleanup["status"] == "PASS" else "FAIL"
    report = {
        "schema_version": "h8-m0f-calibration-sampling-report-1.0",
        "status": status,
        "phase": "H8_M0F_CALIBRATION_SAMPLING_ONLY",
        "audit": audit,
        "attempt_events": {
            "total": len(events),
            "success": len(success_events),
            "technical_failure": len(technical_failures),
            "technical_failure_types": dict(
                sorted(Counter(event.get("exception_type", "") for event in technical_failures).items())
            ),
        },
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_payload_sha256": json.loads(manifest_path.read_text(encoding="utf-8"))[
            "payload_sha256"
        ],
        "responses_file_sha256": file_sha256(responses_path),
        "attempt_events_file_sha256": file_sha256(events_path),
        "formal_reference_responses": 0,
        "attack_responses": 0,
        "heldout_responses": 0,
        "mmd_precalibration_fit_only_responses": len(records),
        "feature_scaler_fit_performed": False,
        "family_balanced_transform_performed": False,
        "bandwidth_estimation_performed": False,
        "bandwidth_stability_performed": False,
        "mmd_pseudo_trials_performed": False,
        "top_r_or_global_test_performed": False,
        "gpu_cleanup_after_worker_exit": gpu_cleanup,
        "authorization_sha256": file_sha256(auth_path),
        "git_commit": verify_git_lineage(auth),
        "created_at_utc": now(),
        "next_gate": "STOP_AND_WAIT_FOR_EXPLICIT_FEATURE_SCALER_OR_BANDWIDTH_APPROVAL",
    }
    atomic_json(report_path, report)
    atomic_json(
        output_dir / "sampling_progress.json",
        {
            "status": "completed",
            "completed_responses": len(records),
            "completed_replicate_rounds": len(records) // 12,
            "last_response_id": records[-1]["response_id"] if records else None,
            "updated_at_utc": now(),
            "formal_reference_responses": 0,
            "attack_responses": 0,
            "heldout_responses": 0,
            "final_report_status": status,
            "final_report_path": auth["artifacts"]["report"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 M0-F calibration-only sampling runner")
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--prepare-manifest-only", action="store_true")
    parser.add_argument("--verify-manifest-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = args.auth.resolve()
    if args.prepare_manifest_only:
        return prepare_manifest(auth_path)
    if args.verify_manifest_only:
        _, _, _, schedule, path = manifest_context(auth_path)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "manifest_path": str(path),
                    "manifest_file_sha256": file_sha256(path),
                    "request_count": len(schedule),
                    "unique_seed_count": len({request.seed for request in schedule}),
                    "generation_started": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.worker:
        return worker(auth_path)
    return parent(auth_path)


if __name__ == "__main__":
    raise SystemExit(main())
