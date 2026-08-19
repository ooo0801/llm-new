#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d1b_formal import (
    GENESIS_CHAIN_SHA256,
    audit_formal_records,
    request_sha256,
    seal_response_record,
    validate_attempt_events,
    verify_completed_prefix,
)
from llm_integrity.h8_d1b_sampling import (
    AUDIT_ROLE,
    FIT_ROLE,
    FORMAL_TOTAL,
    D1BGenerationRequest,
    file_sha256,
    load_manifest,
)
from llm_integrity.h8_sampling import canonical_json_sha256, gpu_compute_processes, runtime_provenance
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement
from run_h8_m0f_calibration_sampling import verify_live_bundle, verify_runtime, verify_snapshot
from run_h8_mmd_smoke import generate_one, load_fingerprint, read_config


DEFAULT_AUTH = ROOT / "configs" / "h8_d1b1_formal_score_calibration_sampling.yaml"
REQUIRED_RESPONSE_FIELDS = {
    "response_id",
    "schedule_position",
    "round_id",
    "prompt_position_in_round",
    "prompt_index",
    "prompt_id",
    "prompt_sha256",
    "seed",
    "seed_digest_sha256",
    "data_role",
    "attempt_records",
    "rendered_prompt_sha256",
    "input_token_ids",
    "input_token_ids_sha256",
    "completion_token_ids",
    "completion_token_ids_sha256",
    "raw_response",
    "stop_reason",
    "response_token_count_including_eos",
    "legal_first_token_eos",
    "runtime_identity_sha256",
    "frozen_provenance",
    "frozen_provenance_sha256",
    "formal_manifest_file_sha256",
    "manifest_request_sha256",
    "record_payload_sha256",
    "record_chain_sha256",
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
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def load_auth(path: Path) -> dict[str, Any]:
    auth = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(auth, dict) or auth.get("phase") != "H8_D1B1_FORMAL_SCORE_CALIBRATION_SAMPLING":
        raise ValueError("Invalid D1-B1 authorization artifact")
    if auth.get("authorization_status") != "user_approved" or auth.get("formal_sampling_authorized") is not True:
        raise PermissionError("D1-B1 formal sampling is not explicitly authorized")
    sampling = auth["sampling"]
    expected = {
        "batch_size": 1,
        "rounds": 200,
        "prompt_count": 12,
        "total_responses": 2400,
        "fit_responses": 1200,
        "audit_responses": 1200,
        "fit_per_prompt": 100,
        "audit_per_prompt": 100,
        "even_round_role": FIT_ROLE,
        "odd_round_role": AUDIT_ROLE,
        "strict_offline": True,
    }
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"D1-B1 frozen sampling setting mismatch: {key}")
    if not all(value is True for value in auth["forbidden_operations"].values()):
        raise PermissionError("All D1-B1 post-sampling operations must remain forbidden")
    failed_path = str(auth["frozen_inputs"].get("d1b0_pass_report", ""))
    if "failed_before_generation" in failed_path or not failed_path.endswith("H8_D1B0_SCORE_CALIBRATION_SAMPLING_PREFLIGHT_REPORT.json"):
        raise ValueError("D1-B1 must bind the final D1-B0 PASS report, never a failed diagnostic")
    return auth


def verify_git_lineage(auth: Mapping[str, Any], auth_path: Path) -> str:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(auth["sampling_code_commit"]), "HEAD"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "ls-files", "--error-unmatch", str(auth_path.relative_to(ROOT))], cwd=ROOT, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree is not clean")
    require_hash(resolve(auth["implementation"]["runner"]), auth["implementation"]["runner_sha256"], "D1-B1 runner")
    require_hash(resolve(auth["implementation"]["formal_module"]), auth["implementation"]["formal_module_sha256"], "D1-B1 formal module")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_immutable_files(auth: Mapping[str, Any]) -> None:
    frozen = auth["frozen_inputs"]
    bindings = (
        ("d1b0_pass_report", "d1b0_pass_report_sha256", "D1-B0 final PASS report"),
        ("d1b0_config", "d1b0_config_sha256", "D1-B0 preflight config"),
        ("formal_manifest", "formal_manifest_sha256", "D1-B0 formal manifest"),
        ("formal_manifest_audit", "formal_manifest_audit_sha256", "D1-B0 formal manifest audit"),
        ("smoke_manifest", "smoke_manifest_sha256", "D1-B0 smoke manifest"),
        ("d1a_report", "d1a_report_sha256", "D1-A score preflight report"),
        ("h8_config", "h8_config_sha256", "H8 generation config"),
        ("runtime_authorization", "runtime_authorization_sha256", "runtime authorization"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12 fingerprint"),
        ("protocol", "protocol_sha256", "D1-B1 protocol"),
    )
    for path_key, hash_key, label in bindings:
        require_hash(resolve(frozen[path_key]), frozen[hash_key], label)
    load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=frozen["mmd_manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )


def frozen_context(
    auth_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[D1BGenerationRequest],
    list[D1BGenerationRequest],
]:
    auth = load_auth(auth_path)
    verify_immutable_files(auth)
    frozen = auth["frozen_inputs"]
    d1b0 = yaml.safe_load(resolve(frozen["d1b0_config"]).read_text(encoding="utf-8"))
    if d1b0.get("formal_sampling_authorized") is not False or d1b0.get("authorization_status") != "user_approved_preflight_only":
        raise PermissionError("D1-B0 preflight-only authorization semantics changed")
    pass_report = json.loads(resolve(frozen["d1b0_pass_report"]).read_text(encoding="utf-8"))
    if pass_report.get("status") != "PASS" or pass_report.get("formal_response_count") != 0:
        raise ValueError("Bound D1-B0 report is not the final successful preflight report")
    if pass_report.get("formal_manifest_file_sha256") != frozen["formal_manifest_sha256"]:
        raise ValueError("D1-B0 report does not bind the authorized formal manifest")
    if pass_report.get("score_schema_sha256") != frozen["score_schema_sha256"]:
        raise ValueError("D1-B0 report score-schema binding mismatch")
    d1a = json.loads(resolve(frozen["d1a_report"]).read_text(encoding="utf-8"))
    if d1a.get("status") != "PASS" or d1a.get("score_schema_sha256") != frozen["score_schema_sha256"]:
        raise ValueError("D1-A PASS/score-schema binding mismatch")
    formal_payload, formal = load_manifest(resolve(frozen["formal_manifest"]), smoke=False)
    smoke_payload, smoke = load_manifest(resolve(frozen["smoke_manifest"]), smoke=True)
    if len(formal) != FORMAL_TOTAL or {row.seed for row in formal} & {row.seed for row in smoke}:
        raise ValueError("Formal schedule size or formal/smoke seed isolation mismatch")
    if formal_payload["frozen_identity"]["score_schema_sha256"] != frozen["score_schema_sha256"]:
        raise ValueError("Formal manifest score-schema binding mismatch")
    h8_config = read_config(resolve(frozen["h8_config"]))
    fingerprint, fingerprint_path = load_fingerprint(h8_config)
    require_hash(fingerprint_path, frozen["fingerprint_sha256"], "H6 MCC12 fingerprint")
    entries = fingerprint["entries"]

    def hydrate(rows: list[D1BGenerationRequest]) -> list[D1BGenerationRequest]:
        hydrated: list[D1BGenerationRequest] = []
        for row in rows:
            entry = entries[row.prompt_index]
            if entry["prompt_id"] != row.prompt_id or entry["metadata"]["prompt_sha256"] != row.prompt_sha256:
                raise ValueError("Frozen schedule no longer matches H6 MCC12")
            hydrated.append(D1BGenerationRequest(**{**row.__dict__, "prompt": entry["prompt"]}))
        return hydrated

    runtime_auth = yaml.safe_load(resolve(frozen["runtime_authorization"]).read_text(encoding="utf-8"))
    verify_git_lineage(auth, auth_path)
    return auth, h8_config, fingerprint, runtime_auth, hydrate(formal), hydrate(smoke)


def artifact_paths(auth: Mapping[str, Any]) -> dict[str, Path]:
    artifacts = auth["artifacts"]
    output = resolve(artifacts["output_dir"])
    archive = resolve(artifacts["archive_dir"])
    return {
        "output": output,
        "responses": output / artifacts["responses"],
        "events": output / artifacts["attempt_events"],
        "provenance": output / artifacts["sampling_provenance"],
        "progress": output / artifacts["progress"],
        "incomplete_report": output / artifacts["incomplete_report"],
        "report": output / artifacts["report"],
        "archive": archive,
        "archive_report": archive / artifacts["report"],
    }


def _write_provenance(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload_dict = dict(payload)
    payload_sha = canonical_json_sha256(payload_dict)
    envelope = {
        "artifact_type": "h8_d1b1_frozen_sampling_provenance",
        "schema_version": "h8-d1b1-sampling-provenance-1.0",
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != envelope:
            raise ValueError("Existing frozen sampling provenance differs from the live authorized identity")
    else:
        atomic_json(path, envelope)
    return payload_dict, payload_sha


def load_provenance(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if envelope.get("artifact_type") != "h8_d1b1_frozen_sampling_provenance" or not isinstance(payload, dict):
        raise ValueError("Frozen sampling provenance type mismatch")
    payload_sha = canonical_json_sha256(payload)
    if envelope.get("payload_sha256") != payload_sha:
        raise ValueError("Frozen sampling provenance payload hash mismatch")
    return payload, payload_sha


def run_attempts(
    bundle: Any,
    request: D1BGenerationRequest,
    h8_config: Mapping[str, Any],
    event_handle: Any,
    previous_events: list[dict[str, Any]],
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    attempts = list(previous_events)
    if any(row.get("status") == "success" for row in attempts):
        raise ValueError("A success attempt event exists without its persisted response record")
    remaining = max_attempts - len(attempts)
    if remaining <= 0:
        raise RuntimeError(f"Same-seed retry budget exhausted for {request.response_id}")
    for _ in range(remaining):
        attempt_index = len(attempts)
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
            generated["retry_seed_changed"] = any(int(row["seed"]) != request.seed for row in attempts)
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
    raise RuntimeError(f"Technical generation failed after {max_attempts} same-seed attempts for {request.response_id}")


def worker(auth_path: Path) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise PermissionError("D1-B1 worker requires strict Hugging Face and Transformers offline mode")
    auth, h8_config, fingerprint, runtime_auth, formal, smoke = frozen_context(auth_path)
    paths = artifact_paths(auth)
    paths["output"].mkdir(parents=True, exist_ok=True)
    records = read_jsonl(paths["responses"])
    events = read_jsonl(paths["events"])
    provenance_payload: dict[str, Any] | None = None
    provenance_sha: str | None = None
    if paths["provenance"].exists():
        provenance_payload, provenance_sha = load_provenance(paths["provenance"])
    final_chain = verify_completed_prefix(
        records,
        formal,
        formal_manifest_sha256=auth["frozen_inputs"]["formal_manifest_sha256"],
        frozen_provenance_sha256=provenance_sha,
    )
    events_by_response = validate_attempt_events(
        events,
        formal,
        len(records),
        max_attempts=int(auth["sampling"]["max_attempts_same_seed"]),
    )
    if len(records) == FORMAL_TOTAL:
        return 0
    verify_immutable_files(auth)
    verify_runtime(runtime_auth)
    snapshot = verify_snapshot(runtime_auth, h8_config)
    from llm_integrity.modeling import load_model

    bundle = load_model(h8_config["model"])
    try:
        live_identity = verify_live_bundle(bundle, runtime_auth, h8_config)
        current_commit = verify_git_lineage(auth, auth_path)
        candidate_provenance = {
            "authorization_file_sha256": file_sha256(auth_path),
            "d1b0_pass_report_sha256": auth["frozen_inputs"]["d1b0_pass_report_sha256"],
            "formal_manifest_file_sha256": auth["frozen_inputs"]["formal_manifest_sha256"],
            "formal_manifest_payload_sha256": json.loads(resolve(auth["frozen_inputs"]["formal_manifest"]).read_text(encoding="utf-8"))["payload_sha256"],
            "formal_seed_set_sha256": canonical_json_sha256(sorted(row.seed for row in formal)),
            "smoke_seed_set_sha256": canonical_json_sha256(sorted(row.seed for row in smoke)),
            "mmd_manifest_sha256": auth["frozen_inputs"]["mmd_manifest_sha256"],
            "mmd_implementation_commit": auth["frozen_inputs"]["mmd_implementation_commit"],
            "d1a_report_sha256": auth["frozen_inputs"]["d1a_report_sha256"],
            "score_schema_sha256": auth["frozen_inputs"]["score_schema_sha256"],
            "fingerprint_sha256": auth["frozen_inputs"]["fingerprint_sha256"],
            "generation_config": h8_config["generation"],
            "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
            "model_snapshot": snapshot,
            "live_identity": live_identity,
            "runtime": runtime_provenance(),
            "sampling_code_commit": auth["sampling_code_commit"],
            "actual_sampling_commit": current_commit,
            "strict_offline": True,
            "created_before_first_response_utc": now(),
        }
        if provenance_payload is None:
            provenance_payload, provenance_sha = _write_provenance(paths["provenance"], candidate_provenance)
        else:
            stable_checks = {
                "authorization_file_sha256": file_sha256(auth_path),
                "formal_manifest_file_sha256": auth["frozen_inputs"]["formal_manifest_sha256"],
                "mmd_manifest_sha256": auth["frozen_inputs"]["mmd_manifest_sha256"],
                "score_schema_sha256": auth["frozen_inputs"]["score_schema_sha256"],
                "fingerprint_sha256": auth["frozen_inputs"]["fingerprint_sha256"],
                "sampling_code_commit": auth["sampling_code_commit"],
                "actual_sampling_commit": current_commit,
            }
            for key, expected in stable_checks.items():
                if provenance_payload.get(key) != expected:
                    raise ValueError(f"Frozen sampling provenance drifted for {key}")
            if provenance_payload["live_identity"]["identity_sha256"] != live_identity["identity_sha256"]:
                raise ValueError("Live model/runtime identity differs from the frozen sampling provenance")
            if provenance_payload["model_snapshot"]["snapshot_listing_sha256"] != snapshot["snapshot_listing_sha256"]:
                raise ValueError("Model snapshot differs from the frozen sampling provenance")
        assert provenance_payload is not None and provenance_sha is not None
        final_chain = verify_completed_prefix(
            records,
            formal,
            formal_manifest_sha256=auth["frozen_inputs"]["formal_manifest_sha256"],
            frozen_provenance_sha256=provenance_sha,
        )
        response_mode = "a" if paths["responses"].exists() else "x"
        event_mode = "a" if paths["events"].exists() else "x"
        with paths["responses"].open(response_mode, encoding="utf-8", newline="\n") as response_handle, paths[
            "events"
        ].open(event_mode, encoding="utf-8", newline="\n") as event_handle:
            for request in formal[len(records) :]:
                verify_immutable_files(auth)
                current_live = verify_live_bundle(bundle, runtime_auth, h8_config)
                if current_live["identity_sha256"] != live_identity["identity_sha256"]:
                    raise ValueError("Live model/tokenizer/runtime identity drifted before the next response")
                if request.prompt_position_in_round == 0:
                    current_snapshot = verify_snapshot(runtime_auth, h8_config)
                    if current_snapshot["snapshot_listing_sha256"] != snapshot["snapshot_listing_sha256"]:
                        raise ValueError("Model snapshot drifted before the next round")
                generated, attempts = run_attempts(
                    bundle,
                    request,
                    h8_config,
                    event_handle,
                    events_by_response.get(request.response_id, []),
                    int(auth["sampling"]["max_attempts_same_seed"]),
                )
                entry = fingerprint["entries"][request.prompt_index]
                record = {
                    "schema_version": "h8-d1b1-formal-response-1.0",
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
                    "data_role": request.data_role,
                    "batch_size": 1,
                    "attempt_records": attempts,
                    "model_name": bundle.name,
                    "model_revision": bundle.revision,
                    "runtime_identity_sha256": live_identity["identity_sha256"],
                    "frozen_provenance": provenance_payload,
                    "frozen_provenance_sha256": provenance_sha,
                    "formal_manifest_file_sha256": auth["frozen_inputs"]["formal_manifest_sha256"],
                    "manifest_request_sha256": request_sha256(request),
                    "eligible_for_score_calibration_fit": request.data_role == FIT_ROLE,
                    "eligible_for_score_stability_audit": request.data_role == AUDIT_ROLE,
                    "eligible_for_formal_reference": False,
                    "eligible_for_heldout_evaluation": False,
                    "eligible_for_target_evaluation": False,
                    "eligible_for_attack_evaluation": False,
                    "created_at_utc": now(),
                    **generated,
                }
                sealed = seal_response_record(record, final_chain)
                append_jsonl(response_handle, sealed)
                records.append(sealed)
                events_by_response[request.response_id] = attempts
                final_chain = sealed["record_chain_sha256"]
                atomic_json(
                    paths["progress"],
                    {
                        "status": "running",
                        "completed_responses": len(records),
                        "completed_rounds": len(records) // 12,
                        "last_response_id": request.response_id,
                        "last_schedule_position": request.schedule_position,
                        "record_chain_sha256": final_chain,
                        "updated_at_utc": now(),
                        "score_parameter_fit_performed": False,
                        "formal_reference_responses": 0,
                        "heldout_responses": 0,
                        "attack_responses": 0,
                    },
                )
        if len(records) != FORMAL_TOTAL:
            raise RuntimeError("D1-B1 worker ended without exactly 2,400 responses")
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
    auth, _, _, _, formal, smoke = frozen_context(auth_path)
    paths = artifact_paths(auth)
    if paths["report"].exists() or paths["archive_report"].exists():
        raise FileExistsError("A final D1-B1 report already exists; refusing to overwrite it")
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_CACHE": str(auth["runtime_execution"]["hf_hub_cache"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--auth", str(auth_path), "--worker"],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    cleanup = gpu_compute_processes()
    records = read_jsonl(paths["responses"])
    events = read_jsonl(paths["events"])
    if completed.returncode != 0:
        report = {
            "schema_version": "h8-d1b1-incomplete-report-1.0",
            "phase": "H8_D1B1_FORMAL_SCORE_CALIBRATION_SAMPLING",
            "status": "NO_GO_TECHNICAL_OR_INTEGRITY_FAILURE",
            "worker_exit_code": completed.returncode,
            "completed_responses": len(records),
            "next_schedule_position": len(records),
            "attempt_event_count": len(events),
            "formal_manifest_file_sha256": auth["frozen_inputs"]["formal_manifest_sha256"],
            "gpu_cleanup": cleanup,
            "score_parameter_fit_performed": False,
            "sample_size_selection_performed": False,
            "top_r_selection_performed": False,
            "formal_reference_responses": 0,
            "heldout_responses": 0,
            "attack_responses": 0,
            "created_at_utc": now(),
            "next_gate": "STOP_REVIEW_AND_RESUME_ONLY_AFTER_PREFIX_VALIDATION",
        }
        atomic_json(paths["incomplete_report"], report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    provenance, provenance_sha = load_provenance(paths["provenance"])
    audit = audit_formal_records(
        records,
        formal,
        events,
        formal_manifest_sha256=auth["frozen_inputs"]["formal_manifest_sha256"],
        smoke_seeds={request.seed for request in smoke},
        frozen_provenance_sha256=provenance_sha,
        required_fields=REQUIRED_RESPONSE_FIELDS,
        max_attempts=int(auth["sampling"]["max_attempts_same_seed"]),
    )
    status = "PASS" if audit["status"] == "PASS" and cleanup["status"] == "PASS" else "FAIL"
    report = {
        "schema_version": "h8-d1b1-formal-sampling-report-1.0",
        "phase": "H8_D1B1_FORMAL_SCORE_CALIBRATION_SAMPLING",
        "status": status,
        "audit": audit,
        "formal_manifest_file_sha256": auth["frozen_inputs"]["formal_manifest_sha256"],
        "d1b0_pass_report_sha256": auth["frozen_inputs"]["d1b0_pass_report_sha256"],
        "mmd_manifest_sha256": auth["frozen_inputs"]["mmd_manifest_sha256"],
        "score_schema_sha256": auth["frozen_inputs"]["score_schema_sha256"],
        "authorization_file_sha256": file_sha256(auth_path),
        "sampling_provenance_sha256": provenance_sha,
        "sampling_provenance": provenance,
        "responses_file_sha256": file_sha256(paths["responses"]),
        "attempt_events_file_sha256": file_sha256(paths["events"]),
        "gpu_cleanup": cleanup,
        "score_parameter_fit_performed": False,
        "sample_size_selection_performed": False,
        "top_r_selection_performed": False,
        "formal_reference_responses": 0,
        "heldout_responses": 0,
        "attack_responses": 0,
        "created_at_utc": now(),
        "next_gate": "STOP_AND_WAIT_FOR_EXPLICIT_D1C_SCORE_PARAMETER_FIT_APPROVAL",
    }
    atomic_json(paths["report"], report)
    paths["archive"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["archive_report"], report)
    report_sha = file_sha256(paths["archive_report"])
    paths["archive_report"].with_suffix(paths["archive_report"].suffix + ".sha256").write_text(
        f"{report_sha}  {paths['archive_report'].name}\n", encoding="ascii", newline="\n"
    )
    atomic_json(
        paths["progress"],
        {
            "status": "completed" if status == "PASS" else "failed_audit",
            "completed_responses": len(records),
            "completed_rounds": len(records) // 12,
            "record_chain_sha256": audit["final_record_chain_sha256"],
            "final_report_sha256": report_sha,
            "updated_at_utc": now(),
            "score_parameter_fit_performed": False,
            "formal_reference_responses": 0,
            "heldout_responses": 0,
            "attack_responses": 0,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def preflight(auth_path: Path) -> int:
    auth, h8_config, _, runtime_auth, formal, smoke = frozen_context(auth_path)
    if len(formal) != 2400 or len(smoke) != 24:
        raise ValueError("D1-B1 schedule counts changed")
    output = {
        "status": "PASS",
        "authorization_file_sha256": file_sha256(auth_path),
        "d1b0_pass_report_sha256": auth["frozen_inputs"]["d1b0_pass_report_sha256"],
        "formal_manifest_file_sha256": file_sha256(resolve(auth["frozen_inputs"]["formal_manifest"])),
        "formal_request_count": len(formal),
        "fit_request_count": sum(row.data_role == FIT_ROLE for row in formal),
        "audit_request_count": sum(row.data_role == AUDIT_ROLE for row in formal),
        "formal_unique_seed_count": len({row.seed for row in formal}),
        "formal_smoke_seed_overlap_count": len({row.seed for row in formal} & {row.seed for row in smoke}),
        "model_name": h8_config["model"]["name"],
        "model_revision": h8_config["model"]["revision"],
        "runtime_status": verify_runtime(runtime_auth),
        "formal_sampling_authorized": True,
        "score_parameter_fit_authorized": False,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 D1-B1 formal score-calibration sampling runner")
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = args.auth.resolve()
    if args.preflight_only:
        return preflight(auth_path)
    if args.worker:
        return worker(auth_path)
    return parent(auth_path)


if __name__ == "__main__":
    raise SystemExit(main())
