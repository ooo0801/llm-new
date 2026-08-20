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
from llm_integrity.h8_d1c import load_frozen_score_calibration
from llm_integrity.h8_d2a import (
    ATTACK_ROLE,
    INTACT_TARGET_ROLE,
    REFERENCE_ROLE,
    load_canonical_envelope as load_d2a_envelope,
    validate_nested_subset_payload,
)
from llm_integrity.h8_d2b0 import file_sha256, load_canonical_envelope as load_d2b0_envelope
from llm_integrity.h8_d2b1 import (
    D2B1GenerationRequest,
    ROLE_TOTALS,
    TOTAL_RESPONSES,
    audit_formal_records,
    combine_and_validate_schedules,
    hydrate_prompts,
    seal_response_record,
    validate_attempt_events,
    verify_completed_prefix,
)
from llm_integrity.h8_sampling import (
    canonical_json_sha256,
    gpu_compute_processes,
    runtime_provenance,
    sha256_text,
)
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement
from llm_integrity.modeling import model_metadata
from run_h8_m0f_calibration_sampling import verify_live_bundle, verify_runtime, verify_snapshot
from run_h8_mmd_smoke import generate_one


DEFAULT_AUTH = ROOT / "configs/h8_d2b1_formal_development_sampling.yaml"
BASE_BANK_END = 1920
REQUIRED_RESPONSE_FIELDS = {
    "response_id",
    "schedule_position",
    "bank_position",
    "replicate_id",
    "prompt_position_in_round",
    "prompt_index",
    "prompt_id",
    "prompt_sha256",
    "generation_seed",
    "seed_digest_sha256",
    "data_role",
    "bank_id",
    "attack_family",
    "attack_instance_id",
    "evaluation_unit_id",
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
    "worker_runtime_identity_sha256",
    "frozen_provenance",
    "frozen_provenance_sha256",
    "generation_manifest_set_sha256",
    "manifest_request_sha256",
    "record_payload_sha256",
    "record_chain_sha256",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured D2-B1 path escapes the repository")
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
        raise ValueError(f"Frozen {label} SHA256 mismatch: {actual} != {expected}")


def load_auth(path: Path) -> dict[str, Any]:
    auth = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(auth, dict) or auth.get("phase") != "H8_D2B1_FORMAL_DEVELOPMENT_SAMPLING":
        raise ValueError("Invalid D2-B1 authorization artifact")
    if auth.get("authorization_status") != "user_approved" or auth.get("formal_development_sampling_authorized") is not True:
        raise PermissionError("D2-B1 formal development sampling is not explicitly authorized")
    sampling = auth.get("sampling", {})
    expected = {
        "batch_size": 1,
        "total_responses": 3840,
        "reference_responses": 720,
        "intact_target_responses": 1200,
        "attack_responses": 1920,
        "reference_per_prompt": 60,
        "intact_target_per_prompt": 100,
        "attack_per_endpoint_prompt": 20,
        "attack_endpoint_count": 8,
        "strict_offline": True,
        "resume_policy": "exact_validated_global_schedule_prefix_only",
        "success_record_policy": "immutable_never_regenerate",
        "technical_retry_policy": "same_seed_only",
    }
    for key, value in expected.items():
        if sampling.get(key) != value:
            raise ValueError(f"D2-B1 frozen sampling setting mismatch: {key}")
    if not all(value is True for value in auth.get("forbidden_operations", {}).values()):
        raise PermissionError("Every D2-B1 post-sampling operation must remain forbidden")
    return auth


def verify_git_lineage(auth: Mapping[str, Any], auth_path: Path) -> str:
    for commit in (auth["implementation_commit"], auth["d2b0_github_archive_commit"]):
        subprocess.run(["git", "merge-base", "--is-ancestor", str(commit), "HEAD"], cwd=ROOT, check=True)
    subprocess.run(["git", "ls-files", "--error-unmatch", str(auth_path.relative_to(ROOT))], cwd=ROOT, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree is not clean before D2-B1 formal sampling")
    require_hash(resolve(auth["implementation"]["runner"]), auth["implementation"]["runner_sha256"], "D2-B1 runner")
    require_hash(resolve(auth["implementation"]["module"]), auth["implementation"]["module_sha256"], "D2-B1 module")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def verify_immutable_files(auth: Mapping[str, Any]) -> None:
    frozen = auth["frozen_inputs"]
    bindings = (
        ("h8_config", "h8_config_sha256", "H8 generation config"),
        ("runtime_authorization", "runtime_authorization_sha256", "runtime authorization"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12"),
        ("d2a_artifact_index", "d2a_artifact_index_sha256", "D2-A artifact index"),
        ("d2a_report", "d2a_report_sha256", "D2-A PASS report"),
        ("d2b0_pre_index", "d2b0_pre_index_sha256", "D2-B0 pre-smoke index"),
        ("d2b0_final_index", "d2b0_final_index_sha256", "D2-B0 final hash index"),
        ("d2b0_final_report", "d2b0_final_report_sha256", "D2-B0 final PASS report"),
        ("d2b0_materialization_audit", "d2b0_materialization_audit_sha256", "D2-B0 materialization audit"),
        ("d2b0_authorization_config", "d2b0_authorization_config_sha256", "D2-B0 authorization config"),
        ("protocol", "protocol_sha256", "D2-B1 protocol"),
    )
    for path_key, hash_key, label in bindings:
        require_hash(resolve(frozen[path_key]), str(frozen[hash_key]), label)
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
        raise ValueError("Frozen score layer status changed")


def _load_d2a_payload(index: Mapping[str, Any], directory: Path, key: str, artifact_type: str) -> dict[str, Any]:
    row = index[key]
    return load_d2a_envelope(
        directory / row["path"],
        expected_artifact_type=artifact_type,
        expected_file_sha256=row["file_sha256"],
        expected_payload_sha256=row["payload_sha256"],
    )


def frozen_context(auth_path: Path) -> dict[str, Any]:
    auth = load_auth(auth_path)
    verify_immutable_files(auth)
    frozen = auth["frozen_inputs"]
    d2a_dir = resolve(frozen["d2a_directory"])
    d2a_index = json.loads(resolve(frozen["d2a_artifact_index"]).read_text(encoding="utf-8"))
    reference = _load_d2a_payload(
        d2a_index, d2a_dir, "reference_manifest", "h8_d2a_reference_generation_manifest"
    )
    intact = _load_d2a_payload(
        d2a_index, d2a_dir, "intact_target_manifest", "h8_d2a_intact_target_generation_manifest"
    )
    attack = _load_d2a_payload(
        d2a_index, d2a_dir, "attack_generation_manifest", "h8_d2a_attack_generation_manifest"
    )
    nested = _load_d2a_payload(
        d2a_index, d2a_dir, "nested_subset_manifest", "h8_d2a_nested_subset_manifest"
    )
    validate_nested_subset_payload(nested)
    instances = _load_d2a_payload(
        d2a_index, d2a_dir, "attack_instance_manifest", "h8_d2a_fresh_attack_instance_manifest"
    )["instances"]
    if any(payload.get("sampling_authorized") is not False for payload in (reference, intact, attack)):
        raise ValueError("D2-A proposed manifests were modified instead of independently authorized by D2-B1")
    schedule = combine_and_validate_schedules(reference["schedule"], intact["schedule"], attack["schedule"])
    manifest_binding = {
        key: {
            "file_sha256": d2a_index[key]["file_sha256"],
            "payload_sha256": d2a_index[key]["payload_sha256"],
        }
        for key in ("reference_manifest", "intact_target_manifest", "attack_generation_manifest")
    }
    manifest_set_sha = canonical_json_sha256(manifest_binding)
    if manifest_set_sha != frozen["generation_manifest_set_sha256"]:
        raise ValueError("D2-B1 generation manifest-set binding mismatch")
    if d2a_index["nested_subset_manifest"]["file_sha256"] != frozen["nested_subset_manifest_sha256"]:
        raise ValueError("D2-B1 nested subset binding mismatch")

    pre_index = json.loads(resolve(frozen["d2b0_pre_index"]).read_text(encoding="utf-8"))
    revised_info = pre_index["revised_configuration_manifest"]
    revised = load_d2b0_envelope(
        resolve(frozen["d2b0_pre_index"]).parent / "D2B0_REVISED_CONFIGURATION_MANIFEST.json",
        expected_artifact_type="h8_d2b0_revised_configuration_manifest",
        expected_file_sha256=revised_info["file_sha256"],
        expected_payload_sha256=revised_info["payload_sha256"],
    )
    selection_rule_sha = canonical_json_sha256(revised["selection_rule"])
    if selection_rule_sha != frozen["configuration_selection_rule_sha256"]:
        raise ValueError("Pre-registered configuration-selection rule changed")

    final_index = json.loads(resolve(frozen["d2b0_final_index"]).read_text(encoding="utf-8"))
    final_report = json.loads(resolve(frozen["d2b0_final_report"]).read_text(encoding="utf-8"))
    materialization_audit = json.loads(resolve(frozen["d2b0_materialization_audit"]).read_text(encoding="utf-8"))
    if (
        final_report.get("status") != "PASS"
        or int(final_report.get("attack_smoke_only_responses", -1)) != 8
        or int(final_report.get("formal_development_responses", -1)) != 0
        or materialization_audit.get("status") != "PASS"
        or int(materialization_audit.get("endpoint_count", -1)) != 8
        or materialization_audit.get("quantization_exact_materialization") is not True
    ):
        raise ValueError("D2-B0 final materialization/smoke gate is not PASS")
    final_dir = resolve(frozen["d2b0_final_index"]).parent
    expected_materialization: dict[str, dict[str, str]] = {}
    smoke_response_ids: set[str] = set()
    smoke_seeds: set[int] = set()
    for instance in instances:
        endpoint = str(instance["attack_instance_id"])
        index_row = final_index["endpoint_artifacts"].get(endpoint)
        if index_row is None:
            raise ValueError(f"D2-B0 final index lacks endpoint {endpoint}")
        materialization_path = final_dir / f"{endpoint}_MATERIALIZATION.json"
        response_path = final_dir / f"{endpoint}_SMOKE_RESPONSE.json"
        require_hash(materialization_path, index_row["materialization_file_sha256"], f"D2-B0 {endpoint} materialization")
        require_hash(response_path, index_row["response_file_sha256"], f"D2-B0 {endpoint} smoke response")
        materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
        smoke = json.loads(response_path.read_text(encoding="utf-8"))
        if (
            materialization.get("status") != "PASS"
            or materialization.get("configuration") != instance["configuration"]
            or int(materialization.get("materialization_seed", -1)) != int(instance["materialization_seed"])
            or smoke.get("data_role") != "detector_development_attack_smoke_only"
        ):
            raise ValueError(f"D2-B0 endpoint identity mismatch: {endpoint}")
        expected_materialization[endpoint] = {
            "d2b0_materialization_file_sha256": index_row["materialization_file_sha256"],
            "d2b0_materialization_payload_sha256": str(materialization["materialization_sha256"]),
            "d2b0_attack_configuration_sha256": str(materialization["configuration_sha256"]),
        }
        smoke_response_ids.add(str(smoke["response_id"]))
        smoke_seeds.add(int(smoke["generation_seed"]))
    if len(expected_materialization) != 8 or len(smoke_response_ids) != 8 or len(smoke_seeds) != 8:
        raise ValueError("D2-B0 endpoint/smoke identity count changed")
    if {row.generation_seed for row in schedule} & smoke_seeds:
        raise ValueError("D2-B0 smoke seed overlaps the formal development schedule")

    fingerprint = json.loads(resolve(frozen["fingerprint"]).read_text(encoding="utf-8"))
    schedule = hydrate_prompts(schedule, fingerprint["entries"])
    h8_config = yaml.safe_load(resolve(frozen["h8_config"]).read_text(encoding="utf-8"))
    runtime_auth = yaml.safe_load(resolve(frozen["runtime_authorization"]).read_text(encoding="utf-8"))
    verify_git_lineage(auth, auth_path)
    return {
        "auth": auth,
        "h8_config": h8_config,
        "runtime_auth": runtime_auth,
        "fingerprint": fingerprint,
        "schedule": schedule,
        "nested": nested,
        "instances": instances,
        "instances_by_id": {str(row["attack_instance_id"]): row for row in instances},
        "manifest_set_sha256": manifest_set_sha,
        "selection_rule_sha256": selection_rule_sha,
        "expected_materialization": expected_materialization,
        "smoke_response_ids": smoke_response_ids,
        "smoke_seeds": smoke_seeds,
        "d2b0_auth": yaml.safe_load(resolve(frozen["d2b0_authorization_config"]).read_text(encoding="utf-8")),
    }


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
        "runner_log": output / artifacts["runner_log"],
        "archive": archive,
        "archive_report": archive / artifacts["report"],
        "archive_index": archive / artifacts["final_hash_index"],
    }


def _write_provenance(path: Path, payload: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload_dict = dict(payload)
    payload_sha = canonical_json_sha256(payload_dict)
    envelope = {
        "artifact_type": "h8_d2b1_frozen_sampling_provenance",
        "schema_version": "h8-d2b1-sampling-provenance-1.0",
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != envelope:
            raise ValueError("Existing D2-B1 frozen provenance differs from authorized identity")
    else:
        atomic_json(path, envelope)
    return payload_dict, payload_sha


def load_provenance(path: Path) -> tuple[dict[str, Any], str]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if envelope.get("artifact_type") != "h8_d2b1_frozen_sampling_provenance" or not isinstance(payload, dict):
        raise ValueError("D2-B1 frozen provenance type mismatch")
    payload_sha = canonical_json_sha256(payload)
    if envelope.get("payload_sha256") != payload_sha:
        raise ValueError("D2-B1 frozen provenance payload hash mismatch")
    return payload, payload_sha


def run_attempts(
    bundle: Any,
    request: D2B1GenerationRequest,
    h8_config: Mapping[str, Any],
    event_handle: Any,
    previous_events: list[dict[str, Any]],
    max_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    attempts = list(previous_events)
    if any(row.get("status") == "success" for row in attempts):
        raise ValueError("A D2-B1 success event exists without its persisted response")
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
                "generation_seed": request.generation_seed,
                "status": "success",
                "created_at_utc": now(),
            }
            append_jsonl(event_handle, event)
            attempts.append(event)
            generated["attempt_count"] = len(attempts)
            generated["retry_seed_changed"] = any(
                int(row["generation_seed"]) != request.generation_seed for row in attempts
            )
            return generated, attempts
        except (RuntimeError, OSError) as exc:
            event = {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "attempt_index": attempt_index,
                "generation_seed": request.generation_seed,
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


def _attack_runtime_identity(
    bundle: Any, runtime_auth: Mapping[str, Any], h8_config: Mapping[str, Any]
) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    expected = runtime_auth["runtime_identity"]
    actual = {
        "model": model_metadata(bundle),
        "first_parameter_dtype": str(next(bundle.model.parameters()).dtype),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        "frozen_eos_token_ids": [int(value) for value in h8_config["generation"]["eos_token_ids"]],
        "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
        "runtime": verify_runtime(runtime_auth),
    }
    required = {
        "tokenizer_class": expected["tokenizer_class"],
        "tokenizer_pad_token_id": expected["tokenizer_pad_token_id"],
        "tokenizer_eos_token_id": expected["tokenizer_eos_token_id"],
        "chat_template_sha256": expected["chat_template_sha256"],
        "frozen_eos_token_ids": expected["frozen_eos_token_ids"],
        "generation_config_sha256": expected["generation_config_sha256"],
    }
    for key, value in required.items():
        if actual[key] != value:
            raise ValueError(f"D2-B1 attack runtime identity mismatch for {key}")
    device_map = dict(getattr(bundle.model, "hf_device_map", {}))
    if {str(value) for value in device_map.values()} & {"cpu", "disk"}:
        raise ValueError("D2-B1 attack model CPU/disk offload is forbidden")
    actual["hf_device_map_sha256"] = canonical_json_sha256(device_map)
    actual["identity_sha256"] = canonical_json_sha256(actual)
    return actual


def _worker_slice(schedule: list[D2B1GenerationRequest], completed: int, worker_kind: str, endpoint_id: str | None) -> tuple[int, int]:
    if completed >= len(schedule):
        return completed, completed
    if worker_kind == "intact":
        if completed >= BASE_BANK_END:
            raise ValueError("Intact worker cannot run after the two intact banks")
        return completed, BASE_BANK_END
    if worker_kind != "attack" or not endpoint_id:
        raise ValueError("D2-B1 worker kind/endpoint is invalid")
    if schedule[completed].attack_instance_id != endpoint_id:
        raise ValueError("D2-B1 attack worker does not match the next unfinished endpoint")
    positions = [row.schedule_position for row in schedule if row.attack_instance_id == endpoint_id]
    if not positions or positions != list(range(min(positions), max(positions) + 1)):
        raise ValueError("D2-B1 endpoint schedule is not contiguous")
    return completed, max(positions) + 1


def worker(auth_path: Path, worker_kind: str, endpoint_id: str | None) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise PermissionError("D2-B1 worker requires strict offline mode")
    context = frozen_context(auth_path)
    auth = context["auth"]
    h8_config = context["h8_config"]
    runtime_auth = context["runtime_auth"]
    schedule = context["schedule"]
    paths = artifact_paths(auth)
    paths["output"].mkdir(parents=True, exist_ok=True)
    records = read_jsonl(paths["responses"])
    events = read_jsonl(paths["events"])
    provenance, provenance_sha = load_provenance(paths["provenance"])
    final_chain = verify_completed_prefix(
        records,
        schedule,
        generation_manifest_set_sha256=context["manifest_set_sha256"],
        frozen_provenance_sha256=provenance_sha,
    )
    events_by_response = validate_attempt_events(
        events,
        schedule,
        len(records),
        max_attempts=int(auth["sampling"]["max_attempts_same_seed"]),
    )
    start, end = _worker_slice(schedule, len(records), worker_kind, endpoint_id)
    if start == end:
        return 0
    verify_immutable_files(auth)
    verify_runtime(runtime_auth)
    snapshot = verify_snapshot(runtime_auth, h8_config)
    bundle = None
    materialization_binding = {
        "d2b0_materialization_file_sha256": None,
        "d2b0_materialization_payload_sha256": None,
        "d2b0_attack_configuration_sha256": None,
    }
    try:
        if worker_kind == "intact":
            from llm_integrity.modeling import load_model

            bundle = load_model(h8_config["model"])
            worker_identity = verify_live_bundle(bundle, runtime_auth, h8_config)
        else:
            import run_h8_d2b0_attack_smoke as d2b0_runner

            assert endpoint_id is not None
            instance = context["instances_by_id"][endpoint_id]
            bundle, materialization = d2b0_runner._materialize_endpoint(
                context["d2b0_auth"], h8_config, instance
            )
            expected = context["expected_materialization"][endpoint_id]
            actual_payload_sha = canonical_json_sha256(materialization)
            if actual_payload_sha != expected["d2b0_materialization_payload_sha256"]:
                raise ValueError(f"D2-B1 rematerialized endpoint differs from D2-B0: {endpoint_id}")
            materialization_binding = dict(expected)
            worker_identity = _attack_runtime_identity(bundle, runtime_auth, h8_config)
        worker_identity_sha = canonical_json_sha256(worker_identity)
        if snapshot["snapshot_listing_sha256"] != provenance["model_snapshot"]["snapshot_listing_sha256"]:
            raise ValueError("D2-B1 local model snapshot differs from frozen provenance")
        response_mode = "a" if paths["responses"].exists() else "x"
        event_mode = "a" if paths["events"].exists() else "x"
        with paths["responses"].open(response_mode, encoding="utf-8", newline="\n") as response_handle, paths[
            "events"
        ].open(event_mode, encoding="utf-8", newline="\n") as event_handle:
            for request in schedule[start:end]:
                verify_immutable_files(auth)
                if worker_kind == "intact":
                    current_identity = verify_live_bundle(bundle, runtime_auth, h8_config)
                else:
                    current_identity = _attack_runtime_identity(bundle, runtime_auth, h8_config)
                if canonical_json_sha256(current_identity) != worker_identity_sha:
                    raise ValueError("D2-B1 live worker identity drifted before the next response")
                if request.prompt_position_in_round == 0:
                    current_snapshot = verify_snapshot(runtime_auth, h8_config)
                    if current_snapshot["snapshot_listing_sha256"] != snapshot["snapshot_listing_sha256"]:
                        raise ValueError("D2-B1 model snapshot drifted before the next generation round")
                generated, attempts = run_attempts(
                    bundle,
                    request,
                    h8_config,
                    event_handle,
                    events_by_response.get(request.response_id, []),
                    int(auth["sampling"]["max_attempts_same_seed"]),
                )
                entry = context["fingerprint"]["entries"][request.prompt_index]
                record = {
                    "schema_version": "h8-d2b1-formal-development-response-1.0",
                    "response_id": request.response_id,
                    "schedule_position": request.schedule_position,
                    "bank_position": request.bank_position,
                    "replicate_id": request.replicate_id,
                    "prompt_position_in_round": request.prompt_position_in_round,
                    "prompt_index": request.prompt_index,
                    "prompt_id": request.prompt_id,
                    "prompt": request.prompt,
                    "prompt_sha256": request.prompt_sha256,
                    "category": entry.get("category"),
                    "task_metadata": entry.get("metadata", {}),
                    "generation_seed": request.generation_seed,
                    "seed_digest_sha256": request.seed_digest_sha256,
                    "derivation_counter": request.derivation_counter,
                    "data_role": request.data_role,
                    "bank_id": request.bank_id,
                    "attack_family": request.attack_family,
                    "attack_instance_id": request.attack_instance_id,
                    "evaluation_unit_id": request.evaluation_unit_id,
                    "batch_size": 1,
                    "attempt_records": attempts,
                    "model_name": bundle.name,
                    "model_revision": bundle.revision,
                    "worker_runtime_identity": worker_identity,
                    "worker_runtime_identity_sha256": worker_identity_sha,
                    "frozen_provenance": provenance,
                    "frozen_provenance_sha256": provenance_sha,
                    "generation_manifest_set_sha256": context["manifest_set_sha256"],
                    "manifest_request_sha256": request.manifest_request_sha256,
                    **materialization_binding,
                    "eligible_for_development_reference": request.data_role == REFERENCE_ROLE,
                    "eligible_for_development_intact_target": request.data_role == INTACT_TARGET_ROLE,
                    "eligible_for_development_attack": request.data_role == ATTACK_ROLE,
                    "eligible_for_future_formal_reference": False,
                    "eligible_for_future_final_heldout": False,
                    "eligible_for_future_final_confirmation": False,
                    "eligible_for_score_parameter_fit": False,
                    "eligible_for_measurement_parameter_fit": False,
                    "development_comparison_performed": False,
                    "detector_statistics_computed": False,
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
                        "next_schedule_position": len(records),
                        "last_response_id": request.response_id,
                        "last_attack_instance_id": request.attack_instance_id,
                        "record_chain_sha256": final_chain,
                        "updated_at_utc": now(),
                        "development_comparison_performed": False,
                        "detector_statistics_computed": False,
                        "sample_size": "not_selected",
                        "aggregation": "not_selected",
                        "detector": "not_frozen",
                    },
                )
        return 0
    finally:
        if bundle is not None:
            bundle.close()
            del bundle
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _incomplete_report(
    auth: Mapping[str, Any], paths: Mapping[str, Path], returncode: int, cleanup: Mapping[str, Any]
) -> dict[str, Any]:
    records = read_jsonl(paths["responses"])
    events = read_jsonl(paths["events"])
    report = {
        "schema_version": "h8-d2b1-incomplete-report-1.0",
        "phase": "H8_D2B1_FORMAL_DEVELOPMENT_SAMPLING",
        "status": "NO_GO_TECHNICAL_OR_INTEGRITY_FAILURE",
        "worker_exit_code": returncode,
        "completed_responses": len(records),
        "next_schedule_position": len(records),
        "attempt_event_count": len(events),
        "gpu_cleanup": dict(cleanup),
        "development_comparison_performed": False,
        "detector_statistics_computed": False,
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "created_at_utc": now(),
        "next_gate": "STOP_REVIEW_AND_RESUME_ONLY_AFTER_EXACT_PREFIX_VALIDATION",
    }
    atomic_json(paths["incomplete_report"], report)
    return report


def parent(auth_path: Path) -> int:
    context = frozen_context(auth_path)
    auth = context["auth"]
    paths = artifact_paths(auth)
    if paths["report"].exists() or paths["archive_report"].exists():
        raise FileExistsError("A final D2-B1 report already exists; refusing to overwrite it")
    paths["output"].mkdir(parents=True, exist_ok=True)
    verify_runtime(context["runtime_auth"])
    snapshot = verify_snapshot(context["runtime_auth"], context["h8_config"])
    current_commit = verify_git_lineage(auth, auth_path)
    provenance_candidate = {
        "authorization_file_sha256": file_sha256(auth_path),
        "d2b0_github_archive_commit": auth["d2b0_github_archive_commit"],
        "d2b0_final_index_sha256": auth["frozen_inputs"]["d2b0_final_index_sha256"],
        "mmd_manifest_sha256": auth["frozen_inputs"]["mmd_manifest_sha256"],
        "score_manifest_sha256": auth["frozen_inputs"]["score_manifest_sha256"],
        "score_schema_sha256": auth["frozen_inputs"]["score_schema_sha256"],
        "generation_manifest_set_sha256": context["manifest_set_sha256"],
        "nested_subset_manifest_sha256": auth["frozen_inputs"]["nested_subset_manifest_sha256"],
        "configuration_selection_rule_sha256": context["selection_rule_sha256"],
        "fingerprint_sha256": auth["frozen_inputs"]["fingerprint_sha256"],
        "generation_config": context["h8_config"]["generation"],
        "generation_config_sha256": canonical_json_sha256(context["h8_config"]["generation"]),
        "model_snapshot": snapshot,
        "runtime": runtime_provenance(),
        "implementation_commit": auth["implementation_commit"],
        "actual_sampling_commit": current_commit,
        "strict_offline": True,
        "created_before_first_response_utc": now(),
    }
    if paths["provenance"].exists():
        provenance, provenance_sha = load_provenance(paths["provenance"])
        stable = {key: value for key, value in provenance_candidate.items() if key != "created_before_first_response_utc"}
        for key, value in stable.items():
            if provenance.get(key) != value:
                raise ValueError(f"D2-B1 frozen provenance drifted for {key}")
    else:
        provenance, provenance_sha = _write_provenance(paths["provenance"], provenance_candidate)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_CACHE": str(auth["runtime_execution"]["hf_hub_cache"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    cleanup_history: list[dict[str, Any]] = []
    while True:
        records = read_jsonl(paths["responses"])
        events = read_jsonl(paths["events"])
        verify_completed_prefix(
            records,
            context["schedule"],
            generation_manifest_set_sha256=context["manifest_set_sha256"],
            frozen_provenance_sha256=provenance_sha,
        )
        validate_attempt_events(
            events,
            context["schedule"],
            len(records),
            max_attempts=int(auth["sampling"]["max_attempts_same_seed"]),
        )
        if len(records) == TOTAL_RESPONSES:
            break
        next_request = context["schedule"][len(records)]
        if next_request.schedule_position < BASE_BANK_END:
            args = ["--worker-kind", "intact"]
            worker_label = "intact_reference_and_target"
        else:
            assert next_request.attack_instance_id is not None
            args = ["--worker-kind", "attack", "--endpoint-id", next_request.attack_instance_id]
            worker_label = next_request.attack_instance_id
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--auth", str(auth_path), "--worker", *args],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        cleanup = gpu_compute_processes()
        cleanup_history.append(
            {
                "worker": worker_label,
                "returncode": completed.returncode,
                "gpu_cleanup": cleanup,
                "completed_responses_after_worker": len(read_jsonl(paths["responses"])),
            }
        )
        if completed.returncode != 0 or cleanup.get("status") != "PASS":
            report = _incomplete_report(auth, paths, completed.returncode, cleanup)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
    records = read_jsonl(paths["responses"])
    events = read_jsonl(paths["events"])
    audit = audit_formal_records(
        records,
        context["schedule"],
        events,
        generation_manifest_set_sha256=context["manifest_set_sha256"],
        frozen_provenance_sha256=provenance_sha,
        smoke_response_ids=context["smoke_response_ids"],
        smoke_seeds=context["smoke_seeds"],
        nested_subset_payload=context["nested"],
        nested_subset_file_sha256=auth["frozen_inputs"]["nested_subset_manifest_sha256"],
        expected_materialization_by_endpoint=context["expected_materialization"],
        required_fields=REQUIRED_RESPONSE_FIELDS,
        max_attempts=int(auth["sampling"]["max_attempts_same_seed"]),
    )
    cleanup_final = gpu_compute_processes()
    status = "PASS" if audit["status"] == "PASS" and cleanup_final["status"] == "PASS" else "FAIL"
    report = {
        "schema_version": "h8-d2b1-formal-development-sampling-report-1.0",
        "phase": "H8_D2B1_FORMAL_DEVELOPMENT_SAMPLING",
        "status": status,
        "audit": audit,
        "formal_development_responses": len(records),
        "reference_responses": audit["role_counts"].get(REFERENCE_ROLE, 0),
        "intact_target_responses": audit["role_counts"].get(INTACT_TARGET_ROLE, 0),
        "attack_responses": audit["role_counts"].get(ATTACK_ROLE, 0),
        "d2b0_smoke_responses_in_formal_bank": audit["d2b0_smoke_response_id_overlap_count"],
        "generation_manifest_set_sha256": context["manifest_set_sha256"],
        "nested_subset_manifest_sha256": auth["frozen_inputs"]["nested_subset_manifest_sha256"],
        "configuration_selection_rule_sha256": context["selection_rule_sha256"],
        "d2b0_final_index_sha256": auth["frozen_inputs"]["d2b0_final_index_sha256"],
        "mmd_manifest_sha256": auth["frozen_inputs"]["mmd_manifest_sha256"],
        "score_manifest_sha256": auth["frozen_inputs"]["score_manifest_sha256"],
        "authorization_file_sha256": file_sha256(auth_path),
        "sampling_provenance_sha256": provenance_sha,
        "sampling_provenance": provenance,
        "responses_file": str(paths["responses"].relative_to(ROOT)),
        "responses_file_sha256": file_sha256(paths["responses"]),
        "attempt_events_file": str(paths["events"].relative_to(ROOT)),
        "attempt_events_file_sha256": file_sha256(paths["events"]),
        "sampling_provenance_file_sha256": file_sha256(paths["provenance"]),
        "worker_cleanup_history": cleanup_history,
        "gpu_cleanup_final": cleanup_final,
        "development_comparison_performed": False,
        "detector_statistics_computed": False,
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "created_at_utc": now(),
        "next_gate": "STOP_WAIT_FOR_EXPLICIT_D2C_DEVELOPMENT_COMPARISON_APPROVAL",
    }
    atomic_json(paths["report"], report)
    paths["archive"].mkdir(parents=True, exist_ok=True)
    atomic_json(paths["archive_report"], report)
    hash_index = {
        "responses": {
            "path": str(paths["responses"].relative_to(ROOT)),
            "file_sha256": file_sha256(paths["responses"]),
        },
        "attempt_events": {
            "path": str(paths["events"].relative_to(ROOT)),
            "file_sha256": file_sha256(paths["events"]),
        },
        "sampling_provenance": {
            "path": str(paths["provenance"].relative_to(ROOT)),
            "file_sha256": file_sha256(paths["provenance"]),
            "payload_sha256": provenance_sha,
        },
        "report": {
            "path": str(paths["archive_report"].relative_to(ROOT)),
            "file_sha256": file_sha256(paths["archive_report"]),
        },
        "generation_manifest_set_sha256": context["manifest_set_sha256"],
        "nested_subset_manifest_sha256": auth["frozen_inputs"]["nested_subset_manifest_sha256"],
        "d2b0_final_index_sha256": auth["frozen_inputs"]["d2b0_final_index_sha256"],
    }
    atomic_json(paths["archive_index"], hash_index)
    atomic_json(
        paths["progress"],
        {
            "status": "completed" if status == "PASS" else "failed_audit",
            "completed_responses": len(records),
            "next_schedule_position": len(records),
            "record_chain_sha256": audit["final_record_chain_sha256"],
            "final_report_sha256": file_sha256(paths["archive_report"]),
            "final_hash_index_sha256": file_sha256(paths["archive_index"]),
            "updated_at_utc": now(),
            "development_comparison_performed": False,
            "detector_statistics_computed": False,
            "sample_size": "not_selected",
            "aggregation": "not_selected",
            "detector": "not_frozen",
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def preflight(auth_path: Path) -> int:
    context = frozen_context(auth_path)
    auth = context["auth"]
    runtime = verify_runtime(context["runtime_auth"])
    output = {
        "status": "PASS",
        "formal_development_sampling_authorized": True,
        "request_count": len(context["schedule"]),
        "role_counts": dict(sorted(Counter(row.data_role for row in context["schedule"]).items())),
        "unique_response_id_count": len({row.response_id for row in context["schedule"]}),
        "unique_generation_seed_count": len({row.generation_seed for row in context["schedule"]}),
        "d2b0_smoke_seed_overlap_count": len(
            {row.generation_seed for row in context["schedule"]} & context["smoke_seeds"]
        ),
        "attack_endpoint_count": len(context["instances"]),
        "generation_manifest_set_sha256": context["manifest_set_sha256"],
        "nested_subset_manifest_sha256": auth["frozen_inputs"]["nested_subset_manifest_sha256"],
        "configuration_selection_rule_sha256": context["selection_rule_sha256"],
        "d2b0_materialization_endpoint_count": len(context["expected_materialization"]),
        "runtime": runtime,
        "development_comparison_performed": False,
        "detector_statistics_computed": False,
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 D2-B1 formal development sampling runner")
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-kind", choices=("intact", "attack"))
    parser.add_argument("--endpoint-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = args.auth.resolve()
    if args.preflight_only:
        return preflight(auth_path)
    if args.worker:
        if not args.worker_kind:
            raise ValueError("D2-B1 worker kind is required")
        return worker(auth_path, args.worker_kind, args.endpoint_id)
    return parent(auth_path)


if __name__ == "__main__":
    raise SystemExit(main())
