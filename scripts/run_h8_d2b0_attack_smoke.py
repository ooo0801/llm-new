#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d1c import load_frozen_score_calibration
from llm_integrity.h8_d2a import load_canonical_envelope as load_d2a_envelope
from llm_integrity.h8_d2b0 import (
    SMOKE_COUNT,
    SMOKE_ROLE,
    file_sha256,
    load_canonical_envelope,
    parameter_state_sketch,
)
from llm_integrity.h8_sampling import (
    canonical_json_sha256,
    gpu_compute_processes,
    run_with_same_seed_retry,
    runtime_provenance,
    sha256_text,
)
from llm_integrity.modeling import model_metadata
from llm_integrity.paper_finetuning import train_lora_manifest_variant
from llm_integrity.paper_in_memory_attacks import apply_manifest_in_memory
from llm_integrity.paper_quantization import load_quantized_manifest_variant
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement
from run_h8_m0f_calibration_sampling import verify_runtime, verify_snapshot
from run_h8_mmd_smoke import cached_snapshot_provenance, generate_one


DEFAULT_AUTH = ROOT / "configs/h8_d2b0_development_sampling_authorization_preflight.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes the repository")
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


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Frozen {label} SHA256 mismatch")


def load_auth(path: Path) -> dict[str, Any]:
    auth = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(auth, dict) or auth.get("phase") != "H8_D2B0_DEVELOPMENT_SAMPLING_AUTHORIZATION_PREFLIGHT":
        raise ValueError("Invalid D2-B0 authorization artifact")
    if auth.get("authorization_status") != "user_approved_preflight_only":
        raise PermissionError("D2-B0 authorization status mismatch")
    if auth.get("formal_development_sampling_authorized") is not False:
        raise PermissionError("The 3,840-response development run is not authorized")
    if auth.get("attack_smoke_sampling_authorized") is not True:
        raise PermissionError("The exactly-eight-response attack smoke is not authorized")
    if int(auth["smoke"]["exact_response_count"]) != SMOKE_COUNT or int(auth["smoke"]["batch_size"]) != 1:
        raise ValueError("D2-B0 smoke count/batch changed")
    if not all(value is True for value in auth["forbidden_operations"].values()):
        raise PermissionError("Every D2-B0 forbidden operation must remain enabled")
    return auth


def verify_implementation(auth: Mapping[str, Any], auth_path: Path) -> str:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(auth["implementation_commit"]), "HEAD"],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(auth_path.relative_to(ROOT))], cwd=ROOT, check=True
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree must be clean before D2-B0 smoke")
    require_hash(resolve(auth["implementation"]["runner"]), auth["implementation"]["runner_sha256"], "runner")
    require_hash(resolve(auth["implementation"]["module"]), auth["implementation"]["module_sha256"], "module")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def load_context(auth_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    auth = load_auth(auth_path)
    frozen = auth["frozen_inputs"]
    bindings = (
        ("h8_config", "h8_config_sha256", "H8 generation config"),
        ("runtime_authorization", "runtime_authorization_sha256", "runtime authorization"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12"),
        ("d2a_artifact_index", "d2a_artifact_index_sha256", "D2-A artifact index"),
        ("d2a_preflight_report", "d2a_preflight_report_sha256", "D2-A preflight report"),
        ("d2b0_pre_smoke_report", "d2b0_pre_smoke_report_sha256", "D2-B0 pre-smoke report"),
        ("protocol", "protocol_sha256", "D2-B0 protocol"),
        ("lora_training_data", "lora_training_data_sha256", "D2 LoRA training data"),
    )
    for path_key, hash_key, label in bindings:
        require_hash(resolve(frozen[path_key]), frozen[hash_key], label)
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
    pre_index = json.loads(resolve(frozen["d2b0_pre_smoke_index"]).read_text(encoding="utf-8"))
    require_hash(resolve(frozen["d2b0_pre_smoke_index"]), frozen["d2b0_pre_smoke_index_sha256"], "D2-B0 pre-smoke index")
    smoke_info = pre_index["attack_smoke_manifest"]
    pre_directory = resolve(frozen["d2b0_pre_smoke_report"]).parent
    revised = load_canonical_envelope(
        pre_directory / "D2B0_REVISED_CONFIGURATION_MANIFEST.json",
        expected_artifact_type="h8_d2b0_revised_configuration_manifest",
        expected_file_sha256=pre_index["revised_configuration_manifest"]["file_sha256"],
        expected_payload_sha256=pre_index["revised_configuration_manifest"]["payload_sha256"],
    )
    freshness = load_canonical_envelope(
        pre_directory / "D2B0_FRESH_ATTACK_PROVENANCE_PREFLIGHT.json",
        expected_artifact_type="h8_d2b0_fresh_attack_provenance_preflight",
        expected_file_sha256=pre_index["freshness_preflight"]["file_sha256"],
        expected_payload_sha256=pre_index["freshness_preflight"]["payload_sha256"],
    )
    lora_isolation = load_canonical_envelope(
        pre_directory / "D2B0_LORA_TRAINING_DATA_ISOLATION_AUDIT.json",
        expected_artifact_type="h8_d2b0_lora_training_isolation_audit",
        expected_file_sha256=pre_index["lora_training_isolation"]["file_sha256"],
        expected_payload_sha256=pre_index["lora_training_isolation"]["payload_sha256"],
    )
    if (
        revised.get("configuration_count") != 12
        or not revised.get("selection_rule", {}).get("primary", "").startswith("minimize_false_positive_count")
        or freshness.get("status") != "PASS"
        or freshness.get("id_level_overlap_status") != "verified_zero"
        or lora_isolation.get("status") != "PASS"
        or lora_isolation.get("mcc12_prompt_or_response_leakage_detected") is not False
        or lora_isolation.get("training_data_sha256") != frozen["lora_training_data_sha256"]
    ):
        raise ValueError("D2-B0 revised selection/freshness/isolation artifact gate failed")
    smoke = load_canonical_envelope(
        resolve(frozen["d2b0_attack_smoke_manifest"]),
        expected_artifact_type="h8_d2b0_attack_smoke_generation_manifest",
        expected_file_sha256=smoke_info["file_sha256"],
        expected_payload_sha256=smoke_info["payload_sha256"],
    )
    if smoke["request_count"] != SMOKE_COUNT or smoke["formal_development_seed_overlap_count"] != 0:
        raise ValueError("Frozen D2-B0 smoke schedule changed")
    d2a_index = json.loads(resolve(frozen["d2a_artifact_index"]).read_text(encoding="utf-8"))
    instance_info = d2a_index["attack_instance_manifest"]
    instances = load_d2a_envelope(
        resolve(frozen["d2a_directory"]) / instance_info["path"],
        expected_artifact_type="h8_d2a_fresh_attack_instance_manifest",
        expected_file_sha256=instance_info["file_sha256"],
        expected_payload_sha256=instance_info["payload_sha256"],
    )["instances"]
    if [row["attack_instance_id"] for row in instances] != [row["attack_instance_id"] for row in smoke["requests"]]:
        raise ValueError("Smoke endpoint order differs from frozen D2-A instances")
    fingerprint = json.loads(resolve(frozen["fingerprint"]).read_text(encoding="utf-8"))
    if len(fingerprint.get("entries", [])) != 12:
        raise ValueError("H6 MCC12 count changed")
    h8_config = yaml.safe_load(resolve(frozen["h8_config"]).read_text(encoding="utf-8"))
    verify_implementation(auth, auth_path)
    return auth, h8_config, instances, smoke["requests"]


def endpoint_paths(auth: Mapping[str, Any], endpoint_id: str) -> dict[str, Path]:
    root = resolve(auth["artifacts"]["result_root"]) / endpoint_id
    return {
        "root": root,
        "response": root / "smoke_response.json",
        "materialization": root / "materialization_report.json",
    }


def _target_scope(instance: Mapping[str, Any]) -> str:
    config = instance["configuration"]
    if instance["family"] == "pruning" and config["type"] == "structured_pruning":
        return "ffn" if config["structure"] == "ffn_channels" else "attention"
    return str(config.get("target_scope", "full_model"))


def _attack_manifest(instance: Mapping[str, Any]) -> dict[str, Any]:
    family = str(instance["family"])
    config = dict(instance["configuration"])
    manifest_family = {
        "gaussian": "gaussian_noise",
        "lora": "finetuning",
    }.get(family, family)
    if family == "pruning":
        manifest_family = str(config["type"])
    return {
        "variant_id": str(instance["attack_instance_id"]),
        "family": manifest_family,
        "seed": int(instance["materialization_seed"]),
        "configuration": config,
    }


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _materialize_lora(
    auth: Mapping[str, Any], h8_config: Mapping[str, Any], instance: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from llm_integrity.modeling import load_model
    from peft import PeftModel

    adapter_root = resolve(auth["artifacts"]["adapter_root"])
    adapter_path = adapter_root / str(instance["attack_instance_id"])
    report_path = adapter_path / "training_report.json"
    manifest = _attack_manifest(instance)
    if adapter_path.exists() and not report_path.is_file():
        raise FileExistsError("Partial LoRA adapter exists without a training report")
    if not report_path.is_file():
        report = train_lora_manifest_variant(
            model_config=h8_config["model"],
            variant=manifest,
            data_path=resolve(auth["frozen_inputs"]["lora_training_data"]),
            output_root=adapter_root,
            max_length=int(auth["lora_training"]["max_length"]),
            batch_size=1,
            gradient_accumulation_steps=int(auth["lora_training"]["gradient_accumulation_steps"]),
        )
        training = report.__dict__
    else:
        training = json.loads(report_path.read_text(encoding="utf-8"))
    expected_steps = int(instance["configuration"]["steps"])
    if int(training["requested_steps"]) != expected_steps or int(training["completed_steps"]) != expected_steps:
        raise RuntimeError("LoRA did not complete the frozen number of steps")
    if training["data_sha256"] != auth["frozen_inputs"]["lora_training_data_sha256"]:
        raise ValueError("LoRA report training-data hash mismatch")
    artifacts = _artifact_hashes(adapter_path)
    if "adapter_model.safetensors" not in artifacts or "adapter_config.json" not in artifacts:
        raise RuntimeError("LoRA adapter payload/config missing")
    bundle = load_model(h8_config["model"])
    bundle.model = PeftModel.from_pretrained(bundle.model, adapter_path, is_trainable=False)
    bundle.model.eval()
    return bundle, {
        "execution_mode": "trained_lora_adapter",
        "training_report": training,
        "adapter_path": str(adapter_path),
        "adapter_artifact_hashes": artifacts,
        "adapter_artifact_set_sha256": canonical_json_sha256(artifacts),
    }


def _materialize_endpoint(
    auth: Mapping[str, Any], h8_config: Mapping[str, Any], instance: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    family = str(instance["family"])
    manifest = _attack_manifest(instance)
    if family == "quantization":
        bundle, quant_report = load_quantized_manifest_variant(h8_config["model"], manifest)
        report = quant_report.to_dict()
        if (
            not report["exact_requested_method"]
            or not report["loaded_in_4bit"]
            or report["loaded_in_8bit"]
            or report["quantized_linear_modules"] <= 0
            or report["double_quant"] is not False
        ):
            bundle.close()
            raise RuntimeError("Quantization endpoint was not exactly materialized in 4-bit")
        state = parameter_state_sketch(
            bundle.model,
            target_scope="full_model",
            seed=int(instance["materialization_seed"]),
        )
        return bundle, {
            "execution_mode": "bitsandbytes_reload_quantization",
            "quantization_load_report": report,
            "quantized_state_sketch": state,
            "bf16_or_fp16_fallback_detected": False,
        }
    if family == "lora":
        bundle, report = _materialize_lora(auth, h8_config, instance)
        report["adapter_loaded_state_sketch"] = parameter_state_sketch(
            bundle.model,
            target_scope="attention_ffn",
            seed=int(instance["materialization_seed"]),
        )
        return bundle, report

    from llm_integrity.modeling import load_model

    bundle = load_model(h8_config["model"])
    scope = _target_scope(instance)
    before = parameter_state_sketch(
        bundle.model, target_scope=scope, seed=int(instance["materialization_seed"])
    )
    attack_report = apply_manifest_in_memory(bundle.model, manifest)
    after = parameter_state_sketch(
        bundle.model, target_scope=scope, seed=int(instance["materialization_seed"])
    )
    report_payload = {
        "name": attack_report.name,
        "changed_parameters": int(attack_report.changed_parameters),
        "total_parameters": int(attack_report.total_parameters),
        "changed_fraction": float(attack_report.changed_fraction),
        "details": attack_report.details,
    }
    if attack_report.changed_parameters <= 0 or before["state_sketch_sha256"] == after["state_sketch_sha256"]:
        bundle.close()
        raise RuntimeError("In-memory attack did not create a verifiable state change")
    return bundle, {
        "execution_mode": "in_memory_weight_modification",
        "target_scope": scope,
        "before_state_sketch": before,
        "after_state_sketch": after,
        "attack_report": report_payload,
        "attack_report_sha256": canonical_json_sha256(report_payload),
    }


def worker(auth_path: Path, endpoint_id: str) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise PermissionError("D2-B0 worker requires strict offline mode")
    auth, h8_config, instances, requests = load_context(auth_path)
    instance = next((row for row in instances if row["attack_instance_id"] == endpoint_id), None)
    request = next((row for row in requests if row["attack_instance_id"] == endpoint_id), None)
    if instance is None or request is None:
        raise ValueError("Unknown frozen D2-B0 endpoint")
    paths = endpoint_paths(auth, endpoint_id)
    if paths["root"].exists():
        raise FileExistsError("Endpoint result directory already exists; successful records are immutable")
    paths["root"].mkdir(parents=True)
    runtime_auth = yaml.safe_load(resolve(auth["frozen_inputs"]["runtime_authorization"]).read_text(encoding="utf-8"))
    verify_runtime(runtime_auth)
    snapshot = verify_snapshot(runtime_auth, h8_config)
    bundle = None
    try:
        bundle, materialization = _materialize_endpoint(auth, h8_config, instance)
        entry = json.loads(resolve(auth["frozen_inputs"]["fingerprint"]).read_text(encoding="utf-8"))["entries"][
            int(request["prompt_index"])
        ]
        if entry["prompt_id"] != request["prompt_id"] or entry["metadata"]["prompt_sha256"] != request["prompt_sha256"]:
            raise ValueError("Smoke request prompt identity mismatch")
        request_object = SimpleNamespace(prompt=entry["prompt"], seed=int(request["generation_seed"]))
        generated, attempts = run_with_same_seed_retry(
            lambda seed, attempt_index: generate_one(bundle, request_object, h8_config),
            int(request["generation_seed"]),
            max_attempts=int(h8_config["generation"]["max_attempts_same_seed"]),
        )
        tokenizer = bundle.tokenizer
        shared_provenance = {
            "actual_execution_commit": verify_implementation(auth, auth_path),
            "model": model_metadata(bundle),
            "model_snapshot": snapshot,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
            "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "frozen_eos_token_ids": [int(value) for value in h8_config["generation"]["eos_token_ids"]],
            "generation_config": h8_config["generation"],
            "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
            "runtime": runtime_provenance(),
            "materialization": materialization,
            "materialization_sha256": canonical_json_sha256(materialization),
        }
        materialization_report = {
            "schema_version": "h8-d2b0-endpoint-materialization-1.0",
            "status": "PASS",
            "attack_instance_id": endpoint_id,
            "family": instance["family"],
            "configuration": instance["configuration"],
            "configuration_sha256": canonical_json_sha256(instance["configuration"]),
            "materialization_seed": instance["materialization_seed"],
            "materialization": materialization,
            "materialization_sha256": shared_provenance["materialization_sha256"],
            "created_at_utc": now(),
        }
        atomic_json(paths["materialization"], materialization_report)
        response = {
            "schema_version": "h8-d2b0-attack-smoke-response-1.0",
            "mode": "smoke_only",
            "data_role": SMOKE_ROLE,
            "response_id": request["response_id"],
            "schedule_position": request["schedule_position"],
            "attack_instance_id": endpoint_id,
            "attack_family": instance["family"],
            "prompt_index": request["prompt_index"],
            "prompt_id": request["prompt_id"],
            "prompt_sha256": request["prompt_sha256"],
            "generation_seed": request["generation_seed"],
            "materialization_seed": instance["materialization_seed"],
            "attempt_records": attempts,
            "batch_size": 1,
            "formal_development_eligible": False,
            "nested_q10_q20_eligible": False,
            "configuration_selection_eligible": False,
            "detector_statistics_computed": False,
            "provenance": shared_provenance,
            "provenance_sha256": canonical_json_sha256(shared_provenance),
            "created_at_utc": now(),
            **generated,
        }
        atomic_json(paths["response"], response)
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
        except Exception:
            pass


def _validate_existing_endpoint(
    auth: Mapping[str, Any], instance: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = endpoint_paths(auth, str(instance["attack_instance_id"]))
    response = json.loads(paths["response"].read_text(encoding="utf-8"))
    materialization = json.loads(paths["materialization"].read_text(encoding="utf-8"))
    expected = {
        "response_id": request["response_id"],
        "generation_seed": request["generation_seed"],
        "materialization_seed": instance["materialization_seed"],
        "attack_instance_id": instance["attack_instance_id"],
        "data_role": SMOKE_ROLE,
    }
    for key, value in expected.items():
        if response.get(key) != value:
            raise ValueError(f"Existing endpoint response mismatch for {key}")
    if response.get("detector_statistics_computed") is not False:
        raise ValueError("Smoke response contains detector statistics")
    if any(response.get(key) is not False for key in (
        "formal_development_eligible", "nested_q10_q20_eligible", "configuration_selection_eligible"
    )):
        raise ValueError("Smoke response eligibility drift")
    if materialization.get("status") != "PASS" or materialization.get("configuration") != instance["configuration"]:
        raise ValueError("Endpoint materialization report mismatch")
    return response, materialization


def _h6_lora_artifacts(auth: Mapping[str, Any]) -> dict[str, str]:
    root = Path(str(auth["freshness"]["h6_server_adapter_root"]))
    if not root.is_dir():
        return {}
    return {
        path.parent.name: file_sha256(path)
        for path in root.rglob("adapter_model.safetensors")
    }


def parent(auth_path: Path) -> int:
    auth, h8_config, instances, requests = load_context(auth_path)
    result_root = resolve(auth["artifacts"]["result_root"])
    result_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_CACHE": str(auth["runtime_execution"]["hf_hub_cache"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    cleanup_by_endpoint: dict[str, Any] = {}
    for instance, request in zip(instances, requests, strict=True):
        endpoint_id = str(instance["attack_instance_id"])
        paths = endpoint_paths(auth, endpoint_id)
        if paths["response"].is_file() and paths["materialization"].is_file():
            _validate_existing_endpoint(auth, instance, request)
            cleanup_by_endpoint[endpoint_id] = {"status": "PASS", "resume_reused_validated_success": True}
            continue
        if paths["root"].exists():
            raise FileExistsError(f"Partial endpoint directory requires review before retry: {paths['root']}")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--auth", str(auth_path), "--worker-endpoint", endpoint_id],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        cleanup = gpu_compute_processes()
        cleanup_by_endpoint[endpoint_id] = cleanup
        if completed.returncode != 0 or cleanup["status"] != "PASS":
            raise RuntimeError(f"D2-B0 endpoint smoke failed or leaked a GPU worker: {endpoint_id}")
        _validate_existing_endpoint(auth, instance, request)

    responses: list[dict[str, Any]] = []
    materializations: list[dict[str, Any]] = []
    for instance, request in zip(instances, requests, strict=True):
        response, materialization = _validate_existing_endpoint(auth, instance, request)
        responses.append(response)
        materializations.append(materialization)
    seeds = {int(row["generation_seed"]) for row in responses}
    if len(responses) != SMOKE_COUNT or len(seeds) != SMOKE_COUNT:
        raise RuntimeError("D2-B0 did not produce exactly eight unique-seed smoke responses")

    h6_lora_hashes = _h6_lora_artifacts(auth)
    d2_lora_hashes = {
        row["attack_instance_id"]: row["materialization"]["adapter_artifact_hashes"]["adapter_model.safetensors"]
        for row in materializations
        if row["family"] == "lora"
    }
    adapter_overlap = sorted(set(h6_lora_hashes.values()) & set(d2_lora_hashes.values()))
    final_cleanup = gpu_compute_processes()
    status = "PASS" if not adapter_overlap and final_cleanup["status"] == "PASS" else "FAIL"
    stop_counts = {
        value: sum(row["stop_reason"] == value for row in responses)
        for value in sorted({row["stop_reason"] for row in responses})
    }
    archive = resolve(auth["artifacts"]["archive_dir"])
    if archive.exists() and any(archive.iterdir()):
        raise FileExistsError("Refusing to overwrite D2-B0 archive")
    archive.mkdir(parents=True, exist_ok=True)
    for row in responses:
        atomic_json(archive / f"{row['attack_instance_id']}_SMOKE_RESPONSE.json", row)
    for row in materializations:
        atomic_json(archive / f"{row['attack_instance_id']}_MATERIALIZATION.json", row)
    materialization_audit = {
        "schema_version": "h8-d2b0-freshness-materialization-audit-1.0",
        "status": status,
        "endpoint_count": len(materializations),
        "endpoints": materializations,
        "h6_persisted_lora_adapter_count": len(h6_lora_hashes),
        "h6_lora_adapter_hashes": h6_lora_hashes,
        "d2_lora_adapter_hashes": d2_lora_hashes,
        "h6_d2_lora_adapter_hash_overlap": adapter_overlap,
        "h6_in_memory_and_quantized_state_artifact_hash_status": "not_archived_unverifiable_at_artifact_level",
        "configuration_seed_freshness_preflight_remains_binding": True,
        "quantization_exact_materialization": all(
            row["materialization"].get("bf16_or_fp16_fallback_detected") is False
            for row in materializations
            if row["family"] == "quantization"
        ),
        "created_at_utc": now(),
    }
    atomic_json(archive / "D2B0_FRESHNESS_MATERIALIZATION_AUDIT.json", materialization_audit)
    response_index = {
        row["attack_instance_id"]: {
            "response_file_sha256": file_sha256(archive / f"{row['attack_instance_id']}_SMOKE_RESPONSE.json"),
            "materialization_file_sha256": file_sha256(archive / f"{row['attack_instance_id']}_MATERIALIZATION.json"),
        }
        for row in responses
    }
    report = {
        "schema_version": "h8-d2b0-development-sampling-authorization-preflight-report-1.0",
        "phase": "H8_D2B0_DEVELOPMENT_SAMPLING_AUTHORIZATION_PREFLIGHT",
        "status": status,
        "actual_execution_commit": verify_implementation(auth, auth_path),
        "formal_development_responses": 0,
        "attack_smoke_only_responses": len(responses),
        "development_comparison_performed": False,
        "detector_statistics_computed": False,
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "response_id_unique_count": len({row["response_id"] for row in responses}),
        "generation_seed_unique_count": len(seeds),
        "formal_development_seed_overlap_count": 0,
        "stop_reason_counts": stop_counts,
        "legal_first_token_eos_count": sum(bool(row["legal_first_token_eos"]) for row in responses),
        "endpoint_materialization_count": len(materializations),
        "families": {
            family: sum(row["family"] == family for row in materializations)
            for family in ("gaussian", "pruning", "lora", "quantization")
        },
        "lora_training_data_sha256": auth["frozen_inputs"]["lora_training_data_sha256"],
        "h6_d2_lora_adapter_hash_overlap": adapter_overlap,
        "freshness_materialization_audit_sha256": file_sha256(
            archive / "D2B0_FRESHNESS_MATERIALIZATION_AUDIT.json"
        ),
        "gpu_cleanup_by_endpoint": cleanup_by_endpoint,
        "gpu_cleanup_final": final_cleanup,
        "response_materialization_index": response_index,
        "created_at_utc": now(),
        "next_gate": "STOP_WAIT_FOR_EXPLICIT_3840_RESPONSE_DEVELOPMENT_SAMPLING_APPROVAL",
    }
    atomic_json(archive / "H8_D2B0_DEVELOPMENT_SAMPLING_AUTHORIZATION_PREFLIGHT_REPORT.json", report)
    atomic_json(
        archive / "D2B0_FINAL_SHA256_INDEX.json",
        {
            "report": file_sha256(
                archive / "H8_D2B0_DEVELOPMENT_SAMPLING_AUTHORIZATION_PREFLIGHT_REPORT.json"
            ),
            "freshness_materialization_audit": report["freshness_materialization_audit_sha256"],
            "endpoint_artifacts": response_index,
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


def preflight_only(auth_path: Path) -> int:
    auth, h8_config, instances, requests = load_context(auth_path)
    runtime_auth = yaml.safe_load(resolve(auth["frozen_inputs"]["runtime_authorization"]).read_text(encoding="utf-8"))
    os.environ["HF_HUB_CACHE"] = str(auth["runtime_execution"]["hf_hub_cache"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    snapshot = cached_snapshot_provenance(h8_config["model"]["name"], h8_config["model"]["revision"])
    if snapshot.get("resolved_snapshot") is None or snapshot.get("resolution_error"):
        raise RuntimeError("Exact offline Qwen2.5-32B snapshot provenance could not be resolved")
    output = {
        "status": "PASS",
        "formal_development_sampling_authorized": False,
        "attack_smoke_sampling_authorized": True,
        "smoke_request_count": len(requests),
        "endpoint_count": len(instances),
        "unique_smoke_seed_count": len({row["generation_seed"] for row in requests}),
        "model": h8_config["model"]["name"],
        "revision": h8_config["model"]["revision"],
        "runtime": verify_runtime(runtime_auth),
        "snapshot": snapshot,
        "formal_development_responses": 0,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--worker-endpoint", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = args.auth.resolve()
    if args.preflight_only:
        return preflight_only(auth_path)
    if args.worker_endpoint:
        return worker(auth_path, args.worker_endpoint)
    return parent(auth_path)


if __name__ == "__main__":
    raise SystemExit(main())
