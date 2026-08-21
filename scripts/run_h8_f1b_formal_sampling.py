#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d2b0 import parameter_state_sketch
from llm_integrity.h8_f1a import load_canonical_envelope, load_frozen_detector
from llm_integrity.h8_f1b import (
    BASE_PARTITION_ID,
    F1B_SCHEMA_VERSION,
    FORMAL_TOTAL,
    audit_formal_records,
    file_sha256,
    partition_requests,
    read_jsonl_strict,
    validate_frozen_inputs,
    validate_response_record,
)
from llm_integrity.h8_precalibration import canonical_sha256
from llm_integrity.h8_sampling import (
    canonical_json_sha256,
    gpu_compute_processes,
    runtime_provenance,
    sha256_text,
)
from llm_integrity.modeling import model_metadata
from llm_integrity.paper_finetuning import train_lora_manifest_variant
from llm_integrity.paper_in_memory_attacks import apply_manifest_in_memory
from llm_integrity.paper_quantization import load_quantized_manifest_variant
from run_h8_m0f_calibration_sampling import verify_runtime, verify_snapshot
from run_h8_mmd_smoke import generate_one


DEFAULT_AUTH = ROOT / "configs/h8_f1b_formal_final_confirmation_sampling.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
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


def append_jsonl_fsync(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Frozen {label} SHA256 mismatch")


def build_file_metadata_seal(paths: Sequence[Path]) -> dict[str, dict[str, int]]:
    seal: dict[str, dict[str, int]] = {}
    for path in sorted({item.resolve() for item in paths}):
        if not path.is_file():
            raise FileNotFoundError(f"F1-B cannot seal missing file: {path}")
        stat = path.stat()
        seal[str(path)] = {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    return seal


def verify_file_metadata_seal(seal: Mapping[str, Mapping[str, int]]) -> None:
    for logical, expected in seal.items():
        path = Path(logical)
        if not path.is_file():
            raise ValueError(f"F1-B sealed input disappeared before generation: {path}")
        stat = path.stat()
        if int(stat.st_size) != int(expected["size_bytes"]) or int(stat.st_mtime_ns) != int(expected["mtime_ns"]):
            raise ValueError(f"F1-B sealed input metadata drifted before generation: {path}")


def load_auth(path: Path, *, require_frozen_implementation: bool = True) -> dict[str, Any]:
    auth = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(auth, dict) or auth.get("phase") != "H8_F1B_FORMAL_FINAL_CONFIRMATION_SAMPLING":
        raise ValueError("Invalid F1-B authorization artifact")
    if auth.get("authorization_status") != "user_approved_formal_sampling_only":
        raise PermissionError("F1-B authorization status mismatch")
    if auth.get("formal_final_confirmation_sampling_authorized") is not True:
        raise PermissionError("F1-B formal sampling is not authorized")
    if auth.get("final_performance_confirmation_authorized") is not False:
        raise PermissionError("F1-B must not authorize F1-C performance confirmation")
    if not all(value is True for value in auth["forbidden_operations"].values()):
        raise PermissionError("Every F1-B performance/selection prohibition must remain enabled")
    if auth["runtime"].get("hf_hub_cache") != "/root/autodl-tmp/huggingface":
        raise ValueError("F1-B HF_HUB_CACHE must remain the verified data-disk cache")
    if require_frozen_implementation:
        commit = auth["git"].get("sampling_implementation_commit")
        runner_sha = auth["implementation"].get("runner_sha256")
        module_sha = auth["implementation"].get("module_sha256")
        protocol_sha = auth["implementation"].get("protocol_sha256")
        if not commit or not runner_sha or not module_sha or not protocol_sha:
            raise PermissionError("F1-B implementation commit/hashes are not frozen")
    return auth


def verify_git_state(auth: Mapping[str, Any], auth_path: Path) -> str:
    required_base = str(auth["git"]["required_f1a_smoke_archive_commit"])
    implementation = str(auth["git"]["sampling_implementation_commit"])
    for commit in (required_base, implementation):
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=True)
    for path in (
        auth_path,
        resolve(auth["implementation"]["runner"]),
        resolve(auth["implementation"]["module"]),
        resolve(auth["implementation"]["protocol"]),
    ):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    require_hash(resolve(auth["implementation"]["runner"]), auth["implementation"]["runner_sha256"], "F1-B runner")
    require_hash(resolve(auth["implementation"]["module"]), auth["implementation"]["module_sha256"], "F1-B module")
    require_hash(resolve(auth["implementation"]["protocol"]), auth["implementation"]["protocol_sha256"], "F1-B protocol")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree must be clean before or during F1-B sampling")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _load_envelope(auth: Mapping[str, Any], prefix: str, artifact_type: str) -> dict[str, Any]:
    frozen = auth["frozen_inputs"]
    return load_canonical_envelope(
        resolve(frozen[prefix]),
        expected_artifact_type=artifact_type,
        expected_file_sha256=frozen[f"{prefix}_sha256"],
        expected_payload_sha256=frozen[f"{prefix}_payload_sha256"],
    )


def load_context(
    auth_path: Path,
    *,
    require_frozen_implementation: bool = True,
) -> dict[str, Any]:
    auth = load_auth(auth_path, require_frozen_implementation=require_frozen_implementation)
    frozen = auth["frozen_inputs"]
    simple_bindings = (
        ("f1a_config", "f1a_config_sha256", "F1-A config"),
        ("f1a_protocol", "f1a_protocol_sha256", "F1-A protocol"),
        ("f1a_preflight_report", "f1a_preflight_report_sha256", "F1-A preflight report"),
        ("f1a_preflight_index", "f1a_preflight_index_sha256", "F1-A preflight index"),
        ("f1a_smoke_report", "f1a_smoke_report_sha256", "F1-A smoke report"),
        ("f1a_smoke_archive_index", "f1a_smoke_archive_index_sha256", "F1-A smoke archive index"),
        ("frozen_detector_manifest", "frozen_detector_manifest_sha256", "frozen detector manifest"),
        ("h8_generation_config", "h8_generation_config_sha256", "H8 generation config"),
        ("runtime_authorization", "runtime_authorization_sha256", "runtime authorization"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12"),
        ("lora_training_data", "lora_training_data_sha256", "F1 LoRA training data"),
    )
    for path_key, hash_key, label in simple_bindings:
        require_hash(resolve(frozen[path_key]), frozen[hash_key], label)

    preflight = json.loads(resolve(frozen["f1a_preflight_report"]).read_text(encoding="utf-8"))
    smoke_report = json.loads(resolve(frozen["f1a_smoke_report"]).read_text(encoding="utf-8"))
    smoke_archive = json.loads(resolve(frozen["f1a_smoke_archive_index"]).read_text(encoding="utf-8"))
    if (
        preflight.get("status") != "PASS_READY_FOR_EXACTLY_SIX_SMOKE_ONLY"
        or preflight.get("formal_final_confirmation_responses") != 0
        or smoke_report.get("status") != "PASS"
        or smoke_report.get("tiny_smoke_responses") != 6
        or smoke_report.get("formal_final_confirmation_responses") != 0
        or smoke_report.get("gpu_cleanup_after_all_workers", {}).get("status") != "PASS"
        or smoke_archive.get("status") != "PASS"
    ):
        raise ValueError("F1-B F1-A preflight/smoke PASS gate failed")

    detector = load_frozen_detector(
        resolve(frozen["frozen_detector_archive"]),
        expected_manifest_sha256=frozen["frozen_detector_manifest_sha256"],
    )
    if detector["manifest"].get("selected_detector_payload_sha256") != frozen["selected_detector_payload_sha256"]:
        raise ValueError("F1-B frozen detector selected payload mismatch")
    generation = _load_envelope(auth, "generation_manifest", "h8_f1a_final_generation_manifest")
    attack = _load_envelope(auth, "attack_manifest", "h8_f1a_fresh_heldout_attack_endpoint_manifest")
    permutation = _load_envelope(auth, "permutation_manifest", "h8_f1a_final_global_permutation_seed_manifest")
    freshness = _load_envelope(auth, "freshness_audit", "h8_f1a_freshness_audit")
    historical = _load_envelope(auth, "historical_identity_index", "h8_f1a_historical_identity_index")
    preflight_index = json.loads(resolve(frozen["f1a_preflight_index"]).read_text(encoding="utf-8"))
    smoke_manifest_info = preflight_index["artifacts"]["smoke_manifest"]
    smoke_manifest = load_canonical_envelope(
        resolve(frozen["f1a_preflight_report"]).parent / "F1A_TINY_SMOKE_MANIFEST.json",
        expected_artifact_type="h8_f1a_tiny_smoke_manifest",
        expected_file_sha256=smoke_manifest_info["file_sha256"],
        expected_payload_sha256=smoke_manifest_info["payload_sha256"],
    )
    frozen_audit = validate_frozen_inputs(
        generation,
        attack,
        permutation,
        freshness,
        historical,
        smoke_manifest,
    )
    h8_config = yaml.safe_load(resolve(frozen["h8_generation_config"]).read_text(encoding="utf-8"))
    runtime_auth = yaml.safe_load(resolve(frozen["runtime_authorization"]).read_text(encoding="utf-8"))
    fingerprint = json.loads(resolve(frozen["fingerprint"]).read_text(encoding="utf-8"))
    if len(fingerprint.get("entries", [])) != 12:
        raise ValueError("F1-B MCC12 count changed")
    frozen_identity = {
        "schema_version": F1B_SCHEMA_VERSION,
        "authorization_sha256": file_sha256(auth_path),
        "sampling_implementation_commit": auth["git"].get("sampling_implementation_commit"),
        "f1a_preflight_report_sha256": frozen["f1a_preflight_report_sha256"],
        "f1a_smoke_report_sha256": frozen["f1a_smoke_report_sha256"],
        "generation_manifest_sha256": frozen["generation_manifest_sha256"],
        "generation_manifest_payload_sha256": frozen["generation_manifest_payload_sha256"],
        "attack_manifest_sha256": frozen["attack_manifest_sha256"],
        "attack_manifest_payload_sha256": frozen["attack_manifest_payload_sha256"],
        "permutation_manifest_sha256": frozen["permutation_manifest_sha256"],
        "permutation_manifest_payload_sha256": frozen["permutation_manifest_payload_sha256"],
        "freshness_audit_sha256": frozen["freshness_audit_sha256"],
        "historical_identity_index_sha256": frozen["historical_identity_index_sha256"],
        "frozen_detector_manifest_sha256": frozen["frozen_detector_manifest_sha256"],
        "selected_detector_payload_sha256": frozen["selected_detector_payload_sha256"],
        "h8_generation_config_sha256": frozen["h8_generation_config_sha256"],
        "runtime_authorization_sha256": frozen["runtime_authorization_sha256"],
        "fingerprint_sha256": frozen["fingerprint_sha256"],
        "lora_training_data_sha256": frozen["lora_training_data_sha256"],
        "hf_hub_cache": auth["runtime"]["hf_hub_cache"],
    }
    sealed_paths = [
        auth_path,
        resolve(auth["implementation"]["runner"]),
        resolve(auth["implementation"]["module"]),
        resolve(auth["implementation"]["protocol"]),
        *(resolve(frozen[path_key]) for path_key, _, _ in simple_bindings),
        *(resolve(frozen[key]) for key in ("generation_manifest", "attack_manifest", "permutation_manifest", "freshness_audit", "historical_identity_index")),
        *[path for path in resolve(frozen["frozen_detector_archive"]).rglob("*") if path.is_file()],
    ]
    return {
        "auth_path": auth_path,
        "auth": auth,
        "generation": generation,
        "attack": attack,
        "permutation": permutation,
        "freshness": freshness,
        "historical": historical,
        "smoke_manifest": smoke_manifest,
        "frozen_audit": frozen_audit,
        "h8_config": h8_config,
        "runtime_auth": runtime_auth,
        "fingerprint": fingerprint,
        "frozen_identity": frozen_identity,
        "frozen_identity_sha256": canonical_sha256(frozen_identity),
        "file_metadata_seal": build_file_metadata_seal(sealed_paths),
    }


def partition_path(auth: Mapping[str, Any], partition_id: str, kind: str) -> Path:
    if partition_id != BASE_PARTITION_ID and not partition_id.startswith("h8f1_final_"):
        raise ValueError("Unsafe F1-B partition ID")
    root = resolve(auth["artifacts"]["output_root"])
    directory = {
        "records": auth["artifacts"]["record_directory"],
        "attempts": auth["artifacts"]["attempt_event_directory"],
        "materialization": auth["artifacts"]["materialization_directory"],
        "provenance": "provenance",
    }[kind]
    extension = ".jsonl" if kind in {"records", "attempts"} else ".json"
    return root / directory / f"{partition_id}{extension}"


def attack_manifest(endpoint: Mapping[str, Any], *, lora_training: bool = False) -> dict[str, Any]:
    family = str(endpoint["family"])
    configuration = dict(endpoint["configuration"])
    manifest_family = {"gaussian": "gaussian_noise", "lora": "finetuning"}.get(family, family)
    if family == "pruning":
        manifest_family = str(configuration["type"])
    seed = endpoint["training_seed"] if lora_training else endpoint["materialization_seed"]
    return {
        "variant_id": str(endpoint["attack_instance_id"]),
        "family": manifest_family,
        "seed": int(seed),
        "configuration": configuration,
    }


def target_scope(endpoint: Mapping[str, Any]) -> str:
    config = endpoint["configuration"]
    if endpoint["family"] == "pruning" and config["type"] == "structured_pruning":
        return "ffn" if config["structure"] == "ffn_channels" else "attention"
    return str(config.get("target_scope", "full_model"))


def artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def live_bundle_identity(bundle: Any, runtime_auth: Mapping[str, Any], h8_config: Mapping[str, Any]) -> dict[str, Any]:
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
    for key in (
        "tokenizer_class",
        "tokenizer_pad_token_id",
        "tokenizer_eos_token_id",
        "chat_template_sha256",
        "frozen_eos_token_ids",
        "generation_config_sha256",
    ):
        if actual[key] != expected[key]:
            raise ValueError(f"F1-B live model/tokenizer identity mismatch for {key}")
    device_map = dict(getattr(bundle.model, "hf_device_map", {}))
    if {str(value) for value in device_map.values()} & {"cpu", "disk"}:
        raise ValueError("F1-B CPU/disk model offload is forbidden")
    actual["hf_device_map_sha256"] = canonical_json_sha256(device_map)
    actual["identity_sha256"] = canonical_json_sha256(actual)
    return actual


def state_monitor_seed(partition_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"h8-f1b-state-monitor\0{partition_id}".encode("utf-8")).digest()[:4], "little")


def train_lora_worker(auth_path: Path, endpoint_id: str) -> int:
    context = load_context(auth_path)
    verify_git_state(context["auth"], auth_path)
    verify_file_metadata_seal(context["file_metadata_seal"])
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("HF_HUB_CACHE") != context["auth"]["runtime"]["hf_hub_cache"]
    ):
        raise PermissionError("F1-B LoRA training requires the frozen strict-offline data-disk cache")
    verify_runtime(context["runtime_auth"])
    verify_snapshot(context["runtime_auth"], context["h8_config"])
    endpoint = next(
        (row for row in context["attack"]["endpoints"] if row["attack_instance_id"] == endpoint_id),
        None,
    )
    if endpoint is None or endpoint.get("family") != "lora":
        raise ValueError("Unknown F1-B LoRA endpoint")
    adapter_root = resolve(context["auth"]["artifacts"]["adapter_root"])
    adapter_path = adapter_root / endpoint_id
    if adapter_path.exists():
        raise FileExistsError("F1-B LoRA adapter path already exists and must be reviewed")
    try:
        report = train_lora_manifest_variant(
            model_config=context["h8_config"]["model"],
            variant=attack_manifest(endpoint, lora_training=True),
            data_path=resolve(context["auth"]["frozen_inputs"]["lora_training_data"]),
            output_root=adapter_root,
            max_length=int(context["auth"]["lora_training"]["max_length"]),
            batch_size=1,
            gradient_accumulation_steps=int(context["auth"]["lora_training"]["gradient_accumulation_steps"]),
        )
        if report.completed_steps != int(endpoint["configuration"]["steps"]):
            raise RuntimeError("F1-B LoRA did not complete its frozen step count")
        if report.seed != int(endpoint["training_seed"]):
            raise ValueError("F1-B LoRA training did not use the frozen training seed")
        if report.data_sha256 != context["auth"]["frozen_inputs"]["lora_training_data_sha256"]:
            raise ValueError("F1-B LoRA training data SHA256 mismatch")
        return 0
    finally:
        cleanup_cuda()


def materialize_partition(context: Mapping[str, Any], partition_id: str) -> tuple[Any, dict[str, Any]]:
    h8_config = context["h8_config"]
    if partition_id == BASE_PARTITION_ID:
        from llm_integrity.modeling import load_model

        return load_model(h8_config["model"]), {
            "execution_mode": "frozen_intact_base_model",
            "shared_reference_bank_count": 1,
        }
    endpoint = next(
        row for row in context["attack"]["endpoints"] if row["attack_instance_id"] == partition_id
    )
    family = str(endpoint["family"])
    manifest = attack_manifest(endpoint)
    if family == "quantization":
        bundle, quant = load_quantized_manifest_variant(h8_config["model"], manifest)
        report = quant.to_dict()
        method = str(endpoint["configuration"]["method"]).lower()
        expected_8bit = method == "int8"
        if (
            report["exact_requested_method"] is not True
            or bool(report["loaded_in_8bit"]) != expected_8bit
            or bool(report["loaded_in_4bit"]) == expected_8bit
            or int(report["quantized_linear_modules"]) <= 0
        ):
            bundle.close()
            raise RuntimeError("F1-B quantization endpoint did not exactly materialize")
        if expected_8bit and float(report["int8_threshold"]) != float(endpoint["configuration"]["llm_int8_threshold"]):
            bundle.close()
            raise RuntimeError("F1-B INT8 endpoint threshold was not realized")
        state = parameter_state_sketch(bundle.model, target_scope="full_model", seed=int(endpoint["materialization_seed"]))
        return bundle, {
            "execution_mode": "bitsandbytes_reload_quantization",
            "quantization_load_report": report,
            "quantized_state_sketch": state,
            "bf16_or_fp16_fallback_detected": False,
        }
    if family == "lora":
        from llm_integrity.modeling import load_model
        from peft import PeftModel

        adapter_path = resolve(context["auth"]["artifacts"]["adapter_root"]) / partition_id
        report_path = adapter_path / "training_report.json"
        if not report_path.is_file():
            raise FileNotFoundError("F1-B LoRA adapter was not trained")
        training = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            int(training["completed_steps"]) != int(endpoint["configuration"]["steps"])
            or int(training["seed"]) != int(endpoint["training_seed"])
            or training["data_sha256"] != context["auth"]["frozen_inputs"]["lora_training_data_sha256"]
        ):
            raise ValueError("F1-B LoRA training report does not match the frozen endpoint")
        artifacts = artifact_hashes(adapter_path)
        if "adapter_model.safetensors" not in artifacts or "adapter_config.json" not in artifacts:
            raise RuntimeError("F1-B LoRA adapter payload is incomplete")
        bundle = load_model(h8_config["model"])
        bundle.model = PeftModel.from_pretrained(bundle.model, adapter_path, is_trainable=False)
        bundle.model.eval()
        return bundle, {
            "execution_mode": "trained_lora_adapter",
            "training_report": training,
            "adapter_artifact_hashes": artifacts,
            "adapter_artifact_set_sha256": canonical_json_sha256(artifacts),
            "adapter_loaded_state_sketch": parameter_state_sketch(
                bundle.model, target_scope="attention_ffn", seed=int(endpoint["materialization_seed"])
            ),
        }

    from llm_integrity.modeling import load_model

    bundle = load_model(h8_config["model"])
    scope = target_scope(endpoint)
    before = parameter_state_sketch(bundle.model, target_scope=scope, seed=int(endpoint["materialization_seed"]))
    attack = apply_manifest_in_memory(bundle.model, manifest)
    after = parameter_state_sketch(bundle.model, target_scope=scope, seed=int(endpoint["materialization_seed"]))
    if int(attack.changed_parameters) <= 0 or before["state_sketch_sha256"] == after["state_sketch_sha256"]:
        bundle.close()
        raise RuntimeError("F1-B in-memory attack did not create a verifiable state change")
    report = {
        "name": attack.name,
        "changed_parameters": int(attack.changed_parameters),
        "total_parameters": int(attack.total_parameters),
        "changed_fraction": float(attack.changed_fraction),
        "details": attack.details,
    }
    return bundle, {
        "execution_mode": "in_memory_weight_modification",
        "target_scope": scope,
        "before_state_sketch": before,
        "after_state_sketch": after,
        "attack_report": report,
        "attack_report_sha256": canonical_json_sha256(report),
    }


def materialization_identity(
    context: Mapping[str, Any], partition_id: str, materialization: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    endpoint = next(
        (row for row in context["attack"]["endpoints"] if row["attack_instance_id"] == partition_id),
        None,
    )
    payload = {
        "schema_version": F1B_SCHEMA_VERSION,
        "partition_id": partition_id,
        "attack_instance_id": None if endpoint is None else endpoint["attack_instance_id"],
        "attack_family": None if endpoint is None else endpoint["family"],
        "configuration": None if endpoint is None else endpoint["configuration"],
        "configuration_sha256": None if endpoint is None else endpoint["configuration_sha256"],
        "materialization_seed": None if endpoint is None else endpoint["materialization_seed"],
        "training_seed": None if endpoint is None else endpoint.get("training_seed"),
        "planned_artifact_identity_sha256": None if endpoint is None else endpoint["planned_artifact_identity_sha256"],
        "materialization": dict(materialization),
        "materialization_sha256": canonical_json_sha256(materialization),
    }
    return payload, canonical_sha256(payload)


def load_existing_prefix(
    path: Path,
    requests: Sequence[Mapping[str, Any]],
    *,
    frozen_identity_sha256: str,
    materialization_identity_sha256: str | None,
) -> list[dict[str, Any]]:
    records = read_jsonl_strict(path)
    if len(records) > len(requests):
        raise ValueError("F1-B partition contains more records than frozen requests")
    for index, record in enumerate(records):
        validate_response_record(
            record,
            requests[index],
            frozen_identity_sha256=frozen_identity_sha256,
            expected_materialization_identity_sha256=materialization_identity_sha256,
        )
    return records


def generate_with_same_seed(
    bundle: Any,
    request: Mapping[str, Any],
    h8_config: Mapping[str, Any],
    attempt_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(request["generation_seed"])
    attempts: list[dict[str, Any]] = []
    max_attempts = int(h8_config["generation"]["max_attempts_same_seed"])
    entry_prompt = str(request["_prompt"])
    for attempt_index in range(max_attempts):
        try:
            generated = generate_one(bundle, SimpleNamespace(prompt=entry_prompt, seed=seed), h8_config)
            attempts.append({"attempt_index": attempt_index, "seed": seed, "status": "success"})
            return generated, attempts
        except (RuntimeError, OSError) as exc:
            event = {
                "schema_version": F1B_SCHEMA_VERSION,
                "response_id": request["response_id"],
                "schedule_position": request["schedule_position"],
                "attempt_index": attempt_index,
                "seed": seed,
                "status": "technical_failure",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "created_at_utc": now(),
            }
            append_jsonl_fsync(attempt_path, event)
            attempts.append({key: value for key, value in event.items() if key not in {"schema_version", "response_id", "schedule_position", "created_at_utc"}})
            if attempt_index + 1 >= max_attempts:
                raise RuntimeError(f"F1-B response failed after {max_attempts} same-seed attempts") from exc
    raise AssertionError("unreachable")


def write_progress(context: Mapping[str, Any], *, active_partition: str | None, status: str) -> dict[str, Any]:
    partitions = partition_requests(context["generation"]["requests"])
    counts: dict[str, int] = {}
    total = 0
    for partition_id in partitions:
        count = len(read_jsonl_strict(partition_path(context["auth"], partition_id, "records")))
        counts[partition_id] = count
        total += count
    payload = {
        "schema_version": F1B_SCHEMA_VERSION,
        "status": status,
        "active_partition": active_partition,
        "completed_response_count": total,
        "planned_response_count": FORMAL_TOTAL,
        "partition_completed_counts": counts,
        "formal_detector_statistics_computed": False,
        "updated_at_utc": now(),
    }
    atomic_json(resolve(context["auth"]["artifacts"]["output_root"]) / context["auth"]["artifacts"]["progress"], payload)
    return payload


def worker(auth_path: Path, partition_id: str) -> int:
    if (
        os.environ.get("HF_HUB_OFFLINE") != "1"
        or os.environ.get("TRANSFORMERS_OFFLINE") != "1"
        or os.environ.get("HF_HUB_CACHE") != "/root/autodl-tmp/huggingface"
    ):
        raise PermissionError("F1-B worker requires the frozen strict-offline data-disk cache")
    context = load_context(auth_path)
    sampling_commit = verify_git_state(context["auth"], auth_path)
    partitions = partition_requests(context["generation"]["requests"])
    if partition_id not in partitions:
        raise ValueError("Unknown F1-B frozen partition")
    requests = partitions[partition_id]
    actual_runtime = verify_runtime(context["runtime_auth"])
    snapshot = verify_snapshot(context["runtime_auth"], context["h8_config"])
    bundle = None
    try:
        bundle, materialization = materialize_partition(context, partition_id)
        monitor_seed = state_monitor_seed(partition_id)
        state_monitor = parameter_state_sketch(bundle.model, target_scope="full_model", seed=monitor_seed)
        materialization = {**materialization, "live_model_state_monitor": state_monitor}
        identity_payload, identity_sha = materialization_identity(context, partition_id, materialization)
        materialization_path = partition_path(context["auth"], partition_id, "materialization")
        if materialization_path.is_file():
            existing = json.loads(materialization_path.read_text(encoding="utf-8"))
            if existing.get("materialization_identity_sha256") != identity_sha or existing.get("payload") != identity_payload:
                raise ValueError("F1-B rematerialized endpoint identity differs from the sealed identity")
        else:
            atomic_json(
                materialization_path,
                {
                    "schema_version": F1B_SCHEMA_VERSION,
                    "status": "PASS_SEALED_BEFORE_FIRST_RESPONSE",
                    "materialization_identity_sha256": identity_sha,
                    "payload": identity_payload,
                    "sealed_at_utc": now(),
                },
            )
        tokenizer = bundle.tokenizer
        live_identity = live_bundle_identity(bundle, context["runtime_auth"], context["h8_config"])
        snapshot_files = [path for path in Path(snapshot["resolved_snapshot"]).rglob("*") if path.is_file()]
        execution_file_seal = {
            **context["file_metadata_seal"],
            **build_file_metadata_seal(snapshot_files),
        }
        provenance = {
            "schema_version": F1B_SCHEMA_VERSION,
            "sampling_code_commit": context["auth"]["git"]["sampling_implementation_commit"],
            "execution_commit": sampling_commit,
            "frozen_identity": context["frozen_identity"],
            "frozen_identity_sha256": context["frozen_identity_sha256"],
            "partition_id": partition_id,
            "materialization_identity_sha256": identity_sha,
            "model": model_metadata(bundle),
            "model_snapshot": snapshot,
            "runtime_identity": actual_runtime,
            "live_bundle_identity": live_identity,
            "live_bundle_identity_sha256": live_identity["identity_sha256"],
            "live_model_state_monitor_sha256": state_monitor["state_sketch_sha256"],
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
            "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "frozen_eos_token_ids": [int(value) for value in context["h8_config"]["generation"]["eos_token_ids"]],
            "generation_config": context["h8_config"]["generation"],
            "generation_config_sha256": canonical_json_sha256(context["h8_config"]["generation"]),
        }
        provenance_sha = canonical_json_sha256(provenance)
        provenance_path = partition_path(context["auth"], partition_id, "provenance")
        if provenance_path.is_file():
            if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
                raise ValueError("F1-B partition provenance changed on resume")
        else:
            atomic_json(provenance_path, provenance)

        record_path = partition_path(context["auth"], partition_id, "records")
        attempt_path = partition_path(context["auth"], partition_id, "attempts")
        existing_records = load_existing_prefix(
            record_path,
            requests,
            frozen_identity_sha256=context["frozen_identity_sha256"],
            materialization_identity_sha256=identity_sha,
        )
        fingerprint_by_id = {str(entry["prompt_id"]): entry for entry in context["fingerprint"]["entries"]}
        previous_unit_key: tuple[str, str, int] | None = None
        for request in requests[len(existing_records):]:
            verify_file_metadata_seal(execution_file_seal)
            if live_bundle_identity(bundle, context["runtime_auth"], context["h8_config"])["identity_sha256"] != live_identity["identity_sha256"]:
                raise ValueError("F1-B live model/tokenizer/runtime identity drifted before the next response")
            unit_key = (
                str(request["data_role"]),
                str(request["evaluation_unit_id"]),
                int(request["replicate_id"]),
            )
            if unit_key != previous_unit_key:
                current_state = parameter_state_sketch(bundle.model, target_scope="full_model", seed=monitor_seed)
                if current_state["state_sketch_sha256"] != state_monitor["state_sketch_sha256"]:
                    raise ValueError("F1-B live model parameter-state monitor drifted before the next response")
                previous_unit_key = unit_key
            entry = fingerprint_by_id[str(request["prompt_id"])]
            if entry["metadata"]["prompt_sha256"] != request["prompt_sha256"]:
                raise ValueError("F1-B prompt SHA256 changed before generation")
            generated, attempts = generate_with_same_seed(
                bundle,
                {**request, "_prompt": entry["prompt"]},
                context["h8_config"],
                attempt_path,
            )
            record = {
                "schema_version": F1B_SCHEMA_VERSION,
                **request,
                "generation_seed": int(request["generation_seed"]),
                "attempt": len(attempts),
                "attempt_records": attempts,
                "batch_size": 1,
                "formal_final_confirmation_eligible": True,
                "eligible_for_measurement_parameter_fit": False,
                "eligible_for_score_parameter_fit": False,
                "eligible_for_detector_selection": False,
                "formal_detector_statistics_computed": False,
                "frozen_identity_sha256": context["frozen_identity_sha256"],
                "materialization_identity_sha256": identity_sha,
                "provenance_path": str(provenance_path.relative_to(ROOT)).replace("\\", "/"),
                "provenance_sha256": provenance_sha,
                "generated_token_count": int(generated["response_token_count_including_eos"]),
                "created_at_utc": now(),
                **generated,
            }
            validate_response_record(
                record,
                request,
                frozen_identity_sha256=context["frozen_identity_sha256"],
                expected_materialization_identity_sha256=identity_sha,
            )
            append_jsonl_fsync(record_path, record)
            success_event = {
                "schema_version": F1B_SCHEMA_VERSION,
                "response_id": request["response_id"],
                "schedule_position": request["schedule_position"],
                "attempt_index": len(attempts) - 1,
                "seed": int(request["generation_seed"]),
                "status": "success_record_fsynced",
                "created_at_utc": now(),
            }
            append_jsonl_fsync(attempt_path, success_event)
            if (len(existing_records) + 1) % 12 == 0:
                write_progress(context, active_partition=partition_id, status="RUNNING")
            existing_records.append(record)
        write_progress(context, active_partition=partition_id, status="PARTITION_COMPLETE")
        return 0
    finally:
        if bundle is not None:
            bundle.close()
            del bundle
        cleanup_cuda()


def validate_partition_complete(context: Mapping[str, Any], partition_id: str) -> int:
    partitions = partition_requests(context["generation"]["requests"])
    materialization_path = partition_path(context["auth"], partition_id, "materialization")
    if not materialization_path.is_file():
        return 0
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    identity_sha = materialization.get("materialization_identity_sha256")
    records = load_existing_prefix(
        partition_path(context["auth"], partition_id, "records"),
        partitions[partition_id],
        frozen_identity_sha256=context["frozen_identity_sha256"],
        materialization_identity_sha256=identity_sha,
    )
    return len(records)


def build_sha_index(root: Path, *, exclude: set[Path]) -> dict[str, Any]:
    files = {
        str(path.relative_to(root)).replace("\\", "/"): {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path not in exclude
    }
    return {
        "schema_version": F1B_SCHEMA_VERSION,
        "file_count": len(files),
        "files": files,
    }


def final_audit(context: Mapping[str, Any], cleanup_events: Sequence[Mapping[str, Any]]) -> Path:
    partitions = partition_requests(context["generation"]["requests"])
    all_records: list[dict[str, Any]] = []
    materialization_identities: dict[str, str] = {}
    for partition_id in partitions:
        materialization_path = partition_path(context["auth"], partition_id, "materialization")
        if not materialization_path.is_file():
            raise ValueError("F1-B final audit lacks a partition materialization identity")
        materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
        identity_sha = str(materialization["materialization_identity_sha256"])
        materialization_identities[partition_id] = identity_sha
        records = load_existing_prefix(
            partition_path(context["auth"], partition_id, "records"),
            partitions[partition_id],
            frozen_identity_sha256=context["frozen_identity_sha256"],
            materialization_identity_sha256=identity_sha,
        )
        if len(records) != len(partitions[partition_id]):
            raise ValueError("F1-B final audit found an incomplete partition")
        all_records.extend(records)
    all_records.sort(key=lambda row: int(row["schedule_position"]))
    audit = audit_formal_records(
        all_records,
        context["generation"]["requests"],
        frozen_identity_sha256=context["frozen_identity_sha256"],
        endpoint_materialization_identities=materialization_identities,
    )
    final_cleanup = gpu_compute_processes()
    if final_cleanup["status"] != "PASS":
        raise RuntimeError("F1-B final GPU cleanup gate failed")
    root = resolve(context["auth"]["artifacts"]["output_root"])
    report_path = root / context["auth"]["artifacts"]["final_report"]
    report = {
        "schema_version": F1B_SCHEMA_VERSION,
        "phase": "H8_F1B_FORMAL_FINAL_CONFIRMATION_SAMPLING",
        "status": "PASS",
        "formal_sampling_authorized": True,
        "sampling_code_commit": context["auth"]["git"]["sampling_implementation_commit"],
        "execution_commit": verify_git_state(context["auth"], context["auth_path"]),
        "frozen_identity": context["frozen_identity"],
        "frozen_identity_sha256": context["frozen_identity_sha256"],
        "frozen_input_audit": context["frozen_audit"],
        "response_integrity_audit": audit,
        "materialization_identity_count": len(materialization_identities) - 1,
        "worker_cleanup_events": list(cleanup_events),
        "gpu_cleanup_after_final_audit": final_cleanup,
        "formal_reference_responses": 720,
        "formal_heldout_intact_responses": 7200,
        "formal_heldout_attack_responses": 4800,
        "formal_final_confirmation_responses": 12720,
        "shared_reference_bank_count": 1,
        "formal_detector_statistics_computed": False,
        "final_features_computed": False,
        "final_mmd_computed": False,
        "final_scores_computed": False,
        "final_global_permutation_computed": False,
        "final_fpr_computed": False,
        "final_attack_detection_rate_computed": False,
        "created_at_utc": now(),
        "next_gate": "STOP_AND_WAIT_FOR_EXPLICIT_F1C_FINAL_PERFORMANCE_CONFIRMATION_APPROVAL",
    }
    atomic_json(report_path, report)
    index_path = root / context["auth"]["artifacts"]["sha256_index"]
    index = build_sha_index(root, exclude={index_path})
    index.update(
        {
            "status": "PASS",
            "formal_response_count": 12720,
            "report": {"path": report_path.name, "sha256": file_sha256(report_path)},
            "formal_detector_statistics_computed": False,
            "created_at_utc": now(),
        }
    )
    atomic_json(index_path, index)
    write_progress(context, active_partition=None, status="PASS_COMPLETE_STOPPED_BEFORE_F1C")
    return report_path


def preflight_only(auth_path: Path, *, allow_unfrozen_implementation: bool = False) -> int:
    context = load_context(auth_path, require_frozen_implementation=not allow_unfrozen_implementation)
    commit = None
    if not allow_unfrozen_implementation:
        commit = verify_git_state(context["auth"], auth_path)
        if os.environ.get("HF_HUB_CACHE") != context["auth"]["runtime"]["hf_hub_cache"]:
            raise PermissionError("F1-B preflight HF_HUB_CACHE differs from the frozen data-disk path")
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise PermissionError("F1-B preflight requires strict offline mode")
        verify_runtime(context["runtime_auth"])
        verify_snapshot(context["runtime_auth"], context["h8_config"])
        if gpu_compute_processes()["status"] != "PASS":
            raise RuntimeError("F1-B preflight requires clean GPUs")
    output_root = resolve(context["auth"]["artifacts"]["output_root"])
    existing_count = sum(
        len(read_jsonl_strict(path)) for path in (output_root / context["auth"]["artifacts"]["record_directory"]).glob("*.jsonl")
    ) if output_root.is_dir() else 0
    if existing_count != 0:
        raise ValueError("F1-B preflight expected zero formal responses before first launch")
    report = {
        "schema_version": F1B_SCHEMA_VERSION,
        "status": "PASS_READY_FOR_12720_FORMAL_RESPONSES" if not allow_unfrozen_implementation else "PASS_DEVELOPMENT_DRY_RUN_ONLY",
        "execution_commit": commit,
        "formal_sampling_authorized": context["auth"]["formal_final_confirmation_sampling_authorized"],
        "existing_formal_response_count": existing_count,
        "planned_formal_response_count": 12720,
        "planned_role_counts": context["frozen_audit"]["role_counts"],
        "partition_count": context["frozen_audit"]["partition_count"],
        "single_shared_reference_bank": True,
        "hf_hub_cache": context["auth"]["runtime"]["hf_hub_cache"],
        "formal_detector_statistics_computed": False,
        "frozen_identity_sha256": context["frozen_identity_sha256"],
        "created_at_utc": now(),
    }
    if not allow_unfrozen_implementation:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(output_root / context["auth"]["artifacts"]["preflight_report"], report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def parent(auth_path: Path) -> int:
    context = load_context(auth_path)
    verify_git_state(context["auth"], auth_path)
    if os.environ.get("HF_HUB_CACHE") != context["auth"]["runtime"]["hf_hub_cache"]:
        raise PermissionError("F1-B parent HF_HUB_CACHE differs from the frozen data-disk path")
    partitions = partition_requests(context["generation"]["requests"])
    endpoints = {row["attack_instance_id"]: row for row in context["attack"]["endpoints"]}
    partition_order = [BASE_PARTITION_ID, *[row["attack_instance_id"] for row in context["attack"]["endpoints"]]]
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_CACHE": context["auth"]["runtime"]["hf_hub_cache"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": context["auth"]["runtime"]["cuda_visible_devices"],
        }
    )
    cleanup_events: list[dict[str, Any]] = []
    write_progress(context, active_partition=None, status="RUNNING_PREFLIGHT_COMPLETE")
    for partition_id in partition_order:
        completed_count = validate_partition_complete(context, partition_id)
        if completed_count == len(partitions[partition_id]):
            cleanup_events.append({"partition_id": partition_id, "resume_reused_validated_complete_partition": True})
            continue
        endpoint = endpoints.get(partition_id)
        if endpoint is not None and endpoint["family"] == "lora":
            adapter_path = resolve(context["auth"]["artifacts"]["adapter_root"]) / partition_id
            if not (adapter_path / "training_report.json").is_file():
                trained = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--auth", str(auth_path), "--train-lora-endpoint", partition_id],
                    cwd=ROOT,
                    env=env,
                    check=False,
                )
                cleanup = gpu_compute_processes()
                cleanup_events.append({"partition_id": partition_id, "stage": "lora_training", "exit_code": trained.returncode, "gpu_cleanup": cleanup})
                if trained.returncode != 0 or cleanup["status"] != "PASS":
                    write_progress(context, active_partition=partition_id, status="FAIL_LORA_TRAINING")
                    raise RuntimeError(f"F1-B LoRA training failed or leaked GPU workers: {partition_id}")
        write_progress(context, active_partition=partition_id, status="RUNNING_PARTITION")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--auth", str(auth_path), "--worker-partition", partition_id],
            cwd=ROOT,
            env=env,
            check=False,
        )
        cleanup = gpu_compute_processes()
        cleanup_events.append({"partition_id": partition_id, "stage": "generation", "exit_code": completed.returncode, "gpu_cleanup": cleanup})
        if completed.returncode != 0 or cleanup["status"] != "PASS":
            write_progress(context, active_partition=partition_id, status="FAIL_PARTITION")
            raise RuntimeError(f"F1-B partition failed or leaked GPU workers: {partition_id}")
        if validate_partition_complete(context, partition_id) != len(partitions[partition_id]):
            raise RuntimeError("F1-B worker exited without completing its frozen partition")
    report_path = final_audit(context, cleanup_events)
    print(json.dumps({"status": "PASS", "formal_responses": 12720, "formal_detector_statistics_computed": False, "report": str(report_path)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 F1-B formal final-confirmation sampling runner")
    parser.add_argument("--auth", type=Path, default=DEFAULT_AUTH)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--development-dry-run-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-partition", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--train-lora-endpoint", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auth_path = args.auth.resolve()
    if args.development_dry_run_only:
        return preflight_only(auth_path, allow_unfrozen_implementation=True)
    if args.preflight_only:
        return preflight_only(auth_path)
    if args.train_lora_endpoint:
        return train_lora_worker(auth_path, args.train_lora_endpoint)
    if args.worker_partition:
        return worker(auth_path, args.worker_partition)
    return parent(auth_path)


if __name__ == "__main__":
    raise SystemExit(main())
