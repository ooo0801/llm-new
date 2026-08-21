#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d2b0 import parameter_state_sketch
from llm_integrity.h8_f1a import (
    F1A_SCHEMA_VERSION,
    SMOKE_TOTAL,
    file_sha256,
    load_canonical_envelope,
    validate_smoke_records,
)
from llm_integrity.h8_precalibration import canonical_sha256
from llm_integrity.h8_sampling import (
    canonical_json_sha256,
    exercise_retry_contract,
    gpu_compute_processes,
    run_with_same_seed_retry,
    runtime_provenance,
    sha256_text,
)
from llm_integrity.modeling import model_metadata
from llm_integrity.paper_finetuning import train_lora_manifest_variant
from llm_integrity.paper_in_memory_attacks import apply_manifest_in_memory
from llm_integrity.paper_quantization import load_quantized_manifest_variant
from run_h8_m0f_calibration_sampling import verify_runtime, verify_snapshot
from run_h8_mmd_smoke import generate_one


DEFAULT_CONFIG = ROOT / "configs/h8_f1a_final_confirmation_preflight.yaml"
REPORT_NAME = "H8_F1A_TINY_SMOKE_REPORT.json"


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


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"Frozen {label} SHA256 mismatch")


def load_context(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != "H8_F1A_FRESH_HELDOUT_FINAL_CONFIRMATION_PROTOCOL_FREEZE_PREFLIGHT":
        raise ValueError("Invalid F1-A phase")
    if config.get("formal_final_confirmation_sampling_authorized") is not False:
        raise PermissionError("F1-A formal 12,720-response sampling is not authorized")
    if config.get("tiny_smoke_sampling_authorized") is not True:
        raise PermissionError("F1-A six-response smoke is not authorized")
    if int(config["tiny_smoke"]["exact_response_count"]) != SMOKE_TOTAL:
        raise ValueError("F1-A smoke count changed")
    if not all(value is True for value in config["forbidden_operations"].values()):
        raise PermissionError("Every F1-A forbidden operation must remain enabled")
    frozen = config["frozen_inputs"]
    for path_key, hash_key, label in (
        ("h8_generation_config", "h8_generation_config_sha256", "H8 generation config"),
        ("runtime_authorization", "runtime_authorization_sha256", "runtime authorization"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12"),
        ("final_lora_training_data", "final_lora_training_data_sha256", "F1 LoRA training data"),
        ("protocol", "protocol_sha256", "F1-A protocol"),
    ):
        require_hash(resolve(frozen[path_key]), frozen[hash_key], label)

    preflight = resolve(config["artifacts"]["preflight_directory"])
    index_path = preflight / config["artifacts"]["sha256_index"]
    report_path = preflight / config["artifacts"]["report"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_READY_FOR_EXACTLY_SIX_SMOKE_ONLY":
        raise ValueError("F1-A preflight is not ready for six smoke responses")
    if report.get("formal_sampling_authorized") is not False or report.get("formal_final_confirmation_responses") != 0:
        raise PermissionError("F1-A preflight report crossed the formal sampling boundary")
    if index["artifacts"]["report"]["file_sha256"] != file_sha256(report_path):
        raise ValueError("F1-A preflight report/index SHA256 mismatch")
    smoke_info = index["artifacts"]["smoke_manifest"]
    smoke = load_canonical_envelope(
        preflight / config["artifacts"]["smoke_manifest"],
        expected_artifact_type="h8_f1a_tiny_smoke_manifest",
        expected_file_sha256=smoke_info["file_sha256"],
        expected_payload_sha256=smoke_info["payload_sha256"],
    )
    if smoke.get("exact_response_count") != SMOKE_TOTAL or len(smoke.get("requests", [])) != SMOKE_TOTAL:
        raise ValueError("F1-A frozen smoke manifest changed")
    h8_config = yaml.safe_load(resolve(frozen["h8_generation_config"]).read_text(encoding="utf-8"))
    fingerprint = json.loads(resolve(frozen["fingerprint"]).read_text(encoding="utf-8"))
    if len(fingerprint.get("entries", [])) != 12:
        raise ValueError("F1-A MCC12 count changed")
    return config, h8_config, fingerprint, list(smoke["requests"])


def verify_git_state(config_path: Path) -> str:
    for path in (config_path, Path(__file__).resolve(), ROOT / "src/llm_integrity/h8_f1a.py"):
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree must be clean before F1-A smoke")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def attack_manifest(endpoint: Mapping[str, Any]) -> dict[str, Any]:
    family = str(endpoint["family"])
    configuration = dict(endpoint["configuration"])
    manifest_family = {"gaussian": "gaussian_noise", "lora": "finetuning"}.get(family, family)
    if family == "pruning":
        manifest_family = str(configuration["type"])
    return {
        "variant_id": str(endpoint["attack_instance_id"]),
        "family": manifest_family,
        "seed": int(endpoint["materialization_seed"]),
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


def train_lora(config_path: Path, endpoint_id: str) -> int:
    config, h8_config, _, requests = load_context(config_path)
    endpoint = next(
        (row for row in config["tiny_smoke"]["attack_endpoints"] if row["attack_instance_id"] == endpoint_id),
        None,
    )
    frozen_request = next((row for row in requests if row.get("attack_instance_id") == endpoint_id), None)
    if endpoint is None or frozen_request is None or endpoint["family"] != "lora":
        raise ValueError("Unknown F1-A smoke LoRA endpoint")
    endpoint = {**endpoint, **next(row for row in load_smoke_endpoints(config_path) if row["attack_instance_id"] == endpoint_id)}
    adapter_root = resolve(config["artifacts"]["adapter_directory"])
    adapter_path = adapter_root / endpoint_id
    if adapter_path.exists():
        raise FileExistsError("F1-A smoke LoRA adapter path already exists")
    try:
        report = train_lora_manifest_variant(
            model_config=h8_config["model"],
            variant=attack_manifest(endpoint),
            data_path=resolve(config["frozen_inputs"]["final_lora_training_data"]),
            output_root=adapter_root,
            max_length=int(config["tiny_smoke"]["lora_training"]["max_length"]),
            batch_size=1,
            gradient_accumulation_steps=int(config["tiny_smoke"]["lora_training"]["gradient_accumulation_steps"]),
        )
        if report.completed_steps != int(endpoint["configuration"]["steps"]):
            raise RuntimeError("F1-A smoke LoRA did not complete frozen steps")
        if report.data_sha256 != config["frozen_inputs"]["final_lora_training_data_sha256"]:
            raise ValueError("F1-A smoke LoRA training-data hash mismatch")
        return 0
    finally:
        cleanup_cuda()


def load_smoke_endpoints(config_path: Path) -> list[dict[str, Any]]:
    config, _, _, _ = load_context(config_path)
    preflight = resolve(config["artifacts"]["preflight_directory"])
    index = json.loads((preflight / config["artifacts"]["sha256_index"]).read_text(encoding="utf-8"))
    info = index["artifacts"]["smoke_manifest"]
    return load_canonical_envelope(
        preflight / config["artifacts"]["smoke_manifest"],
        expected_artifact_type="h8_f1a_tiny_smoke_manifest",
        expected_file_sha256=info["file_sha256"],
        expected_payload_sha256=info["payload_sha256"],
    )["attack_endpoints"]


def materialize(
    config: Mapping[str, Any],
    h8_config: Mapping[str, Any],
    endpoint: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    if endpoint is None:
        from llm_integrity.modeling import load_model

        return load_model(h8_config["model"]), {"execution_mode": "frozen_intact_base_model"}
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
            raise RuntimeError("F1-A smoke quantization did not exactly materialize")
        expected_threshold = endpoint["configuration"].get("llm_int8_threshold")
        if expected_8bit and float(report["int8_threshold"]) != float(expected_threshold):
            bundle.close()
            raise RuntimeError("F1-A smoke INT8 threshold was not realized")
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

        adapter_path = resolve(config["artifacts"]["adapter_directory"]) / str(endpoint["attack_instance_id"])
        report_path = adapter_path / "training_report.json"
        if not report_path.is_file():
            raise FileNotFoundError("F1-A smoke LoRA adapter was not trained")
        training = json.loads(report_path.read_text(encoding="utf-8"))
        if int(training["completed_steps"]) != int(endpoint["configuration"]["steps"]):
            raise RuntimeError("F1-A smoke LoRA training report changed")
        artifacts = artifact_hashes(adapter_path)
        if "adapter_model.safetensors" not in artifacts or "adapter_config.json" not in artifacts:
            raise RuntimeError("F1-A smoke LoRA adapter payload is incomplete")
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
        raise RuntimeError("F1-A smoke in-memory attack did not change model state")
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


def worker(config_path: Path, position: int) -> int:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise PermissionError("F1-A smoke worker requires strict offline mode")
    config, h8_config, fingerprint, requests = load_context(config_path)
    commit = verify_git_state(config_path)
    if not 0 <= position < SMOKE_TOTAL:
        raise ValueError("F1-A smoke position is outside the frozen six requests")
    request = requests[position]
    endpoints = load_smoke_endpoints(config_path)
    endpoint = next((row for row in endpoints if row["attack_instance_id"] == request.get("attack_instance_id")), None)
    if request.get("attack_instance_id") is not None and endpoint is None:
        raise ValueError("F1-A smoke request endpoint is missing")
    result_root = resolve(config["artifacts"]["result_directory"])
    response_path = result_root / "responses" / f"{position:02d}_{request['response_id']}.json"
    materialization_path = result_root / "materialization" / f"{position:02d}_{request['response_id']}.json"
    if response_path.exists() or materialization_path.exists():
        raise FileExistsError("F1-A smoke success/materialization is immutable")
    runtime_auth = yaml.safe_load(resolve(config["frozen_inputs"]["runtime_authorization"]).read_text(encoding="utf-8"))
    actual_runtime = verify_runtime(runtime_auth)
    snapshot = verify_snapshot(runtime_auth, h8_config)
    bundle = None
    try:
        bundle, materialization = materialize(config, h8_config, endpoint)
        entry = fingerprint["entries"][int(request["prompt_index"])]
        if entry["prompt_id"] != request["prompt_id"] or entry["metadata"]["prompt_sha256"] != request["prompt_sha256"]:
            raise ValueError("F1-A smoke prompt identity mismatch")
        generated, attempts = run_with_same_seed_retry(
            lambda seed, attempt_index: generate_one(
                bundle, SimpleNamespace(prompt=entry["prompt"], seed=seed), h8_config
            ),
            int(request["generation_seed"]),
            max_attempts=int(h8_config["generation"]["max_attempts_same_seed"]),
        )
        tokenizer = bundle.tokenizer
        provenance = {
            "sampling_code_commit": commit,
            "f1a_config_sha256": file_sha256(config_path),
            "preflight_report_sha256": file_sha256(
                resolve(config["artifacts"]["preflight_directory"]) / config["artifacts"]["report"]
            ),
            "model": model_metadata(bundle),
            "model_snapshot": snapshot,
            "runtime_identity": actual_runtime,
            "runtime": runtime_provenance(),
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
            "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "frozen_eos_token_ids": [int(value) for value in h8_config["generation"]["eos_token_ids"]],
            "generation_config_sha256": canonical_json_sha256(h8_config["generation"]),
            "materialization_sha256": canonical_json_sha256(materialization),
        }
        record = {
            "schema_version": F1A_SCHEMA_VERSION,
            "mode": "smoke_only",
            **request,
            "generation_seed": int(request["generation_seed"]),
            "attempt_records": attempts,
            "batch_size": 1,
            "formal_final_confirmation_eligible": False,
            "detector_statistics_computed": False,
            "measurement_or_score_fit_performed": False,
            "generated_token_count": int(generated["response_token_count_including_eos"]),
            "provenance": provenance,
            "provenance_sha256": canonical_json_sha256(provenance),
            "created_at_utc": now(),
            **generated,
        }
        materialization_record = {
            "schema_version": F1A_SCHEMA_VERSION,
            "status": "PASS",
            "response_id": request["response_id"],
            "attack_instance_id": request.get("attack_instance_id"),
            "attack_family": request.get("attack_family"),
            "configuration": None if endpoint is None else endpoint["configuration"],
            "configuration_sha256": None if endpoint is None else canonical_sha256(endpoint["configuration"]),
            "materialization_seed": None if endpoint is None else endpoint["materialization_seed"],
            "materialization": materialization,
            "materialization_sha256": canonical_json_sha256(materialization),
            "created_at_utc": now(),
        }
        atomic_json(materialization_path, materialization_record)
        atomic_json(response_path, record)
        return 0
    finally:
        if bundle is not None:
            bundle.close()
            del bundle
        cleanup_cuda()


def validate_existing(path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("response_id", "generation_seed", "data_role", "prompt_id", "prompt_sha256", "attack_instance_id"):
        if record.get(key) != request.get(key):
            raise ValueError(f"Existing F1-A smoke record mismatch: {key}")
    return record


def parent(config_path: Path) -> int:
    config, _, _, requests = load_context(config_path)
    commit = verify_git_state(config_path)
    result_root = resolve(config["artifacts"]["result_directory"])
    result_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    retry_contract = exercise_retry_contract()
    cleanup_events: list[dict[str, Any]] = []
    endpoints = load_smoke_endpoints(config_path)
    for position, request in enumerate(requests):
        response_path = result_root / "responses" / f"{position:02d}_{request['response_id']}.json"
        materialization_path = result_root / "materialization" / f"{position:02d}_{request['response_id']}.json"
        if response_path.is_file() and materialization_path.is_file():
            validate_existing(response_path, request)
            cleanup_events.append({"position": position, "resume_reused_validated_success": True})
            continue
        if response_path.exists() or materialization_path.exists():
            raise FileExistsError("Partial F1-A smoke record requires review")
        if request.get("attack_family") == "lora":
            endpoint_id = str(request["attack_instance_id"])
            adapter_path = resolve(config["artifacts"]["adapter_directory"]) / endpoint_id
            if not (adapter_path / "training_report.json").is_file():
                trained = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--train-lora", endpoint_id],
                    cwd=ROOT,
                    env=env,
                    check=False,
                )
                training_cleanup = gpu_compute_processes()
                if trained.returncode != 0 or training_cleanup["status"] != "PASS":
                    raise RuntimeError("F1-A smoke LoRA training failed or leaked GPU workers")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--worker-position", str(position)],
            cwd=ROOT,
            env=env,
            check=False,
        )
        cleanup = gpu_compute_processes()
        cleanup_events.append({"position": position, "worker_exit_code": completed.returncode, "gpu_cleanup": cleanup})
        if completed.returncode != 0 or cleanup["status"] != "PASS":
            raise RuntimeError(f"F1-A smoke worker failed or leaked GPU workers at position {position}")
    records = [
        validate_existing(result_root / "responses" / f"{position:02d}_{request['response_id']}.json", request)
        for position, request in enumerate(requests)
    ]
    audit = validate_smoke_records(records, requests)
    final_cleanup = gpu_compute_processes()
    status = "PASS" if audit["status"] == "PASS" and final_cleanup["status"] == "PASS" else "FAIL"
    report = {
        "schema_version": F1A_SCHEMA_VERSION,
        "phase": "H8_F1A_TINY_SMOKE_ONLY",
        "status": status,
        "sampling_code_commit": commit,
        "formal_sampling_authorized": False,
        "formal_final_confirmation_responses": 0,
        "formal_reference_responses": 0,
        "formal_heldout_intact_responses": 0,
        "formal_heldout_attack_responses": 0,
        "tiny_smoke_responses": len(records),
        "detector_statistics_computed": False,
        "measurement_or_score_refit_performed": False,
        "sample_size_or_aggregation_changed": False,
        "same_seed_retry_contract": retry_contract,
        "smoke_audit": audit,
        "worker_cleanup_events": cleanup_events,
        "gpu_cleanup_after_all_workers": final_cleanup,
        "response_file_sha256": {
            path.name: file_sha256(path) for path in sorted((result_root / "responses").glob("*.json"))
        },
        "materialization_file_sha256": {
            path.name: file_sha256(path) for path in sorted((result_root / "materialization").glob("*.json"))
        },
        "created_at_utc": now(),
        "next_gate": "STOP_AND_WAIT_FOR_SEPARATE_12720_RESPONSE_FINAL_SAMPLING_AUTHORIZATION",
    }
    atomic_json(result_root / REPORT_NAME, report)
    print(json.dumps({"status": status, "tiny_smoke_responses": len(records), "formal_responses": 0, "report": str(result_root / REPORT_NAME)}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 F1-A exact six-response smoke runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--worker-position", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--train-lora", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config, _, _, requests = load_context(config_path)
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "planned_smoke": len(requests), "formal_responses": 0}, ensure_ascii=False))
        return 0
    if args.train_lora:
        return train_lora(config_path, args.train_lora)
    if args.worker_position is not None:
        return worker(config_path, args.worker_position)
    return parent(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
