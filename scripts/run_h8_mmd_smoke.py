#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_sampling import (
    H8_SMOKE_RESPONSE_LIMIT,
    build_smoke_plan,
    canonical_json_sha256,
    completion_stop_metadata,
    exercise_retry_contract,
    gpu_compute_processes,
    run_with_same_seed_retry,
    runtime_provenance,
    sha256_bytes,
    sha256_text,
    small_file_provenance,
)


DEFAULT_CONFIG = ROOT / "configs" / "h8_qwen32b_mmd_precalibration.yaml"
REPORT_NAME = "H8_M0_PREFLIGHT_REPORT.json"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("H8 config must be a mapping")
    if config.get("formal_calibration_authorized") is not False:
        raise PermissionError("Formal Calibration must remain unauthorized")
    if config.get("calibration", {}).get("sampling_authorized") is not False:
        raise PermissionError("Calibration sampling_authorized must remain false")
    return config


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_fingerprint(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    fingerprint_config = config["fingerprint"]
    path = resolve_repo_path(str(fingerprint_config["source"]))
    raw = path.read_bytes()
    if sha256_bytes(raw) != str(fingerprint_config["source_sha256"]).lower():
        raise ValueError("H6 MCC12 fingerprint artifact SHA256 mismatch")
    value = json.loads(raw)
    if value.get("schema_version") != fingerprint_config["expected_schema_version"]:
        raise ValueError("H6 MCC12 schema version mismatch")
    if len(value.get("entries", [])) != int(fingerprint_config["expected_prompt_count"]):
        raise ValueError("H6 MCC12 prompt count mismatch")
    if value.get("model_revision") != config["model"]["revision"]:
        raise ValueError("H6 fingerprint/model revision mismatch")
    return value, path


def cached_snapshot_provenance(model_name: str, revision: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_name": model_name,
        "requested_revision": revision,
        "resolved_snapshot": None,
        "small_files": [],
    }
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id != model_name:
                continue
            for cached_revision in repo.revisions:
                if cached_revision.commit_hash != revision:
                    continue
                snapshot_path = Path(cached_revision.snapshot_path)
                result["resolved_snapshot"] = str(snapshot_path)
                names = [
                    "config.json",
                    "generation_config.json",
                    "model.safetensors.index.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                ]
                result["small_files"] = small_file_provenance(
                    [snapshot_path / name for name in names]
                )
                result["snapshot_listing_sha256"] = canonical_json_sha256(
                    sorted(
                        {
                            "name": item.name,
                            "size": item.stat().st_size,
                        }
                        for item in snapshot_path.iterdir()
                        if item.is_file()
                    )
                )
                return result
    except Exception as exc:  # provenance failure is reported and made fatal by the worker gate
        result["resolution_error"] = f"{type(exc).__name__}: {exc}"
    return result


def generation_kwargs(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    generation = config["generation"]
    result: dict[str, Any] = {
        "max_new_tokens": int(generation["max_new_tokens"]),
        "do_sample": bool(generation["do_sample"]),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": [int(value) for value in generation["eos_token_ids"]],
    }
    if result["do_sample"]:
        result.update(
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            top_k=int(generation["top_k"]),
        )
    return result


def generate_one(bundle: Any, request: Any, config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from llm_integrity.io import set_seed
    from llm_integrity.modeling import render_prompt

    generation = config["generation"]
    rendered_prompt = render_prompt(
        bundle.tokenizer,
        request.prompt,
        generation.get("system_prompt"),
    )
    encoded_cpu = bundle.tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=int(generation["max_input_tokens"]),
        add_special_tokens=True,
    )
    input_ids = encoded_cpu["input_ids"][0].tolist()
    encoded = {key: value.to(bundle.device) for key, value in encoded_cpu.items()}
    kwargs = generation_kwargs(config, bundle.tokenizer)
    set_seed(request.seed)
    with torch.inference_mode():
        generated = bundle.model.generate(**encoded, **kwargs)
    input_length = int(encoded["input_ids"].shape[1])
    completion_ids = generated[0, input_length:].detach().cpu().tolist()
    raw_response = bundle.tokenizer.decode(completion_ids, skip_special_tokens=True)
    stop = completion_stop_metadata(
        completion_ids,
        generation["eos_token_ids"],
        int(generation["max_new_tokens"]),
    )
    return {
        "raw_response": raw_response,
        "raw_response_sha256": sha256_text(raw_response),
        "rendered_prompt": rendered_prompt,
        "rendered_prompt_sha256": sha256_text(rendered_prompt),
        "input_token_ids": input_ids,
        "input_token_ids_sha256": canonical_json_sha256(input_ids),
        "input_token_count": len(input_ids),
        **stop,
    }


def worker(config_path: Path, output_dir: Path) -> int:
    config = read_config(config_path)
    fingerprint, fingerprint_path = load_fingerprint(config)
    smoke = config["smoke"]
    plan = build_smoke_plan(
        fingerprint["entries"],
        int(smoke["seed_base"]),
        mode=str(smoke["mode"]),
        data_role=str(smoke["data_role"]),
        batch_size=int(config["generation"]["batch_size"]),
        formal_calibration_authorized=bool(config["formal_calibration_authorized"]),
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty smoke directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "smoke_responses.jsonl"
    model_config = config["model"]
    snapshot = cached_snapshot_provenance(model_config["name"], model_config["revision"])
    if snapshot.get("resolved_snapshot") is None:
        raise RuntimeError("Exact local model snapshot provenance could not be resolved")

    from llm_integrity.modeling import load_model, model_metadata

    bundle = load_model(model_config)
    try:
        if bundle.revision != model_config["revision"]:
            raise ValueError("Loaded model revision mismatch")
        tokenizer = bundle.tokenizer
        tokenizer_eos = int(tokenizer.eos_token_id)
        if tokenizer_eos not in {int(value) for value in config["generation"]["eos_token_ids"]}:
            raise ValueError("Tokenizer primary EOS is absent from the frozen EOS list")
        shared_provenance = {
            "model": model_metadata(bundle),
            "model_snapshot": snapshot,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_pad_token_id": int(tokenizer.pad_token_id),
            "tokenizer_eos_token_id": tokenizer_eos,
            "frozen_eos_token_ids": [int(value) for value in config["generation"]["eos_token_ids"]],
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
            "runtime": runtime_provenance(),
            "generation_config": config["generation"],
            "generation_config_sha256": canonical_json_sha256(config["generation"]),
            "fingerprint_path": str(fingerprint_path),
            "fingerprint_sha256": config["fingerprint"]["source_sha256"],
            "data_role": smoke["data_role"],
        }
        records: list[dict[str, Any]] = []
        with response_path.open("x", encoding="utf-8", newline="\n") as handle:
            for request in plan:
                generated, attempt_records = run_with_same_seed_retry(
                    lambda seed, attempt_index: generate_one(bundle, request, config),
                    request.seed,
                    max_attempts=int(config["generation"]["max_attempts_same_seed"]),
                )
                entry = fingerprint["entries"][request.prompt_index]
                record = {
                    "schema_version": config["protocol_version"],
                    "mode": smoke["mode"],
                    "data_role": smoke["data_role"],
                    "response_id": request.response_id,
                    "prompt_index": request.prompt_index,
                    "response_index": request.response_index,
                    "prompt_id": request.prompt_id,
                    "prompt": request.prompt,
                    "prompt_sha256": request.prompt_sha256,
                    "category": entry.get("category"),
                    "task_metadata": entry.get("metadata", {}),
                    "seed": request.seed,
                    "batch_size": 1,
                    "attempt_records": attempt_records,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provenance_sha256": canonical_json_sha256(shared_provenance),
                    **generated,
                }
                handle.write(canonical_json(record) + "\n")
                handle.flush()
                records.append(record)
        if len(records) != H8_SMOKE_RESPONSE_LIMIT:
            raise RuntimeError("Worker did not produce exactly 24 smoke records")
        if len({record["seed"] for record in records}) != H8_SMOKE_RESPONSE_LIMIT:
            raise RuntimeError("Worker produced duplicate response seeds")
        if len({record["response_id"] for record in records}) != H8_SMOKE_RESPONSE_LIMIT:
            raise RuntimeError("Worker produced duplicate response IDs")
        required = {
            "input_token_ids",
            "completion_token_ids",
            "stop_reason",
            "seed",
            "rendered_prompt_sha256",
            "attempt_records",
        }
        if any(not required.issubset(record) for record in records):
            raise RuntimeError("Smoke record provenance is incomplete")
        worker_report = {
            "status": "PASS",
            "response_count": len(records),
            "unique_seed_count": len({record["seed"] for record in records}),
            "batch_size": 1,
            "stop_reason_counts": {
                reason: sum(record["stop_reason"] == reason for record in records)
                for reason in sorted({record["stop_reason"] for record in records})
            },
            "legal_first_token_eos_count": sum(
                bool(record["legal_first_token_eos"]) for record in records
            ),
            "responses_sha256": sha256_bytes(response_path.read_bytes()),
            "shared_provenance": shared_provenance,
        }
        (output_dir / "worker_report.json").write_text(
            json.dumps(worker_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
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
    return 0


def parent(config_path: Path, output_dir: Path) -> int:
    config = read_config(config_path)
    load_fingerprint(config)
    retry_contract = exercise_retry_contract()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty smoke directory: {output_dir}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"H8 smoke worker failed with exit code {completed.returncode}")
    gpu_cleanup = gpu_compute_processes()
    worker_report = json.loads((output_dir / "worker_report.json").read_text(encoding="utf-8"))
    status = "PASS" if worker_report["status"] == "PASS" and gpu_cleanup["status"] == "PASS" else "FAIL"
    report = {
        "experiment": "H8 Qwen2.5-32B MMD pre-sampling smoke",
        "protocol_version": config["protocol_version"],
        "status": status,
        "scope": "24 smoke_only responses; zero mmd_precalibration_fit_only responses",
        "formal_calibration_started": False,
        "formal_calibration_response_count": 0,
        "worker": worker_report,
        "same_seed_retry_contract": retry_contract,
        "gpu_cleanup_after_worker_exit": gpu_cleanup,
        "config_path": str(config_path),
        "config_sha256": sha256_bytes(config_path.read_bytes()),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "git_status_at_report": subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "next_gate": "STOP_AND_WAIT_FOR_EXPLICIT_1200_CALIBRATION_APPROVAL",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "report": str(output_dir / REPORT_NAME)}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-limited H8 24-response smoke runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = read_config(config_path)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_repo_path(str(config["smoke"]["output_dir"])).resolve()
    )
    if args.worker:
        return worker(config_path, output_dir)
    return parent(config_path, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
