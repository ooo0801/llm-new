from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


H8_SMOKE_MODE = "smoke_only"
H8_SMOKE_DATA_ROLE = "smoke_only_not_calibration"
H8_SMOKE_PROMPTS = 12
H8_SMOKE_RESPONSES_PER_PROMPT = 2
H8_SMOKE_RESPONSE_LIMIT = 24


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


@dataclass(frozen=True)
class SmokeRequest:
    prompt_index: int
    prompt_id: str
    prompt: str
    prompt_sha256: str
    response_index: int
    seed: int

    @property
    def response_id(self) -> str:
        return f"h8-smoke-{self.prompt_index:02d}-{self.response_index:02d}-{self.seed}"


def build_smoke_plan(
    entries: Sequence[Mapping[str, Any]],
    seed_base: int,
    *,
    mode: str,
    data_role: str,
    batch_size: int,
    formal_calibration_authorized: bool,
) -> list[SmokeRequest]:
    if mode != H8_SMOKE_MODE:
        raise PermissionError("This runner is hard-limited to smoke_only")
    if data_role != H8_SMOKE_DATA_ROLE:
        raise PermissionError("Smoke responses cannot use a Calibration/Reference/held-out data role")
    if formal_calibration_authorized:
        raise PermissionError("Formal Calibration authorization must remain false in the smoke runner")
    if batch_size != 1:
        raise ValueError("H8 Primary requires batch=1 for response-level unique seeds")
    if len(entries) != H8_SMOKE_PROMPTS:
        raise ValueError("H8 smoke requires exactly the archived MCC12 prompts")
    requests: list[SmokeRequest] = []
    seen_prompt_ids: set[str] = set()
    for prompt_index, entry in enumerate(entries):
        prompt_id = str(entry.get("prompt_id", ""))
        prompt = str(entry.get("prompt", ""))
        metadata = entry.get("metadata") or {}
        archived_hash = str(metadata.get("prompt_sha256", ""))
        if not prompt_id or prompt_id in seen_prompt_ids or not prompt:
            raise ValueError("MCC12 contains an empty or duplicate prompt")
        if sha256_text(prompt) != archived_hash:
            raise ValueError(f"Archived prompt hash mismatch for {prompt_id}")
        seen_prompt_ids.add(prompt_id)
        for response_index in range(H8_SMOKE_RESPONSES_PER_PROMPT):
            seed = int(seed_base) + prompt_index * H8_SMOKE_RESPONSES_PER_PROMPT + response_index
            requests.append(
                SmokeRequest(
                    prompt_index,
                    prompt_id,
                    prompt,
                    archived_hash,
                    response_index,
                    seed,
                )
            )
    if len(requests) != H8_SMOKE_RESPONSE_LIMIT:
        raise RuntimeError("Smoke plan violated the hard response limit")
    seeds = [request.seed for request in requests]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Every smoke response must have a globally unique seed")
    return requests


def normalize_eos_token_ids(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        result = (value,)
    else:
        result = tuple(int(item) for item in value)
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise ValueError("EOS token IDs must be a non-empty unique sequence")
    return result


def completion_stop_metadata(
    completion_token_ids: Sequence[int],
    eos_token_ids: int | Sequence[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    tokens = [int(item) for item in completion_token_ids]
    eos = set(normalize_eos_token_ids(eos_token_ids))
    eos_positions = [index for index, token in enumerate(tokens) if token in eos]
    if eos_positions:
        position = eos_positions[0]
        stop_reason = "eos"
        stop_token_id = tokens[position]
    elif len(tokens) >= max_new_tokens:
        position = None
        stop_reason = "length"
        stop_token_id = None
    else:
        position = None
        stop_reason = "generation_stopped_without_eos_or_length"
        stop_token_id = None
    return {
        "completion_token_ids": tokens,
        "completion_token_ids_sha256": canonical_json_sha256(tokens),
        "response_token_count_including_eos": len(tokens),
        "stop_reason": stop_reason,
        "stop_token_id": stop_token_id,
        "first_eos_position": position,
        "legal_first_token_eos": position == 0,
    }


def run_with_same_seed_retry(
    attempt: Callable[[int, int], Mapping[str, Any]],
    seed: int,
    max_attempts: int = 2,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    attempt_records: list[dict[str, Any]] = []
    for attempt_index in range(max_attempts):
        try:
            result = dict(attempt(seed, attempt_index))
            attempt_records.append(
                {"attempt_index": attempt_index, "seed": seed, "status": "success"}
            )
            result["attempt_count"] = attempt_index + 1
            result["retry_seed_changed"] = False
            return result, attempt_records
        except (RuntimeError, OSError) as exc:
            attempt_records.append(
                {
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "status": "technical_failure",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
            if attempt_index + 1 >= max_attempts:
                raise RuntimeError(
                    f"Technical generation failed after {max_attempts} same-seed attempts"
                ) from exc
    raise AssertionError("unreachable")


def exercise_retry_contract() -> dict[str, Any]:
    observed: list[int] = []

    def attempt(seed: int, attempt_index: int) -> Mapping[str, Any]:
        observed.append(seed)
        if attempt_index == 0:
            raise RuntimeError("synthetic retry-contract check")
        return {"raw_response": ""}

    result, records = run_with_same_seed_retry(attempt, 998877, max_attempts=2)
    if observed != [998877, 998877] or result["retry_seed_changed"]:
        raise RuntimeError("same-seed retry contract failed")
    return {"status": "PASS", "observed_seeds": observed, "attempt_records": records}


def runtime_provenance() -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import numpy

        result["numpy"] = numpy.__version__
    except ImportError:
        result["numpy"] = None
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_available": torch.cuda.is_available(),
                "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        )
    except ImportError:
        result["torch"] = None
    try:
        import transformers

        result["transformers"] = transformers.__version__
    except ImportError:
        result["transformers"] = None
    return result


def small_file_provenance(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value in paths:
        path = Path(value)
        if path.is_file():
            data = path.read_bytes()
            records.append(
                {"name": path.name, "size": len(data), "sha256": sha256_bytes(data)}
            )
    return sorted(records, key=lambda item: item["name"])


def gpu_compute_processes() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNAVAILABLE", "error": str(exc), "processes": []}
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return {
        "status": "PASS" if completed.returncode == 0 and not lines else "FAIL",
        "returncode": completed.returncode,
        "processes": lines,
        "stderr": completed.stderr.strip(),
    }
