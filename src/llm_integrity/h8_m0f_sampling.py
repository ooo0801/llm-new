from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .h8_precalibration import H8_SCHEMA_VERSION, canonical_sha256
from .h8_sampling import canonical_json_sha256, sha256_bytes, sha256_text


M0F_MANIFEST_ARTIFACT_TYPE = "h8_m0f_calibration_seed_manifest"
M0F_MANIFEST_SCHEMA = "h8-m0f-calibration-seed-manifest-1.0"
M0F_DATA_ROLE = "mmd_precalibration_fit_only"
M0F_TOTAL_RESPONSES = 1200
M0F_PROMPT_COUNT = 12
M0F_REPLICATES = 100
UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class CalibrationRequest:
    schedule_position: int
    replicate_id: int
    prompt_position_in_round: int
    prompt_index: int
    prompt_id: str
    prompt: str
    prompt_sha256: str
    seed: int
    derivation_counter: int
    seed_digest_sha256: str

    @property
    def response_id(self) -> str:
        return f"h8-m0f-r{self.replicate_id:03d}-p{self.prompt_index:02d}-s{self.seed:010d}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_position": self.schedule_position,
            "replicate_id": self.replicate_id,
            "prompt_position_in_round": self.prompt_position_in_round,
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
            "seed": self.seed,
            "derivation_counter": self.derivation_counter,
            "seed_digest_sha256": self.seed_digest_sha256,
            "response_id": self.response_id,
            "data_role": M0F_DATA_ROLE,
            "eligible_for_formal_reference": False,
            "eligible_for_heldout_evaluation": False,
            "eligible_for_attack_evaluation": False,
        }


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def verify_file_hash(path: str | Path, expected_sha256: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != str(expected_sha256).lower():
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")


def _domain_digest(domain: str, root_seed: int, *parts: Any) -> bytes:
    material = "\x00".join([domain, str(root_seed), *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).digest()


def _order_for_round(
    entries: Sequence[Mapping[str, Any]], root_seed: int, replicate_id: int
) -> list[int]:
    return sorted(
        range(len(entries)),
        key=lambda index: _domain_digest(
            "h8-m0f-order-v1", root_seed, replicate_id, entries[index]["prompt_id"]
        ),
    )


def build_calibration_schedule(
    entries: Sequence[Mapping[str, Any]],
    root_seed: int,
    forbidden_seeds: Iterable[int] = (),
) -> list[CalibrationRequest]:
    if len(entries) != M0F_PROMPT_COUNT:
        raise ValueError("M0-F requires exactly the archived MCC12 prompts")
    if not 0 <= int(root_seed) <= UINT32_MAX:
        raise ValueError("root_seed must fit uint32")
    seen_prompt_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        prompt_id = str(entry.get("prompt_id", ""))
        prompt = str(entry.get("prompt", ""))
        archived_hash = str((entry.get("metadata") or {}).get("prompt_sha256", ""))
        if not prompt_id or prompt_id in seen_prompt_ids or not prompt:
            raise ValueError("MCC12 contains an empty or duplicate prompt")
        if sha256_text(prompt) != archived_hash:
            raise ValueError(f"Archived prompt hash mismatch for {prompt_id}")
        seen_prompt_ids.add(prompt_id)
        normalized.append({"prompt_id": prompt_id, "prompt": prompt, "prompt_sha256": archived_hash})

    used = {int(seed) for seed in forbidden_seeds if 0 <= int(seed) <= UINT32_MAX}
    schedule: list[CalibrationRequest] = []
    for replicate_id in range(M0F_REPLICATES):
        order = _order_for_round(normalized, int(root_seed), replicate_id)
        for prompt_position, prompt_index in enumerate(order):
            entry = normalized[prompt_index]
            counter = 0
            while True:
                digest = _domain_digest(
                    "h8-m0f-response-seed-v1",
                    int(root_seed),
                    replicate_id,
                    entry["prompt_id"],
                    counter,
                )
                seed = int.from_bytes(digest[:4], "little", signed=False)
                if seed not in used:
                    break
                counter += 1
            used.add(seed)
            schedule.append(
                CalibrationRequest(
                    schedule_position=len(schedule),
                    replicate_id=replicate_id,
                    prompt_position_in_round=prompt_position,
                    prompt_index=prompt_index,
                    prompt_id=entry["prompt_id"],
                    prompt=entry["prompt"],
                    prompt_sha256=entry["prompt_sha256"],
                    seed=seed,
                    derivation_counter=counter,
                    seed_digest_sha256=digest.hex(),
                )
            )
    validate_calibration_schedule(schedule)
    return schedule


def validate_calibration_schedule(schedule: Sequence[CalibrationRequest]) -> None:
    if len(schedule) != M0F_TOTAL_RESPONSES:
        raise ValueError("M0-F schedule must contain exactly 1200 requests")
    positions = [request.schedule_position for request in schedule]
    if positions != list(range(M0F_TOTAL_RESPONSES)):
        raise ValueError("M0-F schedule positions must be contiguous and ordered")
    seeds = [request.seed for request in schedule]
    if len(set(seeds)) != M0F_TOTAL_RESPONSES:
        raise ValueError("M0-F schedule seeds are not unique")
    if any(seed < 0 or seed > UINT32_MAX for seed in seeds):
        raise ValueError("M0-F schedule contains a seed outside uint32")
    response_ids = [request.response_id for request in schedule]
    if len(set(response_ids)) != M0F_TOTAL_RESPONSES:
        raise ValueError("M0-F response IDs are not unique")
    prompt_counts = Counter(request.prompt_id for request in schedule)
    if len(prompt_counts) != M0F_PROMPT_COUNT or set(prompt_counts.values()) != {M0F_REPLICATES}:
        raise ValueError("Each MCC12 prompt must occur exactly 100 times")
    for replicate_id in range(M0F_REPLICATES):
        rows = [request for request in schedule if request.replicate_id == replicate_id]
        if len(rows) != M0F_PROMPT_COUNT or len({row.prompt_id for row in rows}) != M0F_PROMPT_COUNT:
            raise ValueError("Every replicate round must cover all MCC12 prompts exactly once")
        if [row.prompt_position_in_round for row in rows] != list(range(M0F_PROMPT_COUNT)):
            raise ValueError("Prompt positions within a replicate round are invalid")


def _extract_seed_values(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[int, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _extract_seed_values(item, (*path, str(key)))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _extract_seed_values(item, (*path, str(index)))
        return
    if isinstance(value, bool) or not isinstance(value, int):
        return
    if any("seed" in part.lower() for part in path):
        yield int(value), ".".join(path)


def discover_repository_seeds(root: str | Path) -> tuple[set[int], list[dict[str, Any]]]:
    root_path = Path(root).resolve()
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root_path,
        capture_output=True,
        check=True,
    )
    paths = [item for item in completed.stdout.decode("utf-8").split("\x00") if item]
    seeds: set[int] = set()
    sources: list[dict[str, Any]] = []
    for relative in paths:
        path = root_path / relative
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl", ".yaml", ".yml"}:
            continue
        try:
            if suffix == ".jsonl":
                values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            elif suffix in {".yaml", ".yml"}:
                values = [yaml.safe_load(path.read_text(encoding="utf-8"))]
            else:
                values = [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
            continue
        found: list[tuple[int, str]] = []
        for value in values:
            found.extend(_extract_seed_values(value))
        if found:
            for seed, _ in found:
                if 0 <= seed <= UINT32_MAX:
                    seeds.add(seed)
            sources.append(
                {
                    "path": relative.replace("\\", "/"),
                    "valid_uint32_seed_values": len({seed for seed, _ in found if 0 <= seed <= UINT32_MAX}),
                    "content_sha256": file_sha256(path),
                }
            )
    return seeds, sorted(sources, key=lambda item: item["path"])


def make_manifest_payload(
    schedule: Sequence[CalibrationRequest],
    root_seed: int,
    source_fingerprint_sha256: str,
    frozen_identity: Mapping[str, Any],
    repository_seed_sources: Sequence[Mapping[str, Any]],
    forbidden_seed_count: int,
) -> dict[str, Any]:
    validate_calibration_schedule(schedule)
    return {
        "manifest_schema": M0F_MANIFEST_SCHEMA,
        "h8_schema_version": H8_SCHEMA_VERSION,
        "data_role": M0F_DATA_ROLE,
        "eligible_for_formal_reference": False,
        "eligible_for_heldout_evaluation": False,
        "eligible_for_attack_evaluation": False,
        "root_seed": int(root_seed),
        "seed_algorithm": "sha256_domain_separated_uint32_rejection_v1",
        "seed_byte_order": "little_endian_first_4_digest_bytes",
        "collision_resolution": "increment_derivation_counter_until_not_forbidden_or_used",
        "order_algorithm": "sha256_per_replicate_prompt_sort_v1",
        "schedule_organization": "100_replicate_rounds_each_covering_MCC12_once",
        "prompt_count": M0F_PROMPT_COUNT,
        "replicate_rounds": M0F_REPLICATES,
        "total_responses": M0F_TOTAL_RESPONSES,
        "batch_size": 1,
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "frozen_identity": dict(frozen_identity),
        "repository_cross_check": {
            "forbidden_seed_count": forbidden_seed_count,
            "source_file_count": len(repository_seed_sources),
            "source_files": list(repository_seed_sources),
            "collision_count_after_resolution": sum(request.derivation_counter > 0 for request in schedule),
            "final_overlap_count": 0,
        },
        "schedule": [request.as_dict() for request in schedule],
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_dict = dict(payload)
    payload_sha = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": M0F_MANIFEST_ARTIFACT_TYPE,
        "schema_version": H8_SCHEMA_VERSION,
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    raw_sha = file_sha256(target)
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{raw_sha}  {target.name}\n", encoding="utf-8", newline="\n")
    return {"payload_sha256": payload_sha, "file_sha256": raw_sha, "path": str(target)}


def load_manifest(path: str | Path) -> tuple[dict[str, Any], list[CalibrationRequest]]:
    target = Path(path)
    envelope = json.loads(target.read_text(encoding="utf-8"))
    if envelope.get("artifact_type") != M0F_MANIFEST_ARTIFACT_TYPE:
        raise ValueError("M0-F manifest artifact type mismatch")
    if envelope.get("schema_version") != H8_SCHEMA_VERSION:
        raise ValueError("M0-F manifest H8 schema mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or canonical_sha256(payload) != envelope.get("payload_sha256"):
        raise ValueError("M0-F manifest payload hash mismatch")
    if payload.get("manifest_schema") != M0F_MANIFEST_SCHEMA:
        raise ValueError("M0-F manifest schema mismatch")
    schedule = [
        CalibrationRequest(
            schedule_position=int(row["schedule_position"]),
            replicate_id=int(row["replicate_id"]),
            prompt_position_in_round=int(row["prompt_position_in_round"]),
            prompt_index=int(row["prompt_index"]),
            prompt_id=str(row["prompt_id"]),
            prompt="",
            prompt_sha256=str(row["prompt_sha256"]),
            seed=int(row["seed"]),
            derivation_counter=int(row["derivation_counter"]),
            seed_digest_sha256=str(row["seed_digest_sha256"]),
        )
        for row in payload["schedule"]
    ]
    validate_calibration_schedule(schedule)
    return payload, schedule


def quantile_summary(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "q05": None, "median": None, "mean": None, "q95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def audit_calibration_records(
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[CalibrationRequest],
    required_fields: Sequence[str],
) -> dict[str, Any]:
    expected = {request.response_id: request for request in schedule}
    missing_field_count = 0
    schedule_mismatches: list[str] = []
    for record in records:
        missing_field_count += sum(field not in record for field in required_fields)
        response_id = str(record.get("response_id", ""))
        request = expected.get(response_id)
        if request is None:
            schedule_mismatches.append(response_id)
            continue
        if any(
            [
                int(record.get("schedule_position", -1)) != request.schedule_position,
                int(record.get("replicate_id", -1)) != request.replicate_id,
                str(record.get("prompt_id", "")) != request.prompt_id,
                int(record.get("seed", -1)) != request.seed,
                str(record.get("data_role", "")) != M0F_DATA_ROLE,
                bool(record.get("eligible_for_formal_reference", True)),
                bool(record.get("eligible_for_heldout_evaluation", True)),
                bool(record.get("eligible_for_attack_evaluation", True)),
            ]
        ):
            schedule_mismatches.append(response_id)
    prompt_counts = Counter(str(record.get("prompt_id")) for record in records)
    stop_counts = Counter(str(record.get("stop_reason")) for record in records)
    seeds = [int(record.get("seed", -1)) for record in records]
    response_ids = [str(record.get("response_id", "")) for record in records]
    valid = (
        len(records) == M0F_TOTAL_RESPONSES
        and len(set(seeds)) == M0F_TOTAL_RESPONSES
        and len(set(response_ids)) == M0F_TOTAL_RESPONSES
        and len(prompt_counts) == M0F_PROMPT_COUNT
        and set(prompt_counts.values()) == {M0F_REPLICATES}
        and missing_field_count == 0
        and not schedule_mismatches
    )
    return {
        "status": "PASS" if valid else "FAIL",
        "total_responses": len(records),
        "per_prompt_counts": dict(sorted(prompt_counts.items())),
        "unique_seed_count": len(set(seeds)),
        "unique_response_id_count": len(set(response_ids)),
        "missing_required_field_count": missing_field_count,
        "schedule_mismatch_count": len(schedule_mismatches),
        "schedule_mismatch_examples": schedule_mismatches[:10],
        "stop_reason_counts": dict(sorted(stop_counts.items())),
        "empty_response_count": sum(str(record.get("raw_response", "")) == "" for record in records),
        "legal_first_token_eos_count": sum(bool(record.get("legal_first_token_eos")) for record in records),
        "input_token_count": quantile_summary([int(record["input_token_count"]) for record in records]),
        "generated_token_count_including_eos": quantile_summary(
            [int(record["response_token_count_including_eos"]) for record in records]
        ),
        "attempt_count": quantile_summary([int(record["attempt_count"]) for record in records]),
        "responses_with_retry": sum(int(record["attempt_count"]) > 1 for record in records),
        "retry_seed_changed_count": sum(bool(record.get("retry_seed_changed", False)) for record in records),
    }
