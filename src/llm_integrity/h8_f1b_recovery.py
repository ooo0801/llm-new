from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm_integrity.h8_f1b import read_jsonl_strict
from llm_integrity.h8_precalibration import canonical_sha256


RECOVERY_SCHEMA_VERSION = "h8-f1b-tuple-list-recovery-1.0"
NORMALIZED_FIELDS = (
    "materialization.quantization_load_report.notes",
    "materialization.quantization_load_report.quantized_module_examples",
)


def json_roundtrip_normalize(value: Any) -> Any:
    """Return the JSON data-model representation without changing values."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def normalize_identity_payload(payload: Mapping[str, Any], identity_sha256: str) -> dict[str, Any]:
    """Normalize tuple/list containers while preserving the canonical identity."""

    normalized = json_roundtrip_normalize(dict(payload))
    if canonical_sha256(normalized) != identity_sha256:
        raise ValueError("JSON normalization changed the materialization identity")
    return normalized


def validate_normalized_materialization_match(
    sealed: Mapping[str, Any],
    live_payload: Mapping[str, Any],
    live_identity_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless sealed and live materializations are JSON-equivalent."""

    if sealed.get("materialization_identity_sha256") != live_identity_sha256:
        raise ValueError("Recovery materialization identity SHA256 mismatch")
    normalized_live = normalize_identity_payload(live_payload, live_identity_sha256)
    normalized_sealed = json_roundtrip_normalize(sealed.get("payload"))
    if normalized_sealed != normalized_live:
        raise ValueError("Recovery materialization payload has a semantic mismatch")
    return normalized_live


def sha256_prefix_lines(path: Path, line_count: int) -> str | None:
    """Hash the exact first N newline-terminated JSONL records."""

    if line_count == 0:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    seen = 0
    with path.open("rb") as handle:
        for line in handle:
            if seen >= line_count:
                break
            if not line.endswith(b"\n"):
                raise ValueError(f"Partial trailing JSONL line in {path}")
            digest.update(line)
            seen += 1
    if seen != line_count:
        raise ValueError(f"Expected {line_count} prefix lines in {path}, found {seen}")
    return digest.hexdigest()


def validate_attempt_alignment(
    records: Sequence[Mapping[str, Any]],
    attempt_events: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Validate same-seed technical retries and one fsynced success per response."""

    records_by_id = {str(record["response_id"]): record for record in records}
    if len(records_by_id) != len(records):
        raise ValueError("Duplicate response IDs in recovery prefix")

    successes: list[Mapping[str, Any]] = []
    technical_failures = 0
    for event in attempt_events:
        response_id = str(event.get("response_id"))
        if response_id not in records_by_id:
            raise ValueError("Attempt event has no successful response record")
        record = records_by_id[response_id]
        if int(event.get("schedule_position")) != int(record["schedule_position"]):
            raise ValueError("Attempt event schedule position mismatch")
        if int(event.get("seed")) != int(record["generation_seed"]):
            raise ValueError("Attempt event changed the frozen generation seed")
        status = event.get("status")
        if status == "success_record_fsynced":
            successes.append(event)
        elif status == "technical_failure":
            technical_failures += 1
        else:
            raise ValueError(f"Unexpected attempt event status: {status!r}")

    if len(successes) != len(records):
        raise ValueError("Recovery prefix must have one fsynced success per response")
    if [event["response_id"] for event in successes] != [
        record["response_id"] for record in records
    ]:
        raise ValueError("Successful attempt events are not aligned to record order")
    return {
        "success_attempt_event_count": len(successes),
        "technical_failure_event_count": technical_failures,
        "total_attempt_event_count": len(attempt_events),
    }


def load_recovery_authorization(path: Path) -> dict[str, Any]:
    import yaml

    authorization = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(authorization, dict):
        raise ValueError("Invalid recovery authorization")
    if authorization.get("phase") != "H8_F1B_TUPLE_LIST_NORMALIZATION_RECOVERY":
        raise ValueError("Wrong recovery authorization phase")
    if authorization.get("authorization_status") != "user_approved_bounded_recovery_only":
        raise PermissionError("Recovery has not been explicitly authorized")
    if authorization.get("formal_sampling_recovery_authorized") is not True:
        raise PermissionError("Formal recovery is not authorized")
    if authorization.get("final_performance_confirmation_authorized") is not False:
        raise PermissionError("Recovery must not authorize F1-C")
    if not all(authorization.get("forbidden_operations", {}).values()):
        raise PermissionError("Every recovery forbidden-operation gate must remain enabled")
    bounds = authorization.get("recovery_bounds", {})
    if bounds.get("allowed_schedule_positions") != [12583, 12719]:
        raise PermissionError("Recovery schedule bounds changed")
    if bounds.get("allowed_partitions") != [
        "h8f1_final_quantization_09_seed2646328052",
        "h8f1_final_quantization_10_seed0331532823",
    ]:
        raise PermissionError("Recovery partition bounds changed")
    if tuple(bounds.get("normalized_fields", ())) != NORMALIZED_FIELDS:
        raise PermissionError("Recovery normalization scope changed")
    return authorization


def read_attempt_events(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_strict(path)
