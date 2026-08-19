from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .h8_d1b_sampling import AUDIT_ROLE, FIT_ROLE, FORMAL_TOTAL, D1BGenerationRequest
from .h8_precalibration import canonical_sha256
from .h8_sampling import sha256_text


GENESIS_CHAIN_SHA256 = "0" * 64


def request_sha256(request: D1BGenerationRequest) -> str:
    return canonical_sha256(request.as_dict())


def seal_response_record(record: Mapping[str, Any], previous_chain_sha256: str) -> dict[str, Any]:
    if len(previous_chain_sha256) != 64:
        raise ValueError("Previous record-chain SHA256 must contain 64 hexadecimal characters")
    payload = dict(record)
    for forbidden in ("record_payload_sha256", "record_chain_sha256"):
        if forbidden in payload:
            raise ValueError(f"Unsealed response record must not contain {forbidden}")
    payload["previous_record_chain_sha256"] = previous_chain_sha256
    payload_sha = canonical_sha256(payload)
    payload["record_payload_sha256"] = payload_sha
    payload["record_chain_sha256"] = sha256_text(previous_chain_sha256 + "\0" + payload_sha)
    return payload


def _verify_record_hash(record: Mapping[str, Any], previous_chain_sha256: str) -> str:
    if record.get("previous_record_chain_sha256") != previous_chain_sha256:
        raise ValueError("Response record chain predecessor mismatch")
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"record_payload_sha256", "record_chain_sha256"}
    }
    payload_sha = canonical_sha256(payload)
    if record.get("record_payload_sha256") != payload_sha:
        raise ValueError("Response record payload hash mismatch")
    chain_sha = sha256_text(previous_chain_sha256 + "\0" + payload_sha)
    if record.get("record_chain_sha256") != chain_sha:
        raise ValueError("Response record chain hash mismatch")
    return chain_sha


def verify_completed_prefix(
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[D1BGenerationRequest],
    *,
    formal_manifest_sha256: str,
    frozen_provenance_sha256: str | None = None,
) -> str:
    if len(records) > len(schedule):
        raise ValueError("Existing response file is longer than the frozen formal schedule")
    response_ids: set[str] = set()
    seeds: set[int] = set()
    chain_sha = GENESIS_CHAIN_SHA256
    for index, record in enumerate(records):
        request = schedule[index]
        expected = {
            "response_id": request.response_id,
            "schedule_position": index,
            "round_id": request.round_id,
            "prompt_position_in_round": request.prompt_position_in_round,
            "prompt_index": request.prompt_index,
            "prompt_id": request.prompt_id,
            "prompt_sha256": request.prompt_sha256,
            "seed": request.seed,
            "seed_digest_sha256": request.seed_digest_sha256,
            "data_role": request.data_role,
            "manifest_request_sha256": request_sha256(request),
            "formal_manifest_file_sha256": formal_manifest_sha256,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"Existing response prefix mismatch at {index}: {key}")
        if request.data_role not in {FIT_ROLE, AUDIT_ROLE}:
            raise ValueError("Formal response has an invalid data role")
        if bool(record.get("eligible_for_score_calibration_fit")) != (request.data_role == FIT_ROLE):
            raise ValueError("Score-calibration fit eligibility does not match the frozen role")
        if bool(record.get("eligible_for_score_stability_audit")) != (request.data_role == AUDIT_ROLE):
            raise ValueError("Stability-audit eligibility does not match the frozen role")
        for key in (
            "eligible_for_formal_reference",
            "eligible_for_heldout_evaluation",
            "eligible_for_target_evaluation",
            "eligible_for_attack_evaluation",
        ):
            if record.get(key) is not False:
                raise ValueError(f"Forbidden eligibility flag is not false: {key}")
        if frozen_provenance_sha256 is not None and record.get("frozen_provenance_sha256") != frozen_provenance_sha256:
            raise ValueError("Frozen provenance hash changed within the formal response bank")
        chain_sha = _verify_record_hash(record, chain_sha)
        response_id = str(record["response_id"])
        seed = int(record["seed"])
        if response_id in response_ids or seed in seeds:
            raise ValueError("Formal response prefix contains duplicate response ID or seed")
        response_ids.add(response_id)
        seeds.add(seed)
    return chain_sha


def validate_attempt_events(
    events: Sequence[Mapping[str, Any]],
    schedule: Sequence[D1BGenerationRequest],
    completed_response_count: int,
    *,
    max_attempts: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= completed_response_count <= len(schedule):
        raise ValueError("Completed response count is outside the frozen schedule")
    by_response: dict[str, list[dict[str, Any]]] = defaultdict(list)
    request_by_id = {request.response_id: request for request in schedule}
    for raw in events:
        event = dict(raw)
        response_id = str(event.get("response_id", ""))
        if response_id not in request_by_id:
            raise ValueError("Attempt event references an unknown response ID")
        request = request_by_id[response_id]
        if int(event.get("schedule_position", -1)) != request.schedule_position:
            raise ValueError("Attempt event schedule position mismatch")
        if int(event.get("seed", -1)) != request.seed:
            raise ValueError("Attempt event changed the frozen generation seed")
        if request.schedule_position > completed_response_count:
            raise ValueError("Attempt events exist beyond the next unfinished response")
        by_response[response_id].append(event)
    for request in schedule:
        rows = by_response.get(request.response_id, [])
        if [int(row.get("attempt_index", -1)) for row in rows] != list(range(len(rows))):
            raise ValueError("Attempt indices are not contiguous from zero")
        if len(rows) > max_attempts:
            raise ValueError("Same-seed retry budget was exceeded")
        statuses = [row.get("status") for row in rows]
        if any(status not in {"technical_failure", "success"} for status in statuses):
            raise ValueError("Attempt event has an invalid status")
        if request.schedule_position < completed_response_count:
            if statuses.count("success") != 1 or not statuses or statuses[-1] != "success":
                raise ValueError("Every persisted response must have exactly one terminal success event")
        elif request.schedule_position == completed_response_count:
            if "success" in statuses:
                raise ValueError("Orphan success event exists without a persisted response record")
        elif rows:
            raise ValueError("Future schedule positions must not have attempt events")
    return by_response


def _quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def audit_formal_records(
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[D1BGenerationRequest],
    events: Sequence[Mapping[str, Any]],
    *,
    formal_manifest_sha256: str,
    smoke_seeds: Iterable[int],
    frozen_provenance_sha256: str,
    required_fields: Iterable[str],
    max_attempts: int,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        final_chain = verify_completed_prefix(
            records,
            schedule,
            formal_manifest_sha256=formal_manifest_sha256,
            frozen_provenance_sha256=frozen_provenance_sha256,
        )
    except (TypeError, ValueError) as exc:
        failures.append(str(exc))
        final_chain = None
    try:
        validate_attempt_events(events, schedule, len(records), max_attempts=max_attempts)
    except (TypeError, ValueError) as exc:
        failures.append(str(exc))
    required = set(required_fields)
    missing = sum(len(required - set(record)) for record in records)
    if missing:
        failures.append(f"missing_required_fields={missing}")
    roles = Counter(str(record.get("data_role")) for record in records)
    per_prompt_role = Counter((str(record.get("prompt_id")), str(record.get("data_role"))) for record in records)
    if len(records) != FORMAL_TOTAL:
        failures.append(f"response_count={len(records)}")
    if roles != Counter({FIT_ROLE: 1200, AUDIT_ROLE: 1200}):
        failures.append("formal role totals are not 1200/1200")
    prompt_ids = sorted({request.prompt_id for request in schedule})
    for prompt_id in prompt_ids:
        if per_prompt_role[(prompt_id, FIT_ROLE)] != 100 or per_prompt_role[(prompt_id, AUDIT_ROLE)] != 100:
            failures.append(f"per-prompt role count mismatch: {prompt_id}")
    seeds = [int(record.get("seed", -1)) for record in records]
    response_ids = [str(record.get("response_id", "")) for record in records]
    smoke_overlap = len(set(seeds) & {int(seed) for seed in smoke_seeds})
    if smoke_overlap:
        failures.append(f"formal_smoke_seed_overlap={smoke_overlap}")
    lengths = [int(record.get("response_token_count_including_eos", 0)) for record in records]
    technical = [event for event in events if event.get("status") == "technical_failure"]
    retry_seed_changed = sum(bool(record.get("retry_seed_changed")) for record in records)
    if retry_seed_changed:
        failures.append(f"retry_seed_changed={retry_seed_changed}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "total_responses": len(records),
        "fit_responses": roles[FIT_ROLE],
        "audit_responses": roles[AUDIT_ROLE],
        "per_prompt_role_counts": {
            prompt_id: {
                FIT_ROLE: per_prompt_role[(prompt_id, FIT_ROLE)],
                AUDIT_ROLE: per_prompt_role[(prompt_id, AUDIT_ROLE)],
            }
            for prompt_id in prompt_ids
        },
        "unique_response_id_count": len(set(response_ids)),
        "unique_seed_count": len(set(seeds)),
        "fit_audit_seed_overlap_count": len(
            {int(record["seed"]) for record in records if record.get("data_role") == FIT_ROLE}
            & {int(record["seed"]) for record in records if record.get("data_role") == AUDIT_ROLE}
        ),
        "formal_smoke_seed_overlap_count": smoke_overlap,
        "missing_required_field_count": missing,
        "attempt_event_count": len(events),
        "technical_failure_count": len(technical),
        "technical_failure_types": dict(sorted(Counter(str(row.get("exception_type", "")) for row in technical).items())),
        "retry_seed_changed_count": retry_seed_changed,
        "stop_reason_counts": dict(sorted(Counter(str(record.get("stop_reason")) for record in records).items())),
        "legal_first_token_eos_count": sum(bool(record.get("legal_first_token_eos")) for record in records),
        "empty_response_count": sum(str(record.get("raw_response", "")) == "" for record in records),
        "token_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": (sum(lengths) / len(lengths)) if lengths else 0.0,
            "median": float(median(lengths)) if lengths else 0.0,
            "q05": _quantile(lengths, 0.05),
            "q95": _quantile(lengths, 0.95),
        },
        "final_record_chain_sha256": final_chain,
    }
