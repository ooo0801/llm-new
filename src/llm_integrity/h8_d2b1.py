from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .h8_d1b_formal import GENESIS_CHAIN_SHA256, seal_response_record
from .h8_d2a import ATTACK_ROLE, INTACT_TARGET_ROLE, REFERENCE_ROLE
from .h8_precalibration import canonical_sha256
from .h8_sampling import sha256_text


D2B1_SCHEMA_VERSION = "h8-d2b1-formal-development-sampling-1.0"
TOTAL_RESPONSES = 3840
ROLE_TOTALS = {
    REFERENCE_ROLE: 720,
    INTACT_TARGET_ROLE: 1200,
    ATTACK_ROLE: 1920,
}
GENESIS_CHAIN = GENESIS_CHAIN_SHA256


@dataclass(frozen=True)
class D2B1GenerationRequest:
    schedule_position: int
    bank_position: int
    replicate_id: int
    prompt_position_in_round: int
    prompt_index: int
    prompt_id: str
    prompt_sha256: str
    generation_seed: int
    derivation_counter: int
    seed_digest_sha256: str
    response_id: str
    data_role: str
    bank_id: str
    attack_family: str | None
    attack_instance_id: str | None
    manifest_request_sha256: str
    prompt: str = ""

    @property
    def seed(self) -> int:
        return self.generation_seed

    @property
    def evaluation_unit_id(self) -> str | None:
        if self.data_role == REFERENCE_ROLE:
            return None
        if self.data_role == INTACT_TARGET_ROLE:
            return f"development_intact_block_{self.replicate_id // 20:02d}"
        return self.attack_instance_id

    def with_prompt(self, prompt: str) -> "D2B1GenerationRequest":
        return replace(self, prompt=prompt)


def request_from_manifest_row(row: Mapping[str, Any]) -> D2B1GenerationRequest:
    payload = dict(row)
    role = str(payload.get("data_role", ""))
    if role not in ROLE_TOTALS:
        raise ValueError("D2-B1 manifest row has an invalid development role")
    seed = int(payload.get("generation_seed", -1))
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("D2-B1 generation seed is outside uint32")
    response_id = str(payload.get("response_id", ""))
    prompt_id = str(payload.get("prompt_id", ""))
    prompt_sha = str(payload.get("prompt_sha256", ""))
    if not response_id or not prompt_id or len(prompt_sha) != 64:
        raise ValueError("D2-B1 manifest row identity is incomplete")
    for key in (
        "eligible_for_future_formal_reference",
        "eligible_for_future_final_heldout",
        "eligible_for_future_final_confirmation",
        "eligible_for_score_parameter_fit",
        "eligible_for_measurement_parameter_fit",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"D2-B1 frozen manifest eligibility changed: {key}")
    family = payload.get("attack_family")
    endpoint = payload.get("attack_instance_id")
    if role == ATTACK_ROLE:
        if not family or not endpoint:
            raise ValueError("Attack request lacks family or endpoint identity")
    elif family is not None or endpoint is not None:
        raise ValueError("Intact development request carries attack identity")
    return D2B1GenerationRequest(
        schedule_position=int(payload["schedule_position"]),
        bank_position=int(payload["bank_position"]),
        replicate_id=int(payload["replicate_id"]),
        prompt_position_in_round=int(payload["prompt_position_in_round"]),
        prompt_index=int(payload["prompt_index"]),
        prompt_id=prompt_id,
        prompt_sha256=prompt_sha,
        generation_seed=seed,
        derivation_counter=int(payload["derivation_counter"]),
        seed_digest_sha256=str(payload["seed_digest_sha256"]),
        response_id=response_id,
        data_role=role,
        bank_id=str(payload["bank_id"]),
        attack_family=None if family is None else str(family),
        attack_instance_id=None if endpoint is None else str(endpoint),
        manifest_request_sha256=canonical_sha256(payload),
    )


def combine_and_validate_schedules(
    reference_rows: Sequence[Mapping[str, Any]],
    intact_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
) -> list[D2B1GenerationRequest]:
    requests = [request_from_manifest_row(row) for row in [*reference_rows, *intact_rows, *attack_rows]]
    requests.sort(key=lambda row: row.schedule_position)
    if len(requests) != TOTAL_RESPONSES:
        raise ValueError("D2-B1 frozen schedule must contain exactly 3,840 requests")
    if [row.schedule_position for row in requests] != list(range(TOTAL_RESPONSES)):
        raise ValueError("D2-B1 schedule positions must be contiguous from zero")
    if Counter(row.data_role for row in requests) != Counter(ROLE_TOTALS):
        raise ValueError("D2-B1 frozen role totals changed")
    if len({row.response_id for row in requests}) != TOTAL_RESPONSES:
        raise ValueError("D2-B1 response IDs are not globally unique")
    if len({row.generation_seed for row in requests}) != TOTAL_RESPONSES:
        raise ValueError("D2-B1 generation seeds are not globally unique")
    prompt_ids = {row.prompt_id for row in requests}
    if len(prompt_ids) != 12:
        raise ValueError("D2-B1 schedule must cover exactly twelve prompts")
    per_prompt_role = Counter((row.prompt_id, row.data_role) for row in requests)
    for prompt_id in prompt_ids:
        if per_prompt_role[(prompt_id, REFERENCE_ROLE)] != 60:
            raise ValueError("D2-B1 reference per-prompt count changed")
        if per_prompt_role[(prompt_id, INTACT_TARGET_ROLE)] != 100:
            raise ValueError("D2-B1 intact-target per-prompt count changed")
    endpoint_prompt = Counter(
        (row.attack_instance_id, row.prompt_id) for row in requests if row.data_role == ATTACK_ROLE
    )
    if len(endpoint_prompt) != 8 * 12 or set(endpoint_prompt.values()) != {20}:
        raise ValueError("D2-B1 attack endpoint/prompt counts changed")
    return requests


def hydrate_prompts(
    schedule: Sequence[D2B1GenerationRequest], entries: Sequence[Mapping[str, Any]]
) -> list[D2B1GenerationRequest]:
    if len(entries) != 12:
        raise ValueError("D2-B1 requires MCC12")
    hydrated: list[D2B1GenerationRequest] = []
    for request in schedule:
        entry = entries[request.prompt_index]
        if (
            str(entry.get("prompt_id")) != request.prompt_id
            or str((entry.get("metadata") or {}).get("prompt_sha256")) != request.prompt_sha256
        ):
            raise ValueError("D2-B1 schedule no longer matches the frozen MCC12")
        hydrated.append(request.with_prompt(str(entry["prompt"])))
    return hydrated


def _verify_record_hash(record: Mapping[str, Any], previous_chain_sha256: str) -> str:
    if record.get("previous_record_chain_sha256") != previous_chain_sha256:
        raise ValueError("D2-B1 response-chain predecessor mismatch")
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"record_payload_sha256", "record_chain_sha256"}
    }
    payload_sha = canonical_sha256(payload)
    if record.get("record_payload_sha256") != payload_sha:
        raise ValueError("D2-B1 response payload hash mismatch")
    chain = sha256_text(previous_chain_sha256 + "\0" + payload_sha)
    if record.get("record_chain_sha256") != chain:
        raise ValueError("D2-B1 response-chain hash mismatch")
    return chain


def verify_completed_prefix(
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[D2B1GenerationRequest],
    *,
    generation_manifest_set_sha256: str,
    frozen_provenance_sha256: str | None = None,
) -> str:
    if len(records) > len(schedule):
        raise ValueError("Existing D2-B1 response file exceeds the frozen schedule")
    chain = GENESIS_CHAIN
    response_ids: set[str] = set()
    seeds: set[int] = set()
    for index, record in enumerate(records):
        request = schedule[index]
        expected = {
            "response_id": request.response_id,
            "schedule_position": request.schedule_position,
            "bank_position": request.bank_position,
            "replicate_id": request.replicate_id,
            "prompt_position_in_round": request.prompt_position_in_round,
            "prompt_index": request.prompt_index,
            "prompt_id": request.prompt_id,
            "prompt_sha256": request.prompt_sha256,
            "generation_seed": request.generation_seed,
            "seed_digest_sha256": request.seed_digest_sha256,
            "data_role": request.data_role,
            "bank_id": request.bank_id,
            "attack_family": request.attack_family,
            "attack_instance_id": request.attack_instance_id,
            "evaluation_unit_id": request.evaluation_unit_id,
            "manifest_request_sha256": request.manifest_request_sha256,
            "generation_manifest_set_sha256": generation_manifest_set_sha256,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"Existing D2-B1 response prefix mismatch at {index}: {key}")
        eligibility = {
            "eligible_for_development_reference": request.data_role == REFERENCE_ROLE,
            "eligible_for_development_intact_target": request.data_role == INTACT_TARGET_ROLE,
            "eligible_for_development_attack": request.data_role == ATTACK_ROLE,
            "eligible_for_future_formal_reference": False,
            "eligible_for_future_final_heldout": False,
            "eligible_for_future_final_confirmation": False,
            "eligible_for_score_parameter_fit": False,
            "eligible_for_measurement_parameter_fit": False,
        }
        for key, value in eligibility.items():
            if record.get(key) is not value:
                raise ValueError(f"D2-B1 response eligibility mismatch: {key}")
        if frozen_provenance_sha256 is not None and record.get("frozen_provenance_sha256") != frozen_provenance_sha256:
            raise ValueError("D2-B1 frozen provenance changed within the bank")
        chain = _verify_record_hash(record, chain)
        response_id = str(record["response_id"])
        seed = int(record["generation_seed"])
        if response_id in response_ids or seed in seeds:
            raise ValueError("D2-B1 prefix contains duplicate response ID or seed")
        response_ids.add(response_id)
        seeds.add(seed)
    return chain


def validate_attempt_events(
    events: Sequence[Mapping[str, Any]],
    schedule: Sequence[D2B1GenerationRequest],
    completed_response_count: int,
    *,
    max_attempts: int,
) -> dict[str, list[dict[str, Any]]]:
    if not 0 <= completed_response_count <= len(schedule):
        raise ValueError("D2-B1 completed prefix length is invalid")
    request_by_id = {row.response_id: row for row in schedule}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        event = dict(raw)
        response_id = str(event.get("response_id", ""))
        if response_id not in request_by_id:
            raise ValueError("D2-B1 attempt event references an unknown response")
        request = request_by_id[response_id]
        if int(event.get("schedule_position", -1)) != request.schedule_position:
            raise ValueError("D2-B1 attempt schedule position changed")
        if int(event.get("generation_seed", -1)) != request.generation_seed:
            raise ValueError("D2-B1 technical retry changed the generation seed")
        if request.schedule_position > completed_response_count:
            raise ValueError("D2-B1 attempt event jumps beyond the next unfinished response")
        grouped[response_id].append(event)
    for request in schedule:
        rows = grouped.get(request.response_id, [])
        if [int(row.get("attempt_index", -1)) for row in rows] != list(range(len(rows))):
            raise ValueError("D2-B1 attempt indices are not contiguous")
        if len(rows) > max_attempts:
            raise ValueError("D2-B1 same-seed retry budget exceeded")
        statuses = [row.get("status") for row in rows]
        if any(status not in {"technical_failure", "success"} for status in statuses):
            raise ValueError("D2-B1 attempt status is invalid")
        if request.schedule_position < completed_response_count:
            if statuses.count("success") != 1 or statuses[-1:] != ["success"]:
                raise ValueError("Every persisted D2-B1 response needs one terminal success event")
        elif request.schedule_position == completed_response_count:
            if "success" in statuses:
                raise ValueError("D2-B1 orphan success event has no response record")
        elif rows:
            raise ValueError("Future D2-B1 positions have attempt events")
    return grouped


def _quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def validate_nested_membership_against_records(
    nested: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> list[str]:
    failures: list[str] = []
    record_by_id = {str(row.get("response_id", "")): row for row in records}
    for prompt_id, membership in dict(nested.get("reference_membership_by_prompt", {})).items():
        r60 = set(membership.get("R60", []))
        r40 = set(membership.get("R40", []))
        actual = {
            response_id
            for response_id, row in record_by_id.items()
            if row.get("data_role") == REFERENCE_ROLE and row.get("prompt_id") == prompt_id
        }
        if actual != r60 or not r40.issubset(r60):
            failures.append(f"reference nested membership mismatch: {prompt_id}")
    for unit in nested.get("target_evaluation_units", []):
        unit_id = str(unit.get("evaluation_unit_id", ""))
        for prompt_id, membership in dict(unit.get("members_by_prompt", {})).items():
            q20 = set(membership.get("Q20", []))
            q10 = set(membership.get("Q10", []))
            actual = {
                response_id
                for response_id, row in record_by_id.items()
                if row.get("evaluation_unit_id") == unit_id and row.get("prompt_id") == prompt_id
            }
            if actual != q20 or not q10.issubset(q20):
                failures.append(f"target nested membership mismatch: {unit_id}/{prompt_id}")
    return failures


def audit_formal_records(
    records: Sequence[Mapping[str, Any]],
    schedule: Sequence[D2B1GenerationRequest],
    events: Sequence[Mapping[str, Any]],
    *,
    generation_manifest_set_sha256: str,
    frozen_provenance_sha256: str,
    smoke_response_ids: Iterable[str],
    smoke_seeds: Iterable[int],
    nested_subset_payload: Mapping[str, Any],
    nested_subset_file_sha256: str,
    expected_materialization_by_endpoint: Mapping[str, Mapping[str, str]],
    required_fields: Iterable[str],
    max_attempts: int,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        final_chain = verify_completed_prefix(
            records,
            schedule,
            generation_manifest_set_sha256=generation_manifest_set_sha256,
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
    if len(records) != TOTAL_RESPONSES:
        failures.append(f"response_count={len(records)}")
    roles = Counter(str(record.get("data_role")) for record in records)
    if roles != Counter(ROLE_TOTALS):
        failures.append("formal development role totals changed")
    seeds = [int(record.get("generation_seed", -1)) for record in records]
    response_ids = [str(record.get("response_id", "")) for record in records]
    smoke_seed_overlap = len(set(seeds) & {int(seed) for seed in smoke_seeds})
    smoke_id_overlap = len(set(response_ids) & {str(value) for value in smoke_response_ids})
    if smoke_seed_overlap or smoke_id_overlap:
        failures.append("D2-B0 smoke entered the formal development bank")
    role_seed_sets = {
        role: {int(record["generation_seed"]) for record in records if record.get("data_role") == role}
        for role in ROLE_TOTALS
    }
    role_seed_overlap = sum(
        len(role_seed_sets[left] & role_seed_sets[right])
        for index, left in enumerate(ROLE_TOTALS)
        for right in list(ROLE_TOTALS)[index + 1 :]
    )
    if role_seed_overlap:
        failures.append(f"cross_role_seed_overlap={role_seed_overlap}")
    per_prompt_role = Counter((str(row.get("prompt_id")), str(row.get("data_role"))) for row in records)
    prompt_ids = sorted({row.prompt_id for row in schedule})
    for prompt_id in prompt_ids:
        if per_prompt_role[(prompt_id, REFERENCE_ROLE)] != 60:
            failures.append(f"reference per-prompt mismatch: {prompt_id}")
        if per_prompt_role[(prompt_id, INTACT_TARGET_ROLE)] != 100:
            failures.append(f"intact per-prompt mismatch: {prompt_id}")
    endpoint_prompt = Counter(
        (str(row.get("attack_instance_id")), str(row.get("prompt_id")))
        for row in records
        if row.get("data_role") == ATTACK_ROLE
    )
    if len(endpoint_prompt) != 96 or set(endpoint_prompt.values()) != {20}:
        failures.append("attack endpoint/prompt counts changed")
    nested_failures = validate_nested_membership_against_records(nested_subset_payload, records)
    failures.extend(nested_failures)
    materialization_failures: list[str] = []
    for record in records:
        if record.get("data_role") != ATTACK_ROLE:
            continue
        endpoint = str(record.get("attack_instance_id"))
        expected = expected_materialization_by_endpoint.get(endpoint)
        if expected is None or any(record.get(key) != value for key, value in expected.items()):
            materialization_failures.append(endpoint)
    if materialization_failures:
        failures.append("attack materialization binding mismatch")
    technical = [event for event in events if event.get("status") == "technical_failure"]
    retry_seed_changed = sum(bool(record.get("retry_seed_changed")) for record in records)
    if retry_seed_changed:
        failures.append(f"retry_seed_changed={retry_seed_changed}")
    lengths = [int(record.get("response_token_count_including_eos", 0)) for record in records]
    endpoint_counts = Counter(
        str(row.get("attack_instance_id")) for row in records if row.get("data_role") == ATTACK_ROLE
    )
    family_counts = Counter(
        str(row.get("attack_family")) for row in records if row.get("data_role") == ATTACK_ROLE
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "total_responses": len(records),
        "role_counts": dict(sorted(roles.items())),
        "per_prompt_role_counts": {
            prompt_id: {
                REFERENCE_ROLE: per_prompt_role[(prompt_id, REFERENCE_ROLE)],
                INTACT_TARGET_ROLE: per_prompt_role[(prompt_id, INTACT_TARGET_ROLE)],
                ATTACK_ROLE: per_prompt_role[(prompt_id, ATTACK_ROLE)],
            }
            for prompt_id in prompt_ids
        },
        "attack_endpoint_counts": dict(sorted(endpoint_counts.items())),
        "attack_family_counts": dict(sorted(family_counts.items())),
        "unique_response_id_count": len(set(response_ids)),
        "unique_generation_seed_count": len(set(seeds)),
        "cross_role_seed_overlap_count": role_seed_overlap,
        "d2b0_smoke_seed_overlap_count": smoke_seed_overlap,
        "d2b0_smoke_response_id_overlap_count": smoke_id_overlap,
        "nested_subset_file_sha256": nested_subset_file_sha256,
        "nested_membership_failure_count": len(nested_failures),
        "missing_required_field_count": missing,
        "attack_materialization_binding_failure_count": len(materialization_failures),
        "attempt_event_count": len(events),
        "technical_failure_count": len(technical),
        "technical_failure_types": dict(
            sorted(Counter(str(row.get("exception_type", "")) for row in technical).items())
        ),
        "retry_seed_changed_count": retry_seed_changed,
        "stop_reason_counts": dict(sorted(Counter(str(row.get("stop_reason")) for row in records).items())),
        "legal_first_token_eos_count": sum(bool(row.get("legal_first_token_eos")) for row in records),
        "empty_response_count": sum(str(row.get("raw_response", "")) == "" for row in records),
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


__all__ = [
    "ATTACK_ROLE",
    "D2B1GenerationRequest",
    "D2B1_SCHEMA_VERSION",
    "GENESIS_CHAIN",
    "INTACT_TARGET_ROLE",
    "REFERENCE_ROLE",
    "ROLE_TOTALS",
    "TOTAL_RESPONSES",
    "audit_formal_records",
    "combine_and_validate_schedules",
    "hydrate_prompts",
    "request_from_manifest_row",
    "seal_response_record",
    "validate_attempt_events",
    "validate_nested_membership_against_records",
    "verify_completed_prefix",
]
