from __future__ import annotations

from collections import Counter
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .h8_f1a import (
    ATTACK_ENDPOINTS_PER_FAMILY,
    ATTACK_FAMILIES,
    ATTACK_TOTAL,
    FINAL_ATTACK_ROLE,
    FINAL_INTACT_ROLE,
    FINAL_REFERENCE_ROLE,
    FORMAL_TOTAL,
    INTACT_TOTAL,
    INTACT_UNIT_COUNT,
    PERMUTATIONS,
    PROMPT_COUNT,
    REFERENCE_TOTAL,
    UINT32_MAX,
    build_permutation_seed_manifest,
    validate_attack_endpoints,
    validate_generation_schedule,
)
from .h8_precalibration import canonical_sha256


F1B_SCHEMA_VERSION = "h8-f1b-formal-final-confirmation-sampling-1.0"
BASE_PARTITION_ID = "frozen_intact_base_model"
EXPECTED_PARTITION_COUNT = 1 + len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uint64(domain: str, root_seed: int, *parts: object) -> int:
    material = "\0".join([domain, str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")


def partition_requests(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    validate_generation_schedule(rows)
    partitions: dict[str, list[dict[str, Any]]] = {BASE_PARTITION_ID: []}
    for raw in rows:
        row = dict(raw)
        role = str(row["data_role"])
        if role in {FINAL_REFERENCE_ROLE, FINAL_INTACT_ROLE}:
            partition = BASE_PARTITION_ID
        elif role == FINAL_ATTACK_ROLE:
            partition = str(row.get("attack_instance_id", ""))
            if not partition:
                raise ValueError("F1-B attack request lacks its frozen endpoint ID")
        else:
            raise ValueError("F1-B request has an unsupported role")
        partitions.setdefault(partition, []).append(row)
    if len(partitions) != EXPECTED_PARTITION_COUNT:
        raise ValueError("F1-B must contain one base and forty attack partitions")
    if len(partitions[BASE_PARTITION_ID]) != REFERENCE_TOTAL + INTACT_TOTAL:
        raise ValueError("F1-B base partition count changed")
    for partition, requests in partitions.items():
        requests.sort(key=lambda row: int(row["schedule_position"]))
        if partition != BASE_PARTITION_ID and len(requests) != ATTACK_TOTAL // 40:
            raise ValueError("F1-B attack partition must contain exactly 120 requests")
    return partitions


def _permutation_seed_set(payload: Mapping[str, Any], prompt_ids: Sequence[str]) -> set[int]:
    result: set[int] = set()
    for stream in payload["streams"]:
        stream_seed = int(stream["stream_seed_uint64"])
        for permutation_id in range(PERMUTATIONS):
            for prompt_id in sorted(prompt_ids):
                seed = _uint64(
                    "h8-f1a-within-prompt-permutation-v1",
                    stream_seed,
                    permutation_id,
                    prompt_id,
                )
                if seed in result:
                    raise ValueError("F1-B frozen permutation seed collision")
                result.add(seed)
    return result


def validate_frozen_inputs(
    generation: Mapping[str, Any],
    attack: Mapping[str, Any],
    permutation: Mapping[str, Any],
    freshness: Mapping[str, Any],
    historical: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    requests = list(generation.get("requests", []))
    validate_generation_schedule(requests)
    if generation.get("response_count") != FORMAL_TOTAL or generation.get("formal_sampling_authorized") is not False:
        raise ValueError("F1-B frozen F1-A generation manifest changed")
    endpoints = list(attack.get("endpoints", []))
    endpoint_audit = validate_attack_endpoints(endpoints)
    if attack.get("formal_endpoint_count") != 40 or endpoint_audit["endpoint_count"] != 40:
        raise ValueError("F1-B frozen attack endpoint manifest changed")
    if freshness.get("status") != "PASS_FRESH_BY_CONSTRUCTION_PRE_MATERIALIZATION":
        raise ValueError("F1-B attack freshness gate is not PASS")
    if any(
        int(freshness.get(key, -1)) != 0
        for key in (
            "historical_id_overlap_count",
            "historical_configuration_overlap_count",
            "historical_materialization_seed_overlap_count",
            "formal_generation_historical_overlap_count",
            "formal_generation_attack_seed_overlap_count",
            "formal_response_id_historical_overlap_count",
            "smoke_formal_generation_seed_overlap_count",
            "smoke_historical_seed_overlap_count",
        )
    ):
        raise ValueError("F1-B frozen freshness overlap count is nonzero")
    if freshness.get("lora_training_isolation", {}).get("status") != "PASS":
        raise ValueError("F1-B LoRA training isolation gate changed")

    endpoint_ids = {str(row["attack_instance_id"]) for row in endpoints}
    request_endpoint_ids = {
        str(row["attack_instance_id"])
        for row in requests
        if row["data_role"] == FINAL_ATTACK_ROLE
    }
    if request_endpoint_ids != endpoint_ids:
        raise ValueError("F1-B generation and endpoint manifests disagree")
    reference_banks = {
        str(row["bank_id"])
        for row in requests
        if row["data_role"] == FINAL_REFERENCE_ROLE
    }
    if reference_banks != {"final_reference_bank"}:
        raise ValueError("F1-B must generate exactly one shared Reference bank")

    generation_seeds = {int(row["generation_seed"]) for row in requests}
    generation_ids = {str(row["response_id"]) for row in requests}
    endpoint_seeds = {int(row["materialization_seed"]) for row in endpoints}
    endpoint_seeds.update(int(row["training_seed"]) for row in endpoints if row.get("training_seed") is not None)
    smoke_requests = list(smoke.get("requests", []))
    smoke_seeds = {int(row["generation_seed"]) for row in smoke_requests}
    smoke_seeds.update(int(row["materialization_seed"]) for row in smoke.get("attack_endpoints", []))
    smoke_seeds.update(
        int(row["training_seed"])
        for row in smoke.get("attack_endpoints", [])
        if row.get("training_seed") is not None
    )
    smoke_ids = {str(row["response_id"]) for row in smoke_requests}
    historical_seeds = {int(seed) for seed in historical.get("seeds", [])}
    historical_ids = {str(value) for value in historical.get("response_ids", [])}
    if generation_seeds & endpoint_seeds or generation_seeds & smoke_seeds or generation_seeds & historical_seeds:
        raise ValueError("F1-B generation seed namespace overlaps a forbidden namespace")
    if endpoint_seeds & smoke_seeds or endpoint_seeds & historical_seeds:
        raise ValueError("F1-B endpoint seed namespace overlaps a forbidden namespace")
    if generation_ids & smoke_ids or generation_ids & historical_ids:
        raise ValueError("F1-B response ID namespace overlaps smoke/history")

    prompt_ids = sorted({str(row["prompt_id"]) for row in requests})
    units = [
        {"evaluation_unit_id": f"final_intact_unit_{index:03d}", "data_role": FINAL_INTACT_ROLE, "attack_family": None}
        for index in range(INTACT_UNIT_COUNT)
    ] + [
        {
            "evaluation_unit_id": row["attack_instance_id"],
            "data_role": FINAL_ATTACK_ROLE,
            "attack_family": row["family"],
        }
        for row in endpoints
    ]
    rebuilt_permutation = build_permutation_seed_manifest(
        units,
        prompt_ids,
        permutation_root_seed=int(permutation["permutation_root_seed_uint64"]),
    )
    if canonical_sha256(rebuilt_permutation) != canonical_sha256(dict(permutation)):
        raise ValueError("F1-B frozen permutation manifest cannot be deterministically reproduced")
    permutation_seeds = _permutation_seed_set(permutation, prompt_ids)
    if permutation_seeds & generation_seeds or permutation_seeds & endpoint_seeds or permutation_seeds & smoke_seeds:
        raise ValueError("F1-B CPU permutation namespace overlaps a generation/materialization namespace")

    partitions = partition_requests(requests)
    return {
        "schema_version": F1B_SCHEMA_VERSION,
        "status": "PASS",
        "formal_response_count": len(requests),
        "role_counts": dict(sorted(Counter(str(row["data_role"]) for row in requests).items())),
        "partition_count": len(partitions),
        "base_partition_count": len(partitions[BASE_PARTITION_ID]),
        "attack_partition_count": len(partitions) - 1,
        "generation_seed_unique_count": len(generation_seeds),
        "generation_response_id_unique_count": len(generation_ids),
        "historical_seed_overlap_count": 0,
        "historical_response_id_overlap_count": 0,
        "smoke_seed_overlap_count": 0,
        "smoke_response_id_overlap_count": 0,
        "endpoint_seed_overlap_count": 0,
        "permutation_seed_count": len(permutation_seeds),
        "permutation_seed_unique": True,
        "permutation_cross_namespace_overlap_count": 0,
        "single_shared_reference_bank": True,
    }


def validate_response_record(
    record: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    frozen_identity_sha256: str,
    expected_materialization_identity_sha256: str | None,
) -> None:
    for key in (
        "schedule_position",
        "response_id",
        "generation_seed",
        "data_role",
        "bank_id",
        "evaluation_unit_id",
        "attack_family",
        "attack_instance_id",
        "replicate_id",
        "prompt_index",
        "prompt_id",
        "prompt_sha256",
    ):
        if record.get(key) != request.get(key):
            raise ValueError(f"F1-B response/request mismatch: {key}")
    if record.get("frozen_identity_sha256") != frozen_identity_sha256:
        raise ValueError("F1-B response frozen-identity SHA256 mismatch")
    if record.get("materialization_identity_sha256") != expected_materialization_identity_sha256:
        raise ValueError("F1-B response materialization identity mismatch")
    if (
        record.get("formal_final_confirmation_eligible") is not True
        or record.get("eligible_for_measurement_parameter_fit") is not False
        or record.get("eligible_for_score_parameter_fit") is not False
        or record.get("eligible_for_detector_selection") is not False
        or record.get("formal_detector_statistics_computed") is not False
    ):
        raise ValueError("F1-B response eligibility/analysis boundary changed")
    seed = int(record.get("generation_seed", -1))
    if not 0 <= seed <= UINT32_MAX:
        raise ValueError("F1-B response generation seed is outside uint32")
    attempts = list(record.get("attempt_records", []))
    if not attempts or any(int(attempt.get("seed", -1)) != seed for attempt in attempts):
        raise ValueError("F1-B technical retry changed the generation seed")
    if attempts[-1].get("status") != "success":
        raise ValueError("F1-B successful response lacks a final successful attempt")
    if any(attempt.get("status") != "technical_failure" for attempt in attempts[:-1]):
        raise ValueError("F1-B retry history contains a non-technical failed attempt")
    required = (
        "rendered_prompt_sha256",
        "input_token_ids",
        "input_token_ids_sha256",
        "completion_token_ids",
        "completion_token_ids_sha256",
        "generated_token_count",
        "stop_reason",
        "provenance_sha256",
    )
    if any(key not in record for key in required):
        raise ValueError("F1-B response provenance is incomplete")
    completion = list(record["completion_token_ids"])
    if int(record["generated_token_count"]) != len(completion):
        raise ValueError("F1-B response token count is inconsistent")
    if record.get("legal_first_token_eos"):
        if len(completion) != 1 or str(record.get("raw_response", "")) != "":
            raise ValueError("F1-B legal first-token EOS response was not preserved")


def audit_formal_records(
    records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    *,
    frozen_identity_sha256: str,
    endpoint_materialization_identities: Mapping[str, str],
) -> dict[str, Any]:
    if len(records) != FORMAL_TOTAL or len(requests) != FORMAL_TOTAL:
        raise ValueError("F1-B final audit requires exactly 12,720 records/requests")
    expected = {str(row["response_id"]): dict(row) for row in requests}
    if len(expected) != FORMAL_TOTAL:
        raise ValueError("F1-B frozen request IDs are duplicated")
    ids: set[str] = set()
    seeds: set[int] = set()
    role_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    reference_prompts: Counter[str] = Counter()
    intact_units: Counter[tuple[str, str]] = Counter()
    attack_units: Counter[tuple[str, str]] = Counter()
    stop_reasons: Counter[str] = Counter()
    empty_count = 0
    retry_count = 0
    token_lengths: list[int] = []
    for raw in records:
        record = dict(raw)
        response_id = str(record.get("response_id", ""))
        if response_id in ids or response_id not in expected:
            raise ValueError("F1-B response ID is duplicated or unexpected")
        request = expected[response_id]
        endpoint_id = request.get("attack_instance_id")
        materialization_identity = (
            endpoint_materialization_identities.get(BASE_PARTITION_ID)
            if endpoint_id is None
            else endpoint_materialization_identities.get(str(endpoint_id))
        )
        if materialization_identity is None:
            raise ValueError("F1-B response lacks a sealed partition materialization identity")
        validate_response_record(
            record,
            request,
            frozen_identity_sha256=frozen_identity_sha256,
            expected_materialization_identity_sha256=materialization_identity,
        )
        seed = int(record["generation_seed"])
        if seed in seeds:
            raise ValueError("F1-B response generation seed is duplicated")
        ids.add(response_id)
        seeds.add(seed)
        role = str(record["data_role"])
        role_counts[role] += 1
        prompt_counts[str(record["prompt_id"])] += 1
        if role == FINAL_REFERENCE_ROLE:
            reference_prompts[str(record["prompt_id"])] += 1
        if role == FINAL_INTACT_ROLE:
            intact_units[(str(record["evaluation_unit_id"]), str(record["prompt_id"]))] += 1
        if role == FINAL_ATTACK_ROLE:
            attack_units[(str(record["attack_instance_id"]), str(record["prompt_id"]))] += 1
        stop_reasons[str(record["stop_reason"])] += 1
        empty_count += int(str(record.get("raw_response", "")) == "")
        retry_count += max(0, len(record["attempt_records"]) - 1)
        token_lengths.append(int(record["generated_token_count"]))
    expected_roles = Counter(
        {FINAL_REFERENCE_ROLE: REFERENCE_TOTAL, FINAL_INTACT_ROLE: INTACT_TOTAL, FINAL_ATTACK_ROLE: ATTACK_TOTAL}
    )
    if role_counts != expected_roles:
        raise ValueError("F1-B final role totals changed")
    if len(prompt_counts) != PROMPT_COUNT:
        raise ValueError("F1-B final prompt coverage changed")
    if set(prompt_counts.values()) != {FORMAL_TOTAL // PROMPT_COUNT}:
        raise ValueError("F1-B per-prompt total count changed")
    if len(reference_prompts) != PROMPT_COUNT or set(reference_prompts.values()) != {60}:
        raise ValueError("F1-B shared Reference bank membership changed")
    if len(intact_units) != INTACT_UNIT_COUNT * PROMPT_COUNT or set(intact_units.values()) != {10}:
        raise ValueError("F1-B intact unit membership changed")
    if len(attack_units) != 40 * PROMPT_COUNT or set(attack_units.values()) != {10}:
        raise ValueError("F1-B attack endpoint membership changed")
    return {
        "schema_version": F1B_SCHEMA_VERSION,
        "status": "PASS",
        "total_response_count": len(records),
        "role_counts": dict(sorted(role_counts.items())),
        "prompt_count": len(prompt_counts),
        "intact_unit_count": len({key[0] for key in intact_units}),
        "attack_endpoint_count": len({key[0] for key in attack_units}),
        "response_id_unique_count": len(ids),
        "generation_seed_unique_count": len(seeds),
        "missing_required_field_count": 0,
        "technical_retry_count": retry_count,
        "same_seed_retry_compliant": True,
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "empty_response_count": empty_count,
        "generated_token_count_summary": {
            "min": min(token_lengths),
            "median": float(statistics.median(token_lengths)),
            "max": max(token_lengths),
        },
        "materialization_identity_count": len(endpoint_materialization_identities) - 1,
        "materialization_identity_consistent": True,
        "formal_detector_statistics_computed": False,
    }


def read_jsonl_strict(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"F1-B JSONL has a partial trailing line at {target}:{line_number}")
            if not line.strip():
                raise ValueError(f"F1-B JSONL contains a blank record at {target}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("F1-B JSONL record is not an object")
            rows.append(value)
    return rows
