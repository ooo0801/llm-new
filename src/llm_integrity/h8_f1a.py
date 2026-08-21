from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .h8_d2a import global_permutation_test
from .h8_precalibration import canonical_sha256


F1A_SCHEMA_VERSION = "h8-f1a-final-confirmation-preflight-1.0"
PROMPT_COUNT = 12
FINAL_REFERENCE_ROLE = "final_reference_only"
FINAL_INTACT_ROLE = "final_heldout_intact_target_only"
FINAL_ATTACK_ROLE = "final_heldout_attack_only"
FINAL_ROLES = (FINAL_REFERENCE_ROLE, FINAL_INTACT_ROLE, FINAL_ATTACK_ROLE)
ATTACK_FAMILIES = ("gaussian", "pruning", "lora", "quantization")
REFERENCE_PER_PROMPT = 60
INTACT_UNIT_COUNT = 60
TARGET_PER_PROMPT_UNIT = 10
ATTACK_ENDPOINTS_PER_FAMILY = 10
ATTACK_PER_ENDPOINT_PROMPT = 10
REFERENCE_TOTAL = 720
INTACT_TOTAL = 7200
ATTACK_TOTAL = 4800
FORMAL_TOTAL = 12720
SMOKE_TOTAL = 6
N_REFERENCE = 60
N_TARGET = 10
TOP_R = 2
PERMUTATIONS = 999
ALPHA = 0.05
UINT32_MAX = 2**32 - 1
FINAL_ESTIMAND = "conditional_performance_of_frozen_detector_with_single_shared_reference_bank"
ATTACK_RATE_SCOPE = "preregistered_fresh_heldout_endpoint_panel_detection_rate"
INTERVAL_SCOPE = "conditional_on_single_frozen_shared_reference_bank"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uint64(domain: str, root_seed: int, *parts: object) -> int:
    material = "\0".join([domain, str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")


def _derive_uint32(
    domain: str,
    root_seed: int,
    forbidden: set[int],
    *parts: object,
) -> tuple[int, int, str]:
    counter = 0
    while True:
        material = "\0".join(
            [domain, str(root_seed), *(str(part) for part in parts), str(counter)]
        )
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        value = int.from_bytes(digest[:4], "little")
        if value not in forbidden:
            forbidden.add(value)
            return value, counter, hashlib.sha256(material.encode("utf-8")).hexdigest()
        counter += 1


def _prompt_rows(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) != PROMPT_COUNT:
        raise ValueError("F1-A requires exactly the frozen MCC12 prompts")
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        prompt_id = str(entry.get("prompt_id", ""))
        prompt = str(entry.get("prompt", ""))
        prompt_sha = str((entry.get("metadata") or {}).get("prompt_sha256", ""))
        if not prompt_id or not prompt or len(prompt_sha) != 64:
            raise ValueError("Frozen MCC12 prompt identity is incomplete")
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha:
            raise ValueError("Frozen MCC12 prompt SHA256 mismatch")
        rows.append(
            {
                "prompt_index": index,
                "prompt_id": prompt_id,
                "prompt_sha256": prompt_sha,
            }
        )
    if len({row["prompt_id"] for row in rows}) != PROMPT_COUNT:
        raise ValueError("Frozen MCC12 prompt IDs are not unique")
    return rows


def _ordered_prompt_indices(
    prompt_count: int,
    *,
    order_root_seed: int,
    round_id: str,
) -> list[int]:
    return sorted(
        range(prompt_count),
        key=lambda index: hashlib.sha256(
            f"h8-f1a-prompt-order-v1\0{order_root_seed}\0{round_id}\0{index}".encode("utf-8")
        ).digest(),
    )


def load_frozen_detector(
    archive: str | Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    root = Path(archive)
    manifest_path = root / "DETECTOR_FROZEN_MANIFEST.json"
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Frozen detector manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in dict(manifest.get("files", {})).items():
        payload_path = root / relative
        if not payload_path.is_file() or file_sha256(payload_path) != expected:
            raise ValueError(f"Frozen detector payload mismatch: {relative}")
    selected_path = root / "H8_D2C_SELECTED_DETECTOR.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    payload = selected.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Frozen selected-detector payload is missing")
    actual_payload_sha = canonical_sha256(payload)
    if (
        actual_payload_sha != selected.get("payload_sha256")
        or actual_payload_sha != manifest.get("selected_detector_payload_sha256")
    ):
        raise ValueError("Frozen selected-detector payload SHA256 mismatch")
    invariants = {
        "selected_configuration_id": "r60_q10_top2",
        "n_reference": N_REFERENCE,
        "n_target": N_TARGET,
        "top_r": TOP_R,
        "aggregation": "top_2_sum",
        "permutations": PERMUTATIONS,
        "alpha": ALPHA,
        "global_p_value_formula": "(1 + count(T_perm >= T_observed)) / 1000",
    }
    for key, expected in invariants.items():
        if payload.get(key) != expected:
            raise ValueError(f"Frozen detector invariant changed: {key}")
    if manifest.get("measurement_layer") != "frozen" or manifest.get("score_layer") != "frozen":
        raise ValueError("Frozen measurement/score layer status changed")
    return {"manifest": manifest, "selected_detector": payload}


def _normalized_family(value: str) -> str:
    lowered = value.lower()
    aliases = {
        "gaussian_noise": "gaussian",
        "finetuning": "lora",
        "unstructured_pruning": "pruning",
        "structured_pruning": "pruning",
    }
    return aliases.get(lowered, lowered)


def endpoint_configuration_sha256(endpoint: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(endpoint["configuration"]))


def build_attack_endpoints(
    attack_design: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    materialization_root_seed: int,
    lora_training_root_seed: int,
    forbidden_seeds: Iterable[int] = (),
) -> list[dict[str, Any]]:
    used = {int(seed) for seed in forbidden_seeds}
    rows: list[dict[str, Any]] = []
    for family in ATTACK_FAMILIES:
        designs = list(attack_design.get(family, []))
        if len(designs) != ATTACK_ENDPOINTS_PER_FAMILY:
            raise ValueError(f"F1-A attack design must contain ten {family} configurations")
        for index, raw_config in enumerate(designs):
            config = dict(raw_config)
            if family == "gaussian":
                config = {
                    "type": "gaussian_noise",
                    "scale_rule": "parameter_tensor_std",
                    **config,
                }
            elif family == "lora":
                config = {
                    "type": "finetuning",
                    "method": "lora",
                    "target_scope": "attention_ffn",
                    "data_source": "h8_f1_final_isolated_attack_training_data_v1",
                    **config,
                }
            elif family == "quantization":
                config = {"type": "quantization", **config}
            material_seed, _, _ = _derive_uint32(
                "h8-f1a-final-attack-materialization-v1",
                materialization_root_seed,
                used,
                family,
                index,
                canonical_sha256(config),
            )
            training_seed: int | None = None
            if family == "lora":
                training_seed, _, _ = _derive_uint32(
                    "h8-f1a-final-lora-training-v1",
                    lora_training_root_seed,
                    used,
                    family,
                    index,
                    canonical_sha256(config),
                )
            rows.append(
                {
                    "attack_instance_id": f"h8f1_final_{family}_{index + 1:02d}_seed{material_seed:010d}",
                    "family": family,
                    "materialization_seed": material_seed,
                    "training_seed": training_seed,
                    "configuration": config,
                    "configuration_sha256": canonical_sha256(config),
                }
            )
    return rows


def validate_attack_endpoints(
    endpoints: Sequence[Mapping[str, Any]],
    *,
    historical_endpoint_ids: Iterable[str] = (),
    historical_materialization_seeds: Iterable[int] = (),
    historical_configuration_sha256: Iterable[str] = (),
) -> dict[str, Any]:
    if len(endpoints) != len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY:
        raise ValueError("F1-A requires exactly forty formal attack endpoints")
    historical_ids = {str(value) for value in historical_endpoint_ids}
    historical_seeds = {int(value) for value in historical_materialization_seeds}
    historical_configs = {str(value) for value in historical_configuration_sha256}
    ids: set[str] = set()
    materialization_seeds: set[int] = set()
    training_seeds: set[int] = set()
    configuration_hashes: set[str] = set()
    family_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for raw in endpoints:
        row = dict(raw)
        endpoint_id = str(row.get("attack_instance_id", ""))
        family = _normalized_family(str(row.get("family", "")))
        config = row.get("configuration")
        material_seed = int(row.get("materialization_seed", -1))
        training_seed_raw = row.get("training_seed")
        training_seed = None if training_seed_raw is None else int(training_seed_raw)
        if not endpoint_id or family not in ATTACK_FAMILIES or not isinstance(config, Mapping):
            raise ValueError("Final attack endpoint identity/configuration is invalid")
        if endpoint_id in ids or endpoint_id in historical_ids:
            raise ValueError("Final attack endpoint ID is reused")
        if not 0 <= material_seed <= UINT32_MAX or material_seed in materialization_seeds or material_seed in historical_seeds:
            raise ValueError("Final attack materialization seed is reused or outside uint32")
        if training_seed is not None:
            if family != "lora" or not 0 <= training_seed <= UINT32_MAX:
                raise ValueError("Only LoRA endpoints may carry a uint32 training seed")
            if training_seed in training_seeds or training_seed in historical_seeds or training_seed == material_seed:
                raise ValueError("Final LoRA training seed is reused")
            training_seeds.add(training_seed)
        elif family == "lora":
            raise ValueError("Final LoRA endpoint lacks its independent training seed")
        config_sha = canonical_sha256(dict(config))
        if config_sha in configuration_hashes or config_sha in historical_configs:
            raise ValueError("Final attack configuration is not fresh and unique")
        declared_sha = row.get("configuration_sha256")
        if declared_sha is not None and declared_sha != config_sha:
            raise ValueError("Final attack configuration SHA256 mismatch")
        ids.add(endpoint_id)
        materialization_seeds.add(material_seed)
        configuration_hashes.add(config_sha)
        family_counts[family] += 1
        rows.append(
            {
                **row,
                "family": family,
                "configuration": dict(config),
                "configuration_sha256": config_sha,
                "planned_artifact_identity_sha256": canonical_sha256(
                    {
                        "attack_instance_id": endpoint_id,
                        "family": family,
                        "configuration_sha256": config_sha,
                        "materialization_seed": material_seed,
                        "training_seed": training_seed,
                    }
                ),
                "materialization_status": "not_materialized_preflight_only",
            }
        )
    if family_counts != Counter({family: ATTACK_ENDPOINTS_PER_FAMILY for family in ATTACK_FAMILIES}):
        raise ValueError("Final attack family counts changed")
    if materialization_seeds & training_seeds:
        raise ValueError("Final materialization and training seed namespaces overlap")
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "status": "PASS_FRESH_BY_CONSTRUCTION_PRE_MATERIALIZATION",
        "endpoint_count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "endpoint_ids_unique": True,
        "configuration_sha256_unique": True,
        "materialization_seeds_unique": True,
        "training_seeds_unique": True,
        "historical_id_overlap_count": 0,
        "historical_configuration_overlap_count": 0,
        "historical_materialization_seed_overlap_count": 0,
        "artifact_identity_status": "planned_identity_frozen_materialized_hash_pending_formal_authorization",
        "endpoints": rows,
    }


@dataclass(frozen=True)
class FinalGenerationRequest:
    schedule_position: int
    bank_position: int
    data_role: str
    bank_id: str
    evaluation_unit_id: str | None
    attack_family: str | None
    attack_instance_id: str | None
    replicate_id: int
    prompt_position_in_round: int
    prompt_index: int
    prompt_id: str
    prompt_sha256: str
    generation_seed: int
    derivation_counter: int
    seed_digest_sha256: str
    response_id: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "eligible_for_final_confirmation": True,
                "eligible_for_final_reference": self.data_role == FINAL_REFERENCE_ROLE,
                "eligible_for_final_intact_evaluation": self.data_role == FINAL_INTACT_ROLE,
                "eligible_for_final_attack_evaluation": self.data_role == FINAL_ATTACK_ROLE,
                "eligible_for_measurement_parameter_fit": False,
                "eligible_for_score_parameter_fit": False,
                "eligible_for_detector_selection": False,
                "eligible_for_development": False,
                "formal_sampling_authorized": False,
            }
        )
        return payload


def build_generation_schedule(
    entries: Sequence[Mapping[str, Any]],
    attack_endpoints: Sequence[Mapping[str, Any]],
    *,
    generation_root_seed: int,
    order_root_seed: int,
    forbidden_generation_seeds: Iterable[int] = (),
    forbidden_response_ids: Iterable[str] = (),
) -> dict[str, Any]:
    prompts = _prompt_rows(entries)
    endpoint_audit = validate_attack_endpoints(attack_endpoints)
    endpoints = endpoint_audit["endpoints"]
    used_seeds = {int(seed) for seed in forbidden_generation_seeds}
    forbidden_ids = {str(value) for value in forbidden_response_ids}
    rows: list[FinalGenerationRequest] = []
    bank_positions: Counter[str] = Counter()

    def append_round(
        *,
        round_id: str,
        data_role: str,
        bank_id: str,
        replicate_id: int,
        evaluation_unit_id: str | None,
        attack_family: str | None = None,
        attack_instance_id: str | None = None,
    ) -> None:
        order = _ordered_prompt_indices(PROMPT_COUNT, order_root_seed=order_root_seed, round_id=round_id)
        for prompt_position, prompt_index in enumerate(order):
            prompt = prompts[prompt_index]
            seed, counter, digest = _derive_uint32(
                "h8-f1a-final-generation-v1",
                generation_root_seed,
                used_seeds,
                data_role,
                bank_id,
                replicate_id,
                prompt["prompt_id"],
            )
            response_id = (
                f"h8-f1-{data_role.replace('_only', '').replace('_', '-')}-"
                f"b{bank_positions[bank_id]:05d}-p{prompt_index:02d}-s{seed:010d}"
            )
            if response_id in forbidden_ids:
                raise ValueError("Final response ID overlaps a historical response")
            forbidden_ids.add(response_id)
            rows.append(
                FinalGenerationRequest(
                    schedule_position=len(rows),
                    bank_position=bank_positions[bank_id],
                    data_role=data_role,
                    bank_id=bank_id,
                    evaluation_unit_id=evaluation_unit_id,
                    attack_family=attack_family,
                    attack_instance_id=attack_instance_id,
                    replicate_id=replicate_id,
                    prompt_position_in_round=prompt_position,
                    prompt_index=prompt_index,
                    prompt_id=prompt["prompt_id"],
                    prompt_sha256=prompt["prompt_sha256"],
                    generation_seed=seed,
                    derivation_counter=counter,
                    seed_digest_sha256=digest,
                    response_id=response_id,
                )
            )
            bank_positions[bank_id] += 1

    # Each intact unit contributes ten target rounds. One reference round is inserted at a
    # deterministic location among the eleven rounds, distributing the shared Reference bank
    # across the complete intact acquisition period without altering unit membership.
    for unit_index in range(INTACT_UNIT_COUNT):
        reference_slot = _uint64("h8-f1a-reference-slot-v1", order_root_seed, unit_index) % 11
        target_replicate = 0
        for local_round in range(11):
            if local_round == reference_slot:
                append_round(
                    round_id=f"intact-superblock-{unit_index:02d}-reference",
                    data_role=FINAL_REFERENCE_ROLE,
                    bank_id="final_reference_bank",
                    replicate_id=unit_index,
                    evaluation_unit_id=None,
                )
            else:
                unit_id = f"final_intact_unit_{unit_index:03d}"
                append_round(
                    round_id=f"{unit_id}-replicate-{target_replicate:02d}",
                    data_role=FINAL_INTACT_ROLE,
                    bank_id=unit_id,
                    replicate_id=target_replicate,
                    evaluation_unit_id=unit_id,
                )
                target_replicate += 1

    for endpoint in endpoints:
        endpoint_id = str(endpoint["attack_instance_id"])
        for replicate_id in range(ATTACK_PER_ENDPOINT_PROMPT):
            append_round(
                round_id=f"{endpoint_id}-replicate-{replicate_id:02d}",
                data_role=FINAL_ATTACK_ROLE,
                bank_id=endpoint_id,
                replicate_id=replicate_id,
                evaluation_unit_id=endpoint_id,
                attack_family=str(endpoint["family"]),
                attack_instance_id=endpoint_id,
            )

    payload_rows = [row.to_dict() for row in rows]
    validate_generation_schedule(payload_rows)
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "artifact_status": "frozen_before_first_formal_response",
        "formal_sampling_authorized": False,
        "derivation": "domain_separated_sha256_first_uint32_with_collision_counter",
        "generation_root_seed_uint32": generation_root_seed,
        "order_root_seed_uint64": order_root_seed,
        "historical_forbidden_generation_seed_count": len(set(int(seed) for seed in forbidden_generation_seeds)),
        "historical_forbidden_response_id_count": len(set(str(value) for value in forbidden_response_ids)),
        "response_count": len(payload_rows),
        "unique_generation_seed_count": len({row["generation_seed"] for row in payload_rows}),
        "unique_response_id_count": len({row["response_id"] for row in payload_rows}),
        "role_counts": dict(sorted(Counter(row["data_role"] for row in payload_rows).items())),
        "requests": payload_rows,
    }


def validate_generation_schedule(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != FORMAL_TOTAL:
        raise ValueError("F1-A formal schedule must contain exactly 12,720 requests")
    if [int(row.get("schedule_position", -1)) for row in rows] != list(range(FORMAL_TOTAL)):
        raise ValueError("F1-A schedule positions are not contiguous")
    roles = Counter(str(row.get("data_role")) for row in rows)
    expected_roles = Counter(
        {
            FINAL_REFERENCE_ROLE: REFERENCE_TOTAL,
            FINAL_INTACT_ROLE: INTACT_TOTAL,
            FINAL_ATTACK_ROLE: ATTACK_TOTAL,
        }
    )
    if roles != expected_roles:
        raise ValueError("F1-A formal role totals changed")
    seeds = [int(row.get("generation_seed", -1)) for row in rows]
    ids = [str(row.get("response_id", "")) for row in rows]
    if any(not 0 <= seed <= UINT32_MAX for seed in seeds) or len(set(seeds)) != FORMAL_TOTAL:
        raise ValueError("F1-A generation seeds are invalid or duplicated")
    if any(not value for value in ids) or len(set(ids)) != FORMAL_TOTAL:
        raise ValueError("F1-A response IDs are missing or duplicated")
    prompt_ids = {str(row.get("prompt_id")) for row in rows}
    if len(prompt_ids) != PROMPT_COUNT:
        raise ValueError("F1-A schedule does not cover MCC12")
    per_prompt_role = Counter((str(row["prompt_id"]), str(row["data_role"])) for row in rows)
    for prompt_id in prompt_ids:
        if per_prompt_role[(prompt_id, FINAL_REFERENCE_ROLE)] != REFERENCE_PER_PROMPT:
            raise ValueError("F1-A reference per-prompt count changed")
        if per_prompt_role[(prompt_id, FINAL_INTACT_ROLE)] != INTACT_UNIT_COUNT * TARGET_PER_PROMPT_UNIT:
            raise ValueError("F1-A intact per-prompt count changed")
    units = Counter(
        (str(row.get("evaluation_unit_id")), str(row["prompt_id"]))
        for row in rows
        if row["data_role"] == FINAL_INTACT_ROLE
    )
    if len(units) != INTACT_UNIT_COUNT * PROMPT_COUNT or set(units.values()) != {TARGET_PER_PROMPT_UNIT}:
        raise ValueError("F1-A intact evaluation-unit membership changed")
    endpoints = Counter(
        (str(row.get("attack_instance_id")), str(row["prompt_id"]))
        for row in rows
        if row["data_role"] == FINAL_ATTACK_ROLE
    )
    if len(endpoints) != len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY * PROMPT_COUNT:
        raise ValueError("F1-A attack endpoint membership changed")
    if set(endpoints.values()) != {ATTACK_PER_ENDPOINT_PROMPT}:
        raise ValueError("F1-A attack endpoint/prompt count changed")
    for row in rows:
        for key in (
            "eligible_for_measurement_parameter_fit",
            "eligible_for_score_parameter_fit",
            "eligible_for_detector_selection",
            "eligible_for_development",
            "formal_sampling_authorized",
        ):
            if row.get(key) is not False:
                raise ValueError(f"F1-A eligibility/authorization changed: {key}")


def validate_smoke_records(
    records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit the six-role plumbing smoke without computing detector statistics."""
    if len(records) != SMOKE_TOTAL or len(requests) != SMOKE_TOTAL:
        raise ValueError("F1-A tiny smoke requires exactly six records and six requests")
    expected = {str(row["response_id"]): dict(row) for row in requests}
    if len(expected) != SMOKE_TOTAL:
        raise ValueError("F1-A smoke request IDs are duplicated")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    stop_reasons: Counter[str] = Counter()
    legal_empty_eos = 0
    retry_count = 0
    for raw in records:
        record = dict(raw)
        response_id = str(record.get("response_id", ""))
        if response_id in seen_ids or response_id not in expected:
            raise ValueError("F1-A smoke response ID is missing, duplicated, or unexpected")
        request = expected[response_id]
        seed = int(record.get("generation_seed", -1))
        if seed != int(request["generation_seed"]) or seed in seen_seeds:
            raise ValueError("F1-A smoke generation seed changed or was duplicated")
        for key in ("data_role", "prompt_id", "prompt_sha256", "attack_instance_id", "attack_family"):
            if record.get(key) != request.get(key):
                raise ValueError(f"F1-A smoke record/request mismatch: {key}")
        if (
            record.get("formal_final_confirmation_eligible") is not False
            or record.get("detector_statistics_computed") is not False
            or record.get("measurement_or_score_fit_performed") is not False
        ):
            raise ValueError("F1-A smoke eligibility or analysis boundary changed")
        required = (
            "rendered_prompt_sha256",
            "input_token_ids",
            "input_token_ids_sha256",
            "completion_token_ids",
            "completion_token_ids_sha256",
            "generated_token_count",
            "stop_reason",
            "attempt_records",
            "provenance_sha256",
        )
        if any(key not in record for key in required):
            raise ValueError("F1-A smoke record provenance is incomplete")
        attempts = list(record["attempt_records"])
        if not attempts or any(int(attempt.get("seed", -1)) != seed for attempt in attempts):
            raise ValueError("F1-A smoke retry changed the frozen response seed")
        retry_count += max(0, len(attempts) - 1)
        completion = list(record["completion_token_ids"])
        generated_count = int(record["generated_token_count"])
        if generated_count != len(completion):
            raise ValueError("F1-A smoke generated-token count is inconsistent")
        if record.get("legal_first_token_eos"):
            if generated_count != 1 or str(record.get("raw_response", "")) != "":
                raise ValueError("F1-A legal first-token EOS empty response was not preserved")
            legal_empty_eos += 1
        stop_reasons[str(record["stop_reason"])] += 1
        seen_ids.add(response_id)
        seen_seeds.add(seed)
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "status": "PASS",
        "response_count": len(records),
        "unique_response_id_count": len(seen_ids),
        "unique_generation_seed_count": len(seen_seeds),
        "stop_reason_counts": dict(sorted(stop_reasons.items())),
        "legal_first_token_eos_empty_response_count": legal_empty_eos,
        "technical_retry_count": retry_count,
        "same_seed_retry_compliant": True,
        "formal_final_confirmation_responses": 0,
        "detector_statistics_computed": False,
        "measurement_or_score_fit_performed": False,
    }


def build_permutation_seed_manifest(
    evaluation_units: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[str],
    *,
    permutation_root_seed: int,
) -> dict[str, Any]:
    if len(evaluation_units) != INTACT_UNIT_COUNT + len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY:
        raise ValueError("F1-A must define exactly one hundred evaluation units")
    if len(prompt_ids) != PROMPT_COUNT or len(set(prompt_ids)) != PROMPT_COUNT:
        raise ValueError("F1-A permutation manifest requires MCC12")
    seen: set[int] = set()
    overall = hashlib.sha256()
    streams: list[dict[str, Any]] = []
    for unit in evaluation_units:
        unit_id = str(unit["evaluation_unit_id"])
        stream_seed = _uint64("h8-f1a-global-permutation-stream-v1", permutation_root_seed, unit_id)
        stream_digest = hashlib.sha256()
        for permutation_id in range(PERMUTATIONS):
            for prompt_id in sorted(prompt_ids):
                seed = _uint64(
                    "h8-f1a-within-prompt-permutation-v1",
                    stream_seed,
                    permutation_id,
                    prompt_id,
                )
                if seed in seen:
                    raise ValueError("F1-A permutation seed collision")
                seen.add(seed)
                encoded = seed.to_bytes(8, "little")
                stream_digest.update(encoded)
                overall.update(encoded)
        streams.append(
            {
                "evaluation_unit_id": unit_id,
                "data_role": unit["data_role"],
                "attack_family": unit.get("attack_family"),
                "stream_seed_uint64": stream_seed,
                "permutations": PERMUTATIONS,
                "prompt_count": PROMPT_COUNT,
                "derived_prompt_permutation_seed_set_sha256": stream_digest.hexdigest(),
            }
        )
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "algorithm": "domain_separated_sha256_uint64_to_numpy_pcg64",
        "permutation_root_seed_uint64": permutation_root_seed,
        "n_reference": N_REFERENCE,
        "n_target": N_TARGET,
        "top_r": TOP_R,
        "permutations": PERMUTATIONS,
        "within_prompt_only": True,
        "group_sizes_preserved": True,
        "stream_count": len(streams),
        "derived_seed_count": len(seen),
        "derived_seed_unique": True,
        "derived_seed_set_sha256": overall.hexdigest(),
        "streams": streams,
    }


def final_global_permutation_test(
    kernels_by_prompt: Mapping[str, Any],
    score_parameters_by_prompt: Mapping[str, Mapping[str, Any]],
    *,
    root_seed: int,
) -> Any:
    return global_permutation_test(
        kernels_by_prompt,
        score_parameters_by_prompt,
        n_reference=N_REFERENCE,
        n_target=N_TARGET,
        top_r=TOP_R,
        permutations=PERMUTATIONS,
        alpha=ALPHA,
        permutation_root_seed=root_seed,
    )


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.fsum(
        math.comb(n, index)
        * probability**index
        * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def exact_binomial_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("Invalid exact-binomial interval arguments")
    tail = (1.0 - confidence) / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(120):
            mid = (lo + hi) / 2.0
            upper_tail = 1.0 - _binomial_cdf(successes - 1, trials, mid)
            if upper_tail < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2.0
    if successes == trials:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(120):
            mid = (lo + hi) / 2.0
            cdf = _binomial_cdf(successes, trials, mid)
            if cdf > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2.0
    return float(lower), float(upper)


def summarize_final_decisions(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != INTACT_UNIT_COUNT + len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY:
        raise ValueError("Final decision record count must equal one hundred")
    ids = [str(record.get("evaluation_unit_id", "")) for record in records]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Final evaluation-unit IDs are missing or duplicated")
    normalized: list[dict[str, Any]] = []
    for record in records:
        p_value = float(record.get("p_global", math.nan))
        if not math.isfinite(p_value) or not 0.0 < p_value <= 1.0:
            raise ValueError("Final global p-value is invalid")
        detected = bool(record.get("detected"))
        if detected != (p_value <= ALPHA):
            raise ValueError("Final detector decision does not match the frozen alpha")
        normalized.append({**dict(record), "p_global": p_value, "detected": detected})
    intact = [row for row in normalized if row.get("data_role") == FINAL_INTACT_ROLE]
    attacks = [row for row in normalized if row.get("data_role") == FINAL_ATTACK_ROLE]
    if len(intact) != INTACT_UNIT_COUNT or len(attacks) != len(ATTACK_FAMILIES) * ATTACK_ENDPOINTS_PER_FAMILY:
        raise ValueError("Final intact/attack decision totals changed")
    false_positives = sum(int(row["detected"]) for row in intact)
    fp_ci = exact_binomial_interval(false_positives, INTACT_UNIT_COUNT)
    family_rows: dict[str, Any] = {}
    for family in ATTACK_FAMILIES:
        selected = [row for row in attacks if _normalized_family(str(row.get("attack_family", ""))) == family]
        if len(selected) != ATTACK_ENDPOINTS_PER_FAMILY:
            raise ValueError("Final family endpoint decision count changed")
        detected = sum(int(row["detected"]) for row in selected)
        interval = exact_binomial_interval(detected, ATTACK_ENDPOINTS_PER_FAMILY)
        family_rows[family] = {
            "detected": detected,
            "total": ATTACK_ENDPOINTS_PER_FAMILY,
            "heldout_endpoint_panel_detection_rate": detected / ATTACK_ENDPOINTS_PER_FAMILY,
            "operational_tpr_alias": detected / ATTACK_ENDPOINTS_PER_FAMILY,
            "conditional_clopper_pearson_95ci": [interval[0], interval[1]],
            "rate_scope": ATTACK_RATE_SCOPE,
            "iid_attack_superpopulation_claimed": False,
        }
    return {
        "schema_version": F1A_SCHEMA_VERSION,
        "frozen_detector": "r60_q10_top2",
        "alpha": ALPHA,
        "permutations": PERMUTATIONS,
        "global_p_value_formula": "(1 + count(T_perm >= T_observed)) / 1000",
        "primary_estimand": FINAL_ESTIMAND,
        "reference_bank_count": 1,
        "reference_bank_shared_across_all_evaluation_units": True,
        "interval_scope": INTERVAL_SCOPE,
        "unconditional_reference_regeneration_coverage_claimed": False,
        "intact": {
            "false_positive_count": false_positives,
            "total": INTACT_UNIT_COUNT,
            "fpr": false_positives / INTACT_UNIT_COUNT,
            "conditional_clopper_pearson_95ci": [fp_ci[0], fp_ci[1]],
        },
        "attack_families": family_rows,
        "overall_attack_detected": sum(int(row["detected"]) for row in attacks),
        "overall_attack_total": len(attacks),
        "overall_attack_result_does_not_replace_family_specific_results": True,
        "shared_reference_dependence_note": (
            "Clopper-Pearson intervals are conditional descriptive binomial intervals for the single "
            "frozen shared 60-response-per-prompt Reference bank. They do not include uncertainty from "
            "regenerating a new Reference bank, and shared-Reference dependence must not be hidden or "
            "interpreted as unconditional exact coverage."
        ),
        "attack_panel_note": (
            "Family-specific detected/10 values describe the preregistered fresh held-out endpoint "
            "panel. They are not an iid superpopulation TPR claim because endpoint configurations are "
            "intentionally heterogeneous."
        ),
    }


def save_canonical_envelope(path: str | Path, artifact_type: str, payload: Mapping[str, Any]) -> dict[str, str]:
    payload_dict = dict(payload)
    payload_sha = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": artifact_type,
        "schema_version": F1A_SCHEMA_VERSION,
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(target)
    return {"payload_sha256": payload_sha, "file_sha256": file_sha256(target)}


def load_canonical_envelope(
    path: str | Path,
    *,
    expected_artifact_type: str,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    target = Path(path)
    if file_sha256(target) != expected_file_sha256:
        raise ValueError("F1-A artifact file SHA256 mismatch")
    envelope = json.loads(target.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if (
        envelope.get("artifact_type") != expected_artifact_type
        or envelope.get("schema_version") != F1A_SCHEMA_VERSION
        or not isinstance(payload, dict)
    ):
        raise ValueError("F1-A artifact type/schema mismatch")
    actual = canonical_sha256(payload)
    if actual != envelope.get("payload_sha256") or actual != expected_payload_sha256:
        raise ValueError("F1-A artifact payload SHA256 mismatch")
    return payload
