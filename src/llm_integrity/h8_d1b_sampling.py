from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .h8_precalibration import canonical_sha256
from .h8_sampling import sha256_bytes, sha256_text


FIT_ROLE = "score_calibration_fit_only"
AUDIT_ROLE = "score_calibration_stability_audit_only"
SMOKE_ROLE = "score_calibration_smoke_only"
FORMAL_MANIFEST_TYPE = "h8_d1b_score_calibration_generation_manifest"
SMOKE_MANIFEST_TYPE = "h8_d1b_score_calibration_smoke_manifest"
MANIFEST_SCHEMA = "h8-d1b-generation-manifest-1.0"
PROMPT_COUNT = 12
FORMAL_ROUNDS = 200
FORMAL_TOTAL = 2400
SMOKE_ROUNDS = 2
SMOKE_TOTAL = 24
UINT32_MAX = 2**32 - 1


@dataclass(frozen=True)
class D1BGenerationRequest:
    schedule_position: int
    round_id: int
    prompt_position_in_round: int
    prompt_index: int
    prompt_id: str
    prompt: str
    prompt_sha256: str
    seed: int
    derivation_counter: int
    seed_digest_sha256: str
    data_role: str
    intended_bank_role: str
    is_smoke: bool

    @property
    def response_id(self) -> str:
        phase = "smoke" if self.is_smoke else "formal"
        role = "fit" if self.intended_bank_role == FIT_ROLE else "audit"
        return (
            f"h8-d1b-{phase}-{role}-r{self.round_id:03d}-"
            f"p{self.prompt_index:02d}-s{self.seed:010d}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schedule_position": self.schedule_position,
            "round_id": self.round_id,
            "prompt_position_in_round": self.prompt_position_in_round,
            "prompt_index": self.prompt_index,
            "prompt_id": self.prompt_id,
            "prompt_sha256": self.prompt_sha256,
            "seed": self.seed,
            "derivation_counter": self.derivation_counter,
            "seed_digest_sha256": self.seed_digest_sha256,
            "response_id": self.response_id,
            "data_role": self.data_role,
            "intended_bank_role": self.intended_bank_role,
            "is_smoke": self.is_smoke,
            "eligible_for_formal_reference": False,
            "eligible_for_heldout_evaluation": False,
            "eligible_for_target_evaluation": False,
            "eligible_for_attack_evaluation": False,
        }


def file_sha256(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _digest(domain: str, root_seed: int, *parts: object) -> bytes:
    value = "\0".join([domain, str(root_seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).digest()


def _normalize_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) != PROMPT_COUNT:
        raise ValueError("D1-B requires exactly twelve frozen MCC prompts")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        prompt_id = str(entry.get("prompt_id", ""))
        prompt = str(entry.get("prompt", ""))
        prompt_sha = str((entry.get("metadata") or {}).get("prompt_sha256", ""))
        if not prompt_id or prompt_id in seen or not prompt or sha256_text(prompt) != prompt_sha:
            raise ValueError(f"Invalid frozen prompt at index {index}")
        seen.add(prompt_id)
        result.append(
            {"prompt_index": index, "prompt_id": prompt_id, "prompt": prompt, "prompt_sha256": prompt_sha}
        )
    return result


def _round_order(entries: Sequence[Mapping[str, Any]], root_seed: int, round_id: int, smoke: bool) -> list[int]:
    domain = "h8-d1b-smoke-order-v1" if smoke else "h8-d1b-formal-order-v1"
    return sorted(
        range(len(entries)),
        key=lambda index: _digest(domain, root_seed, round_id, entries[index]["prompt_id"]),
    )


def _build_schedule(
    entries: Sequence[Mapping[str, Any]],
    root_seed: int,
    *,
    rounds: int,
    smoke: bool,
    forbidden_seeds: Iterable[int],
) -> list[D1BGenerationRequest]:
    if not 0 <= int(root_seed) <= UINT32_MAX:
        raise ValueError("Generation root seed must fit uint32")
    normalized = _normalize_entries(entries)
    used = {int(seed) for seed in forbidden_seeds if 0 <= int(seed) <= UINT32_MAX}
    schedule: list[D1BGenerationRequest] = []
    domain = "h8-d1b-smoke-generation-seed-v1" if smoke else "h8-d1b-formal-generation-seed-v1"
    for round_id in range(rounds):
        intended_role = FIT_ROLE if round_id % 2 == 0 else AUDIT_ROLE
        data_role = SMOKE_ROLE if smoke else intended_role
        for prompt_position, prompt_index in enumerate(
            _round_order(normalized, int(root_seed), round_id, smoke)
        ):
            entry = normalized[prompt_index]
            counter = 0
            while True:
                digest = _digest(domain, int(root_seed), round_id, entry["prompt_id"], counter)
                seed = int.from_bytes(digest[:4], "little", signed=False)
                if seed not in used:
                    break
                counter += 1
            used.add(seed)
            schedule.append(
                D1BGenerationRequest(
                    schedule_position=len(schedule),
                    round_id=round_id,
                    prompt_position_in_round=prompt_position,
                    prompt_index=prompt_index,
                    prompt_id=entry["prompt_id"],
                    prompt=entry["prompt"],
                    prompt_sha256=entry["prompt_sha256"],
                    seed=seed,
                    derivation_counter=counter,
                    seed_digest_sha256=digest.hex(),
                    data_role=data_role,
                    intended_bank_role=intended_role,
                    is_smoke=smoke,
                )
            )
    validate_schedule(schedule, smoke=smoke)
    return schedule


def build_formal_schedule(
    entries: Sequence[Mapping[str, Any]], root_seed: int, forbidden_seeds: Iterable[int] = ()
) -> list[D1BGenerationRequest]:
    return _build_schedule(
        entries, root_seed, rounds=FORMAL_ROUNDS, smoke=False, forbidden_seeds=forbidden_seeds
    )


def build_smoke_schedule(
    entries: Sequence[Mapping[str, Any]], root_seed: int, forbidden_seeds: Iterable[int] = ()
) -> list[D1BGenerationRequest]:
    return _build_schedule(
        entries, root_seed, rounds=SMOKE_ROUNDS, smoke=True, forbidden_seeds=forbidden_seeds
    )


def validate_schedule(schedule: Sequence[D1BGenerationRequest], *, smoke: bool) -> None:
    expected_rounds = SMOKE_ROUNDS if smoke else FORMAL_ROUNDS
    expected_total = SMOKE_TOTAL if smoke else FORMAL_TOTAL
    if len(schedule) != expected_total:
        raise ValueError("D1-B schedule length mismatch")
    if [request.schedule_position for request in schedule] != list(range(expected_total)):
        raise ValueError("D1-B schedule positions are not contiguous")
    seeds = [request.seed for request in schedule]
    if len(set(seeds)) != expected_total or any(not 0 <= seed <= UINT32_MAX for seed in seeds):
        raise ValueError("D1-B seeds must be unique uint32 values")
    if len({request.response_id for request in schedule}) != expected_total:
        raise ValueError("D1-B response IDs are not unique")
    for round_id in range(expected_rounds):
        rows = [request for request in schedule if request.round_id == round_id]
        if len(rows) != PROMPT_COUNT or len({row.prompt_id for row in rows}) != PROMPT_COUNT:
            raise ValueError("Every D1-B round must contain all prompts exactly once")
        if [row.prompt_position_in_round for row in rows] != list(range(PROMPT_COUNT)):
            raise ValueError("Prompt positions within D1-B round are invalid")
        intended = FIT_ROLE if round_id % 2 == 0 else AUDIT_ROLE
        if any(row.intended_bank_role != intended for row in rows):
            raise ValueError("D1-B role does not match even/odd round rule")
        expected_data_role = SMOKE_ROLE if smoke else intended
        if any(row.data_role != expected_data_role or row.is_smoke != smoke for row in rows):
            raise ValueError("D1-B data role/smoke flag mismatch")
    per_prompt = Counter(request.prompt_id for request in schedule)
    if len(per_prompt) != PROMPT_COUNT or set(per_prompt.values()) != {expected_rounds}:
        raise ValueError("D1-B per-prompt counts are invalid")
    if not smoke:
        fit = Counter(request.prompt_id for request in schedule if request.data_role == FIT_ROLE)
        audit = Counter(request.prompt_id for request in schedule if request.data_role == AUDIT_ROLE)
        if set(fit.values()) != {100} or set(audit.values()) != {100}:
            raise ValueError("Every prompt must have 100 fit and 100 audit requests")


def make_manifest_payload(
    schedule: Sequence[D1BGenerationRequest],
    *,
    root_seed: int,
    frozen_identity: Mapping[str, Any],
    source_fingerprint_sha256: str,
    repository_seed_sources: Sequence[Mapping[str, Any]],
    forbidden_seed_count: int,
    cpu_resampling_root_seed_reserved: int,
) -> dict[str, Any]:
    smoke = bool(schedule[0].is_smoke)
    validate_schedule(schedule, smoke=smoke)
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_kind": "smoke_only" if smoke else "formal_2400_not_yet_authorized",
        "formal_sampling_authorized_at_manifest_freeze": False,
        "generation_root_seed": int(root_seed),
        "generation_seed_algorithm": "sha256_domain_separated_uint32_rejection_v1",
        "generation_seed_byte_order": "little_endian_first_4_digest_bytes",
        "collision_resolution": "increment_counter_until_not_forbidden_or_used",
        "prompt_order_algorithm": "sha256_per_round_prompt_sort_v1",
        "cpu_resampling_seed_namespace": {
            "root_seed_reserved": int(cpu_resampling_root_seed_reserved),
            "seeds_generated_in_d1b": False,
            "may_overlap_generation_seeds": False,
        },
        "bank_design": {
            "single_response_bank_not_four_structure_banks": True,
            "future_structures": ["r40_q10", "r40_q20", "r60_q10", "r60_q20"],
            "cross_structure_statistic_or_parameter_pooling_forbidden": True,
        },
        "prompt_count": PROMPT_COUNT,
        "round_count": SMOKE_ROUNDS if smoke else FORMAL_ROUNDS,
        "total_responses": SMOKE_TOTAL if smoke else FORMAL_TOTAL,
        "batch_size": 1,
        "roles": {
            "actual": [SMOKE_ROLE] if smoke else [FIT_ROLE, AUDIT_ROLE],
            "even_round_intended": FIT_ROLE,
            "odd_round_intended": AUDIT_ROLE,
            "fit_is_only_final_parameter_fit_role": True,
            "audit_is_stability_only": True,
        },
        "eligibility": {
            "formal_reference": False,
            "heldout": False,
            "target": False,
            "attack": False,
        },
        "source_fingerprint_sha256": source_fingerprint_sha256,
        "frozen_identity": dict(frozen_identity),
        "repository_seed_cross_check": {
            "forbidden_seed_count": forbidden_seed_count,
            "source_file_count": len(repository_seed_sources),
            "source_files": list(repository_seed_sources),
            "collision_count_after_resolution": sum(row.derivation_counter > 0 for row in schedule),
            "final_overlap_count": 0,
        },
        "schedule": [request.as_dict() for request in schedule],
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any], *, smoke: bool) -> dict[str, str]:
    payload_dict = dict(payload)
    payload_sha = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": SMOKE_MANIFEST_TYPE if smoke else FORMAL_MANIFEST_TYPE,
        "schema_version": MANIFEST_SCHEMA,
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
    file_sha = file_sha256(target)
    target.with_suffix(target.suffix + ".sha256").write_text(
        f"{file_sha}  {target.name}\n", encoding="ascii", newline="\n"
    )
    return {"file_sha256": file_sha, "payload_sha256": payload_sha}


def load_manifest(path: str | Path, *, smoke: bool) -> tuple[dict[str, Any], list[D1BGenerationRequest]]:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_type = SMOKE_MANIFEST_TYPE if smoke else FORMAL_MANIFEST_TYPE
    if envelope.get("artifact_type") != expected_type or envelope.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("D1-B manifest type/schema mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, dict) or canonical_sha256(payload) != envelope.get("payload_sha256"):
        raise ValueError("D1-B manifest payload hash mismatch")
    schedule = [
        D1BGenerationRequest(
            schedule_position=int(row["schedule_position"]),
            round_id=int(row["round_id"]),
            prompt_position_in_round=int(row["prompt_position_in_round"]),
            prompt_index=int(row["prompt_index"]),
            prompt_id=str(row["prompt_id"]),
            prompt="",
            prompt_sha256=str(row["prompt_sha256"]),
            seed=int(row["seed"]),
            derivation_counter=int(row["derivation_counter"]),
            seed_digest_sha256=str(row["seed_digest_sha256"]),
            data_role=str(row["data_role"]),
            intended_bank_role=str(row["intended_bank_role"]),
            is_smoke=bool(row["is_smoke"]),
        )
        for row in payload["schedule"]
    ]
    validate_schedule(schedule, smoke=smoke)
    return payload, schedule


def audit_schedule(
    schedule: Sequence[D1BGenerationRequest],
    *,
    smoke: bool,
    forbidden_seeds: Iterable[int],
) -> dict[str, Any]:
    validate_schedule(schedule, smoke=smoke)
    seeds = {row.seed for row in schedule}
    forbidden = {int(seed) for seed in forbidden_seeds if 0 <= int(seed) <= UINT32_MAX}
    roles = Counter(row.data_role for row in schedule)
    intended = Counter(row.intended_bank_role for row in schedule)
    return {
        "status": "PASS" if not seeds & forbidden else "FAIL",
        "smoke": smoke,
        "total_responses": len(schedule),
        "round_count": len({row.round_id for row in schedule}),
        "prompt_count": len({row.prompt_id for row in schedule}),
        "per_prompt_counts": dict(sorted(Counter(row.prompt_id for row in schedule).items())),
        "actual_role_counts": dict(sorted(roles.items())),
        "intended_role_counts": dict(sorted(intended.items())),
        "unique_seed_count": len(seeds),
        "unique_response_id_count": len({row.response_id for row in schedule}),
        "uint32_seed_gate": all(0 <= seed <= UINT32_MAX for seed in seeds),
        "forbidden_seed_overlap_count": len(seeds & forbidden),
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "seed_set_sha256": canonical_sha256(sorted(seeds)),
        "schedule_sha256": canonical_sha256([row.as_dict() for row in schedule]),
        "formal_reference_eligible_count": 0,
        "heldout_eligible_count": 0,
        "target_eligible_count": 0,
        "attack_eligible_count": 0,
        "generation_started": False,
    }
