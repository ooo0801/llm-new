from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm_integrity.h8_d2a import ATTACK_ROLE, INTACT_TARGET_ROLE, REFERENCE_ROLE
from llm_integrity.h8_d2b1 import (
    GENESIS_CHAIN,
    audit_formal_records,
    combine_and_validate_schedules,
    seal_response_record,
    validate_attempt_events,
    validate_nested_membership_against_records,
    verify_completed_prefix,
)


ROOT = Path(__file__).resolve().parents[1]
D2A = ROOT / "reproducibility/h8_qwen32b_detector_development_20260820/d2a"
MANIFEST_SET_SHA = "a" * 64
PROVENANCE_SHA = "b" * 64


def envelope_payload(name: str) -> dict:
    return json.loads((D2A / name).read_text(encoding="utf-8"))["payload"]


def schedule():
    return combine_and_validate_schedules(
        envelope_payload("D2A_DEVELOPMENT_REFERENCE_GENERATION_MANIFEST.json")["schedule"],
        envelope_payload("D2A_DEVELOPMENT_INTACT_TARGET_GENERATION_MANIFEST.json")["schedule"],
        envelope_payload("D2A_DEVELOPMENT_ATTACK_GENERATION_MANIFEST.json")["schedule"],
    )


def expected_materialization(rows) -> dict[str, dict[str, str]]:
    return {
        str(row.attack_instance_id): {
            "d2b0_materialization_file_sha256": "c" * 64,
            "d2b0_materialization_payload_sha256": "d" * 64,
            "d2b0_attack_configuration_sha256": "e" * 64,
        }
        for row in rows
        if row.attack_instance_id is not None
    }


def response_record(request, previous_chain: str, materialization: dict[str, dict[str, str]]) -> dict:
    binding = materialization.get(
        str(request.attack_instance_id),
        {
            "d2b0_materialization_file_sha256": None,
            "d2b0_materialization_payload_sha256": None,
            "d2b0_attack_configuration_sha256": None,
        },
    )
    record = {
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
        "generation_manifest_set_sha256": MANIFEST_SET_SHA,
        "frozen_provenance_sha256": PROVENANCE_SHA,
        "eligible_for_development_reference": request.data_role == REFERENCE_ROLE,
        "eligible_for_development_intact_target": request.data_role == INTACT_TARGET_ROLE,
        "eligible_for_development_attack": request.data_role == ATTACK_ROLE,
        "eligible_for_future_formal_reference": False,
        "eligible_for_future_final_heldout": False,
        "eligible_for_future_final_confirmation": False,
        "eligible_for_score_parameter_fit": False,
        "eligible_for_measurement_parameter_fit": False,
        "retry_seed_changed": False,
        "stop_reason": "eos",
        "legal_first_token_eos": False,
        "raw_response": "ok",
        "response_token_count_including_eos": 2,
        **binding,
    }
    return seal_response_record(record, previous_chain)


def test_frozen_generation_manifests_form_exact_global_schedule() -> None:
    rows = schedule()
    assert len(rows) == 3840
    assert [row.schedule_position for row in rows] == list(range(3840))
    assert sum(row.data_role == REFERENCE_ROLE for row in rows) == 720
    assert sum(row.data_role == INTACT_TARGET_ROLE for row in rows) == 1200
    assert sum(row.data_role == ATTACK_ROLE for row in rows) == 1920
    assert len({row.response_id for row in rows}) == 3840
    assert len({row.generation_seed for row in rows}) == 3840


def test_manifest_tampering_is_rejected() -> None:
    reference = envelope_payload("D2A_DEVELOPMENT_REFERENCE_GENERATION_MANIFEST.json")["schedule"]
    intact = envelope_payload("D2A_DEVELOPMENT_INTACT_TARGET_GENERATION_MANIFEST.json")["schedule"]
    attack = envelope_payload("D2A_DEVELOPMENT_ATTACK_GENERATION_MANIFEST.json")["schedule"]
    changed = copy.deepcopy(reference)
    changed[1]["generation_seed"] = changed[0]["generation_seed"]
    with pytest.raises(ValueError, match="seeds"):
        combine_and_validate_schedules(changed, intact, attack)
    changed = copy.deepcopy(reference)
    changed[0]["eligible_for_future_formal_reference"] = True
    with pytest.raises(ValueError, match="eligibility"):
        combine_and_validate_schedules(changed, intact, attack)


def test_global_prefix_hash_chain_and_manifest_identity_fail_closed() -> None:
    rows = schedule()
    materialization = expected_materialization(rows)
    records = []
    chain = GENESIS_CHAIN
    for request in rows[:3]:
        record = response_record(request, chain, materialization)
        records.append(record)
        chain = record["record_chain_sha256"]
    assert (
        verify_completed_prefix(
            records,
            rows,
            generation_manifest_set_sha256=MANIFEST_SET_SHA,
            frozen_provenance_sha256=PROVENANCE_SHA,
        )
        == chain
    )
    changed = copy.deepcopy(records)
    changed[1]["raw_response"] = "tampered"
    with pytest.raises(ValueError, match="payload hash"):
        verify_completed_prefix(
            changed,
            rows,
            generation_manifest_set_sha256=MANIFEST_SET_SHA,
            frozen_provenance_sha256=PROVENANCE_SHA,
        )


def test_attempt_events_only_allow_same_seed_and_next_position() -> None:
    rows = schedule()
    events = [
        {
            "response_id": rows[0].response_id,
            "schedule_position": 0,
            "attempt_index": 0,
            "generation_seed": rows[0].generation_seed,
            "status": "success",
        },
        {
            "response_id": rows[1].response_id,
            "schedule_position": 1,
            "attempt_index": 0,
            "generation_seed": rows[1].generation_seed,
            "status": "technical_failure",
        },
    ]
    validate_attempt_events(events, rows, 1, max_attempts=2)
    changed = copy.deepcopy(events)
    changed[1]["generation_seed"] += 1
    with pytest.raises(ValueError, match="changed"):
        validate_attempt_events(changed, rows, 1, max_attempts=2)
    jumped = copy.deepcopy(events)
    jumped[1].update(
        response_id=rows[2].response_id,
        schedule_position=2,
        generation_seed=rows[2].generation_seed,
    )
    with pytest.raises(ValueError, match="beyond"):
        validate_attempt_events(jumped, rows, 1, max_attempts=2)


def test_nested_membership_matches_frozen_maximal_banks() -> None:
    rows = schedule()
    minimal = [
        {
            "response_id": row.response_id,
            "data_role": row.data_role,
            "prompt_id": row.prompt_id,
            "evaluation_unit_id": row.evaluation_unit_id,
        }
        for row in rows
    ]
    nested = envelope_payload("D2A_NESTED_SUBSET_MANIFEST.json")
    assert validate_nested_membership_against_records(nested, minimal) == []
    minimal[0]["response_id"] = "changed"
    assert validate_nested_membership_against_records(nested, minimal)


def test_full_integrity_audit_passes_exact_frozen_counts() -> None:
    rows = schedule()
    materialization = expected_materialization(rows)
    records = []
    events = []
    chain = GENESIS_CHAIN
    for request in rows:
        record = response_record(request, chain, materialization)
        records.append(record)
        chain = record["record_chain_sha256"]
        events.append(
            {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "attempt_index": 0,
                "generation_seed": request.generation_seed,
                "status": "success",
            }
        )
    audit = audit_formal_records(
        records,
        rows,
        events,
        generation_manifest_set_sha256=MANIFEST_SET_SHA,
        frozen_provenance_sha256=PROVENANCE_SHA,
        smoke_response_ids={"smoke"},
        smoke_seeds={1},
        nested_subset_payload=envelope_payload("D2A_NESTED_SUBSET_MANIFEST.json"),
        nested_subset_file_sha256="f" * 64,
        expected_materialization_by_endpoint=materialization,
        required_fields={"response_id", "generation_seed", "stop_reason"},
        max_attempts=2,
    )
    assert audit["status"] == "PASS"
    assert audit["total_responses"] == 3840
    assert audit["unique_response_id_count"] == 3840
    assert audit["unique_generation_seed_count"] == 3840
    assert audit["nested_membership_failure_count"] == 0
    assert audit["attack_materialization_binding_failure_count"] == 0


def test_runner_cannot_compute_or_select_detector() -> None:
    source = (ROOT / "scripts/run_h8_d2b1_formal_sampling.py").read_text(encoding="utf-8")
    forbidden_calls = (
        "global_permutation_test(",
        "development_global_permutation_test(",
        "evaluate_development_unit(",
        "select_development_configuration(",
        "top_r_sum(",
    )
    assert not any(call in source for call in forbidden_calls)
    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source
    assert '"development_comparison_performed": False' in source
    assert '"detector_statistics_computed": False' in source


def test_protocol_keeps_detector_unfrozen() -> None:
    protocol = (ROOT / "docs/H8_Qwen32B_D2B1_Formal_Development_Sampling_Protocol.md").read_text(
        encoding="utf-8"
    )
    assert "exactly 3,840" in protocol
    assert "detector=not_frozen" in protocol
    assert "must not compute MMD" in protocol
