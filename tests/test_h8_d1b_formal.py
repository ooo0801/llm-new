from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from llm_integrity.h8_d1b_formal import (
    GENESIS_CHAIN_SHA256,
    audit_formal_records,
    request_sha256,
    seal_response_record,
    validate_attempt_events,
    verify_completed_prefix,
)
from llm_integrity.h8_d1b_sampling import AUDIT_ROLE, FIT_ROLE, build_formal_schedule
from llm_integrity.h8_sampling import sha256_text
from run_h8_d1b1_formal_sampling import run_attempts


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA = "e6db0eda9d373c7d62f2a2efeacf4ca127476a0ab91d104665bcaa7354577002"
PROVENANCE_SHA = "a" * 64
AUTH_PATH = ROOT / "configs" / "h8_d1b1_formal_score_calibration_sampling.yaml"


def entries() -> list[dict]:
    result = []
    for index in range(12):
        prompt = f"prompt {index}"
        result.append(
            {
                "prompt_id": f"p{index:02d}",
                "prompt": prompt,
                "metadata": {"prompt_sha256": sha256_text(prompt)},
            }
        )
    return result


def base_record(request, previous_chain: str) -> dict:
    record = {
        "response_id": request.response_id,
        "schedule_position": request.schedule_position,
        "round_id": request.round_id,
        "prompt_position_in_round": request.prompt_position_in_round,
        "prompt_index": request.prompt_index,
        "prompt_id": request.prompt_id,
        "prompt_sha256": request.prompt_sha256,
        "seed": request.seed,
        "seed_digest_sha256": request.seed_digest_sha256,
        "data_role": request.data_role,
        "manifest_request_sha256": request_sha256(request),
        "formal_manifest_file_sha256": MANIFEST_SHA,
        "frozen_provenance_sha256": PROVENANCE_SHA,
        "eligible_for_score_calibration_fit": request.data_role == FIT_ROLE,
        "eligible_for_score_stability_audit": request.data_role == AUDIT_ROLE,
        "eligible_for_formal_reference": False,
        "eligible_for_heldout_evaluation": False,
        "eligible_for_target_evaluation": False,
        "eligible_for_attack_evaluation": False,
        "raw_response": "x",
        "stop_reason": "eos",
        "legal_first_token_eos": False,
        "response_token_count_including_eos": 5,
        "retry_seed_changed": False,
    }
    return seal_response_record(record, previous_chain)


def prefix(count: int = 3):
    schedule = build_formal_schedule(entries(), 2026081903)
    records = []
    chain = GENESIS_CHAIN_SHA256
    events = []
    for request in schedule[:count]:
        record = base_record(request, chain)
        records.append(record)
        chain = record["record_chain_sha256"]
        events.append(
            {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "attempt_index": 0,
                "seed": request.seed,
                "status": "success",
            }
        )
    return schedule, records, events, chain


def test_record_chain_and_prefix_are_fail_closed() -> None:
    schedule, records, _, chain = prefix()
    assert (
        verify_completed_prefix(
            records,
            schedule,
            formal_manifest_sha256=MANIFEST_SHA,
            frozen_provenance_sha256=PROVENANCE_SHA,
        )
        == chain
    )
    changed = copy.deepcopy(records)
    changed[1]["raw_response"] = "tampered"
    with pytest.raises(ValueError, match="payload hash"):
        verify_completed_prefix(
            changed,
            schedule,
            formal_manifest_sha256=MANIFEST_SHA,
            frozen_provenance_sha256=PROVENANCE_SHA,
        )


@pytest.mark.parametrize("key,value", [("seed", 1), ("schedule_position", 9), ("data_role", AUDIT_ROLE)])
def test_manifest_fields_cannot_change_on_resume(key: str, value) -> None:
    schedule, records, _, _ = prefix(1)
    records[0][key] = value
    with pytest.raises(ValueError, match="prefix mismatch"):
        verify_completed_prefix(records, schedule, formal_manifest_sha256=MANIFEST_SHA)


def test_attempt_events_allow_only_completed_prefix_and_next_failures() -> None:
    schedule, records, events, _ = prefix(2)
    next_request = schedule[2]
    events.append(
        {
            "response_id": next_request.response_id,
            "schedule_position": next_request.schedule_position,
            "attempt_index": 0,
            "seed": next_request.seed,
            "status": "technical_failure",
        }
    )
    grouped = validate_attempt_events(events, schedule, len(records), max_attempts=2)
    assert len(grouped[next_request.response_id]) == 1
    events[-1]["status"] = "success"
    with pytest.raises(ValueError, match="Orphan success"):
        validate_attempt_events(events, schedule, len(records), max_attempts=2)


def test_events_cannot_change_seed_or_jump_ahead() -> None:
    schedule, records, events, _ = prefix(1)
    event = {
        "response_id": schedule[2].response_id,
        "schedule_position": schedule[2].schedule_position,
        "attempt_index": 0,
        "seed": schedule[2].seed,
        "status": "technical_failure",
    }
    with pytest.raises(ValueError, match="beyond"):
        validate_attempt_events([*events, event], schedule, len(records), max_attempts=2)
    events[0]["seed"] += 1
    with pytest.raises(ValueError, match="changed"):
        validate_attempt_events(events, schedule, len(records), max_attempts=2)


def test_same_seed_retry_budget_is_cumulative_across_resume() -> None:
    request = build_formal_schedule(entries(), 2026081903)[0]
    failures = [
        {"response_id": request.response_id, "seed": request.seed, "status": "technical_failure"},
        {"response_id": request.response_id, "seed": request.seed, "status": "technical_failure"},
    ]
    with pytest.raises(RuntimeError, match="budget exhausted"):
        run_attempts(None, request, {}, None, failures, max_attempts=2)
    with pytest.raises(ValueError, match="success attempt"):
        run_attempts(
            None,
            request,
            {},
            None,
            [{"response_id": request.response_id, "seed": request.seed, "status": "success"}],
            max_attempts=2,
        )


def test_exact_role_schedule_and_seed_isolation() -> None:
    schedule = build_formal_schedule(entries(), 2026081903)
    assert len(schedule) == 2400
    assert sum(row.data_role == FIT_ROLE for row in schedule) == 1200
    assert sum(row.data_role == AUDIT_ROLE for row in schedule) == 1200
    for prompt_id in {row.prompt_id for row in schedule}:
        assert sum(row.prompt_id == prompt_id and row.data_role == FIT_ROLE for row in schedule) == 100
        assert sum(row.prompt_id == prompt_id and row.data_role == AUDIT_ROLE for row in schedule) == 100


def test_audit_detects_missing_record_and_smoke_overlap() -> None:
    schedule, records, events, _ = prefix(2)
    result = audit_formal_records(
        records,
        schedule,
        events,
        formal_manifest_sha256=MANIFEST_SHA,
        smoke_seeds={records[0]["seed"]},
        frozen_provenance_sha256=PROVENANCE_SHA,
        required_fields={"response_id", "seed"},
        max_attempts=2,
    )
    assert result["status"] == "FAIL"
    assert result["formal_smoke_seed_overlap_count"] == 1


def test_runner_has_no_score_fit_or_selection_calls() -> None:
    source = (ROOT / "scripts" / "run_h8_d1b1_formal_sampling.py").read_text(encoding="utf-8")
    forbidden_calls = (
        "fit_score_calibration(",
        "fit_robust_score_parameters(",
        "select_sample_size(",
        "select_top_r(",
        "h8_mmd2_unbiased_unequal(",
    )
    assert not any(call in source for call in forbidden_calls)
    assert '"HF_HUB_OFFLINE": "1"' in source
    assert '"TRANSFORMERS_OFFLINE": "1"' in source


def test_d1b0_preflight_authorization_remains_false() -> None:
    import yaml

    d1b0 = yaml.safe_load((ROOT / "configs" / "h8_d1b0_score_calibration_sampling.yaml").read_text(encoding="utf-8"))
    assert d1b0["formal_sampling_authorized"] is False
    assert d1b0["authorization_status"] == "user_approved_preflight_only"


def test_d1b1_authorization_is_independent_and_binds_only_final_pass() -> None:
    import hashlib
    import yaml

    auth = yaml.safe_load(AUTH_PATH.read_text(encoding="utf-8"))
    assert auth["phase"] == "H8_D1B1_FORMAL_SCORE_CALIBRATION_SAMPLING"
    assert auth["formal_sampling_authorized"] is True
    assert auth["sampling_code_commit"] == "19b3e447a73b0e933ddefeac1ccba7b77f8c4bfc"
    assert "failed_before_generation" not in auth["frozen_inputs"]["d1b0_pass_report"]
    assert auth["frozen_inputs"]["formal_manifest_sha256"] == MANIFEST_SHA
    assert auth["sampling"]["resume_policy"] == "exact_validated_schedule_prefix_only"
    assert auth["sampling"]["success_record_policy"] == "immutable_never_regenerate"
    for path_key, hash_key in (
        ("d1b0_pass_report", "d1b0_pass_report_sha256"),
        ("formal_manifest", "formal_manifest_sha256"),
        ("d1a_report", "d1a_report_sha256"),
        ("h8_config", "h8_config_sha256"),
        ("fingerprint", "fingerprint_sha256"),
        ("protocol", "protocol_sha256"),
    ):
        path = ROOT / auth["frozen_inputs"][path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == auth["frozen_inputs"][hash_key]
    assert all(auth["forbidden_operations"].values())
