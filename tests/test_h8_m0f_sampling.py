from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from llm_integrity.h8_m0f_sampling import (
    M0F_DATA_ROLE,
    M0F_TOTAL_RESPONSES,
    audit_calibration_records,
    build_calibration_schedule,
    load_manifest,
    make_manifest_payload,
    validate_calibration_schedule,
    write_manifest,
)
from llm_integrity.h8_sampling import sha256_text
from run_h8_m0f_calibration_sampling import run_attempts


ROOT = Path(__file__).resolve().parents[1]
AUTH_PATH = ROOT / "configs" / "h8_m0f_calibration_sampling.yaml"


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


def test_schedule_is_deterministic_round_balanced_and_unique() -> None:
    first = build_calibration_schedule(entries(), 2026081801, range(100, 150))
    second = build_calibration_schedule(entries(), 2026081801, range(100, 150))
    assert [row.as_dict() for row in first] == [row.as_dict() for row in second]
    assert len(first) == M0F_TOTAL_RESPONSES
    assert len({row.seed for row in first}) == M0F_TOTAL_RESPONSES
    assert len({row.response_id for row in first}) == M0F_TOTAL_RESPONSES
    assert not ({row.seed for row in first} & set(range(100, 150)))
    assert all(0 <= row.seed <= 2**32 - 1 for row in first)
    for replicate in range(100):
        rows = [row for row in first if row.replicate_id == replicate]
        assert len(rows) == 12
        assert len({row.prompt_id for row in rows}) == 12


def test_seed_collision_is_resolved_without_changing_scope() -> None:
    original = build_calibration_schedule(entries(), 2026081801)
    forbidden = {original[0].seed}
    rebuilt = build_calibration_schedule(entries(), 2026081801, forbidden)
    assert not ({row.seed for row in rebuilt} & forbidden)
    assert rebuilt[0].derivation_counter >= 1
    validate_calibration_schedule(rebuilt)


@pytest.mark.parametrize("root_seed", [-1, 2**32])
def test_root_seed_must_fit_uint32(root_seed: int) -> None:
    with pytest.raises(ValueError, match="root_seed"):
        build_calibration_schedule(entries(), root_seed)


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    schedule = build_calibration_schedule(entries(), 2026081801, {10, 20})
    payload = make_manifest_payload(
        schedule,
        2026081801,
        "a" * 64,
        {"identity": "frozen"},
        [{"path": "x.json", "content_sha256": "b" * 64, "valid_uint32_seed_values": 2}],
        2,
    )
    path = tmp_path / "manifest.json"
    info = write_manifest(path, payload)
    loaded_payload, loaded_schedule = load_manifest(path)
    assert loaded_payload == payload
    assert len(loaded_schedule) == 1200
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info["file_sha256"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["root_seed"] += 1
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_manifest(path)


def test_integrity_audit_passes_only_exact_schedule() -> None:
    schedule = build_calibration_schedule(entries(), 2026081801)
    required = ("response_id", "seed", "input_token_count", "response_token_count_including_eos", "attempt_count")
    records = []
    for request in schedule:
        records.append(
            {
                "response_id": request.response_id,
                "schedule_position": request.schedule_position,
                "replicate_id": request.replicate_id,
                "prompt_id": request.prompt_id,
                "seed": request.seed,
                "data_role": M0F_DATA_ROLE,
                "eligible_for_formal_reference": False,
                "eligible_for_heldout_evaluation": False,
                "eligible_for_attack_evaluation": False,
                "raw_response": "x",
                "stop_reason": "eos",
                "legal_first_token_eos": False,
                "input_token_count": 20,
                "response_token_count_including_eos": 5,
                "attempt_count": 1,
            }
        )
    audit = audit_calibration_records(records, schedule, required)
    assert audit["status"] == "PASS"
    assert audit["total_responses"] == 1200
    assert set(audit["per_prompt_counts"].values()) == {100}
    bad = records[:-1]
    assert audit_calibration_records(bad, schedule, required)["status"] == "FAIL"


def test_authorization_binds_preflight_and_forbids_post_sampling_operations() -> None:
    auth = yaml.safe_load(AUTH_PATH.read_text(encoding="utf-8"))
    assert auth["preflight_base_commit"] == "919b445f38a9ddf5e13423d5c1fb3aad6eebea19"
    assert auth["sampling"]["data_role"] == M0F_DATA_ROLE
    assert auth["sampling"]["total_responses"] == 1200
    assert auth["sampling"]["batch_size"] == 1
    assert auth["sampling"]["seed_algorithm"] == "sha256_domain_separated_uint32_rejection_v1"
    assert auth["sampling"]["order_algorithm"] == "sha256_per_replicate_prompt_sort_v1"
    assert all(value is True for value in auth["forbidden_operations"].values())
    for key in ("h8_config", "protocol", "preflight_archive"):
        path = ROOT / auth["frozen_inputs"][key]
        expected = auth["frozen_inputs"][f"{key}_sha256"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_sampling_runner_does_not_call_forbidden_statistical_stages() -> None:
    source = (ROOT / "scripts" / "run_h8_m0f_calibration_sampling.py").read_text(encoding="utf-8")
    forbidden_calls = (
        "fit_family_balanced_scaler(",
        "h8_median_positive_pairwise_distance(",
        "h8_bandwidth_stability(",
        "h8_mmd2_unbiased_unequal(",
        "h8_mmd2_biased(",
    )
    assert not any(call in source for call in forbidden_calls)


def test_same_seed_attempt_budget_is_cumulative_across_resume() -> None:
    request = build_calibration_schedule(entries(), 2026081801)[0]
    failures = [
        {"response_id": request.response_id, "seed": request.seed, "status": "technical_failure"},
        {"response_id": request.response_id, "seed": request.seed, "status": "technical_failure"},
    ]
    with pytest.raises(RuntimeError, match="budget already exhausted"):
        run_attempts(None, request, {}, None, failures, max_attempts=2)
    with pytest.raises(ValueError, match="success attempt event"):
        run_attempts(
            None,
            request,
            {},
            None,
            [{"response_id": request.response_id, "seed": request.seed, "status": "success"}],
            max_attempts=2,
        )
