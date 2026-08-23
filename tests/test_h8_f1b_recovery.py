from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from llm_integrity.h8_f1b_recovery import (
    NORMALIZED_FIELDS,
    json_roundtrip_normalize,
    load_recovery_authorization,
    normalize_identity_payload,
    sha256_prefix_lines,
    validate_attempt_alignment,
    validate_normalized_materialization_match,
)
from llm_integrity.h8_precalibration import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_AUTH_PATH = ROOT / "configs/h8_f1b_tuple_list_recovery.yaml"


def live_payload() -> dict:
    return {
        "schema_version": "x",
        "materialization": {
            "quantization_load_report": {
                "notes": (),
                "quantized_module_examples": ("a", "b"),
                "quantized_linear_modules": 448,
            },
            "state": {"sha256": "f" * 64},
        },
    }


def test_json_normalization_repairs_only_container_representation() -> None:
    payload = live_payload()
    identity = canonical_sha256(payload)
    sealed = {
        "materialization_identity_sha256": identity,
        "payload": json_roundtrip_normalize(payload),
    }
    assert sealed["payload"] != payload
    normalized = validate_normalized_materialization_match(sealed, payload, identity)
    assert normalized == sealed["payload"]
    assert canonical_sha256(normalized) == identity
    assert isinstance(normalized["materialization"]["quantization_load_report"]["notes"], list)


def test_normalization_fails_closed_on_semantic_change() -> None:
    payload = live_payload()
    identity = canonical_sha256(payload)
    sealed = {
        "materialization_identity_sha256": identity,
        "payload": json_roundtrip_normalize(payload),
    }
    changed = copy.deepcopy(payload)
    changed["materialization"]["quantization_load_report"]["quantized_linear_modules"] += 1
    changed_identity = canonical_sha256(changed)
    with pytest.raises(ValueError, match="identity SHA256 mismatch"):
        validate_normalized_materialization_match(sealed, changed, changed_identity)
    with pytest.raises(ValueError, match="normalization changed"):
        normalize_identity_payload(changed, identity)


def test_prefix_hash_preserves_exact_immutable_jsonl_bytes(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    prefix = b'{"x":1}\n{"x":2}\n'
    path.write_bytes(prefix + b'{"x":3}\n')
    assert sha256_prefix_lines(path, 2) == hashlib.sha256(prefix).hexdigest()
    assert sha256_prefix_lines(path, 0) is None
    with pytest.raises(ValueError, match="Expected 4 prefix lines"):
        sha256_prefix_lines(path, 4)


def test_attempt_alignment_allows_same_seed_technical_retry() -> None:
    records = [
        {"response_id": "a", "schedule_position": 12583, "generation_seed": 11},
        {"response_id": "b", "schedule_position": 12584, "generation_seed": 12},
    ]
    attempts = [
        {
            "response_id": "a",
            "schedule_position": 12583,
            "seed": 11,
            "status": "technical_failure",
        },
        {
            "response_id": "a",
            "schedule_position": 12583,
            "seed": 11,
            "status": "success_record_fsynced",
        },
        {
            "response_id": "b",
            "schedule_position": 12584,
            "seed": 12,
            "status": "success_record_fsynced",
        },
    ]
    result = validate_attempt_alignment(records, attempts)
    assert result == {
        "success_attempt_event_count": 2,
        "technical_failure_event_count": 1,
        "total_attempt_event_count": 3,
    }
    changed = copy.deepcopy(attempts)
    changed[0]["seed"] = 99
    with pytest.raises(ValueError, match="changed the frozen generation seed"):
        validate_attempt_alignment(records, changed)


def test_authorization_gate_is_bounded_and_sampling_only(tmp_path: Path) -> None:
    authorization = {
        "phase": "H8_F1B_TUPLE_LIST_NORMALIZATION_RECOVERY",
        "authorization_status": "user_approved_bounded_recovery_only",
        "formal_sampling_recovery_authorized": True,
        "final_performance_confirmation_authorized": False,
        "recovery_bounds": {
            "allowed_schedule_positions": [12583, 12719],
            "allowed_partitions": [
                "h8f1_final_quantization_09_seed2646328052",
                "h8f1_final_quantization_10_seed0331532823",
            ],
            "normalized_fields": list(NORMALIZED_FIELDS),
        },
        "forbidden_operations": {
            "feature_extraction": True,
            "mmd": True,
            "score": True,
            "permutation": True,
            "f1c": True,
        },
    }
    path = tmp_path / "auth.yaml"
    path.write_text(yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8")
    loaded = load_recovery_authorization(path)
    assert loaded["recovery_bounds"]["allowed_schedule_positions"] == [12583, 12719]

    authorization["recovery_bounds"]["allowed_schedule_positions"] = [12582, 12719]
    path.write_text(yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8")
    with pytest.raises(PermissionError, match="schedule bounds changed"):
        load_recovery_authorization(path)


def test_recovery_runner_has_no_detector_performance_path() -> None:
    authorization = load_recovery_authorization(RECOVERY_AUTH_PATH)
    assert authorization["formal_sampling_recovery_authorized"] is True
    assert authorization["final_performance_confirmation_authorized"] is False
    assert all(authorization["forbidden_operations"].values())
    source = (ROOT / "scripts/run_h8_f1b_tuple_list_recovery.py").read_text(encoding="utf-8")
    assert "final_global_permutation_test" not in source
    assert "summarize_final_decisions" not in source
    assert "run_f1c" not in source.lower()
