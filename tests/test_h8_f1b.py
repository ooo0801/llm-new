from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from llm_integrity.h8_f1a import load_canonical_envelope
from llm_integrity.h8_f1b import (
    BASE_PARTITION_ID,
    FORMAL_TOTAL,
    audit_formal_records,
    partition_requests,
    read_jsonl_strict,
    validate_frozen_inputs,
    validate_response_record,
)


ROOT = Path(__file__).resolve().parents[1]
AUTH_PATH = ROOT / "configs/h8_f1b_formal_final_confirmation_sampling.yaml"


def auth() -> dict:
    return yaml.safe_load(AUTH_PATH.read_text(encoding="utf-8"))


def envelope(config: dict, key: str, artifact_type: str) -> dict:
    frozen = config["frozen_inputs"]
    return load_canonical_envelope(
        ROOT / frozen[key],
        expected_artifact_type=artifact_type,
        expected_file_sha256=frozen[f"{key}_sha256"],
        expected_payload_sha256=frozen[f"{key}_payload_sha256"],
    )


@pytest.fixture(scope="module")
def frozen() -> dict:
    config = auth()
    smoke_index = json.loads(
        (ROOT / config["frozen_inputs"]["f1a_preflight_index"]).read_text(encoding="utf-8")
    )
    smoke_info = smoke_index["artifacts"]["smoke_manifest"]
    smoke = load_canonical_envelope(
        ROOT
        / "reproducibility/h8_qwen32b_final_confirmation_20260821/f1a/F1A_TINY_SMOKE_MANIFEST.json",
        expected_artifact_type="h8_f1a_tiny_smoke_manifest",
        expected_file_sha256=smoke_info["file_sha256"],
        expected_payload_sha256=smoke_info["payload_sha256"],
    )
    return {
        "generation": envelope(config, "generation_manifest", "h8_f1a_final_generation_manifest"),
        "attack": envelope(config, "attack_manifest", "h8_f1a_fresh_heldout_attack_endpoint_manifest"),
        "permutation": envelope(config, "permutation_manifest", "h8_f1a_final_global_permutation_seed_manifest"),
        "freshness": envelope(config, "freshness_audit", "h8_f1a_freshness_audit"),
        "historical": envelope(config, "historical_identity_index", "h8_f1a_historical_identity_index"),
        "smoke": smoke,
    }


def valid_record(request: dict, frozen_sha: str, materialization_sha: str) -> dict:
    return {
        **request,
        "formal_final_confirmation_eligible": True,
        "eligible_for_measurement_parameter_fit": False,
        "eligible_for_score_parameter_fit": False,
        "eligible_for_detector_selection": False,
        "formal_detector_statistics_computed": False,
        "frozen_identity_sha256": frozen_sha,
        "materialization_identity_sha256": materialization_sha,
        "attempt_records": [{"attempt_index": 0, "seed": request["generation_seed"], "status": "success"}],
        "rendered_prompt_sha256": "a" * 64,
        "input_token_ids": [1, 2],
        "input_token_ids_sha256": "b" * 64,
        "completion_token_ids": [151645],
        "completion_token_ids_sha256": "c" * 64,
        "generated_token_count": 1,
        "stop_reason": "eos_token",
        "legal_first_token_eos": True,
        "raw_response": "",
        "provenance_sha256": "d" * 64,
    }


def test_frozen_f1a_inputs_are_reproducible_and_disjoint(frozen: dict) -> None:
    result = validate_frozen_inputs(**frozen)
    assert result["status"] == "PASS"
    assert result["formal_response_count"] == FORMAL_TOTAL == 12720
    assert result["partition_count"] == 41
    assert result["base_partition_count"] == 7920
    assert result["attack_partition_count"] == 40
    assert result["permutation_seed_count"] == 100 * 999 * 12
    assert result["permutation_cross_namespace_overlap_count"] == 0
    assert result["single_shared_reference_bank"] is True


def test_partitioning_is_exact_and_attack_balanced(frozen: dict) -> None:
    partitions = partition_requests(frozen["generation"]["requests"])
    assert list(partitions)[0] == BASE_PARTITION_ID
    assert len(partitions[BASE_PARTITION_ID]) == 7920
    assert len(partitions) == 41
    assert {len(rows) for key, rows in partitions.items() if key != BASE_PARTITION_ID} == {120}
    assert all(
        [row["schedule_position"] for row in rows]
        == sorted(row["schedule_position"] for row in rows)
        for rows in partitions.values()
    )


def test_response_validation_preserves_first_token_eos_and_same_seed_retry(frozen: dict) -> None:
    request = frozen["generation"]["requests"][0]
    record = valid_record(request, "e" * 64, "f" * 64)
    record["attempt_records"] = [
        {"attempt_index": 0, "seed": request["generation_seed"], "status": "technical_failure"},
        {"attempt_index": 1, "seed": request["generation_seed"], "status": "success"},
    ]
    validate_response_record(
        record,
        request,
        frozen_identity_sha256="e" * 64,
        expected_materialization_identity_sha256="f" * 64,
    )
    changed = copy.deepcopy(record)
    changed["attempt_records"][0]["seed"] += 1
    with pytest.raises(ValueError, match="retry changed"):
        validate_response_record(
            changed,
            request,
            frozen_identity_sha256="e" * 64,
            expected_materialization_identity_sha256="f" * 64,
        )
    changed = copy.deepcopy(record)
    changed["completion_token_ids"] = [151645, 42]
    changed["generated_token_count"] = 2
    with pytest.raises(ValueError, match="first-token EOS"):
        validate_response_record(
            changed,
            request,
            frozen_identity_sha256="e" * 64,
            expected_materialization_identity_sha256="f" * 64,
        )


def test_full_integrity_audit_enforces_all_counts_without_detector_statistics(frozen: dict) -> None:
    requests = frozen["generation"]["requests"]
    identities = {BASE_PARTITION_ID: "0" * 64}
    identities.update({row["attack_instance_id"]: row["planned_artifact_identity_sha256"] for row in frozen["attack"]["endpoints"]})
    records = []
    for request in requests:
        partition = request.get("attack_instance_id") or BASE_PARTITION_ID
        records.append(valid_record(request, "9" * 64, identities[partition]))
    result = audit_formal_records(
        records,
        requests,
        frozen_identity_sha256="9" * 64,
        endpoint_materialization_identities=identities,
    )
    assert result["status"] == "PASS"
    assert result["total_response_count"] == 12720
    assert result["intact_unit_count"] == 60
    assert result["attack_endpoint_count"] == 40
    assert result["materialization_identity_count"] == 40
    assert result["formal_detector_statistics_computed"] is False


def test_strict_jsonl_rejects_partial_trailing_record(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_text('{"response_id":"x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="partial trailing line"):
        read_jsonl_strict(path)


def test_authorization_is_sampling_only_and_runner_has_no_performance_path() -> None:
    config = auth()
    assert config["formal_final_confirmation_sampling_authorized"] is True
    assert config["final_performance_confirmation_authorized"] is False
    assert all(config["forbidden_operations"].values())
    source = (ROOT / config["implementation"]["runner"]).read_text(encoding="utf-8")
    assert "final_global_permutation_test" not in source
    assert "summarize_final_decisions" not in source

