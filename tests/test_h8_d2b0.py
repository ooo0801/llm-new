from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_integrity.h8_d2b0 import (
    D2B0_SCHEMA_VERSION,
    SMOKE_ROLE,
    audit_fresh_attack_provenance,
    audit_lora_training_isolation,
    build_attack_smoke_manifest,
    load_canonical_envelope,
    save_canonical_envelope,
)


ROOT = Path(__file__).resolve().parents[1]


def instances() -> list[dict[str, object]]:
    return [
        {
            "attack_instance_id": f"h8d2_{family}_{index}",
            "family": family,
            "materialization_seed": 3000 + offset * 10 + index,
            "configuration": {"type": family, "strength": offset + index / 10},
        }
        for offset, family in enumerate(("gaussian", "pruning", "lora", "quantization"))
        for index in range(2)
    ]


def entries() -> list[dict[str, object]]:
    return [
        {
            "prompt_id": f"prompt_{index:02d}",
            "prompt": f"protected prompt {index}",
            "reference_responses": [],
            "metadata": {"prompt_sha256": f"{index:064x}"},
        }
        for index in range(12)
    ]


def write_h6_manifest(path: Path) -> None:
    rows = [
        {
            "variant_id": f"h6_{family}_{index}",
            "family": {
                "gaussian": "gaussian_noise",
                "pruning": "structured_pruning",
                "lora": "finetuning",
                "quantization": "quantization",
            }[family],
            "seed": 1000 + offset * 10 + index,
            "configuration": {"type": "historical", "strength": offset + index / 10},
        }
        for offset, family in enumerate(("gaussian", "pruning", "lora", "quantization"))
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_freshness_audit_uses_explicit_h6_ids_configurations_and_seeds(tmp_path: Path) -> None:
    manifest = tmp_path / "h6.jsonl"
    write_h6_manifest(manifest)
    result = audit_fresh_attack_provenance(instances(), [manifest])
    assert result["status"] == "PASS"
    assert result["h6_explicit_id_provenance_available"] is True
    assert result["id_level_overlap_status"] == "verified_zero"
    assert result["materialization_seed_overlap"] == []
    assert all(row["fresh_by_configuration_seed"] for row in result["configuration_seed_comparisons"])


def test_freshness_audit_detects_same_quantization_instance(tmp_path: Path) -> None:
    d2 = instances()
    quant = d2[-1]
    manifest = tmp_path / "h6.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "variant_id": "h6_quant_same",
                "family": "quantization",
                "seed": quant["materialization_seed"],
                "configuration": quant["configuration"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = audit_fresh_attack_provenance(d2, [manifest])
    assert result["status"] == "FAIL"
    assert result["configuration_seed_comparisons"][-1]["quantization_same_deterministic_instance"] is True


def test_lora_training_isolation_passes_and_detects_protected_prompt() -> None:
    fingerprint = {"entries": entries()}
    training = [
        {"id": "train_1", "prompt": "isolated unique prompt", "expected_answer": "isolated-answer"}
    ]
    assert audit_lora_training_isolation(training, fingerprint, [])["status"] == "PASS"
    training[0]["prompt"] = "protected prompt 0"
    with pytest.raises(ValueError, match="overlaps"):
        audit_lora_training_isolation(training, fingerprint, [])


def test_smoke_manifest_has_eight_unique_nonformal_seeds() -> None:
    formal = set(range(100, 200))
    payload = build_attack_smoke_manifest(
        instances(), entries(), root_seed=99, forbidden_seeds={1, 2, 3}, formal_development_seeds=formal
    )
    requests = payload["requests"]
    seeds = {row["generation_seed"] for row in requests}
    assert payload["schema_version"] == D2B0_SCHEMA_VERSION
    assert len(requests) == len(seeds) == 8
    assert not seeds & formal
    assert all(row["data_role"] == SMOKE_ROLE for row in requests)
    assert all(row["formal_development_eligible"] is False for row in requests)
    assert all(row["nested_q10_q20_eligible"] is False for row in requests)


def test_canonical_artifact_fails_closed_after_tampering(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    info = save_canonical_envelope(path, "d2b0_test", {"value": 3})
    assert load_canonical_envelope(
        path,
        expected_artifact_type="d2b0_test",
        expected_file_sha256=info["file_sha256"],
        expected_payload_sha256=info["payload_sha256"],
    ) == {"value": 3}
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA256 mismatch"):
        load_canonical_envelope(
            path,
            expected_artifact_type="d2b0_test",
            expected_file_sha256=info["file_sha256"],
            expected_payload_sha256=info["payload_sha256"],
        )


def test_frozen_pre_smoke_artifacts_reverse_load_and_bind_revised_rule() -> None:
    directory = ROOT / "reproducibility/h8_qwen32b_detector_development_20260820/d2b0_authorization"
    index = json.loads((directory / "D2B0_PRE_SMOKE_SHA256_INDEX.json").read_text(encoding="utf-8"))
    bindings = {
        "revised_configuration_manifest": (
            "D2B0_REVISED_CONFIGURATION_MANIFEST.json",
            "h8_d2b0_revised_configuration_manifest",
        ),
        "freshness_preflight": (
            "D2B0_FRESH_ATTACK_PROVENANCE_PREFLIGHT.json",
            "h8_d2b0_fresh_attack_provenance_preflight",
        ),
        "lora_training_isolation": (
            "D2B0_LORA_TRAINING_DATA_ISOLATION_AUDIT.json",
            "h8_d2b0_lora_training_isolation_audit",
        ),
        "attack_smoke_manifest": (
            "D2B0_ATTACK_SMOKE_GENERATION_MANIFEST.json",
            "h8_d2b0_attack_smoke_generation_manifest",
        ),
    }
    payloads = {}
    for key, (filename, artifact_type) in bindings.items():
        payloads[key] = load_canonical_envelope(
            directory / filename,
            expected_artifact_type=artifact_type,
            expected_file_sha256=index[key]["file_sha256"],
            expected_payload_sha256=index[key]["payload_sha256"],
        )
    rule = payloads["revised_configuration_manifest"]["selection_rule"]
    assert rule["primary"].startswith("minimize_false_positive_count")
    assert rule["tie_1"].startswith("maximize_minimum_detected_endpoint_count")
    assert rule["tie_2"].startswith("maximize_total_detected_count")
    assert payloads["freshness_preflight"]["id_level_overlap_status"] == "verified_zero"
    assert payloads["lora_training_isolation"]["mcc12_prompt_or_response_leakage_detected"] is False
    assert payloads["attack_smoke_manifest"]["request_count"] == 8
