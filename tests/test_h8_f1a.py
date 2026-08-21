from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from llm_integrity.h8_f1a import (
    ALPHA,
    ATTACK_FAMILIES,
    ATTACK_TOTAL,
    FINAL_ATTACK_ROLE,
    FINAL_INTACT_ROLE,
    FORMAL_TOTAL,
    INTACT_TOTAL,
    REFERENCE_TOTAL,
    build_attack_endpoints,
    build_generation_schedule,
    build_permutation_seed_manifest,
    exact_binomial_interval,
    final_global_permutation_test,
    load_canonical_envelope,
    load_frozen_detector,
    save_canonical_envelope,
    summarize_final_decisions,
    validate_attack_endpoints,
    validate_smoke_records,
)
from llm_integrity.paper_quantization import _build_model_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/h8_f1a_final_confirmation_preflight.yaml"
FINGERPRINT = (
    ROOT
    / "reproducibility/fingerprint_h6_qwen32b_mcc_20260812/components/fingerprints/global_mcc_32b_h6.json"
)
DETECTOR_ARCHIVE = (
    ROOT
    / "reproducibility/h8_qwen32b_detector_development_20260820/d2c_detector_frozen_v1"
)


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def entries() -> list[dict]:
    return json.loads(FINGERPRINT.read_text(encoding="utf-8"))["entries"]


def endpoints() -> list[dict]:
    cfg = config()
    rows = build_attack_endpoints(
        cfg["attack_design"],
        materialization_root_seed=cfg["seed_roots"]["formal_attack_materialization_uint32"],
        lora_training_root_seed=cfg["seed_roots"]["formal_lora_training_uint32"],
        forbidden_seeds={1, 2, 3},
    )
    return validate_attack_endpoints(rows)["endpoints"]


def test_frozen_detector_fail_closed_and_exact_invariants(tmp_path: Path) -> None:
    cfg = config()
    loaded = load_frozen_detector(
        DETECTOR_ARCHIVE,
        expected_manifest_sha256=cfg["frozen_detector"]["manifest_sha256"],
    )
    assert loaded["selected_detector"]["selected_configuration_id"] == "r60_q10_top2"
    assert loaded["selected_detector"]["n_reference"] == 60
    assert loaded["selected_detector"]["n_target"] == 10
    assert loaded["selected_detector"]["top_r"] == 2
    with pytest.raises(ValueError, match="manifest SHA256"):
        load_frozen_detector(DETECTOR_ARCHIVE, expected_manifest_sha256="0" * 64)


def test_forty_attack_endpoints_are_unique_and_family_balanced() -> None:
    rows = endpoints()
    assert len(rows) == 40
    assert {row["family"] for row in rows} == set(ATTACK_FAMILIES)
    assert len({row["attack_instance_id"] for row in rows}) == 40
    assert len({row["configuration_sha256"] for row in rows}) == 40
    assert len({row["materialization_seed"] for row in rows}) == 40
    assert len({row["training_seed"] for row in rows if row["training_seed"] is not None}) == 10


def test_attack_freshness_rejects_historical_configuration_and_seed() -> None:
    rows = endpoints()
    with pytest.raises(ValueError, match="configuration"):
        validate_attack_endpoints(rows, historical_configuration_sha256={rows[0]["configuration_sha256"]})
    with pytest.raises(ValueError, match="materialization seed"):
        validate_attack_endpoints(rows, historical_materialization_seeds={rows[0]["materialization_seed"]})


def test_generation_manifest_exact_counts_and_seed_isolation() -> None:
    forbidden = set(range(1000, 1100))
    rows = endpoints()
    attack_seeds = {row["materialization_seed"] for row in rows} | {
        row["training_seed"] for row in rows if row["training_seed"] is not None
    }
    manifest = build_generation_schedule(
        entries(),
        rows,
        generation_root_seed=2026082601,
        order_root_seed=2026082602000001,
        forbidden_generation_seeds=forbidden | attack_seeds,
        forbidden_response_ids={"historical-response"},
    )
    requests = manifest["requests"]
    assert len(requests) == FORMAL_TOTAL
    assert manifest["role_counts"] == {
        "final_heldout_attack_only": ATTACK_TOTAL,
        "final_heldout_intact_target_only": INTACT_TOTAL,
        "final_reference_only": REFERENCE_TOTAL,
    }
    seeds = {row["generation_seed"] for row in requests}
    assert len(seeds) == FORMAL_TOTAL
    assert not seeds & forbidden
    assert not seeds & attack_seeds
    assert all(row["formal_sampling_authorized"] is False for row in requests)
    assert all(row["eligible_for_detector_selection"] is False for row in requests)


def test_permutation_manifest_has_one_hundred_unique_streams() -> None:
    attack_rows = endpoints()
    units = [
        {"evaluation_unit_id": f"final_intact_unit_{index:03d}", "data_role": FINAL_INTACT_ROLE, "attack_family": None}
        for index in range(60)
    ] + [
        {
            "evaluation_unit_id": row["attack_instance_id"],
            "data_role": FINAL_ATTACK_ROLE,
            "attack_family": row["family"],
        }
        for row in attack_rows
    ]
    payload = build_permutation_seed_manifest(
        units,
        [row["prompt_id"] for row in entries()],
        permutation_root_seed=2026082603000001,
    )
    assert payload["stream_count"] == 100
    assert payload["derived_seed_count"] == 100 * 999 * 12
    assert payload["derived_seed_unique"] is True


def test_global_permutation_is_fixed_top2_and_reproducible() -> None:
    values = np.arange(70, dtype=np.float64)
    base_kernel = np.exp(-((values[:, None] - values[None, :]) ** 2) / 100.0)
    kernels = {f"prompt_{index:02d}": base_kernel.copy() for index in range(12)}
    parameters = {
        prompt_id: {
            "mapping": "nondegenerate_centered",
            "null_median_m": 0.0,
            "effective_scale": 1.0,
        }
        for prompt_id in kernels
    }
    first = final_global_permutation_test(kernels, parameters, root_seed=42)
    second = final_global_permutation_test(kernels, parameters, root_seed=42)
    assert first.observed_top_r == second.observed_top_r
    assert first.global_p_value == second.global_p_value
    assert len(first.permutation_top_r) == 999
    assert first.permutations == 999
    assert first.alpha == 0.05
    assert 0.0 < first.global_p_value <= 1.0


def test_exact_clopper_pearson_boundaries() -> None:
    assert exact_binomial_interval(0, 60)[0] == 0.0
    assert exact_binomial_interval(0, 60)[1] == pytest.approx(0.0596294923, rel=1e-9)
    assert exact_binomial_interval(0, 10)[1] == pytest.approx(0.3084971, rel=1e-5)
    assert exact_binomial_interval(10, 10)[0] == pytest.approx(0.6915029, rel=1e-5)
    assert exact_binomial_interval(10, 10)[1] == 1.0


def test_final_summary_is_family_specific_and_consistent() -> None:
    records = [
        {
            "evaluation_unit_id": f"final_intact_unit_{index:03d}",
            "data_role": FINAL_INTACT_ROLE,
            "p_global": 0.01 if index == 0 else 0.5,
            "detected": index == 0,
        }
        for index in range(60)
    ]
    for family in ATTACK_FAMILIES:
        records.extend(
            {
                "evaluation_unit_id": f"{family}_{index:02d}",
                "data_role": FINAL_ATTACK_ROLE,
                "attack_family": family,
                "p_global": 0.01 if index < 7 else 0.5,
                "detected": index < 7,
            }
            for index in range(10)
        )
    summary = summarize_final_decisions(records)
    assert summary["intact"]["false_positive_count"] == 1
    assert summary["intact"]["fpr"] == pytest.approx(1 / 60)
    assert all(row["detected"] == 7 for row in summary["attack_families"].values())
    assert summary["overall_attack_detected"] == 28
    assert summary["primary_estimand"].startswith("conditional_performance")
    assert summary["unconditional_reference_regeneration_coverage_claimed"] is False
    assert all(
        row["iid_attack_superpopulation_claimed"] is False
        for row in summary["attack_families"].values()
    )
    assert "shared" in summary["shared_reference_dependence_note"].lower()


def test_canonical_envelope_tamper_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    info = save_canonical_envelope(path, "toy", {"value": 1})
    assert load_canonical_envelope(
        path,
        expected_artifact_type="toy",
        expected_file_sha256=info["file_sha256"],
        expected_payload_sha256=info["payload_sha256"],
    ) == {"value": 1}
    path.write_text(path.read_text(encoding="utf-8").replace('"value":1', '"value":2'), encoding="utf-8")
    with pytest.raises(ValueError, match="file SHA256"):
        load_canonical_envelope(
            path,
            expected_artifact_type="toy",
            expected_file_sha256=info["file_sha256"],
            expected_payload_sha256=info["payload_sha256"],
        )


def test_smoke_audit_preserves_legal_empty_eos_and_rejects_changed_retry_seed() -> None:
    requests = []
    records = []
    for index in range(6):
        request = {
            "response_id": f"smoke-{index}",
            "generation_seed": 100 + index,
            "data_role": "final_reference_smoke_only" if index == 0 else "final_heldout_intact_smoke_only",
            "prompt_id": f"prompt-{index}",
            "prompt_sha256": f"{index:064x}",
            "attack_instance_id": None,
            "attack_family": None,
        }
        requests.append(request)
        records.append(
            {
                **request,
                "formal_final_confirmation_eligible": False,
                "detector_statistics_computed": False,
                "measurement_or_score_fit_performed": False,
                "rendered_prompt_sha256": "a" * 64,
                "input_token_ids": [1, 2],
                "input_token_ids_sha256": "b" * 64,
                "completion_token_ids": [151645] if index == 0 else [3, 4],
                "completion_token_ids_sha256": "c" * 64,
                "generated_token_count": 1 if index == 0 else 2,
                "stop_reason": "eos" if index == 0 else "length",
                "legal_first_token_eos": index == 0,
                "raw_response": "" if index == 0 else "ok",
                "attempt_records": [{"attempt_index": 0, "seed": 100 + index, "status": "success"}],
                "provenance_sha256": "d" * 64,
            }
        )
    audit = validate_smoke_records(records, requests)
    assert audit["legal_first_token_eos_empty_response_count"] == 1
    assert audit["formal_final_confirmation_responses"] == 0
    records[1]["attempt_records"][0]["seed"] = 999
    with pytest.raises(ValueError, match="retry changed"):
        validate_smoke_records(records, requests)


def test_int8_threshold_is_bound_into_quantized_model_config() -> None:
    base = {"name": "toy", "revision": "frozen"}
    model_config, realized, exact, notes = _build_model_config(
        base,
        {
            "method": "int8",
            "compute_dtype": "bfloat16",
            "llm_int8_threshold": 4.0,
        },
    )
    assert model_config["llm_int8_threshold"] == 4.0
    assert realized == "bitsandbytes_int8"
    assert exact is True
    assert notes == ()
