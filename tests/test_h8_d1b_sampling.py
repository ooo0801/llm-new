from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from llm_integrity.h8_d1b_sampling import (
    AUDIT_ROLE,
    FIT_ROLE,
    FORMAL_TOTAL,
    SMOKE_ROLE,
    SMOKE_TOTAL,
    audit_schedule,
    build_formal_schedule,
    build_smoke_schedule,
    load_manifest,
    make_manifest_payload,
    validate_schedule,
    write_manifest,
)
from llm_integrity.h8_sampling import sha256_text
from run_h8_d1b0_sampling_preflight import formal_parent, load_config, run_attempts


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "h8_d1b0_score_calibration_sampling.yaml"


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


def test_formal_schedule_is_deterministic_balanced_and_interleaved() -> None:
    first = build_formal_schedule(entries(), 2026081903, range(100, 200))
    second = build_formal_schedule(entries(), 2026081903, range(100, 200))
    assert [row.as_dict() for row in first] == [row.as_dict() for row in second]
    assert len(first) == FORMAL_TOTAL
    assert len({row.seed for row in first}) == FORMAL_TOTAL
    assert len({row.response_id for row in first}) == FORMAL_TOTAL
    assert not ({row.seed for row in first} & set(range(100, 200)))
    for round_id in range(200):
        rows = [row for row in first if row.round_id == round_id]
        assert len(rows) == 12 and len({row.prompt_id for row in rows}) == 12
        assert {row.data_role for row in rows} == {FIT_ROLE if round_id % 2 == 0 else AUDIT_ROLE}
    fit = Counter(row.prompt_id for row in first if row.data_role == FIT_ROLE)
    audit = Counter(row.prompt_id for row in first if row.data_role == AUDIT_ROLE)
    assert set(fit.values()) == {100}
    assert set(audit.values()) == {100}


def test_smoke_schedule_is_separate_and_never_bank_eligible() -> None:
    formal = build_formal_schedule(entries(), 2026081903)
    formal_seeds = {row.seed for row in formal}
    smoke = build_smoke_schedule(entries(), 2026081904, formal_seeds)
    assert len(smoke) == SMOKE_TOTAL
    assert not ({row.seed for row in smoke} & formal_seeds)
    assert {row.data_role for row in smoke} == {SMOKE_ROLE}
    assert Counter(row.intended_bank_role for row in smoke) == {FIT_ROLE: 12, AUDIT_ROLE: 12}
    assert all(row.as_dict()["eligible_for_target_evaluation"] is False for row in smoke)


def test_seed_collision_is_resolved_without_scope_change() -> None:
    original = build_formal_schedule(entries(), 2026081903)
    rebuilt = build_formal_schedule(entries(), 2026081903, {original[0].seed})
    assert original[0].seed not in {row.seed for row in rebuilt}
    assert rebuilt[0].derivation_counter >= 1
    validate_schedule(rebuilt, smoke=False)


@pytest.mark.parametrize("root_seed", [-1, 2**32])
def test_generation_root_seed_must_fit_uint32(root_seed: int) -> None:
    with pytest.raises(ValueError, match="root seed"):
        build_formal_schedule(entries(), root_seed)


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    schedule = build_formal_schedule(entries(), 2026081903)
    payload = make_manifest_payload(
        schedule,
        root_seed=2026081903,
        frozen_identity={"mmd": "frozen"},
        source_fingerprint_sha256="a" * 64,
        repository_seed_sources=[],
        forbidden_seed_count=0,
        cpu_resampling_root_seed_reserved=2026081905,
    )
    path = tmp_path / "manifest.json"
    info = write_manifest(path, payload, smoke=False)
    loaded_payload, loaded = load_manifest(path, smoke=False)
    assert loaded_payload == payload and len(loaded) == 2400
    assert info["file_sha256"]
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["generation_root_seed"] += 1
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_manifest(path, smoke=False)


def test_manifest_distinguishes_generation_and_cpu_seed_namespaces() -> None:
    schedule = build_formal_schedule(entries(), 2026081903)
    payload = make_manifest_payload(
        schedule,
        root_seed=2026081903,
        frozen_identity={},
        source_fingerprint_sha256="a" * 64,
        repository_seed_sources=[],
        forbidden_seed_count=0,
        cpu_resampling_root_seed_reserved=2026081905,
    )
    cpu = payload["cpu_resampling_seed_namespace"]
    assert cpu["root_seed_reserved"] == 2026081905
    assert cpu["seeds_generated_in_d1b"] is False
    assert cpu["may_overlap_generation_seeds"] is False
    assert payload["bank_design"]["single_response_bank_not_four_structure_banks"] is True
    assert payload["bank_design"]["cross_structure_statistic_or_parameter_pooling_forbidden"] is True


def test_schedule_audit_detects_forbidden_overlap() -> None:
    schedule = build_smoke_schedule(entries(), 2026081904)
    assert audit_schedule(schedule, smoke=True, forbidden_seeds=[])["status"] == "PASS"
    audit = audit_schedule(schedule, smoke=True, forbidden_seeds={schedule[0].seed})
    assert audit["status"] == "FAIL"
    assert audit["forbidden_seed_overlap_count"] == 1


def test_preflight_config_hard_blocks_formal_sampling_and_binds_protocol() -> None:
    config = load_config(CONFIG)
    assert config["formal_sampling_authorized"] is False
    assert config["smoke_sampling_authorized"] is True
    assert config["generation"]["formal_total_responses"] == 2400
    assert all(config["forbidden_operations"].values())
    protocol = ROOT / config["frozen_inputs"]["protocol"]
    import hashlib

    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == config["frozen_inputs"]["protocol_sha256"]
    with pytest.raises(PermissionError, match="not authorized"):
        formal_parent(CONFIG)


def test_same_seed_retry_keeps_seed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request = build_smoke_schedule(entries(), 2026081904)[0]
    calls = []

    def fake_generate(bundle, incoming, config):
        calls.append(incoming.seed)
        if len(calls) == 1:
            raise RuntimeError("technical")
        return {"raw_response": "ok"}

    monkeypatch.setattr("run_h8_d1b0_sampling_preflight.generate_one", fake_generate)
    event_path = tmp_path / "events.jsonl"
    with event_path.open("x", encoding="utf-8") as handle:
        result, attempts = run_attempts(None, request, {}, handle, max_attempts=2)
    assert result["raw_response"] == "ok"
    assert calls == [request.seed, request.seed]
    assert [row["seed"] for row in attempts] == [request.seed, request.seed]
    assert [row["status"] for row in attempts] == ["technical_failure", "success"]


def test_runner_contains_no_score_fit_or_downstream_selection() -> None:
    source = (ROOT / "scripts" / "run_h8_d1b0_sampling_preflight.py").read_text(encoding="utf-8")
    assert "fit_score_calibration_parameters(" not in source
    assert "score_value(" not in source
    assert "fit_top_r(" not in source
    assert '"formal_generation_started": False' in source
    assert '"score_parameter_fit_performed": False' in source
