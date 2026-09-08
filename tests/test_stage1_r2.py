import copy
import hashlib
import json
from pathlib import Path

import pytest

from llm_integrity.stage1_r1 import canonical
from llm_integrity.stage1_r2 import schedule, schedule_hash, read_chain, freeze_json, summarize_validation, confidence_interval

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/stage1_r2_independent_null.json").read_text())
PROMPTS = [{"id": str(i)} for i in range(12)]


def test_schedule_disjointness_and_budget():
    roles = [schedule(CONFIG, PROMPTS, r) for r in ("fit", "calibration", "validation")]
    seeds = [r["generation_seed"] for bank in roles for r in bank]
    assert len(seeds) == len(set(seeds)) == 35712
    assert max(seeds) < 2**32
    assert min(seeds) > 910000 + 12000
    assert [len(r) for r in roles] == [576, 576, 34560]
    assert roles[2][0]["side"] == "reference" and roles[2][1]["side"] == "target"
    assert schedule_hash(roles[0]) == schedule_hash(schedule(CONFIG, PROMPTS, "fit"))


def test_response_chain_rejects_tampering_and_partial_tail(tmp_path):
    expected = schedule(CONFIG, PROMPTS, "fit")[:1]
    record = {**expected[0], "manifest_sha256": "frozen", "previous_sha256": "0" * 64,
              "response": "hello", "response_sha256": hashlib.sha256(b"hello").hexdigest()}
    record["record_sha256"] = hashlib.sha256(canonical(record).encode()).hexdigest()
    p = tmp_path / "rows.jsonl"
    p.write_text(canonical(record) + "\n", encoding="utf-8")
    assert len(read_chain(p, expected, "frozen", True)) == 1
    with pytest.raises(ValueError):
        read_chain(p, expected, "changed")
    p.write_text(canonical(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Partial"):
        read_chain(p, expected, "frozen")
    record["response"] = "changed"
    p.write_text(canonical(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="text hash"):
        read_chain(p, expected, "frozen")


def test_freeze_refuses_mutation(tmp_path):
    p = tmp_path / "freeze.json"
    freeze_json(p, {"alpha": .05})
    freeze_json(p, {"alpha": .05})
    with pytest.raises(ValueError):
        freeze_json(p, {"alpha": .1})


def records():
    return [{"prompt_id": p["id"], "unit": i, "raw": {"detected": False}, "h8": {"detected": False}}
            for p in PROMPTS for i in range(60)]


def test_fixed_sample_acceptance_and_non_binomial_macro_bound():
    rows = records()
    summary = summarize_validation(rows, PROMPTS, CONFIG)
    assert summary["status"] == "PASS_MACRO_ENGINEERING_CHECK_ONLY"
    assert .05 < summary["macro"]["h8"]["simultaneous_Hoeffding_upper"] < .051
    assert summary["formal_every_prompt_FPR_le_5_percent_proven"] is False
    assert confidence_interval(0, 60)[1] > .05
    with pytest.raises(ValueError):
        summarize_validation(rows[:-1], PROMPTS, CONFIG)
    duplicate = rows[:-1] + [rows[0]]
    with pytest.raises(ValueError):
        summarize_validation(duplicate, PROMPTS, CONFIG)


def test_unknown_does_not_count_as_negative_and_local_inflation_is_flagged():
    rows = records()
    rows[0]["h8"]["detected"] = None
    summary = summarize_validation(rows, PROMPTS, CONFIG)
    assert summary["status"] == "TECHNICAL_FAILURE_UNEVALUABLE_UNITS"
    assert summary["macro"]["h8"]["fpr"] is None
    rows = records()
    for r in rows[:20]:
        r["raw"]["detected"] = True
    assert summarize_validation(rows, PROMPTS, CONFIG)["status"] == "FAIL_LOCALIZED_INFLATION_FLAG"
