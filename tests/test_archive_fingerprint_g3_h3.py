from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_fingerprint_g3_h3",
    ROOT / "scripts/archive_fingerprint_g3_h3.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def seed_g3(root: Path, *, passed: bool) -> None:
    write_json(root / "activation_audit.json", {"passed": passed})
    write_json(root / "global_component_universe.json", {"gates": {"passed": passed}})
    write_json(root / "mcc_selection.json", {"passed": passed})


def test_g3_go_and_h3_intact_false_positive_no_go(tmp_path: Path) -> None:
    seed_g3(tmp_path, passed=True)
    write_json(
        tmp_path / "final_mmd_report.json",
        {"technical_passed": True, "hypothesis_supported": False, "intact_correct": False},
    )
    g3 = MODULE.detect_g3_terminal(tmp_path)
    h3 = MODULE.detect_h3_terminal(tmp_path, g3)
    assert g3["terminal_state"] == "g3_go"
    assert h3["terminal_state"] == "h3_no_go"
    assert h3["mmd_report"]["intact_correct"] is False


def test_h3_rejected_when_g3_did_not_pass(tmp_path: Path) -> None:
    seed_g3(tmp_path, passed=False)
    write_json(tmp_path / "final_mmd_report.json", {"technical_passed": True})
    with pytest.raises(RuntimeError, match="G3 did not pass"):
        MODULE.detect_h3_terminal(tmp_path, MODULE.detect_g3_terminal(tmp_path))
