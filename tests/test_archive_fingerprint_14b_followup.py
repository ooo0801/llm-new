from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_fingerprint_14b_followup",
    ROOT / "scripts/archive_fingerprint_14b_followup.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_g1_activation_no_go_is_terminal(tmp_path: Path) -> None:
    write_json(tmp_path / "activation_audit.json", {"passed": False})
    report = MODULE.detect_g1_terminal(tmp_path)
    assert report["terminal_state"] == "activation_no_go"
    assert report["hypothesis_supported"] is False


def test_g1_requires_mcc_after_universe_go(tmp_path: Path) -> None:
    write_json(tmp_path / "activation_audit.json", {"passed": True})
    write_json(tmp_path / "global_component_universe.json", {"gates": {"passed": True}})
    with pytest.raises(RuntimeError, match="no terminal MCC report"):
        MODULE.detect_g1_terminal(tmp_path)


def test_g1_go_and_h1_no_go_are_preserved(tmp_path: Path) -> None:
    write_json(tmp_path / "activation_audit.json", {"passed": True})
    write_json(tmp_path / "global_component_universe.json", {"gates": {"passed": True}})
    write_json(tmp_path / "mcc_selection.json", {"passed": True})
    write_json(
        tmp_path / "final_mmd_report.json",
        {"technical_passed": True, "hypothesis_supported": False},
    )
    g1 = MODULE.detect_g1_terminal(tmp_path)
    h1 = MODULE.detect_h1_terminal(tmp_path, g1)
    assert g1["terminal_state"] == "g1_go"
    assert h1["terminal_state"] == "h1_no_go"


def test_h1_is_rejected_when_g1_did_not_pass(tmp_path: Path) -> None:
    write_json(tmp_path / "final_mmd_report.json", {"technical_passed": True})
    with pytest.raises(RuntimeError, match="conditional G1 gate"):
        MODULE.detect_h1_terminal(tmp_path, {"hypothesis_supported": False})
