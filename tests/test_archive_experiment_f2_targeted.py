from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_experiment_f2_targeted",
    ROOT / "scripts/archive_experiment_f2_targeted.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_gate(root: Path, stage: str, value: str) -> None:
    path = root / stage / "gate_status.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def seed_parent_go(root: Path) -> None:
    write_gate(root, "00_construction", "CONSTRUCTION_GO")
    write_gate(root, "01_development", "DEVELOPMENT_GO")
    write_json(root / "01_development/analysis/report.json", {"technical_passed": True, "gate_passed": True})


def test_terminal_state_preserves_confirmation_no_go(tmp_path: Path) -> None:
    seed_parent_go(tmp_path)
    write_gate(tmp_path, "02_confirmation", "CONFIRMATION_NO_GO")
    write_json(
        tmp_path / "02_confirmation/analysis/report.json",
        {"technical_passed": True, "gate_passed": False, "hypothesis_supported": False},
    )
    _, confirmation, state = MODULE.validate_terminal_state(tmp_path)
    assert confirmation["hypothesis_supported"] is False
    assert state == "confirmation_no_go"


def test_terminal_state_accepts_confirmation_go(tmp_path: Path) -> None:
    seed_parent_go(tmp_path)
    write_gate(tmp_path, "02_confirmation", "CONFIRMATION_GO")
    write_json(
        tmp_path / "02_confirmation/analysis/report.json",
        {"technical_passed": True, "gate_passed": True, "hypothesis_supported": True},
    )
    assert MODULE.validate_terminal_state(tmp_path)[2] == "confirmation_go"


def test_terminal_state_rejects_report_gate_mismatch(tmp_path: Path) -> None:
    seed_parent_go(tmp_path)
    write_gate(tmp_path, "02_confirmation", "CONFIRMATION_GO")
    write_json(
        tmp_path / "02_confirmation/analysis/report.json",
        {"technical_passed": True, "gate_passed": False, "hypothesis_supported": False},
    )
    with pytest.raises(RuntimeError, match="report/gate mismatch"):
        MODULE.validate_terminal_state(tmp_path)
