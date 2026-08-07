from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_experiment_f_robust",
    ROOT / "scripts/archive_experiment_f_robust.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_terminal_state_requires_confirmation_after_development_go(tmp_path: Path) -> None:
    write_json(
        tmp_path / "01_development/analysis/report.json",
        {"technical_passed": True, "gate_passed": True},
    )
    with pytest.raises(RuntimeError, match="confirmation has no terminal report"):
        MODULE.validate_terminal_state(tmp_path)


def test_terminal_state_preserves_confirmation_no_go(tmp_path: Path) -> None:
    write_json(
        tmp_path / "01_development/analysis/report.json",
        {"technical_passed": True, "gate_passed": True},
    )
    write_json(
        tmp_path / "02_confirmation/analysis/report.json",
        {"technical_passed": True, "gate_passed": False, "hypothesis_supported": False},
    )
    development, confirmation, state = MODULE.validate_terminal_state(tmp_path)
    assert development["gate_passed"] is True
    assert confirmation is not None and confirmation["hypothesis_supported"] is False
    assert state == "confirmation_no_go"


def test_terminal_state_accepts_development_no_go_without_confirmation(tmp_path: Path) -> None:
    write_json(
        tmp_path / "01_development/analysis/report.json",
        {"technical_passed": True, "gate_passed": False},
    )
    _, confirmation, state = MODULE.validate_terminal_state(tmp_path)
    assert confirmation is None
    assert state == "development_no_go"
