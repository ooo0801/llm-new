from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_fingerprint_h4",
    ROOT / "scripts/archive_fingerprint_h4.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_report(path: Path, **overrides: object) -> None:
    payload = {
        "technical_passed": True,
        "hypothesis_supported": False,
        "intact_correct": True,
        "modified_detected": 8,
        "required_modified_detected": 9,
        "family_coverage_passed": False,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_h4_specificity_pass_sensitivity_no_go(tmp_path: Path) -> None:
    write_report(tmp_path / "final_mmd_report.json")
    result = MODULE.detect_h4_terminal(tmp_path)
    assert result["terminal_state"] == "h4_no_go"
    assert result["specificity_gate_passed"] is True
    assert result["sensitivity_count_gate_passed"] is False
    assert result["family_coverage_gate_passed"] is False


def test_h4_go_requires_report_support(tmp_path: Path) -> None:
    write_report(
        tmp_path / "final_mmd_report.json",
        hypothesis_supported=True,
        modified_detected=9,
        family_coverage_passed=True,
    )
    result = MODULE.detect_h4_terminal(tmp_path)
    assert result["terminal_state"] == "h4_go"
    assert result["sensitivity_count_gate_passed"] is True
