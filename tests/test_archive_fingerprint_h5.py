from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("archive_fingerprint_h5", ROOT / "scripts/archive_fingerprint_h5.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_h5_go_records_all_four_scientific_gates(tmp_path: Path) -> None:
    report = {
        "technical_passed": True,
        "hypothesis_supported": True,
        "intact_correct": True,
        "modified_detected": 11,
        "required_modified_detected": 9,
        "family_coverage_passed": True,
        "family_minimums_passed": True,
    }
    (tmp_path / "final_mmd_report.json").write_text(json.dumps(report), encoding="utf-8")
    result = MODULE.detect_h5_terminal(tmp_path)
    assert result["terminal_state"] == "h5_go"
    assert result["specificity_gate_passed"] is True
    assert result["sensitivity_count_gate_passed"] is True
    assert result["family_coverage_gate_passed"] is True
    assert result["gaussian_gate_passed"] is True
