from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_fingerprint_h6_archive",
    ROOT / "scripts/finalize_fingerprint_h6_archive.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_h6_terminal_archive_is_self_consistent() -> None:
    release = ROOT / "reproducibility/fingerprint_h6_qwen32b_mcc_20260812"
    report = MODULE.validate_release(release)
    assert report["terminal_state"] == "h6_go_mcc12_complete"
    assert report["hypotheses"]["H6-P"]["confirmed_rows"] == 39
    assert report["hypotheses"]["H6-P"]["minimum_confirmed"] == 19
    assert report["hypotheses"]["H6-MCC12"]["selected_prompts"] == 12
    assert report["hypotheses"]["H6-MCC12"]["selection_deterministic"] is True
    assert report["hypotheses"]["H6-MCC12"]["factorization_passed"] is True
