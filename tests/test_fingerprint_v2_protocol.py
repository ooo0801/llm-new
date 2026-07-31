from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_fingerprint_v2 import TASK_FAMILIES, calibration_prompt, normalize_text  # noqa: E402


def test_expanded_calibration_is_balanced_and_unique() -> None:
    rows = [
        (family, round_index, calibration_prompt(family, round_index)[0])
        for round_index in range(24)
        for family in TASK_FAMILIES
    ]
    assert len(TASK_FAMILIES) == 14
    assert len(rows) == 336
    assert len({normalize_text(prompt) for _, _, prompt in rows}) == 336
    assert all(sum(family == target for family, _, _ in rows) == 24 for target in TASK_FAMILIES)


def test_recycled_static_templates_are_explicit_variants() -> None:
    for family in ("knowledge_qa", "code_generation", "translation", "safety_alignment"):
        original = calibration_prompt(family, 0)[0]
        expanded = calibration_prompt(family, 16)[0]
        assert normalize_text(expanded) != normalize_text(original)
        assert "17" in expanded
