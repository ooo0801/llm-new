from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_experiment_f_robust", ROOT / "scripts/analyze_experiment_f_robust.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def decision(prompt_id: str, category: str, source: str, structured: bool, families: int, score: float) -> dict:
    return {
        "prompt_id": prompt_id,
        "source_prompt_id": source,
        "category": category,
        "optimized_prompt_sha256": f"hash-{prompt_id}",
        "development_eligible": True,
        "structured_nondegraded": structured,
        "nondegraded_families": families,
        "worst_family_median_relative_gain": score,
        "families": {"structured_pruning": {"robust_median_relative_gain": score}},
        "macro": {"relative_gain": score},
        "micro": {"relative_gain": score},
    }


def test_selection_enforces_quota_and_unique_sources() -> None:
    rows = []
    pairs = []
    for category in MODULE.CATEGORIES:
        for index in range(4):
            prompt_id = f"{category}-{index}"
            rows.append(decision(prompt_id, category, f"source-{category}-{index}", index < 3, 5 - index % 2, 0.3 - index * 0.01))
            pairs.append({"prompt_id": prompt_id, "category": category})
    frozen, failures = MODULE.select_development(pairs, rows, selected_size=30, quota=3)
    assert failures == []
    assert len(frozen) == 30
    assert len({row["source_prompt_id"] for row in rows if row["prompt_id"] in {item["prompt_id"] for item in frozen}}) == 30
    counts = {category: sum(row["category"] == category for row in frozen) for category in MODULE.CATEGORIES}
    assert all(value >= 3 for value in counts.values())


def test_structured_robustness_has_ranking_priority() -> None:
    strong_macro = decision("a", "code", "source-a", False, 5, 10.0)
    structured = decision("b", "code", "source-b", True, 3, -0.005)
    assert MODULE.ranking_key(structured) < MODULE.ranking_key(strong_macro)
