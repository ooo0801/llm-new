import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_stratified_prompt_pool import select_stratified


def test_select_stratified_is_balanced_unique_and_reproducible():
    rows = [
        {
            "id": f"{category}_{index}",
            "category": category,
            "prompt": f"prompt {category} {index}",
        }
        for category in ("a", "b", "c", "d")
        for index in range(20)
    ]
    first, first_counts = select_stratified(rows, size=18, seed=42)
    second, second_counts = select_stratified(rows, size=18, seed=42)

    first_ids = [row["id"] for row in first]
    assert first_ids == [row["id"] for row in second]
    assert len(first_ids) == len(set(first_ids)) == 18
    assert max(first_counts.values()) - min(first_counts.values()) <= 1
    assert first_counts == second_counts
