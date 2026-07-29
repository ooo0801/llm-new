import json
from collections import Counter
from pathlib import Path


def test_dataset_splits_are_unique_and_disjoint():
    root = Path(__file__).resolve().parents[1] / "data"
    split_ids = {}
    for split in ["candidate", "validation", "test"]:
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = {row["id"] for row in rows}
        assert len(ids) == len(rows)
        assert all(row["split"] == split for row in rows)
        assert len(Counter(row["category"] for row in rows)) >= 10
        split_ids[split] = ids
    assert split_ids["candidate"].isdisjoint(split_ids["validation"])
    assert split_ids["candidate"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
