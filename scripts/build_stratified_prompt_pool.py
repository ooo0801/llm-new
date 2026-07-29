from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from _bootstrap import project_path
from llm_integrity.io import read_jsonl, write_jsonl


def select_stratified(
    rows: list[dict],
    *,
    size: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if size > len(rows):
        raise ValueError(
            f"Requested {size} prompts from only {len(rows)} rows"
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in rows:
        prompt_id = str(row.get("id", row.get("prompt_id", "")))
        if not prompt_id:
            raise ValueError("Every candidate must have an id or prompt_id")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate candidate id: {prompt_id}")
        seen_ids.add(prompt_id)
        grouped[str(row.get("category", "unknown"))].append(dict(row))

    categories = sorted(grouped)
    if not categories:
        raise ValueError("No candidate categories were found")

    rng = random.Random(seed)
    for category in categories:
        rng.shuffle(grouped[category])

    base, _ = divmod(size, len(categories))
    allocation = {
        category: min(base, len(grouped[category]))
        for category in categories
    }
    remaining = size - sum(allocation.values())

    category_order = list(categories)
    rng.shuffle(category_order)
    while remaining > 0:
        progress = False
        for category in category_order:
            if allocation[category] >= len(grouped[category]):
                continue
            allocation[category] += 1
            remaining -= 1
            progress = True
            if remaining == 0:
                break
        if not progress:
            raise RuntimeError("Unable to satisfy stratified allocation")

    selected_by_category = {
        category: grouped[category][: allocation[category]]
        for category in categories
    }
    selected: list[dict] = []
    max_count = max(allocation.values())
    for rank in range(max_count):
        for category in category_order:
            if rank >= len(selected_by_category[category]):
                continue
            row = dict(selected_by_category[category][rank])
            row["pool_selection"] = {
                "method": "stratified_seeded",
                "seed": seed,
                "target_size": size,
                "category_rank": rank,
            }
            selected.append(row)

    selected_ids = {
        str(row.get("id", row.get("prompt_id")))
        for row in selected
    }
    if len(selected) != size or len(selected_ids) != size:
        raise RuntimeError("Selected prompt count or uniqueness check failed")

    counts = Counter(
        str(row.get("category", "unknown"))
        for row in selected
    )
    return selected, dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible category-stratified prompt pool"
    )
    parser.add_argument("--input", default="data/candidate.jsonl")
    parser.add_argument(
        "--output",
        default=(
            "results/paper_aligned_qwen_7b/prompt_optimization/"
            "seeds_stratified_k60.jsonl"
        ),
    )
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = project_path(args.input)
    output = project_path(args.output)
    rows = read_jsonl(source)
    selected, category_counts = select_stratified(
        rows,
        size=args.size,
        seed=args.seed,
    )
    write_jsonl(output, selected)

    summary = {
        "input": str(source),
        "output": str(output),
        "candidate_rows": len(rows),
        "selected_rows": len(selected),
        "unique_ids": len({
            str(row.get("id", row.get("prompt_id")))
            for row in selected
        }),
        "seed": args.seed,
        "method": "stratified_seeded",
        "category_counts": category_counts,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
