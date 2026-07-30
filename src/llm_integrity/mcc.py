from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping


DEFAULT_WEIGHTS = {
    "attention": 1.2,
    "ffn": 1.0,
    "attention_residual": 0.8,
    "mlp_residual": 0.8,
}


@dataclass
class SelectionResult:
    selected_ids: list[str]
    covered_components: set[str]
    marginal_gains: list[float]
    trace: list[dict[str, object]]


def component_weight(component: str, weights: Mapping[str, float]) -> float:
    kind = component.split(":", 1)[0]
    return float(weights.get(kind, 1.0))


def greedy_mcc(
    prompt_components: Mapping[str, Iterable[str]],
    k: int,
    weights: Mapping[str, float] | None = None,
    metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> SelectionResult:
    if k <= 0:
        raise ValueError("k must be positive")
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    normalized = {key: set(value) for key, value in prompt_components.items()}
    if k > len(normalized):
        raise ValueError("k exceeds candidate count")
    metadata = metadata or {}
    selected: list[str] = []
    covered: set[str] = set()
    gains: list[float] = []
    trace: list[dict[str, object]] = []
    selected_categories: set[str] = set()
    while len(selected) < k:
        ranked: list[tuple[tuple[float, int, float, int], str, float]] = []
        for prompt_id, components in sorted(normalized.items()):
            if prompt_id in selected:
                continue
            gain = sum(component_weight(c, weights) for c in components - covered)
            row = metadata.get(prompt_id, {})
            category = str(row.get("category", ""))
            category_novelty = int(bool(category and category not in selected_categories))
            stability = float(row.get("stability", 0.0))
            token_count = int(row.get("token_count", 10**9))
            ranked.append(
                ((gain, category_novelty, stability, -token_count), prompt_id, gain)
            )
        if not ranked:
            raise RuntimeError("MCC exhausted candidates before reaching k")
        # prompt_id is kept outside the descending score tuple so exact ties
        # are resolved by the lexicographically smallest stable identifier.
        best_score = max(item[0] for item in ranked)
        tied = [item for item in ranked if item[0] == best_score]
        _, best_id, best_gain = min(tied, key=lambda item: item[1])
        new_components = normalized[best_id] - covered
        selected.append(best_id)
        covered.update(normalized[best_id])
        gains.append(best_gain)
        category = str(metadata.get(best_id, {}).get("category", ""))
        if category:
            selected_categories.add(category)
        trace.append(
            {
                "step": len(selected),
                "prompt_id": best_id,
                "marginal_gain": best_gain,
                "new_component_count": len(new_components),
                "covered_component_count": len(covered),
                "category": category or None,
            }
        )
    return SelectionResult(selected, covered, gains, trace)


def top_sensitivity(scores: Mapping[str, float], k: int) -> list[str]:
    return [key for key, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]


def random_selection(ids: Iterable[str], k: int, seed: int = 42) -> list[str]:
    values = list(ids)
    if k > len(values):
        raise ValueError("k exceeds candidate count")
    return random.Random(seed).sample(values, k)


def coverage_rate(selected_ids: Iterable[str], prompt_components: Mapping[str, Iterable[str]]) -> float:
    all_components = set().union(*(set(value) for value in prompt_components.values()))
    if not all_components:
        return 0.0
    selected = set().union(*(set(prompt_components[item]) for item in selected_ids))
    return len(selected) / len(all_components)


def complementarity(selected_ids: Iterable[str], prompt_components: Mapping[str, Iterable[str]]) -> float:
    sets = [set(prompt_components[item]) for item in selected_ids]
    union = set().union(*sets) if sets else set()
    if not union:
        return 0.0
    duplicated = set()
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            duplicated.update(left & right)
    return 1.0 - len(duplicated) / len(union)
