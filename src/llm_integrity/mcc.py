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


def component_weight(component: str, weights: Mapping[str, float]) -> float:
    kind = component.split(":", 1)[0]
    return float(weights.get(kind, 1.0))


def greedy_mcc(
    prompt_components: Mapping[str, Iterable[str]],
    k: int,
    weights: Mapping[str, float] | None = None,
) -> SelectionResult:
    if k <= 0:
        raise ValueError("k must be positive")
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    normalized = {key: set(value) for key, value in prompt_components.items()}
    selected: list[str] = []
    covered: set[str] = set()
    gains: list[float] = []
    while len(selected) < k:
        best_id = None
        best_gain = 0.0
        for prompt_id, components in sorted(normalized.items()):
            if prompt_id in selected:
                continue
            gain = sum(component_weight(c, weights) for c in components - covered)
            if gain > best_gain:
                best_id, best_gain = prompt_id, gain
        if best_id is None:
            break
        selected.append(best_id)
        covered.update(normalized[best_id])
        gains.append(best_gain)
    return SelectionResult(selected, covered, gains)


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
