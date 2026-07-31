from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping, Sequence


COMPONENT_KINDS = (
    "attention",
    "ffn",
    "attention_residual",
    "mlp_residual",
)


def component_kind(component: str) -> str:
    return str(component).split(":", 1)[0]


def validate_component_id(component: str) -> None:
    parts = str(component).split(":")
    kind = parts[0]
    expected_parts = 3 if kind in {"attention", "ffn"} else 2
    if kind not in COMPONENT_KINDS or len(parts) != expected_parts:
        raise ValueError(f"Unsupported component identifier: {component!r}")
    try:
        indices = [int(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"Non-integer component identifier: {component!r}") from exc
    if any(value < 0 for value in indices):
        raise ValueError(f"Negative component identifier: {component!r}")


def component_counts(components: Iterable[str]) -> dict[str, int]:
    counts = Counter(component_kind(component) for component in set(components))
    return {kind: int(counts.get(kind, 0)) for kind in COMPONENT_KINDS}


def stable_prompt_components(
    profiles: Iterable[Mapping[str, object]],
    *,
    repetitions: int,
    stable_frequency: float,
) -> dict[str, set[str]]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < stable_frequency <= 1.0:
        raise ValueError("stable_frequency must be in (0, 1]")
    grouped: dict[str, list[set[str]]] = defaultdict(list)
    repeats: dict[str, set[int]] = defaultdict(set)
    for row in profiles:
        prompt_id = str(row.get("id", row.get("prompt_id", "")))
        if not prompt_id:
            raise ValueError("Activation profile is missing a prompt identifier")
        repeat = int(row.get("profile_repeat", -1))
        if repeat < 0 or repeat >= repetitions:
            raise ValueError(f"Invalid repeat {repeat} for {prompt_id}")
        if repeat in repeats[prompt_id]:
            raise ValueError(f"Duplicate repeat {repeat} for {prompt_id}")
        values = {str(value) for value in row.get("components", [])}
        if not values:
            raise ValueError(f"Empty component profile for {prompt_id}")
        for component in values:
            validate_component_id(component)
        grouped[prompt_id].append(values)
        repeats[prompt_id].add(repeat)

    required = max(1, int(math.ceil(repetitions * stable_frequency)))
    stable: dict[str, set[str]] = {}
    for prompt_id, values in grouped.items():
        if len(values) != repetitions or repeats[prompt_id] != set(range(repetitions)):
            raise ValueError(
                f"Expected repeats 0..{repetitions - 1} for {prompt_id}; "
                f"got {sorted(repeats[prompt_id])}"
            )
        counts = Counter(component for current in values for component in current)
        selected = {component for component, count in counts.items() if count >= required}
        if not selected:
            raise ValueError(f"No stable components for {prompt_id}")
        stable[prompt_id] = selected
    return stable


def union_components(prompt_components: Mapping[str, Iterable[str]]) -> set[str]:
    values = [set(components) for components in prompt_components.values()]
    return set().union(*values) if values else set()


def restrict_to_universe(
    prompt_components: Mapping[str, Iterable[str]],
    universe: Iterable[str],
) -> dict[str, set[str]]:
    frozen = set(universe)
    if not frozen:
        raise ValueError("Global component universe is empty")
    return {
        str(prompt_id): set(components) & frozen
        for prompt_id, components in prompt_components.items()
    }


@dataclass(frozen=True)
class CoverageMetrics:
    global_components: int
    candidate_components: int
    selected_components: int
    candidate_global_components: int
    selected_global_components: int
    candidate_outside_components: int
    selected_outside_components: int
    candidate_reachable_coverage: float
    selection_efficiency: float
    final_global_coverage: float
    factorization_error: float

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def coverage_metrics(
    prompt_components: Mapping[str, Iterable[str]],
    selected_ids: Sequence[str],
    global_components: Iterable[str],
) -> CoverageMetrics:
    universe = set(global_components)
    if not universe:
        raise ValueError("Global component universe is empty")
    missing = [prompt_id for prompt_id in selected_ids if prompt_id not in prompt_components]
    if missing:
        raise ValueError(f"Selected prompt IDs are missing component profiles: {missing}")
    candidate_union = union_components(prompt_components)
    selected_sets = [set(prompt_components[prompt_id]) for prompt_id in selected_ids]
    selected_union = set().union(*selected_sets) if selected_sets else set()
    candidate_global = candidate_union & universe
    selected_global = selected_union & universe
    reachable = len(candidate_global) / len(universe)
    efficiency = len(selected_global) / len(candidate_global) if candidate_global else 0.0
    final = len(selected_global) / len(universe)
    return CoverageMetrics(
        global_components=len(universe),
        candidate_components=len(candidate_union),
        selected_components=len(selected_union),
        candidate_global_components=len(candidate_global),
        selected_global_components=len(selected_global),
        candidate_outside_components=len(candidate_union - universe),
        selected_outside_components=len(selected_union - universe),
        candidate_reachable_coverage=reachable,
        selection_efficiency=efficiency,
        final_global_coverage=final,
        factorization_error=abs(final - reachable * efficiency),
    )


def per_type_coverage(
    prompt_components: Mapping[str, Iterable[str]],
    selected_ids: Sequence[str],
    global_components: Iterable[str],
) -> dict[str, dict[str, int | float]]:
    universe = set(global_components)
    candidate_union = union_components(prompt_components)
    selected_union = set().union(
        *(set(prompt_components[prompt_id]) for prompt_id in selected_ids)
    ) if selected_ids else set()
    result: dict[str, dict[str, int | float]] = {}
    for kind in COMPONENT_KINDS:
        global_kind = {value for value in universe if component_kind(value) == kind}
        candidate_kind = candidate_union & global_kind
        selected_kind = selected_union & global_kind
        result[kind] = {
            "global_components": len(global_kind),
            "candidate_global_components": len(candidate_kind),
            "selected_global_components": len(selected_kind),
            "candidate_reachable_coverage": (
                len(candidate_kind) / len(global_kind) if global_kind else 0.0
            ),
            "selection_efficiency": (
                len(selected_kind) / len(candidate_kind) if candidate_kind else 0.0
            ),
            "final_global_coverage": (
                len(selected_kind) / len(global_kind) if global_kind else 0.0
            ),
        }
    return result


def type_balanced_weights(global_components: Iterable[str]) -> dict[str, float]:
    counts = component_counts(global_components)
    nonempty = [kind for kind, count in counts.items() if count > 0]
    total = sum(counts.values())
    if not nonempty or total <= 0:
        raise ValueError("Global component universe is empty")
    target_total = total / len(nonempty)
    return {
        kind: target_total / counts[kind]
        for kind in nonempty
    }


def cumulative_coverage_curve(
    prompt_components: Mapping[str, Iterable[str]],
    ordered_ids: Sequence[str],
    *,
    batch_size: int,
) -> list[dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if set(ordered_ids) != set(prompt_components) or len(ordered_ids) != len(prompt_components):
        raise ValueError("ordered_ids must contain every prompt exactly once")
    covered: set[str] = set()
    curve: list[dict[str, object]] = []
    for start in range(0, len(ordered_ids), batch_size):
        batch = ordered_ids[start : start + batch_size]
        before = set(covered)
        for prompt_id in batch:
            covered.update(prompt_components[prompt_id])
        new = covered - before
        curve.append(
            {
                "prompts": min(start + len(batch), len(ordered_ids)),
                "batch_prompts": list(batch),
                "component_count": len(covered),
                "new_components": len(new),
                "relative_gain": len(new) / len(covered) if covered else 0.0,
                "component_counts_by_type": component_counts(covered),
                "new_component_counts_by_type": component_counts(new),
            }
        )
    return curve
