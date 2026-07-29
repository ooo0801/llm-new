from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MacroDistanceResult:
    macro_l2_raw: float
    macro_l2_per_dimension: float
    macro_probability_l2: float
    macro_js: float
    output_dimension: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyMacroSummary:
    family: str
    samples: int
    weight: float
    mean: float
    standard_deviation: float
    standard_error: float
    confidence_lower: float
    confidence_upper: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StratifiedMacroSummary:
    metric: str
    estimate: float
    standard_error: float
    confidence_level: float
    confidence_lower: float
    confidence_upper: float
    bootstrap_samples: int
    family_summaries: tuple[FamilyMacroSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["family_summaries"] = [
            summary.to_dict()
            for summary in self.family_summaries
        ]
        return result


def _as_numpy_vector(value: Any, name: str) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
    except ImportError:
        pass

    array = np.asarray(value, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"received shape {array.shape}"
        )

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")

    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials)

    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("Unable to normalize logits")

    return exponentials / denominator


def compute_macro_distances(
    reference_logits: Any,
    variant_logits: Any,
    epsilon: float = 1e-12,
) -> MacroDistanceResult:
    reference = _as_numpy_vector(
        reference_logits,
        "reference_logits",
    )
    variant = _as_numpy_vector(
        variant_logits,
        "variant_logits",
    )

    if reference.shape != variant.shape:
        raise ValueError(
            "reference_logits and variant_logits must have "
            f"the same shape; received {reference.shape} and "
            f"{variant.shape}"
        )

    difference = variant - reference
    l2_raw = float(np.dot(difference, difference))
    output_dimension = int(reference.size)

    reference_probability = _softmax(reference)
    variant_probability = _softmax(variant)

    probability_difference = (
        variant_probability - reference_probability
    )
    probability_l2 = float(
        np.dot(
            probability_difference,
            probability_difference,
        )
    )

    midpoint = 0.5 * (
        reference_probability + variant_probability
    )

    reference_safe = np.clip(
        reference_probability,
        epsilon,
        1.0,
    )
    variant_safe = np.clip(
        variant_probability,
        epsilon,
        1.0,
    )
    midpoint_safe = np.clip(midpoint, epsilon, 1.0)

    reference_kl = np.sum(
        reference_safe
        * np.log(reference_safe / midpoint_safe)
    )
    variant_kl = np.sum(
        variant_safe
        * np.log(variant_safe / midpoint_safe)
    )
    js = float(0.5 * (reference_kl + variant_kl))

    return MacroDistanceResult(
        macro_l2_raw=l2_raw,
        macro_l2_per_dimension=(
            l2_raw / float(output_dimension)
        ),
        macro_probability_l2=probability_l2,
        macro_js=js,
        output_dimension=output_dimension,
    )


def _validate_family_weights(
    family_weights: Mapping[str, float],
) -> dict[str, float]:
    if not family_weights:
        raise ValueError("family_weights cannot be empty")

    weights = {
        str(family): float(weight)
        for family, weight in family_weights.items()
    }

    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("Family weights cannot be negative")

    total = sum(weights.values())

    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            "Family weights must sum to 1.0; "
            f"received {total}"
        )

    return weights


def summarize_stratified_macro(
    records: Sequence[Mapping[str, Any]],
    family_weights: Mapping[str, float],
    metric: str = "macro_l2_raw",
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> StratifiedMacroSummary:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1"
        )

    weights = _validate_family_weights(family_weights)
    values_by_family: dict[str, list[float]] = {
        family: []
        for family in weights
    }

    for record in records:
        family = str(record.get("family", ""))

        if family not in values_by_family:
            continue

        if metric not in record:
            raise ValueError(
                f"Record for family {family!r} "
                f"does not contain metric {metric!r}"
            )

        value = float(record[metric])

        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite {metric} for family {family!r}"
            )

        values_by_family[family].append(value)

    missing = [
        family
        for family, values in values_by_family.items()
        if not values
    ]

    if missing:
        raise ValueError(
            "No successful samples for families: "
            + ", ".join(missing)
        )

    rng = np.random.default_rng(seed)
    lower_quantile = (1.0 - confidence_level) / 2.0
    upper_quantile = 1.0 - lower_quantile

    family_bootstrap: dict[str, np.ndarray] = {}
    family_summaries: list[FamilyMacroSummary] = []

    for family, weight in weights.items():
        values = np.asarray(
            values_by_family[family],
            dtype=np.float64,
        )
        sample_count = int(values.size)

        sampled_indices = rng.integers(
            0,
            sample_count,
            size=(bootstrap_samples, sample_count),
        )
        bootstrap_means = values[sampled_indices].mean(axis=1)
        family_bootstrap[family] = bootstrap_means

        standard_deviation = (
            float(np.std(values, ddof=1))
            if sample_count > 1
            else 0.0
        )
        standard_error = (
            standard_deviation / np.sqrt(sample_count)
            if sample_count > 1
            else 0.0
        )

        family_summaries.append(
            FamilyMacroSummary(
                family=family,
                samples=sample_count,
                weight=weight,
                mean=float(np.mean(values)),
                standard_deviation=standard_deviation,
                standard_error=float(standard_error),
                confidence_lower=float(
                    np.quantile(
                        bootstrap_means,
                        lower_quantile,
                    )
                ),
                confidence_upper=float(
                    np.quantile(
                        bootstrap_means,
                        upper_quantile,
                    )
                ),
            )
        )

    estimate = float(
        sum(
            summary.weight * summary.mean
            for summary in family_summaries
        )
    )

    stratified_bootstrap = np.zeros(
        bootstrap_samples,
        dtype=np.float64,
    )

    for family, weight in weights.items():
        stratified_bootstrap += (
            weight * family_bootstrap[family]
        )

    standard_error = (
        float(np.std(stratified_bootstrap, ddof=1))
        if bootstrap_samples > 1
        else 0.0
    )

    return StratifiedMacroSummary(
        metric=metric,
        estimate=estimate,
        standard_error=standard_error,
        confidence_level=confidence_level,
        confidence_lower=float(
            np.quantile(
                stratified_bootstrap,
                lower_quantile,
            )
        ),
        confidence_upper=float(
            np.quantile(
                stratified_bootstrap,
                upper_quantile,
            )
        ),
        bootstrap_samples=bootstrap_samples,
        family_summaries=tuple(family_summaries),
    )
