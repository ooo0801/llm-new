from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TestResult:
    statistic: float
    p_value: float
    reject: bool
    alpha: float
    method: str
    effect_size: float | None = None

    def as_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "reject": self.reject,
            "alpha": self.alpha,
            "method": self.method,
            "effect_size": self.effect_size,
        }


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0.0)


def median_bandwidth(x: np.ndarray, y: np.ndarray | None = None) -> float:
    combined = np.asarray(x if y is None else np.concatenate([x, y], axis=0), dtype=np.float64)
    distances = squared_distances(combined, combined)
    positive = distances[np.triu_indices_from(distances, k=1)]
    positive = positive[positive > 0]
    if positive.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(positive)))


def rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    return np.exp(-squared_distances(x, y) / (2.0 * bandwidth**2))


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, bandwidth: float | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Unbiased MMD requires at least two samples per group")
    bandwidth = bandwidth or median_bandwidth(x, y)
    kxx = rbf_kernel(x, x, bandwidth)
    kyy = rbf_kernel(y, y, bandwidth)
    kxy = rbf_kernel(x, y, bandwidth)
    x_term = (kxx.sum() - np.trace(kxx)) / (len(x) * (len(x) - 1))
    y_term = (kyy.sum() - np.trace(kyy)) / (len(y) * (len(y) - 1))
    return float(x_term + y_term - 2.0 * kxy.mean())


def mmd_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    bandwidth: float | None = None,
) -> TestResult:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bandwidth = bandwidth or median_bandwidth(x, y)
    observed = mmd2_unbiased(x, y, bandwidth)
    combined = np.concatenate([x, y], axis=0)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        indices = rng.permutation(len(combined))
        perm_x = combined[indices[: len(x)]]
        perm_y = combined[indices[len(x) :]]
        exceedances += mmd2_unbiased(perm_x, perm_y, bandwidth) >= observed
    p_value = (exceedances + 1) / (permutations + 1)
    effect = float(np.linalg.norm(x.mean(0) - y.mean(0)))
    return TestResult(observed, p_value, p_value < alpha, alpha, "pooled_mmd", effect)


def prompt_stratified_mmd_test(
    x: np.ndarray,
    y: np.ndarray,
    strata: list[str],
    permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """
    Prompt-stratified MMD permutation test.

    Each stratum corresponds to one fingerprint prompt. Labels are
    permuted only within the same prompt, so every permuted group keeps
    exactly the same prompt composition.

    The test statistic is the mean of prompt-wise unbiased MMD values.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    strata_array = np.asarray(strata, dtype=object)

    if x.shape != y.shape:
        raise ValueError(
            "Prompt-stratified MMD requires x and y to have equal shapes"
        )

    if x.ndim != 2:
        raise ValueError(
            "Prompt-stratified MMD expects two-dimensional feature arrays"
        )

    if len(strata_array) != len(x):
        raise ValueError(
            "Number of strata labels must equal the number of samples"
        )

    labels = list(dict.fromkeys(strata_array.tolist()))
    groups: list[np.ndarray] = []
    bandwidths: list[float] = []

    for label in labels:
        indices = np.flatnonzero(strata_array == label)

        if len(indices) < 2:
            raise ValueError(
                f"Prompt {label!r} has fewer than two samples per group"
            )

        groups.append(indices)
        bandwidths.append(
            median_bandwidth(
                x[indices],
                y[indices],
            )
        )

    def grouped_statistic(
        left: np.ndarray,
        right: np.ndarray,
    ) -> float:
        values = []

        for indices, bandwidth in zip(groups, bandwidths):
            values.append(
                mmd2_unbiased(
                    left[indices],
                    right[indices],
                    bandwidth,
                )
            )

        return float(np.mean(values))

    observed = grouped_statistic(x, y)

    rng = np.random.default_rng(seed)
    exceedances = 0

    for _ in range(permutations):
        perm_x = x.copy()
        perm_y = y.copy()

        for indices in groups:
            group_size = len(indices)

            combined = np.concatenate(
                [
                    x[indices],
                    y[indices],
                ],
                axis=0,
            )

            order = rng.permutation(len(combined))

            perm_x[indices] = combined[
                order[:group_size]
            ]

            perm_y[indices] = combined[
                order[group_size:]
            ]

        permuted_statistic = grouped_statistic(
            perm_x,
            perm_y,
        )

        exceedances += (
            permuted_statistic >= observed
        )

    p_value = (
        exceedances + 1
    ) / (
        permutations + 1
    )

    prompt_effects = []

    for indices in groups:
        prompt_effects.append(
            float(
                np.linalg.norm(
                    x[indices].mean(axis=0)
                    - y[indices].mean(axis=0)
                )
            )
        )

    effect_size = float(
        np.mean(prompt_effects)
    )

    return TestResult(
        statistic=observed,
        p_value=p_value,
        reject=p_value < alpha,
        alpha=alpha,
        method="prompt_stratified_mmd",
        effect_size=effect_size,
    )


def paired_sign_flip_test(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Prompt-conditioned correction: signs are flipped within prompt pairs."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("Paired test requires equal shapes")
    differences = y - x
    observed = float(np.linalg.norm(differences.mean(axis=0)))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(len(differences), 1))
        statistic = float(np.linalg.norm((differences * signs).mean(axis=0)))
        exceedances += statistic >= observed
    p_value = (exceedances + 1) / (permutations + 1)
    within_scale = float(np.sqrt(np.mean(np.sum(differences**2, axis=1))))
    effect = observed / max(within_scale, 1e-12)
    return TestResult(observed, p_value, p_value < alpha, alpha, "paired_sign_flip", effect)


def normal_power(k: int, delta: float, sigma: float, alpha: float = 0.05) -> float:
    from scipy.stats import norm

    if k <= 0 or sigma <= 0:
        raise ValueError("k and sigma must be positive")
    return float(norm.cdf((np.sqrt(k) * delta - norm.ppf(1 - alpha) * sigma) / sigma))


def required_sample_size(delta: float, sigma: float, alpha: float = 0.05, power: float = 0.8) -> int:
    from scipy.stats import norm

    if delta <= 0 or sigma <= 0 or not 0 < power < 1:
        raise ValueError("Invalid effect, noise, or power")
    value = ((norm.ppf(1 - alpha) + norm.ppf(power)) / (delta / sigma)) ** 2
    return int(np.ceil(value))
