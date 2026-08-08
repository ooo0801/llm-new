from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Any

import numpy as np


@dataclass
class TestResult:
    statistic: float
    p_value: float
    reject: bool
    alpha: float
    method: str
    effect_size: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "reject": self.reject,
            "alpha": self.alpha,
            "method": self.method,
            "effect_size": self.effect_size,
            "diagnostics": self.diagnostics,
        }


def _validate_test_settings(permutations: int, alpha: float) -> None:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")


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
    _validate_test_settings(permutations, alpha)
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
    return TestResult(
        observed,
        p_value,
        p_value < alpha,
        alpha,
        "pooled_mmd",
        effect,
        {
            "permutations": permutations,
            "exceedances": exceedances,
            "bandwidth": bandwidth,
            "minimum_attainable_p": 1.0 / (permutations + 1),
        },
    )


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
    _validate_test_settings(permutations, alpha)
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
        diagnostics={
            "permutations": permutations,
            "exceedances": exceedances,
            "strata": len(groups),
            "samples_per_group": {
                str(label): int(len(indices)) for label, indices in zip(labels, groups)
            },
            "bandwidths": {
                str(label): float(value) for label, value in zip(labels, bandwidths)
            },
            "minimum_attainable_p": 1.0 / (permutations + 1),
        },
    )


def prompt_stratified_block_mmd_test(
    x: np.ndarray,
    y: np.ndarray,
    strata: list[str],
    blocks: list[str | int],
    permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Prompt-wise MMD with whole generation batches as exchangeable units.

    A single generation seed can produce one response for every fingerprint
    prompt in the same model call. Those prompt responses share a random
    generation block and are not independently exchangeable. This test keeps
    every prompt from a seed block together when permuting group labels.
    """
    _validate_test_settings(permutations, alpha)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    strata_array = np.asarray(strata, dtype=object)
    blocks_array = np.asarray(blocks, dtype=object)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Block-stratified MMD requires equal two-dimensional arrays")
    if len(strata_array) != len(x) or len(blocks_array) != len(x):
        raise ValueError("Strata and block labels must match the sample count")

    prompt_labels = list(dict.fromkeys(strata_array.tolist()))
    block_labels = list(dict.fromkeys(blocks_array.tolist()))
    if len(block_labels) < 2:
        raise ValueError("Block-stratified MMD requires at least two blocks per group")

    def cube(values: np.ndarray) -> np.ndarray:
        result = np.empty((len(block_labels), len(prompt_labels), values.shape[1]), dtype=np.float64)
        for block_index, block in enumerate(block_labels):
            for prompt_index, prompt in enumerate(prompt_labels):
                indices = np.flatnonzero((blocks_array == block) & (strata_array == prompt))
                if len(indices) != 1:
                    raise ValueError("Every block must contain exactly one sample per prompt")
                result[block_index, prompt_index] = values[indices[0]]
        return result

    x_cube = cube(x)
    y_cube = cube(y)
    combined = np.concatenate([x_cube, y_cube], axis=0)
    group_blocks = len(block_labels)
    bandwidths = [
        median_bandwidth(x_cube[:, prompt_index], y_cube[:, prompt_index])
        for prompt_index in range(len(prompt_labels))
    ]

    def statistic(left: np.ndarray, right: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    mmd2_unbiased(
                        left[:, prompt_index],
                        right[:, prompt_index],
                        bandwidths[prompt_index],
                    )
                    for prompt_index in range(len(prompt_labels))
                ]
            )
        )

    observed = statistic(x_cube, y_cube)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(permutations):
        order = rng.permutation(len(combined))
        permuted = statistic(combined[order[:group_blocks]], combined[order[group_blocks:]])
        exceedances += permuted >= observed
    p_value = (exceedances + 1) / (permutations + 1)
    prompt_effects = [
        float(np.linalg.norm(x_cube[:, index].mean(0) - y_cube[:, index].mean(0)))
        for index in range(len(prompt_labels))
    ]
    return TestResult(
        statistic=observed,
        p_value=p_value,
        reject=p_value < alpha,
        alpha=alpha,
        method="prompt_stratified_block_mmd",
        effect_size=float(np.mean(prompt_effects)),
        diagnostics={
            "permutations": permutations,
            "exceedances": exceedances,
            "strata": len(prompt_labels),
            "blocks_per_group": group_blocks,
            "exchangeable_unit": "generation_seed_block",
            "bandwidths": {
                str(label): float(value) for label, value in zip(prompt_labels, bandwidths)
            },
            "minimum_attainable_p": 1.0 / (permutations + 1),
        },
    )


def paired_sign_flip_test(
    x: np.ndarray,
    y: np.ndarray,
    permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> TestResult:
    """Prompt-conditioned correction: signs are flipped within prompt pairs."""
    _validate_test_settings(permutations, alpha)
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


def paired_block_sign_flip_test(
    x: np.ndarray,
    y: np.ndarray,
    blocks: list[str | int],
    permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    exact: bool = True,
) -> TestResult:
    """Paired feature test that flips every prompt in a seed block together."""
    _validate_test_settings(permutations, alpha)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    block_array = np.asarray(blocks, dtype=object)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Paired block test requires equal two-dimensional arrays")
    if len(block_array) != len(x):
        raise ValueError("Block labels must match the number of paired samples")
    labels = list(dict.fromkeys(block_array.tolist()))
    if len(labels) < 2:
        raise ValueError("Paired block test requires at least two blocks")
    groups = [np.flatnonzero(block_array == label) for label in labels]
    sizes = {len(indices) for indices in groups}
    if len(sizes) != 1:
        raise ValueError("Every paired block must contain the same number of samples")
    differences = y - x
    observed = float(np.linalg.norm(differences.mean(axis=0)))

    evaluated = 0
    exceedances = 0
    if exact and len(labels) <= 20:
        for mask in range(1 << len(labels)):
            signed = differences.copy()
            for index, indices in enumerate(groups):
                if not (mask >> index) & 1:
                    signed[indices] *= -1.0
            exceedances += float(np.linalg.norm(signed.mean(axis=0))) >= observed
            evaluated += 1
        p_value = exceedances / evaluated
        mode = "exact"
    else:
        rng = np.random.default_rng(seed)
        for _ in range(permutations):
            signed = differences.copy()
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(groups))
            for sign, indices in zip(signs, groups):
                signed[indices] *= sign
            exceedances += float(np.linalg.norm(signed.mean(axis=0))) >= observed
        evaluated = permutations
        p_value = (exceedances + 1) / (permutations + 1)
        mode = "monte_carlo"
    within_scale = float(np.sqrt(np.mean(np.sum(differences**2, axis=1))))
    effect = observed / max(within_scale, 1e-12)
    return TestResult(
        observed,
        p_value,
        p_value < alpha,
        alpha,
        "paired_block_sign_flip",
        effect,
        {
            "mode": mode,
            "evaluated_sign_patterns": evaluated,
            "exceedances": exceedances,
            "blocks": len(labels),
            "samples_per_block": next(iter(sizes)),
            "exchangeable_unit": "generation_seed_block",
            "minimum_attainable_p": 1.0 / evaluated,
        },
    )


def paired_block_mismatch_binomial_test(
    reference_texts: list[str],
    target_texts: list[str],
    blocks: list[str | int],
    null_block_rate: float = 0.1,
    alpha: float = 0.05,
) -> TestResult:
    """Detect paired output changes using seed blocks and an operational null.

    A block is positive when at least one prompt response differs byte-for-byte
    from the response generated with the same prompt and random seed.  The
    one-sided exact binomial tail tests the number of positive seed blocks
    against a preregistered tolerated intact block-mismatch rate.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not 0.0 < null_block_rate < 1.0:
        raise ValueError("null_block_rate must be between zero and one")
    if len(reference_texts) != len(target_texts):
        raise ValueError("Paired mismatch test requires equal response counts")
    block_array = np.asarray(blocks, dtype=object)
    if len(block_array) != len(reference_texts):
        raise ValueError("Block labels must match the number of paired responses")
    labels = list(dict.fromkeys(block_array.tolist()))
    if len(labels) < 2:
        raise ValueError("Paired mismatch test requires at least two seed blocks")
    groups = [np.flatnonzero(block_array == label) for label in labels]
    sizes = {len(indices) for indices in groups}
    if len(sizes) != 1:
        raise ValueError("Every paired mismatch block must contain the same number of responses")

    mismatches = np.asarray(
        [left != right for left, right in zip(reference_texts, target_texts)],
        dtype=bool,
    )
    positive_by_block = [bool(mismatches[indices].any()) for indices in groups]
    positive_blocks = sum(positive_by_block)
    block_count = len(groups)

    def upper_tail(successes: int) -> float:
        return min(1.0, float(
            sum(
                comb(block_count, value)
                * null_block_rate**value
                * (1.0 - null_block_rate) ** (block_count - value)
                for value in range(successes, block_count + 1)
            )
        ))

    p_value = upper_tail(positive_blocks)
    minimum_reject_blocks = next(
        (value for value in range(block_count + 1) if upper_tail(value) < alpha),
        block_count + 1,
    )
    mismatch_count = int(mismatches.sum())
    return TestResult(
        statistic=float(positive_blocks),
        p_value=p_value,
        reject=p_value < alpha,
        alpha=alpha,
        method="paired_block_mismatch_binomial",
        effect_size=mismatch_count / len(mismatches) if len(mismatches) else 0.0,
        diagnostics={
            "blocks": block_count,
            "samples_per_block": next(iter(sizes)),
            "exchangeable_unit": "generation_seed_block",
            "null_block_mismatch_rate": null_block_rate,
            "mismatch_blocks": positive_blocks,
            "mismatch_responses": mismatch_count,
            "total_responses": len(mismatches),
            "minimum_reject_blocks": minimum_reject_blocks,
            "positive_by_block": positive_by_block,
            "comparison": "utf8_response_byte_exact",
        },
    )


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
