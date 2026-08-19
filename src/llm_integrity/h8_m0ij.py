from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def deterministic_uint64(root_seed: int, *parts: object) -> int:
    domain = "h8-m0ij-v1\0" + "\0".join([str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[:8], "little")


def frozen_split(
    population_size: int,
    n_reference: int,
    n_target: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if min(n_reference, n_target) < 2 or n_reference + n_target > population_size:
        raise ValueError("Invalid pseudo-MMD group sizes")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(population_size)
    return order[:n_reference], order[n_reference : n_reference + n_target]


def mmd2_unbiased_from_kernel(
    kernel: np.ndarray,
    reference_indices: Sequence[int],
    target_indices: Sequence[int],
) -> float:
    k = np.asarray(kernel, dtype=np.float64)
    ref = np.asarray(reference_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or not np.isfinite(k).all():
        raise ValueError("kernel must be a finite square matrix")
    if len(ref) < 2 or len(target) < 2:
        raise ValueError("Unbiased MMD requires at least two samples per group")
    if len(set(ref.tolist()) | set(target.tolist())) != len(ref) + len(target):
        raise ValueError("reference and target indices must be unique and disjoint")
    kxx = k[np.ix_(ref, ref)]
    kyy = k[np.ix_(target, target)]
    kxy = k[np.ix_(ref, target)]
    n, m = len(ref), len(target)
    x_term = (kxx.sum(dtype=np.float64) - np.trace(kxx)) / (n * (n - 1))
    y_term = (kyy.sum(dtype=np.float64) - np.trace(kyy)) / (m * (m - 1))
    cross_term = 2.0 * kxy.sum(dtype=np.float64) / (n * m)
    # Deliberately preserve negative finite-sample estimates.
    return float(x_term + y_term - cross_term)


def mmd2_biased_from_kernel(
    kernel: np.ndarray,
    reference_indices: Sequence[int],
    target_indices: Sequence[int],
) -> float:
    k = np.asarray(kernel, dtype=np.float64)
    ref = np.asarray(reference_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or not np.isfinite(k).all():
        raise ValueError("kernel must be a finite square matrix")
    if len(ref) < 1 or len(target) < 1:
        raise ValueError("Biased MMD requires non-empty groups")
    if len(set(ref.tolist()) | set(target.tolist())) != len(ref) + len(target):
        raise ValueError("reference and target indices must be unique and disjoint")
    return float(
        k[np.ix_(ref, ref)].mean(dtype=np.float64)
        + k[np.ix_(target, target)].mean(dtype=np.float64)
        - 2.0 * k[np.ix_(ref, target)].mean(dtype=np.float64)
    )


def numerical_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": int(array.size), "nan_count": nan_count, "inf_count": inf_count}
    median = float(np.median(finite))
    return {
        "count": int(array.size),
        "mean": float(finite.mean(dtype=np.float64)),
        "median": median,
        "std": float(finite.std(ddof=0, dtype=np.float64)),
        "mad": float(np.median(np.abs(finite - median))),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "negative_fraction": float(np.mean(finite < 0.0)),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


@dataclass(frozen=True)
class PermutationSanityResult:
    observed: float
    permutation_values: tuple[float, ...]
    exceedance_count: int
    p_value: float
    reproducible: bool
    group_size_correct: bool
    observed_evaluation_count: int = 1


def permutation_sanity(
    kernel: np.ndarray,
    reference_indices: Sequence[int],
    target_indices: Sequence[int],
    *,
    permutation_seed: int,
    permutations: int = 999,
) -> PermutationSanityResult:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    ref = np.asarray(reference_indices, dtype=np.int64)
    target = np.asarray(target_indices, dtype=np.int64)
    pooled = np.concatenate([ref, target])
    observed = mmd2_unbiased_from_kernel(kernel, ref, target)

    def run_once() -> tuple[float, ...]:
        rng = np.random.Generator(np.random.PCG64(permutation_seed))
        output: list[float] = []
        for _ in range(permutations):
            permuted = pooled[rng.permutation(len(pooled))]
            output.append(
                mmd2_unbiased_from_kernel(kernel, permuted[: len(ref)], permuted[len(ref) :])
            )
        return tuple(output)

    values = run_once()
    reproducible = values == run_once()
    if not math.isfinite(observed) or not np.isfinite(np.asarray(values)).all():
        raise ValueError("Permutation sanity produced NaN or infinity")
    exceedance = sum(value >= observed for value in values)
    p_value = (1.0 + exceedance) / (permutations + 1.0)
    return PermutationSanityResult(
        observed=observed,
        permutation_values=values,
        exceedance_count=exceedance,
        p_value=float(p_value),
        reproducible=reproducible,
        group_size_correct=len(ref) + len(target) == len(pooled),
    )
