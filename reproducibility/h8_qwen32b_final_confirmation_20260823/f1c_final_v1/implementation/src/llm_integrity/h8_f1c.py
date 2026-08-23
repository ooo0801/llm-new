from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .h8_f1a import (
    ALPHA,
    N_REFERENCE,
    N_TARGET,
    PERMUTATIONS,
    PROMPT_COUNT,
    TOP_R,
)
from .h8_m0ij import mmd2_unbiased_from_kernel
from .h8_score_calibration import score_value


F1C_SCHEMA_VERSION = "h8-f1c-final-performance-confirmation-1.0"
PROMPT_PERMUTATION_DOMAIN = "h8-f1a-within-prompt-permutation-v1"
EXPECTED_P_GRID = 1.0 / (PERMUTATIONS + 1)


def uint64_seed(domain: str, root_seed: int, *parts: object) -> int:
    material = "\0".join([domain, str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "little")


def expand_frozen_prompt_seeds(
    stream_seed: int,
    prompt_ids: Sequence[str],
    *,
    permutations: int = PERMUTATIONS,
) -> tuple[dict[str, np.ndarray], str]:
    """Expand one compressed F1-A stream in its frozen iteration order.

    F1-A stores a stream seed plus the SHA256 of all prompt-level seeds.  This
    function materializes those already-frozen seeds without introducing a new
    seed root, domain, ordering rule, or collision policy.
    """
    ordered = tuple(sorted(str(value) for value in prompt_ids))
    if len(ordered) != PROMPT_COUNT or len(set(ordered)) != PROMPT_COUNT:
        raise ValueError("F1-C requires exactly twelve unique frozen prompt IDs")
    if permutations != PERMUTATIONS:
        raise ValueError("F1-C must use exactly 999 frozen permutations")
    output = {prompt_id: np.empty(permutations, dtype=np.uint64) for prompt_id in ordered}
    digest = hashlib.sha256()
    seen: set[int] = set()
    for permutation_id in range(permutations):
        for prompt_id in ordered:
            seed = uint64_seed(PROMPT_PERMUTATION_DOMAIN, stream_seed, permutation_id, prompt_id)
            if seed in seen:
                raise ValueError("F1-C prompt-level permutation seed collision")
            seen.add(seed)
            output[prompt_id][permutation_id] = seed
            digest.update(seed.to_bytes(8, "little"))
    if len(seen) != permutations * PROMPT_COUNT:
        raise ValueError("F1-C frozen prompt-level seed count changed")
    return output, digest.hexdigest()


def _validate_kernel(kernel: np.ndarray, n_reference: int, n_target: int) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=np.float64)
    expected = n_reference + n_target
    if matrix.shape != (expected, expected) or not np.isfinite(matrix).all():
        raise ValueError("F1-C pooled kernel must be a finite 70x70 float64 matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("F1-C pooled kernel is not symmetric")
    return matrix


def permutation_mmd2_from_explicit_seeds(
    kernel: np.ndarray,
    seeds: Sequence[int] | np.ndarray,
    *,
    n_reference: int = N_REFERENCE,
    n_target: int = N_TARGET,
) -> np.ndarray:
    """Vectorized unequal-size unbiased MMD² for explicit frozen seeds."""
    if n_reference != N_REFERENCE or n_target != N_TARGET:
        raise ValueError("F1-C group sizes are frozen at 60/10")
    matrix = _validate_kernel(kernel, n_reference, n_target)
    seed_array = np.asarray(seeds, dtype=np.uint64)
    if seed_array.shape != (PERMUTATIONS,) or len(set(map(int, seed_array))) != PERMUTATIONS:
        raise ValueError("F1-C requires 999 unique explicit prompt-level seeds")
    pooled = n_reference + n_target
    membership = np.zeros((PERMUTATIONS, pooled), dtype=np.float64)
    for row_index, seed in enumerate(seed_array):
        order = np.random.Generator(np.random.PCG64(int(seed))).permutation(pooled)
        membership[row_index, order[:n_reference]] = 1.0

    weighted = membership @ matrix
    ref_full = np.sum(weighted * membership, axis=1, dtype=np.float64)
    ref_to_all = np.sum(weighted, axis=1, dtype=np.float64)
    diagonal = np.diag(matrix)
    ref_diagonal = membership @ diagonal
    total_full = float(matrix.sum(dtype=np.float64))
    total_diagonal = float(diagonal.sum(dtype=np.float64))
    target_full = total_full - 2.0 * ref_to_all + ref_full
    target_diagonal = total_diagonal - ref_diagonal
    cross = ref_to_all - ref_full
    values = (
        (ref_full - ref_diagonal) / (n_reference * (n_reference - 1))
        + (target_full - target_diagonal) / (n_target * (n_target - 1))
        - 2.0 * cross / (n_reference * n_target)
    )
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (PERMUTATIONS,) or not np.isfinite(values).all():
        raise ValueError("F1-C permutation MMD² contains non-finite values")
    return values


def score_array(raw_values: np.ndarray, prompt_parameters: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(raw_values, dtype=np.float64)
    if raw.shape != (PERMUTATIONS,) or not np.isfinite(raw).all():
        raise ValueError("F1-C raw permutation measurements are invalid")
    scale = float(prompt_parameters["effective_scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("F1-C frozen score scale is invalid")
    mapping = prompt_parameters.get("mapping")
    if mapping == "nondegenerate_centered":
        standardized = (raw - float(prompt_parameters["null_median_m"])) / scale
    elif mapping == "degenerate_global_scale":
        standardized = raw / scale
    else:
        raise ValueError("F1-C encountered an unknown frozen D-to-S mapping")
    result = np.maximum(0.0, standardized).astype(np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("F1-C permutation scores are invalid")
    return result


def numeric_summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("F1-C numeric summary requires a finite nonempty vector")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean(dtype=np.float64)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "negative_fraction": float(np.mean(array < 0.0)),
    }


@dataclass(frozen=True)
class FinalUnitResult:
    observed_raw_mmd2: Mapping[str, float]
    observed_scores: Mapping[str, float]
    observed_top2: float
    top_prompt_ids: tuple[str, str]
    permutation_top2: np.ndarray
    permutation_raw_mmd2: Mapping[str, np.ndarray]
    permutation_scores: Mapping[str, np.ndarray]
    exceedance_count: int
    p_global: float
    alarm: bool
    seed_stream_digest_sha256: str


def evaluate_final_unit(
    kernels_by_prompt: Mapping[str, np.ndarray],
    score_parameters_by_prompt: Mapping[str, Mapping[str, Any]],
    *,
    stream_seed: int,
    expected_stream_digest_sha256: str,
) -> FinalUnitResult:
    prompt_ids = tuple(sorted(kernels_by_prompt))
    if len(prompt_ids) != PROMPT_COUNT or set(prompt_ids) != set(score_parameters_by_prompt):
        raise ValueError("F1-C requires twelve matched prompt kernels and score parameters")
    seeds_by_prompt, stream_digest = expand_frozen_prompt_seeds(stream_seed, prompt_ids)
    if stream_digest != expected_stream_digest_sha256:
        raise ValueError("F1-C expanded permutation seed stream SHA256 mismatch")

    reference = np.arange(N_REFERENCE, dtype=np.int64)
    target = np.arange(N_REFERENCE, N_REFERENCE + N_TARGET, dtype=np.int64)
    observed_raw: dict[str, float] = {}
    observed_scores: dict[str, float] = {}
    permutation_raw: dict[str, np.ndarray] = {}
    permutation_scores: dict[str, np.ndarray] = {}
    score_matrix = np.empty((PERMUTATIONS, PROMPT_COUNT), dtype=np.float64)
    for prompt_position, prompt_id in enumerate(prompt_ids):
        matrix = _validate_kernel(kernels_by_prompt[prompt_id], N_REFERENCE, N_TARGET)
        raw = float(mmd2_unbiased_from_kernel(matrix, reference, target))
        observed_raw[prompt_id] = raw
        observed_scores[prompt_id] = score_value(raw, score_parameters_by_prompt[prompt_id])
        raw_perm = permutation_mmd2_from_explicit_seeds(matrix, seeds_by_prompt[prompt_id])
        score_perm = score_array(raw_perm, score_parameters_by_prompt[prompt_id])
        permutation_raw[prompt_id] = raw_perm
        permutation_scores[prompt_id] = score_perm
        score_matrix[:, prompt_position] = score_perm

    ranked = sorted(observed_scores.items(), key=lambda item: (-item[1], item[0]))
    top_ids = (ranked[0][0], ranked[1][0])
    observed_top2 = float(ranked[0][1] + ranked[1][1])
    sorted_permutation_scores = np.sort(score_matrix, axis=1)
    permutation_top2 = np.asarray(
        sorted_permutation_scores[:, -1] + sorted_permutation_scores[:, -2], dtype=np.float64
    )
    if not np.isfinite(permutation_top2).all():
        raise ValueError("F1-C Top-2 permutation distribution is non-finite")
    exceedance = int(np.sum(permutation_top2 >= observed_top2))
    p_global = float((1 + exceedance) / (PERMUTATIONS + 1))
    alarm = p_global <= ALPHA
    if not math.isclose(p_global * 1000.0, round(p_global * 1000.0), abs_tol=1e-12):
        raise ValueError("F1-C global p-value is off the frozen 0.001 grid")
    if alarm != (exceedance <= 49):
        raise ValueError("F1-C alarm is not equivalent to exceedance_count <= 49")
    return FinalUnitResult(
        observed_raw_mmd2=observed_raw,
        observed_scores=observed_scores,
        observed_top2=observed_top2,
        top_prompt_ids=top_ids,
        permutation_top2=permutation_top2,
        permutation_raw_mmd2=permutation_raw,
        permutation_scores=permutation_scores,
        exceedance_count=exceedance,
        p_global=p_global,
        alarm=alarm,
        seed_stream_digest_sha256=stream_digest,
    )
