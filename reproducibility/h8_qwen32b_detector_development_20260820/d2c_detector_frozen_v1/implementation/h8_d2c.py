from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .h8_d2a import (
    ATTACK_FAMILIES,
    ATTACK_ROLE,
    DEVELOPMENT_ALPHA,
    DEVELOPMENT_PERMUTATIONS,
    INTACT_TARGET_ROLE,
    PROMPT_COUNT,
    TOP_R_VALUES,
    _uint64,
    configuration_id,
    empirical_global_p_value,
    select_development_configuration,
    top_r_sum,
)
from .h8_m0ij import mmd2_unbiased_from_kernel
from .h8_precalibration import canonical_sha256
from .h8_score_calibration import score_value


D2C_SCHEMA_VERSION = "h8-d2c-development-comparison-1.0"
EXPECTED_EVALUATION_UNITS = 13
EXPECTED_RESULT_RECORDS = 156


@dataclass(frozen=True)
class MultiTopRPermutationResult:
    prompt_ids: tuple[str, ...]
    observed_raw_mmd2: tuple[float, ...]
    observed_scores: tuple[float, ...]
    diagnostic_max: float
    observed_top_r: Mapping[int, float]
    permutation_top_r: Mapping[int, np.ndarray]
    global_p_values: Mapping[int, float]
    exceedance_counts: Mapping[int, int]
    detected: Mapping[int, bool]
    derived_seed_set_sha256: str


def evaluate_all_top_r(
    kernels_by_prompt: Mapping[str, np.ndarray],
    score_parameters_by_prompt: Mapping[str, Mapping[str, Any]],
    *,
    n_reference: int,
    n_target: int,
    permutation_stream_seed: int,
    permutations: int = DEVELOPMENT_PERMUTATIONS,
    alpha: float = DEVELOPMENT_ALPHA,
) -> MultiTopRPermutationResult:
    """Run one frozen global-permutation stream and reuse it for Top-2/3/4.

    The function deliberately consumes already-frozen kernels and score parameters.
    No measurement or score-layer fitting is possible through this interface.
    """

    prompt_ids = tuple(sorted(kernels_by_prompt))
    if len(prompt_ids) != PROMPT_COUNT or set(prompt_ids) != set(score_parameters_by_prompt):
        raise ValueError("D2-C requires exactly twelve matched prompt kernels and score parameters")
    if min(n_reference, n_target) < 2 or permutations < 1:
        raise ValueError("Invalid D2-C group sizes or permutation count")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("D2-C alpha must lie strictly between zero and one")

    pooled = n_reference + n_target
    reference = np.arange(n_reference, dtype=np.int64)
    target = np.arange(n_reference, pooled, dtype=np.int64)
    matrices: dict[str, np.ndarray] = {}
    raw_observed: list[float] = []
    score_observed: list[float] = []
    for prompt_id in prompt_ids:
        kernel = np.asarray(kernels_by_prompt[prompt_id], dtype=np.float64)
        if kernel.shape != (pooled, pooled) or not np.isfinite(kernel).all():
            raise ValueError(f"Invalid frozen pooled kernel for {prompt_id}")
        if not np.allclose(kernel, kernel.T, atol=1e-14, rtol=1e-14):
            raise ValueError(f"Pooled kernel is not symmetric for {prompt_id}")
        matrices[prompt_id] = kernel
        raw = float(mmd2_unbiased_from_kernel(kernel, reference, target))
        raw_observed.append(raw)
        score_observed.append(float(score_value(raw, score_parameters_by_prompt[prompt_id])))

    permutation_scores = np.empty((permutations, PROMPT_COUNT), dtype=np.float64)
    seed_hash = hashlib.sha256()
    for permutation_id in range(permutations):
        for prompt_position, prompt_id in enumerate(prompt_ids):
            seed = _uint64(
                "h8-d2a-within-prompt-permutation-v1",
                int(permutation_stream_seed),
                permutation_id,
                prompt_id,
            )
            seed_hash.update(seed.to_bytes(8, "little"))
            order = np.random.Generator(np.random.PCG64(seed)).permutation(pooled)
            perm_reference = order[:n_reference]
            perm_target = order[n_reference:]
            raw = mmd2_unbiased_from_kernel(matrices[prompt_id], perm_reference, perm_target)
            permutation_scores[permutation_id, prompt_position] = score_value(
                raw, score_parameters_by_prompt[prompt_id]
            )
    if not np.isfinite(permutation_scores).all() or np.any(permutation_scores < 0.0):
        raise ValueError("D2-C permutation scores contain invalid values")

    sorted_scores = np.sort(permutation_scores, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_scores, axis=1, dtype=np.float64)
    observed_top: dict[int, float] = {}
    permutation_top: dict[int, np.ndarray] = {}
    p_values: dict[int, float] = {}
    exceedances: dict[int, int] = {}
    detected: dict[int, bool] = {}
    for top_r in TOP_R_VALUES:
        observed = top_r_sum(score_observed, top_r)
        values = np.asarray(cumulative[:, top_r - 1], dtype=np.float64)
        p_value = empirical_global_p_value(observed, values)
        observed_top[top_r] = observed
        permutation_top[top_r] = values
        p_values[top_r] = p_value
        exceedances[top_r] = int(np.sum(values >= observed))
        detected[top_r] = bool(p_value <= alpha)

    return MultiTopRPermutationResult(
        prompt_ids=prompt_ids,
        observed_raw_mmd2=tuple(raw_observed),
        observed_scores=tuple(score_observed),
        diagnostic_max=float(max(score_observed)),
        observed_top_r=observed_top,
        permutation_top_r=permutation_top,
        global_p_values=p_values,
        exceedance_counts=exceedances,
        detected=detected,
        derived_seed_set_sha256=seed_hash.hexdigest(),
    )


def result_records_for_unit(
    result: MultiTopRPermutationResult,
    *,
    data_role: str,
    evaluation_unit_id: str,
    sample_structure: str,
    attack_family: str | None,
    permutations: int = DEVELOPMENT_PERMUTATIONS,
    alpha: float = DEVELOPMENT_ALPHA,
) -> list[dict[str, Any]]:
    if data_role not in {INTACT_TARGET_ROLE, ATTACK_ROLE}:
        raise ValueError("D2-C may read development intact/attack roles only")
    if not evaluation_unit_id:
        raise ValueError("D2-C evaluation unit ID is required")
    if data_role == ATTACK_ROLE and attack_family not in ATTACK_FAMILIES:
        raise ValueError("D2-C attack unit has an unknown family")
    if data_role == INTACT_TARGET_ROLE and attack_family is not None:
        raise ValueError("D2-C intact unit cannot have an attack family")
    records: list[dict[str, Any]] = []
    for top_r in TOP_R_VALUES:
        records.append(
            {
                "schema_version": D2C_SCHEMA_VERSION,
                "data_role": data_role,
                "evaluation_unit_id": evaluation_unit_id,
                "attack_family": attack_family,
                "configuration_id": configuration_id(sample_structure, top_r),
                "sample_structure": sample_structure,
                "top_r": top_r,
                "observed_top_r": result.observed_top_r[top_r],
                "diagnostic_max": result.diagnostic_max,
                "global_p_value": result.global_p_values[top_r],
                "exceedance_count": result.exceedance_counts[top_r],
                "detected": result.detected[top_r],
                "permutations": permutations,
                "alpha": alpha,
                "local_p_values_used": False,
                "measurement_or_score_refit_performed": False,
                "final_or_heldout_data_read": False,
            }
        )
    return records


def validate_and_select(
    records: Sequence[Mapping[str, Any]],
    *,
    frozen_selection_rule: Mapping[str, Any],
    expected_selection_rule_sha256: str,
) -> dict[str, Any]:
    if len(records) != EXPECTED_RESULT_RECORDS:
        raise ValueError("D2-C requires exactly 156 configuration-by-unit result records")
    if canonical_sha256(dict(frozen_selection_rule)) != expected_selection_rule_sha256:
        raise ValueError("Frozen configuration-selection rule SHA256 mismatch")
    if any(
        bool(row.get("measurement_or_score_refit_performed"))
        or bool(row.get("final_or_heldout_data_read"))
        for row in records
    ):
        raise ValueError("D2-C result records violate frozen-data or no-refit boundaries")
    selection = select_development_configuration(records)
    if selection["selection_rule"] != dict(frozen_selection_rule):
        raise ValueError("Selection implementation no longer matches the frozen rule")
    selected_id = str(selection["selected_configuration_id"])
    selected_rows = [row for row in records if row["configuration_id"] == selected_id]
    if len(selected_rows) != EXPECTED_EVALUATION_UNITS:
        raise ValueError("Selected detector does not cover all thirteen development units")
    if not all(math.isfinite(float(row["global_p_value"])) for row in records):
        raise ValueError("D2-C contains non-finite global p-values")
    return selection
