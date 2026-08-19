from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


H8_SCHEMA_VERSION = "h8-mmd-precalibration-1.0"
H8_DTYPE = "float64"
H8_DDOF = 0
H8_EPSILON = 1e-8
H8_BANDWIDTH_CONVENTION = "median_positive_pairwise_euclidean_distance"
H8_GLOBAL_BANDWIDTH_CONVENTION = (
    "median_all_positive_within_prompt_distances_non_degenerate_prompts_only"
)
H8_DATA_ROLE = "mmd_precalibration_fit_only"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class FeatureDefinition:
    index: int
    name: str
    family: str
    feature_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "family": self.family,
            "feature_type": self.feature_type,
        }


@dataclass(frozen=True)
class FeatureSchema:
    features: tuple[FeatureDefinition, ...]
    family_slices: Mapping[str, tuple[int, int]]
    dtype: str = H8_DTYPE

    @property
    def dimension(self) -> int:
        return len(self.features)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": H8_SCHEMA_VERSION,
            "dtype": self.dtype,
            "dimension": self.dimension,
            "features": [feature.as_dict() for feature in self.features],
            "family_slices": {
                family: [int(bounds[0]), int(bounds[1])]
                for family, bounds in self.family_slices.items()
            },
            "family_balance": "divide_each_family_by_sqrt(family_dimension)",
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


_SURFACE_FEATURES = (
    ("character_count", "continuous"),
    ("token_count", "continuous"),
    ("unique_token_ratio", "bounded"),
    ("repeated_token_ratio", "bounded"),
    ("sentence_count", "continuous"),
    ("tokens_per_sentence", "continuous"),
    ("line_count", "continuous"),
    ("digit_ratio", "bounded"),
    ("latin_ratio", "bounded"),
    ("chinese_ratio", "bounded"),
    ("code_mark_count", "continuous"),
)

_TASK_FEATURES = (
    ("exact_answer", "binary"),
    ("expected_answer_contained", "binary"),
    ("expected_contains_rate", "bounded"),
    ("valid_json", "binary"),
    ("refusal", "binary"),
)


def build_h8_feature_schema(semantic_dimension: int) -> FeatureSchema:
    if semantic_dimension <= 0:
        raise ValueError("semantic_dimension must be positive")
    features: list[FeatureDefinition] = []
    for name, feature_type in _SURFACE_FEATURES:
        features.append(FeatureDefinition(len(features), name, "surface", feature_type))
    semantic_start = len(features)
    for semantic_index in range(semantic_dimension):
        features.append(
            FeatureDefinition(
                len(features),
                f"semantic_{semantic_index:04d}",
                "semantic",
                "continuous",
            )
        )
    task_start = len(features)
    for name, feature_type in _TASK_FEATURES:
        features.append(FeatureDefinition(len(features), name, "task", feature_type))
    return FeatureSchema(
        tuple(features),
        {
            "surface": (0, semantic_start),
            "semantic": (semantic_start, task_start),
            "task": (task_start, len(features)),
        },
    )


def _as_feature_matrix(values: np.ndarray, schema: FeatureSchema) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != schema.dimension:
        raise ValueError(
            f"Expected a two-dimensional feature matrix with {schema.dimension} columns"
        )
    if len(array) < 2:
        raise ValueError("At least two fit-only observations are required")
    if not np.isfinite(array).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    return array


@dataclass(frozen=True)
class FamilyBalancedScaler:
    prompt_id: str
    feature_schema_sha256: str
    mean: np.ndarray
    scale: np.ndarray
    scale_source: tuple[str, ...]
    constant_in_prompt: tuple[bool, ...]
    global_exclusion_mask: tuple[bool, ...]
    family_weights: Mapping[str, float]
    data_role: str = H8_DATA_ROLE
    dtype: str = H8_DTYPE
    ddof: int = H8_DDOF

    def transform(self, values: np.ndarray, schema: FeatureSchema) -> np.ndarray:
        if schema.sha256 != self.feature_schema_sha256:
            raise ValueError("Feature schema hash does not match frozen scaler")
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.mean):
            raise ValueError("Feature matrix is incompatible with frozen scaler")
        if not np.isfinite(array).all():
            raise ValueError("Feature matrix contains NaN or infinity")
        transformed = (array - self.mean) / self.scale
        transformed[:, np.asarray(self.global_exclusion_mask, dtype=bool)] = 0.0
        for family, (start, stop) in schema.family_slices.items():
            transformed[:, start:stop] *= self.family_weights[family]
        if not np.isfinite(transformed).all():
            raise ValueError("Scaled feature matrix contains NaN or infinity")
        return np.asarray(transformed, dtype=np.float64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": H8_SCHEMA_VERSION,
            "prompt_id": self.prompt_id,
            "feature_schema_sha256": self.feature_schema_sha256,
            "data_role": self.data_role,
            "dtype": self.dtype,
            "ddof": self.ddof,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "scale_source": list(self.scale_source),
            "constant_in_prompt": list(self.constant_in_prompt),
            "global_exclusion_mask": list(self.global_exclusion_mask),
            "active_dimensions": [
                index for index, excluded in enumerate(self.global_exclusion_mask) if not excluded
            ],
            "family_weights": dict(self.family_weights),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FamilyBalancedScaler":
        if value.get("schema_version") != H8_SCHEMA_VERSION:
            raise ValueError("Unsupported H8 scaler schema version")
        if value.get("data_role") != H8_DATA_ROLE:
            raise ValueError("H8 scaler was not fitted on mmd_precalibration_fit_only data")
        if value.get("dtype") != H8_DTYPE or int(value.get("ddof", -1)) != H8_DDOF:
            raise ValueError("H8 scaler dtype/ddof convention mismatch")
        return cls(
            prompt_id=str(value["prompt_id"]),
            feature_schema_sha256=str(value["feature_schema_sha256"]),
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            scale_source=tuple(str(item) for item in value["scale_source"]),
            constant_in_prompt=tuple(bool(item) for item in value["constant_in_prompt"]),
            global_exclusion_mask=tuple(
                bool(item) for item in value["global_exclusion_mask"]
            ),
            family_weights={str(key): float(item) for key, item in value["family_weights"].items()},
        )


def global_continuous_exclusion_mask(
    pooled_fit_only_values: np.ndarray,
    schema: FeatureSchema,
    epsilon: float = H8_EPSILON,
) -> tuple[bool, ...]:
    pooled = _as_feature_matrix(pooled_fit_only_values, schema)
    pooled_scale = pooled.std(axis=0, ddof=H8_DDOF, dtype=np.float64)
    return tuple(
        bool(feature.feature_type == "continuous" and pooled_scale[feature.index] < epsilon)
        for feature in schema.features
    )


def fit_family_balanced_scaler(
    prompt_id: str,
    values: np.ndarray,
    pooled_fit_only_values: np.ndarray,
    schema: FeatureSchema,
    epsilon: float = H8_EPSILON,
    exclusion_mask: Sequence[bool] | None = None,
) -> FamilyBalancedScaler:
    if not prompt_id:
        raise ValueError("prompt_id must be non-empty")
    array = _as_feature_matrix(values, schema)
    pooled = _as_feature_matrix(pooled_fit_only_values, schema)
    mean = array.mean(axis=0, dtype=np.float64)
    prompt_scale = array.std(axis=0, ddof=H8_DDOF, dtype=np.float64)
    pooled_scale = pooled.std(axis=0, ddof=H8_DDOF, dtype=np.float64)
    excluded = (
        global_continuous_exclusion_mask(pooled, schema, epsilon)
        if exclusion_mask is None
        else tuple(bool(item) for item in exclusion_mask)
    )
    if len(excluded) != schema.dimension:
        raise ValueError("Global exclusion mask dimension mismatch")
    for feature, is_excluded in zip(schema.features, excluded):
        if is_excluded and feature.feature_type != "continuous":
            raise ValueError("Only continuous features may enter the global exclusion mask")
    scale = prompt_scale.copy()
    sources: list[str] = []
    constants: list[bool] = []
    for feature in schema.features:
        index = feature.index
        is_constant = bool(prompt_scale[index] < epsilon)
        constants.append(is_constant)
        if excluded[index]:
            scale[index] = 1.0
            sources.append("global_excluded_pooled_degenerate")
            continue
        if not is_constant:
            sources.append("prompt_std_ddof0")
            continue
        if feature.feature_type in {"binary", "bounded"}:
            scale[index] = 1.0
            sources.append("unit_range_constant")
            continue
        if feature.feature_type != "continuous":
            raise ValueError(f"Unsupported feature type: {feature.feature_type}")
        if pooled_scale[index] < epsilon:
            raise ValueError(f"Continuous feature {feature.name!r} must be globally excluded")
        scale[index] = pooled_scale[index]
        sources.append("pooled_fit_only_std_ddof0")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Scaler contains a non-finite or non-positive scale")
    family_weights: dict[str, float] = {}
    for family, (start, stop) in schema.family_slices.items():
        active_count = sum(not excluded[index] for index in range(start, stop))
        if active_count <= 0:
            raise ValueError(f"Feature family {family!r} has no active dimensions")
        family_weights[family] = 1.0 / math.sqrt(active_count)
    return FamilyBalancedScaler(
        prompt_id=prompt_id,
        feature_schema_sha256=schema.sha256,
        mean=mean,
        scale=scale,
        scale_source=tuple(sources),
        constant_in_prompt=tuple(constants),
        global_exclusion_mask=excluded,
        family_weights=family_weights,
    )


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    distances = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2.0 * x @ y.T
    return np.maximum(distances, 0.0)


def h8_within_prompt_pairwise_distances(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Bandwidth fit requires a finite 2D matrix with at least two rows")
    # Direct subtraction is deliberate: the Gram expansion can create tiny
    # positive cancellation artifacts for two bit-identical feature vectors.
    distances = np.empty(len(array) * (len(array) - 1) // 2, dtype=np.float64)
    cursor = 0
    for index in range(len(array) - 1):
        block = np.linalg.norm(array[index + 1 :] - array[index], axis=1)
        distances[cursor : cursor + len(block)] = block
        cursor += len(block)
    return distances


def h8_positive_within_prompt_distances(values: np.ndarray) -> np.ndarray:
    distances = h8_within_prompt_pairwise_distances(values)
    return distances[distances > 0.0]


def h8_median_positive_pairwise_distance(values: np.ndarray) -> float:
    positive = h8_positive_within_prompt_distances(values)
    if positive.size == 0:
        raise ValueError("No positive pairwise distance; H8 bandwidth is undefined")
    bandwidth = float(np.median(positive))
    if not math.isfinite(bandwidth) or bandwidth <= H8_EPSILON:
        raise ValueError("H8 bandwidth is non-finite or at/below the protocol floor")
    return bandwidth


def h8_global_degenerate_bandwidth(
    transformed_by_prompt: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    positive_by_prompt = {
        prompt_id: h8_positive_within_prompt_distances(values)
        for prompt_id, values in transformed_by_prompt.items()
    }
    non_degenerate = sorted(
        prompt_id for prompt_id, distances in positive_by_prompt.items() if len(distances) > 0
    )
    if not non_degenerate:
        raise ValueError("All fingerprints are feature-degenerate; sigma_global is undefined")
    pooled_positive = np.concatenate([positive_by_prompt[prompt_id] for prompt_id in non_degenerate])
    sigma = float(np.median(pooled_positive))
    if not math.isfinite(sigma) or sigma <= H8_EPSILON:
        raise ValueError("sigma_global is non-finite or at/below the protocol floor")
    return {
        "sigma": sigma,
        "method": H8_GLOBAL_BANDWIDTH_CONVENTION,
        "non_degenerate_prompt_ids": non_degenerate,
        "excluded_degenerate_prompt_ids": sorted(
            prompt_id for prompt_id, distances in positive_by_prompt.items() if len(distances) == 0
        ),
        "positive_within_prompt_distance_count": int(len(pooled_positive)),
        "cross_prompt_distances_used": False,
    }


def h8_rbf_kernel(x: np.ndarray, y: np.ndarray, bandwidth: float) -> np.ndarray:
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 2 or y_array.ndim != 2 or x_array.shape[1] != y_array.shape[1]:
        raise ValueError("Kernel inputs must be compatible two-dimensional matrices")
    if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
        raise ValueError("Kernel inputs contain NaN or infinity")
    return np.exp(-_squared_distances(x_array, y_array) / (2.0 * bandwidth**2))


def h8_mmd2_unbiased_unequal(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Generalized unequal-size unbiased MMD^2; negative estimates are valid."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if len(x_array) < 2 or len(y_array) < 2:
        raise ValueError("Unbiased MMD requires at least two samples in each group")
    kxx = h8_rbf_kernel(x_array, x_array, bandwidth)
    kyy = h8_rbf_kernel(y_array, y_array, bandwidth)
    kxy = h8_rbf_kernel(x_array, y_array, bandwidth)
    n = len(x_array)
    m = len(y_array)
    x_term = (kxx.sum(dtype=np.float64) - np.trace(kxx)) / (n * (n - 1))
    y_term = (kyy.sum(dtype=np.float64) - np.trace(kyy)) / (m * (m - 1))
    cross_term = 2.0 * kxy.sum(dtype=np.float64) / (n * m)
    return float(x_term + y_term - cross_term)


def h8_mmd2_biased(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if len(x_array) < 1 or len(y_array) < 1:
        raise ValueError("Biased MMD requires non-empty groups")
    value = float(
        h8_rbf_kernel(x_array, x_array, bandwidth).mean(dtype=np.float64)
        + h8_rbf_kernel(y_array, y_array, bandwidth).mean(dtype=np.float64)
        - 2.0 * h8_rbf_kernel(x_array, y_array, bandwidth).mean(dtype=np.float64)
    )
    return 0.0 if -1e-12 < value < 0.0 else value


def _stability_gate(ratios: Sequence[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [float("nan") if value is None else float(value) for value in ratios],
        dtype=np.float64,
    )
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        return {"status": "FAIL", "reason": "non_finite_or_empty"}
    q05, q50, q95 = np.quantile(array, [0.05, 0.50, 0.95])
    if q05 >= 0.80 and q95 <= 1.20:
        status = "PASS"
    elif q05 >= 0.70 and q95 <= 1.30:
        status = "WARN"
    else:
        status = "FAIL"
    mad = float(np.median(np.abs(array - q50)))
    mean = float(array.mean())
    return {
        "status": status,
        "q05": float(q05),
        "median": float(q50),
        "q95": float(q95),
        "relative_mad": mad / max(abs(float(q50)), H8_EPSILON),
        "coefficient_of_variation": float(array.std(ddof=0)) / max(abs(mean), H8_EPSILON),
    }


def h8_numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Numeric summary requires a one-dimensional array")
    if len(array) == 0:
        return {
            "count": 0,
            "min": None,
            "q05": None,
            "median": None,
            "mean": None,
            "q95": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("Numeric summary contains NaN or infinity")
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean(dtype=np.float64)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def h8_bandwidth_stability(
    prompt_id: str,
    values_by_prompt: Mapping[str, np.ndarray],
    full_scaler: FamilyBalancedScaler,
    schema: FeatureSchema,
    repetitions: int = 100,
    fraction: float = 0.80,
    seed: int = 2026088101,
) -> dict[str, Any]:
    if prompt_id not in values_by_prompt:
        raise ValueError("prompt_id is absent from values_by_prompt")
    if repetitions <= 0 or not 0.0 < fraction < 1.0:
        raise ValueError("Invalid bandwidth stability settings")
    matrices = {key: _as_feature_matrix(value, schema) for key, value in values_by_prompt.items()}
    full_values = matrices[prompt_id]
    full_bandwidth = h8_median_positive_pairwise_distance(
        full_scaler.transform(full_values, schema)
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    fixed_ratios: list[float] = []
    refit_ratios: list[float] = []
    for _ in range(repetitions):
        subsets: dict[str, np.ndarray] = {}
        for key, matrix in matrices.items():
            subset_size = max(2, int(math.floor(len(matrix) * fraction)))
            indices = rng.choice(len(matrix), size=subset_size, replace=False)
            subsets[key] = matrix[indices]
        prompt_subset = subsets[prompt_id]
        fixed_bandwidth = h8_median_positive_pairwise_distance(
            full_scaler.transform(prompt_subset, schema)
        )
        refit_pooled = np.concatenate(list(subsets.values()), axis=0)
        refit_scaler = fit_family_balanced_scaler(
            prompt_id,
            prompt_subset,
            refit_pooled,
            schema,
        )
        refit_bandwidth = h8_median_positive_pairwise_distance(
            refit_scaler.transform(prompt_subset, schema)
        )
        fixed_ratios.append(fixed_bandwidth / full_bandwidth)
        refit_ratios.append(refit_bandwidth / full_bandwidth)
    fixed_gate = _stability_gate(fixed_ratios)
    refit_gate = _stability_gate(refit_ratios)
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    overall = max((fixed_gate["status"], refit_gate["status"]), key=order.__getitem__)
    return {
        "schema_version": H8_SCHEMA_VERSION,
        "prompt_id": prompt_id,
        "full_bandwidth": full_bandwidth,
        "bandwidth_convention": H8_BANDWIDTH_CONVENTION,
        "repetitions": repetitions,
        "fraction": fraction,
        "seed": seed,
        "fixed_full_scaler": {**fixed_gate, "ratios": fixed_ratios},
        "refit_subsample_scaler": {**refit_gate, "ratios": refit_ratios},
        "overall_status": overall,
        "strict_statistical_confidence_interval": False,
    }


def h8_bandwidth_candidate_stability_suite(
    values_by_prompt: Mapping[str, np.ndarray],
    full_scalers: Mapping[str, FamilyBalancedScaler],
    schema: FeatureSchema,
    repetitions: int = 100,
    fraction: float = 0.80,
    seed: int = 2026088101,
) -> dict[str, Any]:
    """Fit prompt/global bandwidth candidates without any cross-prompt distances.

    The same deterministic subsample is used for every prompt in each repetition.
    A prompt with at least one positive full-data distance remains prompt-specific even
    if its stability fails. Only a structurally all-zero prompt may use sigma_global.
    """
    if repetitions <= 0 or not 0.0 < fraction < 1.0:
        raise ValueError("Invalid bandwidth stability settings")
    prompt_ids = sorted(values_by_prompt)
    if prompt_ids != sorted(full_scalers):
        raise ValueError("values_by_prompt/full_scalers prompt sets differ")
    matrices = {
        prompt_id: _as_feature_matrix(values_by_prompt[prompt_id], schema)
        for prompt_id in prompt_ids
    }
    full_transformed = {
        prompt_id: full_scalers[prompt_id].transform(matrices[prompt_id], schema)
        for prompt_id in prompt_ids
    }
    full_positive = {
        prompt_id: h8_positive_within_prompt_distances(full_transformed[prompt_id])
        for prompt_id in prompt_ids
    }
    non_degenerate = [prompt_id for prompt_id in prompt_ids if len(full_positive[prompt_id]) > 0]
    degenerate = [prompt_id for prompt_id in prompt_ids if len(full_positive[prompt_id]) == 0]
    global_candidate = h8_global_degenerate_bandwidth(full_transformed)
    full_prompt_sigma = {
        prompt_id: float(np.median(full_positive[prompt_id])) for prompt_id in non_degenerate
    }

    fixed_ratios: dict[str, list[float | None]] = {
        prompt_id: [] for prompt_id in non_degenerate
    }
    refit_ratios: dict[str, list[float | None]] = {
        prompt_id: [] for prompt_id in non_degenerate
    }
    global_fixed_ratios: list[float | None] = []
    global_refit_ratios: list[float | None] = []
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(repetitions):
        subsets: dict[str, np.ndarray] = {}
        for prompt_id in prompt_ids:
            matrix = matrices[prompt_id]
            subset_size = max(2, int(math.floor(len(matrix) * fraction)))
            indices = rng.choice(len(matrix), size=subset_size, replace=False)
            subsets[prompt_id] = matrix[indices]

        fixed_positive: dict[str, np.ndarray] = {}
        for prompt_id in prompt_ids:
            transformed = full_scalers[prompt_id].transform(subsets[prompt_id], schema)
            fixed_positive[prompt_id] = h8_positive_within_prompt_distances(transformed)
        fixed_global_parts = [fixed_positive[prompt_id] for prompt_id in non_degenerate if len(fixed_positive[prompt_id])]
        if fixed_global_parts:
            fixed_global = float(np.median(np.concatenate(fixed_global_parts)))
            global_fixed_ratios.append(fixed_global / global_candidate["sigma"])
        else:
            global_fixed_ratios.append(None)

        pooled_subset = np.concatenate([subsets[prompt_id] for prompt_id in prompt_ids], axis=0)
        refit_mask = global_continuous_exclusion_mask(pooled_subset, schema)
        refit_positive: dict[str, np.ndarray] = {}
        for prompt_id in prompt_ids:
            refit_scaler = fit_family_balanced_scaler(
                prompt_id,
                subsets[prompt_id],
                pooled_subset,
                schema,
                exclusion_mask=refit_mask,
            )
            transformed = refit_scaler.transform(subsets[prompt_id], schema)
            refit_positive[prompt_id] = h8_positive_within_prompt_distances(transformed)
        refit_global_parts = [refit_positive[prompt_id] for prompt_id in non_degenerate if len(refit_positive[prompt_id])]
        if refit_global_parts:
            refit_global = float(np.median(np.concatenate(refit_global_parts)))
            global_refit_ratios.append(refit_global / global_candidate["sigma"])
        else:
            global_refit_ratios.append(None)

        for prompt_id in non_degenerate:
            fixed = fixed_positive[prompt_id]
            refit = refit_positive[prompt_id]
            fixed_ratios[prompt_id].append(
                float(np.median(fixed)) / full_prompt_sigma[prompt_id]
                if len(fixed)
                else None
            )
            refit_ratios[prompt_id].append(
                float(np.median(refit)) / full_prompt_sigma[prompt_id]
                if len(refit)
                else None
            )

    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    global_fixed_gate = _stability_gate(global_fixed_ratios)
    global_refit_gate = _stability_gate(global_refit_ratios)
    global_status = max(
        (global_fixed_gate["status"], global_refit_gate["status"]), key=order.__getitem__
    )
    global_stability = {
        **global_candidate,
        "fixed_full_scaler": {**global_fixed_gate, "ratios": global_fixed_ratios},
        "refit_subsample_scaler": {**global_refit_gate, "ratios": global_refit_ratios},
        "overall_status": global_status,
    }

    prompt_results: dict[str, dict[str, Any]] = {}
    for prompt_id in prompt_ids:
        distances = h8_within_prompt_pairwise_distances(full_transformed[prompt_id])
        positives = full_positive[prompt_id]
        if prompt_id in non_degenerate:
            fixed_gate = _stability_gate(fixed_ratios[prompt_id])
            refit_gate = _stability_gate(refit_ratios[prompt_id])
            status = max(
                (fixed_gate["status"], refit_gate["status"]), key=order.__getitem__
            )
            sigma = full_prompt_sigma[prompt_id]
            source = "prompt_specific"
            stability = {
                "fixed_full_scaler": {**fixed_gate, "ratios": fixed_ratios[prompt_id]},
                "refit_subsample_scaler": {**refit_gate, "ratios": refit_ratios[prompt_id]},
            }
        else:
            status = global_status
            sigma = float(global_candidate["sigma"])
            source = "global_degenerate_fallback"
            stability = {
                "fixed_full_scaler": global_stability["fixed_full_scaler"],
                "refit_subsample_scaler": global_stability["refit_subsample_scaler"],
            }
        prompt_results[prompt_id] = {
            "prompt_id": prompt_id,
            "bandwidth_source": source,
            "sigma": sigma,
            "pairwise_distance_count": int(len(distances)),
            "positive_distance_count": int(len(positives)),
            "positive_distance_fraction": float(len(positives) / len(distances)),
            "pairwise_distance_distribution": h8_numeric_summary(distances),
            "positive_distance_distribution": h8_numeric_summary(positives),
            "feature_degenerate": bool(len(positives) == 0),
            "fixed_full_scaler": stability["fixed_full_scaler"],
            "refit_subsample_scaler": stability["refit_subsample_scaler"],
            "overall_status": status,
            "prompt_specific_failure_fell_back_to_global": False,
        }

    overall = max(
        [global_status, *(result["overall_status"] for result in prompt_results.values())],
        key=order.__getitem__,
    )
    return {
        "schema_version": H8_SCHEMA_VERSION,
        "bandwidth_convention": H8_BANDWIDTH_CONVENTION,
        "global_bandwidth_convention": H8_GLOBAL_BANDWIDTH_CONVENTION,
        "repetitions": repetitions,
        "fraction": fraction,
        "seed": seed,
        "prompt_results": prompt_results,
        "global_fallback": global_stability,
        "overall_status": overall,
        "cross_prompt_distances_used": False,
        "prompt_specific_stability_fail_may_fallback": False,
        "strict_statistical_confidence_interval": False,
    }


def make_bandwidth_payload(
    prompt_id: str,
    bandwidth: float,
    scaler_payload_sha256: str,
    calibration_manifest_sha256: str,
    feature_schema_sha256: str,
    bandwidth_source: str = "prompt_specific",
    global_exclusion_mask_sha256: str | None = None,
    global_bandwidth_payload_sha256: str | None = None,
) -> dict[str, Any]:
    if not math.isfinite(bandwidth) or bandwidth <= H8_EPSILON:
        raise ValueError("Cannot freeze an invalid H8 bandwidth")
    if bandwidth_source not in {"prompt_specific", "global_degenerate_fallback"}:
        raise ValueError("Invalid H8 bandwidth source")
    if bandwidth_source == "global_degenerate_fallback" and not global_bandwidth_payload_sha256:
        raise ValueError("Global fallback bandwidth must bind the global bandwidth payload")
    return {
        "schema_version": H8_SCHEMA_VERSION,
        "prompt_id": prompt_id,
        "sigma": float(bandwidth),
        "method": H8_BANDWIDTH_CONVENTION,
        "bandwidth_source": bandwidth_source,
        "dtype": H8_DTYPE,
        "scaler_payload_sha256": scaler_payload_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "feature_schema_sha256": feature_schema_sha256,
        "global_exclusion_mask_sha256": global_exclusion_mask_sha256,
        "global_bandwidth_payload_sha256": global_bandwidth_payload_sha256,
        "data_role": H8_DATA_ROLE,
    }


def save_h8_artifact(path: str | Path, artifact_type: str, payload: Mapping[str, Any]) -> str:
    payload_dict = dict(payload)
    payload_sha256 = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": artifact_type,
        "schema_version": H8_SCHEMA_VERSION,
        "payload_sha256": payload_sha256,
        "payload": payload_dict,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(envelope) + b"\n")
    temporary.replace(target)
    return payload_sha256


def load_h8_artifact(
    path: str | Path,
    expected_artifact_type: str,
    expected_payload_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot load canonical H8 artifact") from exc
    if envelope.get("artifact_type") != expected_artifact_type:
        raise ValueError("H8 artifact type mismatch")
    if envelope.get("schema_version") != H8_SCHEMA_VERSION:
        raise ValueError("H8 artifact schema version mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("H8 artifact payload must be a JSON object")
    actual = canonical_sha256(payload)
    if actual != envelope.get("payload_sha256"):
        raise ValueError("H8 artifact payload hash mismatch")
    if expected_payload_sha256 is not None and actual != expected_payload_sha256:
        raise ValueError("H8 artifact does not match the preregistered payload hash")
    return payload


def load_frozen_scaler_and_bandwidth(
    scaler_path: str | Path,
    bandwidth_path: str | Path,
    schema: FeatureSchema,
    expected_scaler_payload_sha256: str | None = None,
    expected_bandwidth_payload_sha256: str | None = None,
) -> tuple[FamilyBalancedScaler, float]:
    scaler_payload = load_h8_artifact(
        scaler_path, "h8_family_balanced_scaler", expected_scaler_payload_sha256
    )
    scaler_sha = canonical_sha256(scaler_payload)
    scaler = FamilyBalancedScaler.from_dict(scaler_payload)
    if scaler.feature_schema_sha256 != schema.sha256:
        raise ValueError("Frozen scaler feature schema mismatch")
    bandwidth_payload = load_h8_artifact(
        bandwidth_path, "h8_prompt_bandwidth", expected_bandwidth_payload_sha256
    )
    if bandwidth_payload.get("method") != H8_BANDWIDTH_CONVENTION:
        raise ValueError("Forbidden legacy H4 bandwidth convention")
    if bandwidth_payload.get("scaler_payload_sha256") != scaler_sha:
        raise ValueError("Bandwidth artifact is not bound to the loaded scaler")
    if bandwidth_payload.get("feature_schema_sha256") != schema.sha256:
        raise ValueError("Bandwidth feature schema mismatch")
    if bandwidth_payload.get("prompt_id") != scaler.prompt_id:
        raise ValueError("Scaler/bandwidth prompt mismatch")
    source = bandwidth_payload.get("bandwidth_source")
    if source not in {"prompt_specific", "global_degenerate_fallback"}:
        raise ValueError("Frozen H8 bandwidth source is invalid")
    if source == "global_degenerate_fallback" and not bandwidth_payload.get(
        "global_bandwidth_payload_sha256"
    ):
        raise ValueError("Frozen global fallback bandwidth is unbound")
    bandwidth = float(bandwidth_payload["sigma"])
    if not math.isfinite(bandwidth) or bandwidth <= H8_EPSILON:
        raise ValueError("Frozen H8 bandwidth is invalid")
    return scaler, bandwidth
