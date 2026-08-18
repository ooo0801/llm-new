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
            family_weights={str(key): float(item) for key, item in value["family_weights"].items()},
        )


def fit_family_balanced_scaler(
    prompt_id: str,
    values: np.ndarray,
    pooled_fit_only_values: np.ndarray,
    schema: FeatureSchema,
    epsilon: float = H8_EPSILON,
) -> FamilyBalancedScaler:
    if not prompt_id:
        raise ValueError("prompt_id must be non-empty")
    array = _as_feature_matrix(values, schema)
    pooled = _as_feature_matrix(pooled_fit_only_values, schema)
    mean = array.mean(axis=0, dtype=np.float64)
    prompt_scale = array.std(axis=0, ddof=H8_DDOF, dtype=np.float64)
    pooled_scale = pooled.std(axis=0, ddof=H8_DDOF, dtype=np.float64)
    scale = prompt_scale.copy()
    sources: list[str] = []
    constants: list[bool] = []
    for feature in schema.features:
        index = feature.index
        is_constant = bool(prompt_scale[index] < epsilon)
        constants.append(is_constant)
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
            raise ValueError(
                f"Continuous feature {feature.name!r} is constant both per-prompt and pooled"
            )
        scale[index] = pooled_scale[index]
        sources.append("pooled_fit_only_std_ddof0")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Scaler contains a non-finite or non-positive scale")
    family_weights = {
        family: 1.0 / math.sqrt(stop - start)
        for family, (start, stop) in schema.family_slices.items()
    }
    return FamilyBalancedScaler(
        prompt_id=prompt_id,
        feature_schema_sha256=schema.sha256,
        mean=mean,
        scale=scale,
        scale_source=tuple(sources),
        constant_in_prompt=tuple(constants),
        family_weights=family_weights,
    )


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    distances = (x * x).sum(axis=1)[:, None] + (y * y).sum(axis=1)[None, :] - 2.0 * x @ y.T
    return np.maximum(distances, 0.0)


def h8_median_positive_pairwise_distance(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Bandwidth fit requires a finite 2D matrix with at least two rows")
    squared = _squared_distances(array, array)
    distances = np.sqrt(squared[np.triu_indices(len(array), k=1)])
    positive = distances[distances > 0.0]
    if positive.size == 0:
        raise ValueError("No positive pairwise distance; H8 bandwidth is undefined")
    bandwidth = float(np.median(positive))
    if not math.isfinite(bandwidth) or bandwidth <= H8_EPSILON:
        raise ValueError("H8 bandwidth is non-finite or at/below the protocol floor")
    return bandwidth


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


def _stability_gate(ratios: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(ratios, dtype=np.float64)
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


def make_bandwidth_payload(
    prompt_id: str,
    bandwidth: float,
    scaler_payload_sha256: str,
    calibration_manifest_sha256: str,
    feature_schema_sha256: str,
) -> dict[str, Any]:
    if not math.isfinite(bandwidth) or bandwidth <= H8_EPSILON:
        raise ValueError("Cannot freeze an invalid H8 bandwidth")
    return {
        "schema_version": H8_SCHEMA_VERSION,
        "prompt_id": prompt_id,
        "sigma": float(bandwidth),
        "method": H8_BANDWIDTH_CONVENTION,
        "dtype": H8_DTYPE,
        "scaler_payload_sha256": scaler_payload_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "feature_schema_sha256": feature_schema_sha256,
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
    bandwidth = float(bandwidth_payload["sigma"])
    if not math.isfinite(bandwidth) or bandwidth <= H8_EPSILON:
        raise ValueError("Frozen H8 bandwidth is invalid")
    return scaler, bandwidth
