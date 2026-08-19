from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .h8_precalibration import build_h8_feature_schema, canonical_sha256, load_h8_artifact


SCORE_SCHEMA_VERSION = "h8-score-calibration-1.0"
SCORE_ARTIFACT_TYPE = "h8_score_calibration_parameters"
ROBUST_MAD_MULTIPLIER = 1.4826
DEFAULT_NUMERICAL_ZERO_THRESHOLD = 1e-12
SAMPLE_STRUCTURES = ("r40_q10", "r40_q20", "r60_q10", "r60_q20")
ALLOWED_FIT_DATA_ROLES = {"score_calibration_fit_only", "development_dry_run"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_schema_descriptor(numerical_zero_threshold: float) -> dict[str, Any]:
    if not math.isfinite(numerical_zero_threshold) or numerical_zero_threshold <= 0:
        raise ValueError("numerical_zero_threshold must be finite and positive")
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "sample_structures": list(SAMPLE_STRUCTURES),
        "raw_measurement": "generalized_unequal_size_unbiased_mmd2_float64",
        "raw_negative_values_preserved": True,
        "robust_mad_multiplier": ROBUST_MAD_MULTIPLIER,
        "numerical_zero_threshold": numerical_zero_threshold,
        "nondegenerate_mapping": "max(0,(D-m_j)/a_j)",
        "degenerate_mapping": "max(0,D/a_global_structure)",
        "a_global": "median(nondegenerate a_k strictly above threshold within structure)",
        "epsilon_scale_replacement": False,
        "sample_size_selection": False,
    }


def score_schema_sha256(numerical_zero_threshold: float) -> str:
    return canonical_sha256(score_schema_descriptor(numerical_zero_threshold))


@dataclass(frozen=True)
class FrozenMMDMeasurementBinding:
    manifest_sha256: str
    feature_schema_sha256: str
    feature_schema_payload_sha256: str
    global_exclusion_mask_payload_sha256: str
    global_bandwidth_payload_sha256: str
    mmd_implementation_commit: str
    scaler_payload_sha256_by_prompt: Mapping[str, str]
    bandwidth_payload_sha256_by_prompt: Mapping[str, str]
    feature_degenerate_by_prompt: Mapping[str, bool]

    @property
    def prompt_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.scaler_payload_sha256_by_prompt))


def _safe_archive_path(archive: Path, logical: str) -> Path:
    target = (archive / logical).resolve()
    archive = archive.resolve()
    if target != archive and archive not in target.parents:
        raise ValueError("Frozen manifest path escapes its archive")
    return target


def load_frozen_mmd_measurement(
    archive_directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_mmd_implementation_commit: str,
) -> FrozenMMDMeasurementBinding:
    archive = Path(archive_directory).resolve()
    manifest_path = archive / "MMD_FROZEN_MANIFEST.json"
    if _file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Frozen MMD manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot parse frozen MMD manifest") from exc
    if manifest.get("measurement_layer_status") != "frozen":
        raise ValueError("MMD measurement layer is not frozen")
    if manifest.get("detector_status") != "not_frozen":
        raise ValueError("Unexpected detector status in MMD manifest")
    if manifest.get("mmd_implementation_commit") != expected_mmd_implementation_commit:
        raise ValueError("MMD implementation commit mismatch")
    if manifest.get("data_scope") != "mmd_precalibration_fit_only":
        raise ValueError("Frozen measurement data scope mismatch")
    for key in ("new_model_responses", "formal_reference_responses", "heldout_responses", "attack_responses"):
        if int(manifest.get(key, -1)) != 0:
            raise ValueError(f"Forbidden response count in frozen manifest: {key}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen manifest has no file inventory")
    for logical, expected in files.items():
        path = _safe_archive_path(archive, str(logical))
        if not path.is_file() or _file_sha256(path) != expected:
            raise ValueError(f"Frozen MMD file hash mismatch: {logical}")

    report_path = archive / "validation" / "H8_MMD_PRECALIBRATION_FINAL_REPORT.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("measurement_layer_status") != "frozen":
        raise ValueError("Frozen MMD final report is not PASS/frozen")
    if report.get("detector_status") != "not_frozen":
        raise ValueError("Frozen MMD final report detector status mismatch")
    if report.get("mmd_implementation_commit") != expected_mmd_implementation_commit:
        raise ValueError("MMD final report implementation commit mismatch")
    if report.get("sample_size_selection_performed") is not False:
        raise ValueError("Frozen MMD report indicates forbidden sample-size selection")

    measurement = archive / "measurement"
    schema_payload = load_h8_artifact(
        measurement / "feature_schema_candidate.json",
        "h8_feature_schema_candidate",
        None,
    )
    # The report stores the semantic schema SHA separately from the envelope
    # payload SHA. Verify both via the payload and a deterministic rebuild.
    schema = build_h8_feature_schema(512)
    if schema_payload.get("feature_schema_sha256") != report["feature_schema_sha256"]:
        raise ValueError("Feature schema semantic hash mismatch")
    if schema.sha256 != report["feature_schema_sha256"]:
        raise ValueError("Feature schema cannot be deterministically reconstructed")

    mask_payload = load_h8_artifact(
        measurement / "global_exclusion_mask_candidate.json",
        "h8_global_exclusion_mask_candidate",
        report["global_exclusion_mask_payload_sha256"],
    )
    global_bandwidth = load_h8_artifact(
        measurement / "global_bandwidth_candidate.json",
        "h8_global_bandwidth_candidate",
        report["global_bandwidth_payload_sha256"],
    )
    if global_bandwidth.get("overall_status") != "PASS":
        raise ValueError("Frozen global bandwidth did not PASS")
    if mask_payload.get("feature_schema_sha256") != schema.sha256:
        raise ValueError("Global exclusion mask schema binding mismatch")

    scaler_hashes = report.get("scaler_payload_sha256_by_prompt")
    bandwidth_hashes = report.get("bandwidth_payload_sha256_by_prompt")
    if not isinstance(scaler_hashes, dict) or not isinstance(bandwidth_hashes, dict):
        raise ValueError("Missing frozen per-prompt payload hashes")
    if len(scaler_hashes) != 12 or set(scaler_hashes) != set(bandwidth_hashes):
        raise ValueError("Frozen MMD layer must contain exactly twelve paired prompts")
    degenerate: dict[str, bool] = {}
    for prompt_id in sorted(scaler_hashes):
        scaler_payload = load_h8_artifact(
            measurement / "scaler_candidates" / f"{prompt_id}.json",
            "h8_family_balanced_scaler_candidate",
            scaler_hashes[prompt_id],
        )
        bandwidth_payload = load_h8_artifact(
            measurement / "bandwidth_candidates" / f"{prompt_id}.json",
            "h8_prompt_bandwidth_candidate",
            bandwidth_hashes[prompt_id],
        )
        if scaler_payload.get("feature_schema_sha256") != schema.sha256:
            raise ValueError(f"Scaler schema mismatch: {prompt_id}")
        if scaler_payload.get("global_exclusion_mask_payload_sha256") != report["global_exclusion_mask_payload_sha256"]:
            raise ValueError(f"Scaler mask binding mismatch: {prompt_id}")
        if bandwidth_payload.get("scaler_payload_sha256") != scaler_hashes[prompt_id]:
            raise ValueError(f"Bandwidth/scaler binding mismatch: {prompt_id}")
        if bandwidth_payload.get("feature_schema_sha256") != schema.sha256:
            raise ValueError(f"Bandwidth schema mismatch: {prompt_id}")
        if bandwidth_payload.get("stability_status") != "PASS":
            raise ValueError(f"Bandwidth stability is not PASS: {prompt_id}")
        is_degenerate = bool(bandwidth_payload.get("feature_degenerate"))
        expected_source = "global_degenerate_fallback" if is_degenerate else "prompt_specific"
        if bandwidth_payload.get("bandwidth_source") != expected_source:
            raise ValueError(f"Bandwidth degeneracy/source mismatch: {prompt_id}")
        if is_degenerate and bandwidth_payload.get("global_bandwidth_payload_sha256") != report["global_bandwidth_payload_sha256"]:
            raise ValueError(f"Degenerate bandwidth global binding mismatch: {prompt_id}")
        degenerate[prompt_id] = is_degenerate
    return FrozenMMDMeasurementBinding(
        manifest_sha256=expected_manifest_sha256,
        feature_schema_sha256=schema.sha256,
        feature_schema_payload_sha256=canonical_sha256(schema_payload),
        global_exclusion_mask_payload_sha256=report["global_exclusion_mask_payload_sha256"],
        global_bandwidth_payload_sha256=report["global_bandwidth_payload_sha256"],
        mmd_implementation_commit=expected_mmd_implementation_commit,
        scaler_payload_sha256_by_prompt=dict(sorted(scaler_hashes.items())),
        bandwidth_payload_sha256_by_prompt=dict(sorted(bandwidth_hashes.items())),
        feature_degenerate_by_prompt=dict(sorted(degenerate.items())),
    )


def _finite_vector(values: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite one-dimensional null sample")
    return array


def fit_score_calibration_parameters(
    null_by_structure: Mapping[str, Mapping[str, Sequence[float]]],
    binding: FrozenMMDMeasurementBinding,
    *,
    data_role: str,
    numerical_zero_threshold: float = DEFAULT_NUMERICAL_ZERO_THRESHOLD,
) -> dict[str, Any]:
    if data_role not in ALLOWED_FIT_DATA_ROLES:
        raise ValueError("Target, attack, held-out, and formal-reference data cannot fit score parameters")
    schema_hash = score_schema_sha256(numerical_zero_threshold)
    if set(null_by_structure) != set(SAMPLE_STRUCTURES):
        raise ValueError("All four and only the frozen sample structures are required")
    structure_payloads: dict[str, Any] = {}
    for structure in SAMPLE_STRUCTURES:
        prompt_values = null_by_structure[structure]
        if set(prompt_values) != set(binding.prompt_ids):
            raise ValueError(f"Prompt set mismatch for {structure}")
        raw_scales: dict[str, float] = {}
        medians: dict[str, float] = {}
        for prompt_id in binding.prompt_ids:
            values = _finite_vector(prompt_values[prompt_id], f"{structure}:{prompt_id}")
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            medians[prompt_id] = median
            raw_scales[prompt_id] = ROBUST_MAD_MULTIPLIER * mad
        nondegenerate_scales = [
            raw_scales[prompt_id]
            for prompt_id in binding.prompt_ids
            if not binding.feature_degenerate_by_prompt[prompt_id]
            and raw_scales[prompt_id] > numerical_zero_threshold
        ]
        if not nondegenerate_scales:
            raise ValueError(f"No valid nondegenerate scale for {structure}")
        a_global = float(np.median(np.asarray(nondegenerate_scales, dtype=np.float64)))
        if not math.isfinite(a_global) or a_global <= numerical_zero_threshold:
            raise ValueError(f"Invalid a_global for {structure}")
        prompt_parameters: dict[str, Any] = {}
        for prompt_id in binding.prompt_ids:
            is_degenerate = binding.feature_degenerate_by_prompt[prompt_id]
            own_scale = raw_scales[prompt_id]
            if is_degenerate:
                if own_scale > numerical_zero_threshold:
                    raise ValueError(f"Structurally degenerate prompt has nonzero robust scale: {structure}:{prompt_id}")
                mapping = "degenerate_global_scale"
                effective_scale = a_global
            else:
                if not math.isfinite(own_scale) or own_scale <= numerical_zero_threshold:
                    raise ValueError(f"Nondegenerate prompt scale is numerically zero: {structure}:{prompt_id}")
                mapping = "nondegenerate_centered"
                effective_scale = own_scale
            prompt_parameters[prompt_id] = {
                "feature_degenerate": is_degenerate,
                "null_median_m": medians[prompt_id],
                "own_robust_scale_a": own_scale,
                "effective_scale": effective_scale,
                "mapping": mapping,
                "null_count": len(prompt_values[prompt_id]),
            }
        structure_payloads[structure] = {
            "a_global": a_global,
            "a_global_source": "median_positive_nondegenerate_prompt_scales_within_structure",
            "prompt_parameters": prompt_parameters,
        }
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "score_schema_sha256": schema_hash,
        "artifact_status": "development_dry_run" if data_role == "development_dry_run" else "candidate",
        "data_role": data_role,
        "eligible_to_freeze": data_role == "score_calibration_fit_only",
        "measurement_manifest_sha256": binding.manifest_sha256,
        "feature_schema_sha256": binding.feature_schema_sha256,
        "feature_schema_payload_sha256": binding.feature_schema_payload_sha256,
        "global_exclusion_mask_payload_sha256": binding.global_exclusion_mask_payload_sha256,
        "global_bandwidth_payload_sha256": binding.global_bandwidth_payload_sha256,
        "mmd_implementation_commit": binding.mmd_implementation_commit,
        "scaler_payload_sha256_by_prompt": dict(binding.scaler_payload_sha256_by_prompt),
        "bandwidth_payload_sha256_by_prompt": dict(binding.bandwidth_payload_sha256_by_prompt),
        "sample_size_selection_performed": False,
        "top_r_selection_performed": False,
        "raw_negative_mmd_values_preserved": True,
        "structures": structure_payloads,
    }


def score_value(raw_unbiased_mmd2: float, prompt_parameters: Mapping[str, Any]) -> float:
    raw = float(raw_unbiased_mmd2)
    if not math.isfinite(raw):
        raise ValueError("Raw unbiased MMD^2 must be finite")
    scale = float(prompt_parameters["effective_scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Score scale must be finite and positive")
    mapping = prompt_parameters.get("mapping")
    if mapping == "nondegenerate_centered":
        standardized = (raw - float(prompt_parameters["null_median_m"])) / scale
    elif mapping == "degenerate_global_scale":
        standardized = raw / scale
    else:
        raise ValueError("Unknown D-to-S mapping")
    result = max(0.0, standardized)
    if not math.isfinite(result):
        raise ValueError("Score is not finite")
    return float(result)


def save_score_calibration_artifact(path: str | Path, payload: Mapping[str, Any]) -> str:
    payload_dict = dict(payload)
    payload_sha = canonical_sha256(payload_dict)
    envelope = {
        "artifact_type": SCORE_ARTIFACT_TYPE,
        "schema_version": SCORE_SCHEMA_VERSION,
        "payload_sha256": payload_sha,
        "payload": payload_dict,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return payload_sha


def load_score_calibration_artifact(
    path: str | Path,
    *,
    binding: FrozenMMDMeasurementBinding,
    expected_payload_sha256: str,
    expected_score_schema_sha256: str,
) -> dict[str, Any]:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot parse score-calibration artifact") from exc
    if envelope.get("artifact_type") != SCORE_ARTIFACT_TYPE or envelope.get("schema_version") != SCORE_SCHEMA_VERSION:
        raise ValueError("Score-calibration artifact type/schema mismatch")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Score-calibration payload must be an object")
    actual = canonical_sha256(payload)
    if actual != envelope.get("payload_sha256") or actual != expected_payload_sha256:
        raise ValueError("Score-calibration payload SHA256 mismatch")
    if payload.get("score_schema_sha256") != expected_score_schema_sha256:
        raise ValueError("Score schema SHA256 mismatch")
    if payload.get("measurement_manifest_sha256") != binding.manifest_sha256:
        raise ValueError("Score parameters bind a different MMD manifest")
    if payload.get("feature_schema_sha256") != binding.feature_schema_sha256:
        raise ValueError("Score parameters bind a different feature schema")
    if payload.get("feature_schema_payload_sha256") != binding.feature_schema_payload_sha256:
        raise ValueError("Score parameters bind a different feature-schema payload")
    if payload.get("global_exclusion_mask_payload_sha256") != binding.global_exclusion_mask_payload_sha256:
        raise ValueError("Score parameters bind a different global exclusion mask")
    if payload.get("global_bandwidth_payload_sha256") != binding.global_bandwidth_payload_sha256:
        raise ValueError("Score parameters bind a different global bandwidth")
    if payload.get("mmd_implementation_commit") != binding.mmd_implementation_commit:
        raise ValueError("Score parameters bind a different MMD implementation commit")
    if payload.get("scaler_payload_sha256_by_prompt") != dict(binding.scaler_payload_sha256_by_prompt):
        raise ValueError("Score parameters bind different scalers")
    if payload.get("bandwidth_payload_sha256_by_prompt") != dict(binding.bandwidth_payload_sha256_by_prompt):
        raise ValueError("Score parameters bind different bandwidths")
    if set(payload.get("structures", {})) != set(SAMPLE_STRUCTURES):
        raise ValueError("Score parameters do not preserve all four sample structures")
    return payload
