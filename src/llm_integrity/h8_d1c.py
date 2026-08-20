from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .h8_m0ij import mmd2_unbiased_from_kernel
from .h8_precalibration import canonical_sha256
from .h8_score_calibration import (
    DEFAULT_NUMERICAL_ZERO_THRESHOLD,
    ROBUST_MAD_MULTIPLIER,
    SAMPLE_STRUCTURES,
    FrozenMMDMeasurementBinding,
    load_score_calibration_artifact,
    score_value,
)


D1C_SCHEMA_VERSION = "h8-d1c-score-fit-audit-1.0"
FIT_ROLE = "score_calibration_fit_only"
AUDIT_ROLE = "score_calibration_stability_audit_only"
ROLE_ORDER = (FIT_ROLE, AUDIT_ROLE)
STRUCTURE_SIZES = {
    "r40_q10": (40, 10),
    "r40_q20": (40, 20),
    "r60_q10": (60, 10),
    "r60_q20": (60, 20),
}


def deterministic_uint64(root_seed: int, *parts: object) -> int:
    if root_seed < 0 or root_seed > (1 << 64) - 1:
        raise ValueError("root_seed must be uint64")
    domain = "h8-d1c-split-v1\0" + "\0".join([str(root_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(domain.encode("utf-8")).digest()[:8], "little")


def trial_split(population_size: int, n_reference: int, n_target: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if population_size != 100:
        raise ValueError("D1-C requires exactly 100 role-specific responses per prompt")
    if min(n_reference, n_target) < 2 or n_reference + n_target > population_size:
        raise ValueError("Invalid D1-C pseudo-split sizes")
    selected = np.random.Generator(np.random.PCG64(seed)).choice(
        population_size, size=n_reference + n_target, replace=False
    )
    return selected[:n_reference], selected[n_reference:]


def _stream_schedule(
    stream_seed: int,
    *,
    population_size: int,
    n_reference: int,
    n_target: int,
    trials: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, str]:
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    schedule_hash = hashlib.sha256()
    seed_hash = hashlib.sha256()
    for trial_id in range(trials):
        seed = deterministic_uint64(stream_seed, "trial", trial_id)
        seed_hash.update(seed.to_bytes(8, "little"))
        reference, target = trial_split(population_size, n_reference, n_target, seed)
        schedule_hash.update(np.asarray(reference, dtype="<u2").tobytes())
        schedule_hash.update(np.asarray(target, dtype="<u2").tobytes())
        splits.append((reference, target))
    return splits, seed_hash.hexdigest(), schedule_hash.hexdigest()


def make_split_manifest(
    prompt_ids: Sequence[str],
    *,
    fit_root_seed: int,
    audit_root_seed: int,
    trials_per_stream: int = 1000,
) -> dict[str, Any]:
    prompts = tuple(sorted(str(value) for value in prompt_ids))
    if len(prompts) != 12 or len(set(prompts)) != 12 or any(not value for value in prompts):
        raise ValueError("D1-C requires exactly twelve unique prompt IDs")
    if trials_per_stream != 1000:
        raise ValueError("D1-C requires exactly 1,000 trials per role/structure/prompt stream")
    roots = {FIT_ROLE: fit_root_seed, AUDIT_ROLE: audit_root_seed}
    if fit_root_seed == audit_root_seed:
        raise ValueError("Fit and audit CPU split roots must be independent")
    stream_rows: list[dict[str, Any]] = []
    all_stream_seeds: set[int] = set()
    all_trial_seeds: set[int] = set()
    global_schedule_hash = hashlib.sha256()
    for role in ROLE_ORDER:
        for structure in SAMPLE_STRUCTURES:
            n_reference, n_target = STRUCTURE_SIZES[structure]
            for prompt_id in prompts:
                stream_seed = deterministic_uint64(roots[role], role, structure, prompt_id)
                if stream_seed in all_stream_seeds:
                    raise ValueError("CPU split stream seed collision")
                all_stream_seeds.add(stream_seed)
                splits, seed_set_sha, schedule_sha = _stream_schedule(
                    stream_seed,
                    population_size=100,
                    n_reference=n_reference,
                    n_target=n_target,
                    trials=trials_per_stream,
                )
                for trial_id in range(trials_per_stream):
                    trial_seed = deterministic_uint64(stream_seed, "trial", trial_id)
                    if trial_seed in all_trial_seeds:
                        raise ValueError("CPU trial seed collision")
                    all_trial_seeds.add(trial_seed)
                global_schedule_hash.update(bytes.fromhex(schedule_sha))
                # Splits are deliberately not serialized: the frozen PCG64/hash
                # rule plus this digest reconstructs and audits them exactly.
                assert len(splits) == trials_per_stream
                stream_rows.append(
                    {
                        "role": role,
                        "structure": structure,
                        "prompt_id": prompt_id,
                        "population_size": 100,
                        "n_reference": n_reference,
                        "n_target": n_target,
                        "trial_count": trials_per_stream,
                        "stream_seed_uint64": stream_seed,
                        "trial_seed_set_sha256": seed_set_sha,
                        "split_schedule_sha256": schedule_sha,
                    }
                )
    payload = {
        "schema_version": D1C_SCHEMA_VERSION,
        "generator": "numpy.random.Generator(PCG64(per_trial_sha256_derived_uint64))",
        "sampling": "choice(population_size=100,size=n_reference+n_target,replace=False)",
        "fit_root_seed_uint64": fit_root_seed,
        "audit_root_seed_uint64": audit_root_seed,
        "roles": list(ROLE_ORDER),
        "sample_structures": list(SAMPLE_STRUCTURES),
        "prompt_ids": list(prompts),
        "trials_per_stream": trials_per_stream,
        "stream_count": len(stream_rows),
        "trial_seed_count": len(all_trial_seeds),
        "stream_seed_unique": len(all_stream_seeds) == len(stream_rows),
        "trial_seed_unique": len(all_trial_seeds) == len(stream_rows) * trials_per_stream,
        "fit_audit_root_independent": True,
        "global_split_schedule_sha256": global_schedule_hash.hexdigest(),
        "streams": stream_rows,
    }
    return {
        "artifact_type": "h8_d1c_cpu_split_manifest",
        "schema_version": D1C_SCHEMA_VERSION,
        "payload_sha256": canonical_sha256(payload),
        "payload": payload,
    }


def validate_split_manifest(manifest: Mapping[str, Any], prompt_ids: Sequence[str]) -> dict[tuple[str, str, str], int]:
    payload = manifest.get("payload")
    if manifest.get("artifact_type") != "h8_d1c_cpu_split_manifest" or not isinstance(payload, dict):
        raise ValueError("Invalid D1-C split manifest envelope")
    if canonical_sha256(payload) != manifest.get("payload_sha256"):
        raise ValueError("D1-C split manifest payload hash mismatch")
    expected = make_split_manifest(
        prompt_ids,
        fit_root_seed=int(payload["fit_root_seed_uint64"]),
        audit_root_seed=int(payload["audit_root_seed_uint64"]),
        trials_per_stream=int(payload["trials_per_stream"]),
    )
    if expected != manifest:
        raise ValueError("D1-C split manifest is not reproducible from its frozen rule")
    return {
        (str(row["role"]), str(row["structure"]), str(row["prompt_id"])): int(row["stream_seed_uint64"])
        for row in payload["streams"]
    }


def null_distribution_from_kernel(
    kernel: np.ndarray,
    *,
    stream_seed: int,
    n_reference: int,
    n_target: int,
    trials: int = 1000,
) -> np.ndarray:
    matrix = np.asarray(kernel, dtype=np.float64)
    if matrix.shape != (100, 100) or not np.isfinite(matrix).all():
        raise ValueError("D1-C kernel must be a finite 100x100 float64 matrix")
    output = np.empty(trials, dtype=np.float64)
    for trial_id in range(trials):
        seed = deterministic_uint64(stream_seed, "trial", trial_id)
        reference, target = trial_split(100, n_reference, n_target, seed)
        output[trial_id] = mmd2_unbiased_from_kernel(matrix, reference, target)
    if not np.isfinite(output).all():
        raise ValueError("D1-C raw MMD null distribution contains NaN or infinity")
    return output


def robust_location_scale(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != 1000 or not np.isfinite(array).all():
        raise ValueError("D1-C null distribution must contain 1,000 finite values")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return median, float(ROBUST_MAD_MULTIPLIER * mad)


@dataclass(frozen=True)
class StabilityAuditResult:
    status: str
    structures: Mapping[str, Any]
    failures: tuple[str, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_score_calibration(
    archive_directory: str | Path,
    *,
    expected_manifest_sha256: str,
    binding: FrozenMMDMeasurementBinding,
    expected_score_schema_sha256: str,
) -> dict[str, Any]:
    archive = Path(archive_directory).resolve()
    manifest_path = archive / "SCORE_CALIBRATION_FROZEN_MANIFEST.json"
    if _file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Frozen score-calibration manifest SHA256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot parse frozen score-calibration manifest") from exc
    expected_states = {
        "measurement_layer": "frozen",
        "score_layer": "frozen",
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
    }
    for key, expected in expected_states.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Frozen score-calibration state mismatch: {key}")
    if manifest.get("audit_data_used_for_parameter_fit") is not False:
        raise ValueError("Audit data contaminated frozen score parameters")
    for key in ("new_model_responses", "formal_reference_responses", "heldout_responses", "attack_responses"):
        if int(manifest.get(key, -1)) != 0:
            raise ValueError(f"Forbidden response count in score manifest: {key}")
    if manifest.get("mmd_manifest_sha256") != binding.manifest_sha256:
        raise ValueError("Frozen score manifest binds a different MMD layer")
    if manifest.get("score_schema_sha256") != expected_score_schema_sha256:
        raise ValueError("Frozen score manifest schema mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Frozen score manifest has no file inventory")
    for logical, expected_hash in files.items():
        path = (archive / str(logical)).resolve()
        if path != archive and archive not in path.parents:
            raise ValueError("Frozen score manifest path escapes archive")
        if not path.is_file() or _file_sha256(path) != expected_hash:
            raise ValueError(f"Frozen score file hash mismatch: {logical}")
    payload = load_score_calibration_artifact(
        archive / "SCORE_CALIBRATION_PARAMETERS.json",
        binding=binding,
        expected_payload_sha256=str(manifest["score_parameter_payload_sha256"]),
        expected_score_schema_sha256=expected_score_schema_sha256,
    )
    if payload.get("artifact_status") != "frozen" or payload.get("audit_data_used_for_parameter_fit") is not False:
        raise ValueError("Score parameter artifact is not a clean frozen fit")
    if payload.get("d1b1_responses_sha256") != manifest.get("d1b1_responses_sha256"):
        raise ValueError("Frozen score response binding mismatch")
    return payload


def audit_fit_parameters(
    fit_payload: Mapping[str, Any],
    audit_null_by_structure: Mapping[str, Mapping[str, Sequence[float]]],
    binding: FrozenMMDMeasurementBinding,
    *,
    numerical_zero_threshold: float = DEFAULT_NUMERICAL_ZERO_THRESHOLD,
    minimum_scale_ratio: float = 0.5,
    maximum_scale_ratio: float = 2.0,
    maximum_location_shift_in_fit_scale: float = 1.0,
) -> StabilityAuditResult:
    if set(audit_null_by_structure) != set(SAMPLE_STRUCTURES):
        raise ValueError("Audit must preserve all four sample structures")
    if set(fit_payload.get("structures", {})) != set(SAMPLE_STRUCTURES):
        raise ValueError("Fit payload must preserve all four sample structures")
    failures: list[str] = []
    structures: dict[str, Any] = {}
    for structure in SAMPLE_STRUCTURES:
        if set(audit_null_by_structure[structure]) != set(binding.prompt_ids):
            raise ValueError(f"Audit prompt set mismatch: {structure}")
        rows: dict[str, Any] = {}
        for prompt_id in binding.prompt_ids:
            values = np.asarray(audit_null_by_structure[structure][prompt_id], dtype=np.float64)
            if values.shape != (1000,) or not np.isfinite(values).all():
                raise ValueError(f"Non-finite or wrong-size audit null: {structure}:{prompt_id}")
            m_audit, a_audit = robust_location_scale(values)
            fit_params = fit_payload["structures"][structure]["prompt_parameters"][prompt_id]
            scores = np.asarray([score_value(value, fit_params) for value in values], dtype=np.float64)
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite audit score: {structure}:{prompt_id}")
            degenerate = bool(binding.feature_degenerate_by_prompt[prompt_id])
            if degenerate:
                max_abs = float(np.max(np.abs(values)))
                max_score = float(np.max(np.abs(scores)))
                passed = max_abs <= numerical_zero_threshold and max_score == 0.0
                row = {
                    "feature_degenerate": True,
                    "m_audit": m_audit,
                    "a_audit": a_audit,
                    "max_abs_raw_unbiased_mmd2": max_abs,
                    "max_abs_score": max_score,
                    "zero_invariant_pass": passed,
                }
            else:
                a_fit = float(fit_params["own_robust_scale_a"])
                m_fit = float(fit_params["null_median_m"])
                scale_ratio = a_audit / a_fit
                location_shift = abs(m_audit - m_fit) / a_fit
                passed = (
                    math.isfinite(scale_ratio)
                    and math.isfinite(location_shift)
                    and minimum_scale_ratio <= scale_ratio <= maximum_scale_ratio
                    and location_shift <= maximum_location_shift_in_fit_scale
                )
                row = {
                    "feature_degenerate": False,
                    "m_fit": m_fit,
                    "a_fit": a_fit,
                    "m_audit": m_audit,
                    "a_audit": a_audit,
                    "a_audit_over_a_fit": scale_ratio,
                    "absolute_location_shift_over_a_fit": location_shift,
                    "scale_ratio_gate": [minimum_scale_ratio, maximum_scale_ratio],
                    "location_shift_gate_max": maximum_location_shift_in_fit_scale,
                    "stability_pass": passed,
                }
            row.update(
                {
                    "raw_negative_fraction": float(np.mean(values < 0.0)),
                    "raw_min": float(values.min()),
                    "raw_max": float(values.max()),
                    "score_min": float(scores.min()),
                    "score_max": float(scores.max()),
                    "all_raw_and_scores_finite": True,
                }
            )
            if not passed:
                failures.append(f"{structure}:{prompt_id}")
            rows[prompt_id] = row
        structures[structure] = {"prompt_audits": rows}
    return StabilityAuditResult(
        status="PASS" if not failures else "FAIL",
        structures=structures,
        failures=tuple(failures),
    )
