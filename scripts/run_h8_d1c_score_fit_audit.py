#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from _bootstrap import ROOT
from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_d1b_sampling import load_manifest
from llm_integrity.h8_d1c import (
    AUDIT_ROLE,
    FIT_ROLE,
    ROLE_ORDER,
    STRUCTURE_SIZES,
    audit_fit_parameters,
    null_distribution_from_kernel,
    validate_split_manifest,
)
from llm_integrity.h8_precalibration import (
    FamilyBalancedScaler,
    build_h8_feature_schema,
    canonical_sha256,
    h8_rbf_kernel,
    load_h8_artifact,
)
from llm_integrity.h8_score_calibration import (
    SAMPLE_STRUCTURES,
    fit_score_calibration_parameters,
    load_frozen_mmd_measurement,
    save_score_calibration_artifact,
    score_schema_sha256,
)


DEFAULT_CONFIG = ROOT / "configs" / "h8_d1c_score_parameter_fit_audit.yaml"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes repository")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tracked_clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("Tracked worktree is not clean")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def task_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        row = dict(record.get("task_metadata") or {})
        row["category"] = record.get("category")
        output.append(row)
    return output


def deterministic_feature_extract(extractor: FeatureExtractor, records: list[dict[str, Any]]) -> np.ndarray:
    texts = [str(record["raw_response"]) for record in records]
    unique_texts = sorted(set(texts))
    encoded_unique = np.asarray(
        extractor.transform(
            unique_texts,
            rows=None,
            include_surface=True,
            include_semantic=True,
            include_task=False,
        ),
        dtype=np.float64,
    )
    lookup = {text: encoded_unique[index] for index, text in enumerate(unique_texts)}
    surface_semantic = np.stack([lookup[text] for text in texts]).astype(np.float64)
    task = np.asarray(
        extractor.transform(
            texts,
            rows=task_rows(records),
            include_surface=False,
            include_semantic=False,
            include_task=True,
        ),
        dtype=np.float64,
    )
    return np.concatenate([surface_semantic, task], axis=1).astype(np.float64)


def load_and_validate_inputs(config: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    frozen = config["frozen_inputs"]
    bindings = (
        ("d1b1_report", "d1b1_report_sha256", "D1-B1 PASS report"),
        ("responses", "responses_sha256", "D1-B1 responses"),
        ("attempt_events", "attempt_events_sha256", "D1-B1 attempt events"),
        ("sampling_provenance", "sampling_provenance_file_sha256", "D1-B1 sampling provenance"),
        ("formal_generation_manifest", "formal_generation_manifest_sha256", "D1-B1 generation manifest"),
        ("d1a_report", "d1a_report_sha256", "D1-A report"),
        ("h6_fingerprint", "h6_fingerprint_sha256", "H6 MCC12 fingerprint"),
        ("protocol", "protocol_sha256", "D1-C protocol"),
    )
    for path_key, hash_key, label in bindings:
        require_hash(resolve(frozen[path_key]), frozen[hash_key], label)
    report = json.loads(resolve(frozen["d1b1_report"]).read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("responses_file_sha256") != frozen["responses_sha256"]:
        raise ValueError("D1-B1 report is not PASS or does not bind the response file")
    if report.get("score_parameter_fit_performed") is not False:
        raise ValueError("D1-B1 unexpectedly performed score fitting")
    for key in ("formal_reference_responses", "heldout_responses", "attack_responses"):
        if int(report.get(key, -1)) != 0:
            raise ValueError(f"D1-B1 report contains forbidden responses: {key}")
    if report.get("mmd_manifest_sha256") != frozen["mmd_manifest_sha256"]:
        raise ValueError("D1-B1 report MMD binding mismatch")
    if report.get("score_schema_sha256") != frozen["score_schema_sha256"]:
        raise ValueError("D1-B1 report score-schema binding mismatch")
    provenance = json.loads(resolve(frozen["sampling_provenance"]).read_text(encoding="utf-8"))
    if provenance.get("payload_sha256") != frozen["sampling_provenance_payload_sha256"]:
        raise ValueError("D1-B1 provenance payload hash mismatch")
    if canonical_sha256(provenance.get("payload")) != frozen["sampling_provenance_payload_sha256"]:
        raise ValueError("D1-B1 provenance cannot be reconstructed")
    binding = load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=frozen["mmd_manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )
    threshold = float(config["score_fit"]["numerical_zero_threshold"])
    if score_schema_sha256(threshold) != frozen["score_schema_sha256"]:
        raise ValueError("Configured score schema does not reconstruct to D1-A")

    records = read_jsonl(resolve(frozen["responses"]))
    _, schedule = load_manifest(resolve(frozen["formal_generation_manifest"]), smoke=False)
    if len(records) != 2400 or len(schedule) != 2400:
        raise ValueError("D1-C requires exactly 2,400 frozen D1-B1 records")
    counts: Counter[tuple[str, str]] = Counter()
    ids: set[str] = set()
    seeds: set[int] = set()
    required_false = (
        "eligible_for_formal_reference",
        "eligible_for_heldout_evaluation",
        "eligible_for_target_evaluation",
        "eligible_for_attack_evaluation",
    )
    for position, (record, request) in enumerate(zip(records, schedule)):
        expected = {
            "response_id": request.response_id,
            "schedule_position": position,
            "round_id": request.round_id,
            "prompt_id": request.prompt_id,
            "seed": request.seed,
            "data_role": request.data_role,
        }
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"D1-B1 record/schedule mismatch at {position}:{key}")
        role = str(record["data_role"])
        if role not in ROLE_ORDER or any(record.get(key) is not False for key in required_false):
            raise ValueError("D1-C input has a forbidden role or eligibility flag")
        if record.get("eligible_for_score_calibration_fit") is not (role == FIT_ROLE):
            raise ValueError("D1-B1 fit eligibility mismatch")
        if record.get("eligible_for_score_stability_audit") is not (role == AUDIT_ROLE):
            raise ValueError("D1-B1 audit eligibility mismatch")
        ids.add(str(record["response_id"]))
        seeds.add(int(record["seed"]))
        counts[(role, str(record["prompt_id"]))] += 1
    if len(ids) != 2400 or len(seeds) != 2400:
        raise ValueError("D1-B1 response IDs or generation seeds are not unique")
    expected_keys = {(role, prompt) for role in ROLE_ORDER for prompt in binding.prompt_ids}
    if set(counts) != expected_keys or set(counts.values()) != {100}:
        raise ValueError("D1-B1 role/prompt banks are not exactly 100 each")
    return binding, records, report


def load_measurement_payloads(config: Mapping[str, Any], binding: Any) -> tuple[Any, dict[str, Any], dict[str, float]]:
    archive = resolve(config["frozen_inputs"]["mmd_archive"])
    measurement = archive / "measurement"
    schema = build_h8_feature_schema(int(config["feature_extraction"]["semantic_dimension"]))
    if schema.sha256 != binding.feature_schema_sha256:
        raise ValueError("Feature schema differs from frozen MMD binding")
    scalers: dict[str, Any] = {}
    bandwidths: dict[str, float] = {}
    for prompt_id in binding.prompt_ids:
        scaler_payload = load_h8_artifact(
            measurement / "scaler_candidates" / f"{prompt_id}.json",
            "h8_family_balanced_scaler_candidate",
            binding.scaler_payload_sha256_by_prompt[prompt_id],
        )
        bandwidth_payload = load_h8_artifact(
            measurement / "bandwidth_candidates" / f"{prompt_id}.json",
            "h8_prompt_bandwidth_candidate",
            binding.bandwidth_payload_sha256_by_prompt[prompt_id],
        )
        scaler = FamilyBalancedScaler.from_dict(scaler_payload)
        if scaler.prompt_id != prompt_id or scaler.feature_schema_sha256 != schema.sha256:
            raise ValueError(f"Frozen scaler binding mismatch: {prompt_id}")
        if bool(bandwidth_payload["feature_degenerate"]) != binding.feature_degenerate_by_prompt[prompt_id]:
            raise ValueError(f"Frozen structural-degeneracy mismatch: {prompt_id}")
        sigma = float(bandwidth_payload["sigma"])
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid frozen bandwidth: {prompt_id}")
        scalers[prompt_id] = scaler
        bandwidths[prompt_id] = sigma
    return schema, scalers, bandwidths


def save_envelope(path: Path, artifact_type: str, payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    payload_sha = canonical_sha256(value)
    atomic_json(
        path,
        {
            "artifact_type": artifact_type,
            "schema_version": "h8-d1c-score-fit-audit-1.0",
            "payload_sha256": payload_sha,
            "payload": value,
        },
    )
    return payload_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("authorization_status") != "user_approved" or config.get("model_sampling_authorized") is not False:
        raise PermissionError("D1-C authorization/sampling boundary mismatch")
    if not all(value is True for value in config["forbidden_operations"].values()):
        raise PermissionError("Every D1-C forbidden operation must remain true")
    if list(config["score_fit"]["sample_structures"]) != list(SAMPLE_STRUCTURES):
        raise ValueError("D1-C must preserve all four frozen sample structures")
    analysis_commit = tracked_clean_commit()
    binding, records, d1b1_report = load_and_validate_inputs(config)
    split_path = resolve(config["cpu_split"]["manifest"])
    require_hash(split_path, config["cpu_split"]["manifest_sha256"], "D1-C CPU split manifest")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    stream_seeds = validate_split_manifest(split_manifest, binding.prompt_ids)

    output_config = config["output"]
    result_dir = resolve(output_config["result_directory"])
    frozen_dir = resolve(output_config["frozen_archive"])
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError("D1-C result directory is non-empty; refusing result-dependent overwrite")
    if frozen_dir.exists():
        raise FileExistsError("D1-C frozen archive already exists")
    result_dir.mkdir(parents=True, exist_ok=True)

    schema, scalers, bandwidths = load_measurement_payloads(config, binding)
    feature_config = config["feature_extraction"]
    if schema.dimension != int(feature_config["expected_total_dimension"]):
        raise ValueError("D1-C feature dimension mismatch")
    import torch

    torch.set_num_threads(int(feature_config["torch_num_threads"]))
    torch.use_deterministic_algorithms(bool(feature_config["deterministic_torch_algorithms"]))
    np.random.seed(0)
    extractor = FeatureExtractor(
        semantic_model_name=feature_config["semantic_model"],
        semantic_model_revision=feature_config["semantic_model_revision"],
        semantic_device=feature_config["semantic_device"],
        semantic_local_files_only=bool(feature_config["semantic_local_files_only"]),
    )
    features = deterministic_feature_extract(extractor, records)
    if features.shape != (2400, schema.dimension) or features.dtype != np.float64 or not np.isfinite(features).all():
        raise ValueError("D1-C feature extraction produced invalid data")
    feature_path = result_dir / output_config["feature_matrix"]
    with feature_path.open("wb") as handle:
        np.save(handle, np.asarray(features, dtype="<f8"), allow_pickle=False)
    row_index_path = result_dir / output_config["feature_row_index"]
    row_rows = [
        {
            "row_index": index,
            "response_id": record["response_id"],
            "prompt_id": record["prompt_id"],
            "data_role": record["data_role"],
            "schedule_position": record["schedule_position"],
        }
        for index, record in enumerate(records)
    ]
    row_index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in row_rows),
        encoding="utf-8",
        newline="\n",
    )

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(str(record["data_role"]), str(record["prompt_id"]))].append(index)
    raw = np.empty((2, 4, 12, 1000), dtype=np.float64)
    prompt_positions = {prompt: index for index, prompt in enumerate(binding.prompt_ids)}
    for role_index, role in enumerate(ROLE_ORDER):
        for prompt_id in binding.prompt_ids:
            indices = grouped[(role, prompt_id)]
            values = scalers[prompt_id].transform(features[indices], schema)
            kernel = h8_rbf_kernel(values, values, bandwidths[prompt_id])
            for structure_index, structure in enumerate(SAMPLE_STRUCTURES):
                n_reference, n_target = STRUCTURE_SIZES[structure]
                raw[role_index, structure_index, prompt_positions[prompt_id]] = null_distribution_from_kernel(
                    kernel,
                    stream_seed=stream_seeds[(role, structure, prompt_id)],
                    n_reference=n_reference,
                    n_target=n_target,
                    trials=1000,
                )
    if not np.isfinite(raw).all():
        raise ValueError("D1-C raw null tensor contains NaN or infinity")
    raw_path = result_dir / output_config["raw_null_matrix"]
    with raw_path.open("wb") as handle:
        np.save(handle, np.asarray(raw, dtype="<f8"), allow_pickle=False)
    raw_index = {
        "axis_order": ["role", "sample_structure", "prompt_id", "trial"],
        "shape": list(raw.shape),
        "dtype": "float64-little-endian",
        "roles": list(ROLE_ORDER),
        "sample_structures": list(SAMPLE_STRUCTURES),
        "prompt_ids": list(binding.prompt_ids),
        "trial_count": 1000,
        "raw_negative_values_preserved": True,
        "raw_matrix_sha256": file_sha256(raw_path),
    }
    raw_index_path = result_dir / output_config["raw_null_index"]
    atomic_json(raw_index_path, raw_index)
    fit_nulls = {
        structure: {
            prompt_id: raw[0, s_index, p_index].copy()
            for p_index, prompt_id in enumerate(binding.prompt_ids)
        }
        for s_index, structure in enumerate(SAMPLE_STRUCTURES)
    }
    audit_nulls = {
        structure: {
            prompt_id: raw[1, s_index, p_index].copy()
            for p_index, prompt_id in enumerate(binding.prompt_ids)
        }
        for s_index, structure in enumerate(SAMPLE_STRUCTURES)
    }
    try:
        fit_payload = fit_score_calibration_parameters(
            fit_nulls,
            binding,
            data_role=FIT_ROLE,
            numerical_zero_threshold=float(config["score_fit"]["numerical_zero_threshold"]),
        )
    except ValueError as exc:
        failure = {
            "schema_version": "h8-d1c-report-1.0",
            "phase": config["phase"],
            "status": "FAIL_SCORE_FIT_GATE",
            "reason": str(exc),
            "score_layer": "not_frozen",
            "new_model_responses": 0,
            "created_at_utc": now(),
        }
        atomic_json(result_dir / output_config["final_report"], failure)
        print(json.dumps(failure, indent=2))
        return 2
    fit_payload.update(
        {
            "d1b1_report_sha256": config["frozen_inputs"]["d1b1_report_sha256"],
            "d1b1_responses_sha256": config["frozen_inputs"]["responses_sha256"],
            "d1c_cpu_split_manifest_sha256": config["cpu_split"]["manifest_sha256"],
            "fit_trial_count_per_structure_prompt": 1000,
            "audit_data_used_for_parameter_fit": False,
        }
    )
    candidate_path = result_dir / output_config["fit_candidate"]
    candidate_payload_sha = save_score_calibration_artifact(candidate_path, fit_payload)
    audit = audit_fit_parameters(
        fit_payload,
        audit_nulls,
        binding,
        numerical_zero_threshold=float(config["score_fit"]["numerical_zero_threshold"]),
        minimum_scale_ratio=float(config["score_fit"]["stability_scale_ratio"][0]),
        maximum_scale_ratio=float(config["score_fit"]["stability_scale_ratio"][1]),
        maximum_location_shift_in_fit_scale=float(
            config["score_fit"]["stability_location_shift_over_fit_scale_max"]
        ),
    )
    audit_payload = {
        "schema_version": "h8-d1c-stability-audit-1.0",
        "status": audit.status,
        "fit_parameter_payload_sha256": candidate_payload_sha,
        "audit_role": AUDIT_ROLE,
        "audit_data_used_for_parameter_fit": False,
        "audit_trials_per_structure_prompt": 1000,
        "failures": list(audit.failures),
        "structures": audit.structures,
    }
    audit_path = result_dir / output_config["stability_report"]
    audit_payload_sha = save_envelope(audit_path, "h8_d1c_independent_stability_audit", audit_payload)
    common_report = {
        "schema_version": "h8-d1c-report-1.0",
        "phase": config["phase"],
        "status": audit.status,
        "analysis_commit": analysis_commit,
        "measurement_layer": "frozen",
        "score_layer": "frozen" if audit.status == "PASS" else "not_frozen",
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "new_model_responses": 0,
        "formal_reference_responses": 0,
        "heldout_responses": 0,
        "attack_responses": 0,
        "fit_response_count": 1200,
        "audit_response_count": 1200,
        "fit_trials": 48_000,
        "audit_trials": 48_000,
        "total_pseudo_splits": 96_000,
        "sample_size_selection_performed": False,
        "aggregation_selection_performed": False,
        "audit_data_used_for_parameter_fit": False,
        "raw_negative_mmd_values_preserved": True,
        "d1b1_report_sha256": config["frozen_inputs"]["d1b1_report_sha256"],
        "d1b1_responses_sha256": config["frozen_inputs"]["responses_sha256"],
        "mmd_manifest_sha256": binding.manifest_sha256,
        "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
        "protocol_sha256": config["frozen_inputs"]["protocol_sha256"],
        "cpu_split_manifest_sha256": config["cpu_split"]["manifest_sha256"],
        "feature_matrix_sha256": file_sha256(feature_path),
        "raw_null_matrix_sha256": file_sha256(raw_path),
        "fit_candidate_payload_sha256": candidate_payload_sha,
        "stability_audit_payload_sha256": audit_payload_sha,
        "stability_failures": list(audit.failures),
        "feature_extraction_runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "semantic_model": feature_config["semantic_model"],
            "semantic_model_revision": feature_config["semantic_model_revision"],
            "semantic_device": feature_config["semantic_device"],
        },
        "created_at_utc": now(),
    }
    result_report_path = result_dir / output_config["final_report"]
    atomic_json(result_report_path, common_report)
    if audit.status != "PASS":
        print(json.dumps(common_report, indent=2))
        return 2

    frozen_dir.mkdir(parents=True)
    frozen_fit = dict(fit_payload)
    frozen_fit["artifact_status"] = "frozen"
    frozen_fit["eligible_to_freeze"] = True
    frozen_fit["independent_stability_audit_payload_sha256"] = audit_payload_sha
    combined_path = frozen_dir / "SCORE_CALIBRATION_PARAMETERS.json"
    frozen_fit_sha = save_score_calibration_artifact(combined_path, frozen_fit)
    structure_hashes: dict[str, str] = {}
    structure_dir = frozen_dir / "structures"
    for structure in SAMPLE_STRUCTURES:
        structure_payload = {
            "structure": structure,
            "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
            "measurement_manifest_sha256": binding.manifest_sha256,
            "d1b1_responses_sha256": config["frozen_inputs"]["responses_sha256"],
            "a_global": frozen_fit["structures"][structure]["a_global"],
            "a_global_source": frozen_fit["structures"][structure]["a_global_source"],
            "prompt_parameters": frozen_fit["structures"][structure]["prompt_parameters"],
            "audit_results": audit.structures[structure],
            "audit_data_used_for_parameter_fit": False,
        }
        structure_hashes[structure] = save_envelope(
            structure_dir / f"{structure}.json", "h8_d1c_frozen_structure_parameters", structure_payload
        )
    shutil.copy2(raw_path, frozen_dir / raw_path.name)
    shutil.copy2(raw_index_path, frozen_dir / raw_index_path.name)
    shutil.copy2(audit_path, frozen_dir / audit_path.name)
    shutil.copy2(split_path, frozen_dir / split_path.name)
    shutil.copy2(config_path, frozen_dir / config_path.name)
    protocol_path = resolve(config["frozen_inputs"]["protocol"])
    shutil.copy2(protocol_path, frozen_dir / protocol_path.name)
    implementation_paths = (
        ROOT / "src" / "llm_integrity" / "h8_d1c.py",
        ROOT / "scripts" / "run_h8_d1c_score_fit_audit.py",
    )
    implementation_dir = frozen_dir / "implementation"
    implementation_dir.mkdir()
    for implementation_path in implementation_paths:
        shutil.copy2(implementation_path, implementation_dir / implementation_path.name)
    frozen_report = dict(common_report)
    frozen_report["frozen_score_parameter_payload_sha256"] = frozen_fit_sha
    frozen_report["frozen_structure_payload_sha256"] = structure_hashes
    frozen_report_path = frozen_dir / output_config["final_report"]
    atomic_json(frozen_report_path, frozen_report)
    files = {
        str(path.relative_to(frozen_dir)).replace("\\", "/"): file_sha256(path)
        for path in sorted(frozen_dir.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": "h8-score-calibration-frozen-manifest-1.0",
        "measurement_layer": "frozen",
        "score_layer": "frozen",
        "sample_size": "not_selected",
        "aggregation": "not_selected",
        "detector": "not_frozen",
        "data_scope": [FIT_ROLE, AUDIT_ROLE],
        "audit_data_used_for_parameter_fit": False,
        "new_model_responses": 0,
        "formal_reference_responses": 0,
        "heldout_responses": 0,
        "attack_responses": 0,
        "d1b1_report_sha256": config["frozen_inputs"]["d1b1_report_sha256"],
        "d1b1_responses_sha256": config["frozen_inputs"]["responses_sha256"],
        "mmd_manifest_sha256": binding.manifest_sha256,
        "score_schema_sha256": config["frozen_inputs"]["score_schema_sha256"],
        "cpu_split_manifest_sha256": config["cpu_split"]["manifest_sha256"],
        "score_parameter_payload_sha256": frozen_fit_sha,
        "structure_payload_sha256": structure_hashes,
        "files": files,
    }
    manifest_path = frozen_dir / output_config["frozen_manifest"]
    atomic_json(manifest_path, manifest)
    frozen_report["score_calibration_frozen_manifest_sha256"] = file_sha256(manifest_path)
    atomic_json(result_report_path, frozen_report)
    print(json.dumps(frozen_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
