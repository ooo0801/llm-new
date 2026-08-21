from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_d1c import load_frozen_score_calibration
from llm_integrity.h8_d2a import (
    ATTACK_ROLE,
    DEVELOPMENT_ALPHA,
    DEVELOPMENT_PERMUTATIONS,
    INTACT_TARGET_ROLE,
    REFERENCE_ROLE,
    SAMPLE_STRUCTURES,
    STRUCTURE_SIZES,
    TOP_R_VALUES,
    load_canonical_envelope,
    validate_nested_subset_payload,
)
from llm_integrity.h8_d2c import (
    D2C_SCHEMA_VERSION,
    evaluate_all_top_r,
    result_records_for_unit,
    validate_and_select,
)
from llm_integrity.h8_precalibration import (
    FamilyBalancedScaler,
    build_h8_feature_schema,
    canonical_sha256,
    h8_rbf_kernel,
    load_h8_artifact,
)
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement


DEFAULT_CONFIG = ROOT / "configs" / "h8_d2c_development_comparison.yaml"
ARTIFACT_TYPES = {
    "reference_manifest": "h8_d2a_reference_generation_manifest",
    "intact_target_manifest": "h8_d2a_intact_target_generation_manifest",
    "attack_generation_manifest": "h8_d2a_attack_generation_manifest",
    "nested_subset_manifest": "h8_d2a_nested_subset_manifest",
    "permutation_seed_manifest": "h8_d2a_global_permutation_seed_manifest",
    "configuration_manifest": "h8_d2a_configuration_manifest",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Path escapes repository: {logical}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} SHA256 mismatch")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def git_state(config: Mapping[str, Any]) -> str:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if branch != config["git"]["branch"]:
        raise ValueError("D2-C is running on the wrong branch")
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if tracked:
        raise ValueError("Tracked worktree is not clean")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", config["git"]["required_d2b1_archive_commit"], head],
        cwd=ROOT,
    )
    if ancestor.returncode != 0:
        raise ValueError("D2-C HEAD does not descend from the frozen D2-B1 archive commit")
    return head


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


def load_d2a_artifacts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    frozen = config["frozen_inputs"]
    index_path = resolve(frozen["d2a_artifact_index"])
    require_hash(index_path, frozen["d2a_artifact_index_sha256"], "D2-A artifact index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    directory = resolve(frozen["d2a_directory"])
    output: dict[str, dict[str, Any]] = {}
    for key, artifact_type in ARTIFACT_TYPES.items():
        row = index[key]
        output[key] = load_canonical_envelope(
            directory / row["path"],
            expected_artifact_type=artifact_type,
            expected_file_sha256=row["file_sha256"],
            expected_payload_sha256=row["payload_sha256"],
        )
    if file_sha256(directory / index["nested_subset_manifest"]["path"]) != frozen["nested_subset_manifest_sha256"]:
        raise ValueError("Frozen nested-subset manifest file hash mismatch")
    if file_sha256(directory / index["permutation_seed_manifest"]["path"]) != frozen["permutation_seed_manifest_sha256"]:
        raise ValueError("Frozen permutation-seed manifest file hash mismatch")
    validate_nested_subset_payload(output["nested_subset_manifest"])
    return output


def load_revised_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    path = resolve(frozen["revised_configuration_manifest"])
    require_hash(path, frozen["revised_configuration_manifest_sha256"], "D2-B0 revised configuration")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope.get("payload")
    if (
        envelope.get("artifact_type") != "h8_d2b0_revised_configuration_manifest"
        or not isinstance(payload, dict)
        or canonical_sha256(payload) != envelope.get("payload_sha256")
        or envelope.get("payload_sha256") != frozen["revised_configuration_payload_sha256"]
    ):
        raise ValueError("D2-B0 revised configuration envelope mismatch")
    rule = payload.get("selection_rule")
    if not isinstance(rule, dict) or canonical_sha256(rule) != frozen["selection_rule_sha256"]:
        raise ValueError("Frozen D2-B0 selection rule mismatch")
    if int(payload.get("configuration_count", -1)) != 12 or len(payload.get("configurations", [])) != 12:
        raise ValueError("Frozen D2-B0 candidate configuration count changed")
    return payload


def validate_records_and_manifests(
    config: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    frozen = config["frozen_inputs"]
    report_path = resolve(frozen["d2b1_report"])
    index_path = resolve(frozen["d2b1_final_index"])
    response_path = resolve(frozen["responses"])
    require_hash(report_path, frozen["d2b1_report_sha256"], "D2-B1 PASS report")
    require_hash(index_path, frozen["d2b1_final_index_sha256"], "D2-B1 final index")
    require_hash(response_path, frozen["responses_sha256"], "D2-B1 formal development responses")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("responses_file_sha256") != frozen["responses_sha256"]
        or index.get("responses", {}).get("file_sha256") != frozen["responses_sha256"]
        or report.get("development_comparison_performed") is not False
        or report.get("detector_statistics_computed") is not False
        or report.get("sample_size") != "not_selected"
        or report.get("aggregation") != "not_selected"
        or report.get("detector") != "not_frozen"
    ):
        raise ValueError("D2-B1 report is not an eligible frozen pre-comparison bank")
    expected_counts = {
        "formal_development_responses": 3840,
        "reference_responses": 720,
        "intact_target_responses": 1200,
        "attack_responses": 1920,
    }
    if any(int(report.get(key, -1)) != value for key, value in expected_counts.items()):
        raise ValueError("D2-B1 role totals changed")
    if report.get("configuration_selection_rule_sha256") != frozen["selection_rule_sha256"]:
        raise ValueError("D2-B1 report binds a different selection rule")

    schedules = (
        list(artifacts["reference_manifest"]["schedule"])
        + list(artifacts["intact_target_manifest"]["schedule"])
        + list(artifacts["attack_generation_manifest"]["schedule"])
    )
    schedules.sort(key=lambda row: int(row["schedule_position"]))
    records = read_jsonl(response_path)
    if len(schedules) != 3840 or len(records) != 3840:
        raise ValueError("D2-C requires exactly 3,840 manifest-bound records")
    record_by_id: dict[str, dict[str, Any]] = {}
    forbidden_flags = (
        "eligible_for_future_formal_reference",
        "eligible_for_future_final_heldout",
        "eligible_for_future_final_confirmation",
        "eligible_for_score_parameter_fit",
        "eligible_for_measurement_parameter_fit",
    )
    for record in records:
        response_id = str(record.get("response_id", ""))
        if not response_id or response_id in record_by_id:
            raise ValueError("Duplicate or empty D2-B1 response ID")
        if str(record.get("data_role")) not in {REFERENCE_ROLE, INTACT_TARGET_ROLE, ATTACK_ROLE}:
            raise ValueError("D2-C input contains a forbidden response role")
        if any(record.get(flag) is not False for flag in forbidden_flags):
            raise ValueError("D2-C input has forbidden downstream eligibility")
        if record.get("development_comparison_performed") is not False or record.get("detector_statistics_computed") is not False:
            raise ValueError("D2-B1 response was already used for detector comparison")
        record_by_id[response_id] = record
    if set(record_by_id) != {str(row["response_id"]) for row in schedules}:
        raise ValueError("D2-B1 response IDs differ from the frozen generation manifests")
    for row in schedules:
        record = record_by_id[str(row["response_id"])]
        expected = {
            "schedule_position": int(row["schedule_position"]),
            "prompt_id": str(row["prompt_id"]),
            "generation_seed": int(row["generation_seed"]),
            "data_role": str(row["data_role"]),
            "attack_family": row.get("attack_family"),
            "attack_instance_id": row.get("attack_instance_id"),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"D2-B1 record/manifest mismatch: {row['response_id']}")
    return records, record_by_id


def load_measurement_and_score(config: Mapping[str, Any]) -> tuple[Any, Any, Any, dict[str, Any], dict[str, float]]:
    frozen = config["frozen_inputs"]
    binding = load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=frozen["mmd_manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )
    score = load_frozen_score_calibration(
        resolve(frozen["score_archive"]),
        expected_manifest_sha256=frozen["score_manifest_sha256"],
        binding=binding,
        expected_score_schema_sha256=frozen["score_schema_sha256"],
    )
    if canonical_sha256(score) != frozen["score_parameter_payload_sha256"]:
        raise ValueError("Frozen score parameter payload SHA256 mismatch")
    feature = config["feature_extraction"]
    schema = build_h8_feature_schema(int(feature["semantic_dimension"]))
    if schema.sha256 != binding.feature_schema_sha256 or schema.sha256 != feature["expected_feature_schema_sha256"]:
        raise ValueError("D2-C feature schema differs from the frozen MMD layer")
    measurement = resolve(frozen["mmd_archive"]) / "measurement"
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
        sigma = float(bandwidth_payload["sigma"])
        if scaler.prompt_id != prompt_id or scaler.feature_schema_sha256 != schema.sha256 or not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Frozen measurement payload mismatch: {prompt_id}")
        scalers[prompt_id] = scaler
        bandwidths[prompt_id] = sigma
    return binding, score, schema, scalers, bandwidths


def save_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype="<f8"), allow_pickle=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="H8 D2-C frozen development comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        config.get("authorization_status") != "user_approved_formal_comparison"
        or config.get("model_sampling_authorized") is not False
        or config.get("development_comparison_authorized") is not True
        or not all(value is True for value in config["forbidden_operations"].values())
    ):
        raise PermissionError("D2-C authorization or forbidden-operation boundary mismatch")
    detector = config["detector"]
    if (
        list(detector["sample_structures"]) != list(SAMPLE_STRUCTURES)
        or list(detector["top_r_values"]) != list(TOP_R_VALUES)
        or int(detector["permutations"]) != DEVELOPMENT_PERMUTATIONS
        or float(detector["alpha"]) != DEVELOPMENT_ALPHA
        or int(detector["configuration_count"]) != 12
        or int(detector["evaluation_unit_count"]) != 13
        or int(detector["result_record_count"]) != 156
    ):
        raise ValueError("D2-C detector constants differ from the frozen candidate set")
    analysis_commit = git_state(config)
    frozen = config["frozen_inputs"]
    for path_key, hash_key, label in (
        ("protocol", "protocol_sha256", "D2-C protocol"),
        ("fingerprint", "fingerprint_sha256", "H6 MCC12 fingerprint"),
    ):
        require_hash(resolve(frozen[path_key]), frozen[hash_key], label)
    artifacts = load_d2a_artifacts(config)
    revised = load_revised_configuration(config)
    records, record_by_id = validate_records_and_manifests(config, artifacts)
    binding, score_payload, schema, scalers, bandwidths = load_measurement_and_score(config)

    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "mode": "preflight_only",
                    "analysis_commit": analysis_commit,
                    "development_response_count": len(records),
                    "candidate_configuration_count": len(revised["configurations"]),
                    "evaluation_unit_count": len(artifacts["nested_subset_manifest"]["target_evaluation_units"]),
                    "permutation_stream_count": len(artifacts["permutation_seed_manifest"]["streams"]),
                    "selection_rule_sha256": frozen["selection_rule_sha256"],
                    "new_model_responses": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output = config["output"]
    result_dir = resolve(output["result_directory"])
    frozen_dir = resolve(output["frozen_archive"])
    if result_dir.exists() or frozen_dir.exists():
        raise FileExistsError("D2-C output or frozen archive already exists; refusing result-dependent overwrite")
    result_dir.mkdir(parents=True)

    feature_config = config["feature_extraction"]
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
    if features.shape != (3840, schema.dimension) or features.dtype != np.float64 or not np.isfinite(features).all():
        raise ValueError("D2-C deterministic feature extraction produced an invalid matrix")
    feature_path = result_dir / output["feature_matrix"]
    save_npy(feature_path, features)
    row_index_path = result_dir / output["feature_row_index"]
    atomic_jsonl(
        row_index_path,
        (
            {
                "row_index": index,
                "response_id": record["response_id"],
                "prompt_id": record["prompt_id"],
                "data_role": record["data_role"],
                "evaluation_unit_id": record["evaluation_unit_id"],
            }
            for index, record in enumerate(records)
        ),
    )
    feature_by_id = {str(record["response_id"]): features[index] for index, record in enumerate(records)}

    nested = artifacts["nested_subset_manifest"]
    permutation = artifacts["permutation_seed_manifest"]
    stream_by_key = {
        (str(row["evaluation_unit_id"]), str(row["sample_structure"])): row
        for row in permutation["streams"]
    }
    if len(stream_by_key) != 52:
        raise ValueError("Frozen D2-A permutation stream count changed")
    units = list(nested["target_evaluation_units"])
    prompt_ids = tuple(binding.prompt_ids)
    prompt_positions = {prompt_id: position for position, prompt_id in enumerate(prompt_ids)}
    observed_raw = np.empty((13, 4, 12), dtype=np.float64)
    observed_scores = np.empty((13, 4, 12), dtype=np.float64)
    permutation_top = np.empty((13, 4, 3, DEVELOPMENT_PERMUTATIONS), dtype=np.float64)
    result_records: list[dict[str, Any]] = []
    seed_stream_audit: list[dict[str, Any]] = []
    membership_ids: set[str] = set()

    for unit_position, unit in enumerate(units):
        unit_id = str(unit["evaluation_unit_id"])
        role = str(unit["data_role"])
        family = unit.get("attack_family")
        for structure_position, structure in enumerate(SAMPLE_STRUCTURES):
            n_reference, n_target = STRUCTURE_SIZES[structure]
            r_key = "R40" if n_reference == 40 else "R60"
            q_key = "Q10" if n_target == 10 else "Q20"
            kernels: dict[str, np.ndarray] = {}
            parameters: dict[str, Mapping[str, Any]] = {}
            for prompt_id in prompt_ids:
                reference_ids = list(nested["reference_membership_by_prompt"][prompt_id][r_key])
                target_ids = list(unit["members_by_prompt"][prompt_id][q_key])
                if len(reference_ids) != n_reference or len(target_ids) != n_target or set(reference_ids) & set(target_ids):
                    raise ValueError("D2-C frozen nested membership has invalid group sizes or overlap")
                for response_id in reference_ids:
                    record = record_by_id.get(response_id)
                    if record is None or record["data_role"] != REFERENCE_ROLE or record["prompt_id"] != prompt_id:
                        raise ValueError("D2-C reference membership does not resolve to the frozen bank")
                for response_id in target_ids:
                    record = record_by_id.get(response_id)
                    if (
                        record is None
                        or record["data_role"] != role
                        or record["prompt_id"] != prompt_id
                        or record["evaluation_unit_id"] != unit_id
                        or (role == ATTACK_ROLE and record["attack_instance_id"] != unit_id)
                    ):
                        raise ValueError("D2-C target membership does not resolve to the frozen evaluation unit")
                membership_ids.update(reference_ids)
                membership_ids.update(target_ids)
                pooled_ids = reference_ids + target_ids
                matrix = np.stack([feature_by_id[response_id] for response_id in pooled_ids]).astype(np.float64)
                transformed = scalers[prompt_id].transform(matrix, schema)
                kernels[prompt_id] = h8_rbf_kernel(transformed, transformed, bandwidths[prompt_id])
                parameters[prompt_id] = score_payload["structures"][structure]["prompt_parameters"][prompt_id]
            stream = stream_by_key[(unit_id, structure)]
            multi = evaluate_all_top_r(
                kernels,
                parameters,
                n_reference=n_reference,
                n_target=n_target,
                permutation_stream_seed=int(stream["stream_seed_uint64"]),
            )
            if multi.derived_seed_set_sha256 != stream["derived_prompt_permutation_seed_set_sha256"]:
                raise ValueError("D2-C derived permutation seed stream differs from the frozen manifest")
            for prompt_id, raw, score in zip(multi.prompt_ids, multi.observed_raw_mmd2, multi.observed_scores):
                position = prompt_positions[prompt_id]
                observed_raw[unit_position, structure_position, position] = raw
                observed_scores[unit_position, structure_position, position] = score
            for r_position, top_r in enumerate(TOP_R_VALUES):
                permutation_top[unit_position, structure_position, r_position] = multi.permutation_top_r[top_r]
            result_records.extend(
                result_records_for_unit(
                    multi,
                    data_role=role,
                    evaluation_unit_id=unit_id,
                    sample_structure=structure,
                    attack_family=family,
                )
            )
            seed_stream_audit.append(
                {
                    "evaluation_unit_id": unit_id,
                    "sample_structure": structure,
                    "stream_seed_uint64": int(stream["stream_seed_uint64"]),
                    "derived_seed_set_sha256": multi.derived_seed_set_sha256,
                    "status": "PASS",
                }
            )

    if not (np.isfinite(observed_raw).all() and np.isfinite(observed_scores).all() and np.isfinite(permutation_top).all()):
        raise ValueError("D2-C numerical tensors contain NaN or infinity")
    if len(membership_ids) != 3840 or membership_ids != set(record_by_id):
        raise ValueError("D2-C frozen nested membership does not cover the development bank exactly once by role/unit")
    selection = validate_and_select(
        result_records,
        frozen_selection_rule=revised["selection_rule"],
        expected_selection_rule_sha256=frozen["selection_rule_sha256"],
    )
    selected_id = str(selection["selected_configuration_id"])
    selected_config = next(row for row in revised["configurations"] if row["configuration_id"] == selected_id)
    comparison_rows = []
    for row in revised["configurations"]:
        config_id = row["configuration_id"]
        comparison_rows.append({**row, **selection["selection_metrics"][config_id]})

    raw_path = result_dir / output["observed_raw_mmd2"]
    score_path = result_dir / output["observed_scores"]
    permutation_path = result_dir / output["permutation_top_r"]
    save_npy(raw_path, observed_raw)
    save_npy(score_path, observed_scores)
    save_npy(permutation_path, permutation_top)
    records_path = result_dir / output["result_records"]
    atomic_jsonl(records_path, result_records)
    comparison_path = result_dir / output["comparison_table"]
    atomic_json(
        comparison_path,
        {
            "schema_version": D2C_SCHEMA_VERSION,
            "configuration_count": 12,
            "rows": comparison_rows,
            "selection_rule": revised["selection_rule"],
            "selection_rule_sha256": frozen["selection_rule_sha256"],
            "development_counts_are_not_formal_fpr_or_tpr_estimates": True,
        },
    )
    selected_path = result_dir / output["selected_detector"]
    selected_payload = {
        "schema_version": D2C_SCHEMA_VERSION,
        "detector_status": "frozen_development_selected",
        "selected_configuration_id": selected_id,
        "sample_structure": selected_config["sample_structure"],
        "n_reference": selected_config["n_reference"],
        "n_target": selected_config["n_target"],
        "aggregation": selected_config["primary_aggregation"],
        "top_r": selected_config["top_r"],
        "permutations": DEVELOPMENT_PERMUTATIONS,
        "alpha": DEVELOPMENT_ALPHA,
        "global_p_value_formula": "(1 + count(T_perm >= T_observed)) / 1000",
        "local_p_values_used": False,
        "max_diagnostic_only": True,
        "energy_implemented": False,
        "selection_key": selection["selection_key"],
        "selection_metrics": selection["selection_metrics"][selected_id],
        "selection_rule_sha256": frozen["selection_rule_sha256"],
        "formal_fpr_or_tpr_estimation_performed": False,
        "final_heldout_confirmation_status": "not_performed",
    }
    selected_envelope = {
        "artifact_type": "h8_d2c_selected_detector",
        "schema_version": D2C_SCHEMA_VERSION,
        "payload_sha256": canonical_sha256(selected_payload),
        "payload": selected_payload,
    }
    atomic_json(selected_path, selected_envelope)

    result_hashes = {
        "feature_matrix": file_sha256(feature_path),
        "feature_row_index": file_sha256(row_index_path),
        "observed_raw_mmd2": file_sha256(raw_path),
        "observed_scores": file_sha256(score_path),
        "permutation_top_r": file_sha256(permutation_path),
        "result_records": file_sha256(records_path),
        "comparison_table": file_sha256(comparison_path),
        "selected_detector": file_sha256(selected_path),
    }
    report = {
        "schema_version": D2C_SCHEMA_VERSION,
        "phase": "H8_D2C_DEVELOPMENT_COMPARISON",
        "status": "PASS",
        "created_at_utc": now(),
        "analysis_commit": analysis_commit,
        "new_model_responses": 0,
        "development_response_count": len(records),
        "development_reference_responses": Counter(row["data_role"] for row in records)[REFERENCE_ROLE],
        "development_intact_target_responses": Counter(row["data_role"] for row in records)[INTACT_TARGET_ROLE],
        "development_attack_responses": Counter(row["data_role"] for row in records)[ATTACK_ROLE],
        "final_or_heldout_responses": 0,
        "measurement_or_score_refit_performed": False,
        "candidate_configuration_count": 12,
        "evaluation_unit_count": 13,
        "configuration_unit_result_count": len(result_records),
        "global_permutation_stream_count": len(seed_stream_audit),
        "permutations_per_stream": DEVELOPMENT_PERMUTATIONS,
        "derived_prompt_permutation_seed_count": 52 * DEVELOPMENT_PERMUTATIONS * 12,
        "nested_membership_manifest_unchanged": True,
        "unique_membership_response_id_count": len(membership_ids),
        "selection_rule_sha256": frozen["selection_rule_sha256"],
        "selected_detector_payload_sha256": selected_envelope["payload_sha256"],
        "selected_configuration_id": selected_id,
        "selected_configuration": selected_payload,
        "comparison_rows": comparison_rows,
        "development_counts_are_not_formal_fpr_or_tpr_estimates": True,
        "formal_performance_confirmation": "not_performed",
        "frozen_bindings": {
            "d2b1_responses_sha256": frozen["responses_sha256"],
            "mmd_manifest_sha256": frozen["mmd_manifest_sha256"],
            "score_manifest_sha256": frozen["score_manifest_sha256"],
            "score_parameter_payload_sha256": frozen["score_parameter_payload_sha256"],
            "nested_subset_manifest_sha256": frozen["nested_subset_manifest_sha256"],
            "permutation_seed_manifest_sha256": frozen["permutation_seed_manifest_sha256"],
            "revised_configuration_payload_sha256": frozen["revised_configuration_payload_sha256"],
        },
        "tensor_axes": {
            "observed_raw_mmd2": ["evaluation_unit", "sample_structure", "prompt_id"],
            "observed_scores": ["evaluation_unit", "sample_structure", "prompt_id"],
            "permutation_top_r": ["evaluation_unit", "sample_structure", "top_r", "permutation"],
            "evaluation_units": [unit["evaluation_unit_id"] for unit in units],
            "sample_structures": list(SAMPLE_STRUCTURES),
            "prompt_ids": list(prompt_ids),
            "top_r_values": list(TOP_R_VALUES),
        },
        "seed_stream_audit": seed_stream_audit,
        "result_file_sha256": result_hashes,
        "detector_status": "frozen_development_selected",
        "next_gate": "STOP_WAIT_FOR_FRESH_FINAL_HELDOUT_CONFIRMATION_PROTOCOL_APPROVAL",
    }
    report_path = result_dir / output["final_report"]
    atomic_json(report_path, report)

    frozen_dir.mkdir(parents=True)
    archive_sources = {
        output["feature_matrix"]: feature_path,
        output["feature_row_index"]: row_index_path,
        output["observed_raw_mmd2"]: raw_path,
        output["observed_scores"]: score_path,
        output["permutation_top_r"]: permutation_path,
        output["result_records"]: records_path,
        output["comparison_table"]: comparison_path,
        output["selected_detector"]: selected_path,
        output["final_report"]: report_path,
        "H8_Qwen32B_D2C_Development_Comparison_Protocol.md": resolve(frozen["protocol"]),
        "h8_d2c_development_comparison.yaml": config_path,
        "implementation/h8_d2c.py": ROOT / "src" / "llm_integrity" / "h8_d2c.py",
        "implementation/run_h8_d2c_development_comparison.py": Path(__file__).resolve(),
    }
    archive_hashes: dict[str, str] = {}
    for logical, source in archive_sources.items():
        destination = frozen_dir / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        archive_hashes[logical] = file_sha256(destination)
    manifest = {
        "schema_version": D2C_SCHEMA_VERSION,
        "measurement_layer": "frozen",
        "score_layer": "frozen",
        "sample_size": selected_config["sample_structure"],
        "aggregation": selected_config["primary_aggregation"],
        "detector": "frozen_development_selected",
        "selected_configuration_id": selected_id,
        "selected_detector_payload_sha256": selected_envelope["payload_sha256"],
        "selection_rule_sha256": frozen["selection_rule_sha256"],
        "development_response_bank_sha256": frozen["responses_sha256"],
        "formal_performance_confirmation": "not_performed",
        "development_counts_are_not_formal_fpr_or_tpr_estimates": True,
        "new_model_responses": 0,
        "final_or_heldout_responses": 0,
        "files": archive_hashes,
    }
    manifest_path = frozen_dir / output["frozen_manifest"]
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_configuration_id": selected_id,
                "selected_metrics": selection["selection_metrics"][selected_id],
                "result_directory": str(result_dir.relative_to(ROOT)),
                "frozen_archive": str(frozen_dir.relative_to(ROOT)),
                "frozen_manifest_sha256": file_sha256(manifest_path),
                "new_model_responses": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
