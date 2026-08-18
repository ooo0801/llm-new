from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_precalibration import (
    H8_DATA_ROLE,
    H8_DDOF,
    H8_DTYPE,
    H8_EPSILON,
    build_h8_feature_schema,
    canonical_sha256,
    fit_family_balanced_scaler,
    global_continuous_exclusion_mask,
    h8_bandwidth_candidate_stability_suite,
    h8_numeric_summary,
    make_bandwidth_payload,
    save_h8_artifact,
)


DEFAULT_CONFIG = ROOT / "configs" / "h8_m0gh_offline_scaler_bandwidth.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def resolve(logical_path: str) -> Path:
    path = (ROOT / logical_path).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise ValueError("Configured path escapes the repository")
    return path


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def clean_git_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise ValueError("M0-G/H requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_input(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.get("model_sampling_authorized") is not False:
        raise ValueError("M0-G/H must explicitly forbid model sampling")
    input_config = config["input"]
    responses_path = resolve(input_config["responses"])
    report_path = resolve(input_config["sampling_report"])
    manifest_path = resolve(input_config["calibration_manifest"])
    require_hash(responses_path, input_config["responses_sha256"], "M0-F responses")
    require_hash(report_path, input_config["sampling_report_sha256"], "M0-F report")
    require_hash(manifest_path, input_config["calibration_manifest_sha256"], "M0-F manifest")
    provenance = config["provenance"]
    require_hash(resolve(provenance["h8_protocol"]), provenance["h8_protocol_sha256"], "H8 protocol")
    require_hash(resolve(provenance["h8_config"]), provenance["h8_config_sha256"], "H8 config")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or int(report.get("mmd_precalibration_fit_only_responses", -1)) != 1200:
        raise ValueError("M0-F PASS report/count is not frozen as expected")
    for key in ("formal_reference_responses", "attack_responses", "heldout_responses"):
        if int(report.get(key, -1)) != 0:
            raise ValueError(f"Forbidden M0-F response role is nonzero: {key}")

    envelope = json.loads(manifest_path.read_text(encoding="utf-8"))
    if envelope.get("payload_sha256") != input_config["calibration_manifest_payload_sha256"]:
        raise ValueError("M0-F manifest payload SHA mismatch")
    if canonical_sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("M0-F manifest canonical payload hash mismatch")
    schedule = envelope["payload"]["schedule"]
    rows = read_jsonl(responses_path)
    expected_count = int(input_config["required_response_count"])
    if len(rows) != expected_count or len(schedule) != expected_count:
        raise ValueError("M0-F responses/schedule must both contain exactly 1200 rows")
    prompt_counts: Counter[str] = Counter()
    seeds: set[int] = set()
    response_ids: set[str] = set()
    for position, (row, request) in enumerate(zip(rows, schedule)):
        if (
            int(row.get("schedule_position", -1)) != position
            or row.get("response_id") != request.get("response_id")
            or row.get("prompt_id") != request.get("prompt_id")
            or int(row.get("seed", -1)) != int(request.get("seed", -2))
            or int(row.get("replicate_id", -1)) != int(request.get("replicate_id", -2))
        ):
            raise ValueError(f"M0-F schedule mismatch at position {position}")
        if row.get("data_role") != input_config["required_data_role"]:
            raise ValueError("M0-G/H input contains a forbidden data role")
        if any(
            bool(row.get(key))
            for key in (
                "eligible_for_formal_reference",
                "eligible_for_heldout_evaluation",
                "eligible_for_attack_evaluation",
            )
        ):
            raise ValueError("M0-G/H input contains an eligible formal/heldout/attack row")
        prompt_counts[str(row["prompt_id"])] += 1
        seeds.add(int(row["seed"]))
        response_ids.add(str(row["response_id"]))
    if len(prompt_counts) != int(input_config["required_prompt_count"]):
        raise ValueError("M0-F prompt count mismatch")
    if set(prompt_counts.values()) != {int(input_config["required_responses_per_prompt"])}:
        raise ValueError("M0-F per-prompt count mismatch")
    if len(seeds) != expected_count or len(response_ids) != expected_count:
        raise ValueError("M0-F seed/response IDs are not unique")
    return rows, {
        "responses_sha256": input_config["responses_sha256"],
        "sampling_report_sha256": input_config["sampling_report_sha256"],
        "manifest_file_sha256": input_config["calibration_manifest_sha256"],
        "manifest_payload_sha256": input_config["calibration_manifest_payload_sha256"],
        "response_count": len(rows),
        "prompt_counts": dict(sorted(prompt_counts.items())),
    }


def task_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        row = dict(record.get("task_metadata") or {})
        row["category"] = record.get("category")
        result.append(row)
    return result


def deterministic_feature_extract(
    extractor: FeatureExtractor,
    records: list[dict[str, Any]],
) -> np.ndarray:
    texts = [str(record["raw_response"]) for record in records]
    unique_texts = sorted(set(texts))
    surface_semantic_unique = np.asarray(
        extractor.transform(
            unique_texts,
            rows=None,
            include_surface=True,
            include_semantic=True,
            include_task=False,
        ),
        dtype=np.float64,
    )
    lookup = {
        text: surface_semantic_unique[index] for index, text in enumerate(unique_texts)
    }
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


def little_endian_row_sha256(row: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(row, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def vector_unique_count(matrix: np.ndarray) -> int:
    return len({little_endian_row_sha256(row) for row in matrix})


def family_dimension_qa(
    matrix: np.ndarray,
    schema: Any,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for family, (start, stop) in schema.family_slices.items():
        block = matrix[:, start:stop]
        std = block.std(axis=0, ddof=0, dtype=np.float64)
        result[family] = {
            "dimension": int(stop - start),
            "constant_dimension_count": int(np.count_nonzero(std == 0.0)),
            "near_zero_nonconstant_dimension_count": int(
                np.count_nonzero((std > 0.0) & (std < threshold))
            ),
            "nan_value_count": int(np.isnan(block).sum()),
            "inf_value_count": int(np.isinf(block).sum()),
        }
    return result


def feature_qa(
    records: list[dict[str, Any]],
    features: np.ndarray,
    schema: Any,
    threshold: float,
) -> dict[str, Any]:
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_prompt[str(record["prompt_id"])].append(index)
    prompts: dict[str, Any] = {}
    for prompt_id in sorted(by_prompt):
        indices = by_prompt[prompt_id]
        subset_records = [records[index] for index in indices]
        subset = features[indices]
        stop_counts = Counter(str(record["stop_reason"]) for record in subset_records)
        token_lengths = np.asarray(
            [int(record["response_token_count_including_eos"]) for record in subset_records],
            dtype=np.float64,
        )
        prompts[prompt_id] = {
            "raw_response_unique_count": len({str(record["raw_response"]) for record in subset_records}),
            "completion_token_sequence_unique_count": len(
                {tuple(int(token) for token in record["completion_token_ids"]) for record in subset_records}
            ),
            "feature_vector_unique_count": vector_unique_count(subset),
            "feature_dimensions": {
                "surface": schema.family_slices["surface"][1] - schema.family_slices["surface"][0],
                "semantic": schema.family_slices["semantic"][1] - schema.family_slices["semantic"][0],
                "task": schema.family_slices["task"][1] - schema.family_slices["task"][0],
                "total": schema.dimension,
            },
            "family_dimension_qa": family_dimension_qa(subset, schema, threshold),
            "nan_value_count": int(np.isnan(subset).sum()),
            "inf_value_count": int(np.isinf(subset).sum()),
            "eos_stop_count": int(stop_counts.get("eos", 0)),
            "eos_stop_fraction": float(stop_counts.get("eos", 0) / len(indices)),
            "length_stop_count": int(stop_counts.get("length", 0)),
            "length_stop_fraction": float(stop_counts.get("length", 0) / len(indices)),
            "other_stop_reason_counts": {
                key: value for key, value in sorted(stop_counts.items()) if key not in {"eos", "length"}
            },
            "generated_token_count_including_eos": h8_numeric_summary(token_lengths),
        }
    return {
        "data_role": H8_DATA_ROLE,
        "response_count": len(records),
        "prompt_count": len(prompts),
        "dtype": str(features.dtype),
        "ddof": H8_DDOF,
        "near_zero_scale_threshold": threshold,
        "overall_nan_value_count": int(np.isnan(features).sum()),
        "overall_inf_value_count": int(np.isinf(features).sum()),
        "overall_feature_vector_unique_count": vector_unique_count(features),
        "prompts": prompts,
    }


def write_feature_row_index(path: Path, records: list[dict[str, Any]], features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (record, feature_row) in enumerate(zip(records, features)):
            value = {
                "row_index": index,
                "response_id": record["response_id"],
                "prompt_id": record["prompt_id"],
                "replicate_id": record["replicate_id"],
                "seed": record["seed"],
                "feature_row_float64_le_sha256": little_endian_row_sha256(feature_row),
                "data_role": H8_DATA_ROLE,
            }
            handle.write(canonical_bytes(value).decode("utf-8") + "\n")
    temporary.replace(path)


def version_info(extractor: FeatureExtractor) -> dict[str, Any]:
    import sentence_transformers
    import torch

    encoder = extractor._encoder
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "semantic_encoder_class": type(encoder).__name__,
        "semantic_device": str(encoder.device),
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }


def main(config_path: Path) -> int:
    analysis_git_commit = clean_git_commit()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    records, input_audit = validate_input(config)
    feature_config = config["features"]
    import torch

    torch.set_num_threads(int(feature_config["torch_num_threads"]))
    torch.use_deterministic_algorithms(bool(feature_config["deterministic_torch_algorithms"]))
    np.random.seed(0)
    schema = build_h8_feature_schema(int(feature_config["semantic_dimension"]))
    if schema.dimension != int(feature_config["expected_total_dimension"]):
        raise ValueError("Feature schema dimension mismatch")
    if schema.sha256 != feature_config["expected_feature_schema_sha256"]:
        raise ValueError("Feature schema SHA256 mismatch")
    extractor = FeatureExtractor(
        semantic_model_name=feature_config["semantic_model"],
        semantic_model_revision=feature_config["semantic_model_revision"],
        semantic_device=feature_config["semantic_device"],
        semantic_local_files_only=bool(feature_config["semantic_local_files_only"]),
    )
    if not bool(feature_config["deduplicate_text_before_semantic_encoding"]):
        raise ValueError("H8 M0-G/H requires unique-text semantic encoding")
    features = deterministic_feature_extract(extractor, records)
    if features.shape != (len(records), schema.dimension) or not np.isfinite(features).all():
        raise ValueError("Deterministic feature extraction produced invalid features")

    output_config = config["output"]
    output_dir = resolve(output_config["directory"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("M0-G/H output directory is non-empty; refusing result-dependent overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / output_config["feature_matrix"]
    with feature_path.open("wb") as handle:
        np.save(handle, np.asarray(features, dtype="<f8"), allow_pickle=False)
    row_index_path = output_dir / output_config["feature_row_index"]
    write_feature_row_index(row_index_path, records, features)

    common_binding = {
        **input_audit,
        "feature_schema_sha256": schema.sha256,
        "feature_matrix_sha256": file_sha256(feature_path),
        "feature_row_index_sha256": file_sha256(row_index_path),
        "feature_extraction_runtime": version_info(extractor),
        "semantic_model": feature_config["semantic_model"],
        "semantic_model_revision": feature_config["semantic_model_revision"],
        "unique_text_semantic_encoding": True,
        "m0gh_config_sha256": file_sha256(config_path),
        "analysis_git_commit": analysis_git_commit,
    }
    schema_payload = {**schema.as_dict(), **common_binding, "artifact_status": "candidate"}
    schema_path = output_dir / output_config["feature_schema"]
    schema_payload_sha = save_h8_artifact(schema_path, "h8_feature_schema_candidate", schema_payload)

    qa_payload = {
        **feature_qa(
            records, features, schema, float(feature_config["near_zero_scale_threshold"])
        ),
        **common_binding,
        "feature_schema_candidate_payload_sha256": schema_payload_sha,
        "artifact_status": "candidate",
    }
    qa_path = output_dir / output_config["feature_qa"]
    qa_payload_sha = save_h8_artifact(qa_path, "h8_feature_qa", qa_payload)

    pooled = features
    exclusion_mask = global_continuous_exclusion_mask(
        pooled, schema, float(feature_config["near_zero_scale_threshold"])
    )
    active_by_family: dict[str, list[int]] = {}
    excluded_by_family: dict[str, list[int]] = {}
    for family, (start, stop) in schema.family_slices.items():
        active_by_family[family] = [index for index in range(start, stop) if not exclusion_mask[index]]
        excluded_by_family[family] = [index for index in range(start, stop) if exclusion_mask[index]]
    exclusion_payload = {
        "artifact_status": "candidate",
        "data_role": H8_DATA_ROLE,
        "dtype": H8_DTYPE,
        "ddof": H8_DDOF,
        "epsilon": float(feature_config["near_zero_scale_threshold"]),
        "feature_schema_sha256": schema.sha256,
        "mask": list(exclusion_mask),
        "excluded_dimensions": [index for index, excluded in enumerate(exclusion_mask) if excluded],
        "excluded_feature_names": [
            schema.features[index].name for index, excluded in enumerate(exclusion_mask) if excluded
        ],
        "active_dimensions": [index for index, excluded in enumerate(exclusion_mask) if not excluded],
        "active_dimensions_by_family": active_by_family,
        "excluded_dimensions_by_family": excluded_by_family,
        "continuous_only": True,
        **common_binding,
    }
    exclusion_path = output_dir / output_config["exclusion_mask"]
    exclusion_payload_sha = save_h8_artifact(
        exclusion_path, "h8_global_exclusion_mask_candidate", exclusion_payload
    )

    indices_by_prompt: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indices_by_prompt[str(record["prompt_id"])].append(index)
    values_by_prompt = {
        prompt_id: features[indices] for prompt_id, indices in sorted(indices_by_prompt.items())
    }
    scaler_dir = output_dir / output_config["scaler_directory"]
    scaler_payload_shas: dict[str, str] = {}
    scalers: dict[str, Any] = {}
    for prompt_id, values in values_by_prompt.items():
        scaler = fit_family_balanced_scaler(
            prompt_id,
            values,
            pooled,
            schema,
            epsilon=float(feature_config["near_zero_scale_threshold"]),
            exclusion_mask=exclusion_mask,
        )
        payload = {
            **scaler.as_dict(),
            "artifact_status": "candidate",
            "global_exclusion_mask_payload_sha256": exclusion_payload_sha,
            "calibration_manifest_sha256": input_audit["manifest_file_sha256"],
            "responses_sha256": input_audit["responses_sha256"],
        }
        scaler_payload_shas[prompt_id] = save_h8_artifact(
            scaler_dir / f"{prompt_id}.json", "h8_family_balanced_scaler_candidate", payload
        )
        scalers[prompt_id] = scaler

    bandwidth_config = config["bandwidth"]
    suite = h8_bandwidth_candidate_stability_suite(
        values_by_prompt,
        scalers,
        schema,
        repetitions=int(bandwidth_config["repetitions"]),
        fraction=float(bandwidth_config["subsample_fraction"]),
        seed=int(bandwidth_config["seed"]),
    )
    global_payload = {
        **suite["global_fallback"],
        "artifact_status": "candidate",
        "data_role": H8_DATA_ROLE,
        "feature_schema_sha256": schema.sha256,
        "global_exclusion_mask_payload_sha256": exclusion_payload_sha,
        "calibration_manifest_sha256": input_audit["manifest_file_sha256"],
        "responses_sha256": input_audit["responses_sha256"],
        "cross_prompt_distances_used": False,
    }
    global_path = output_dir / output_config["global_bandwidth"]
    global_payload_sha = save_h8_artifact(
        global_path, "h8_global_bandwidth_candidate", global_payload
    )

    bandwidth_dir = output_dir / output_config["bandwidth_directory"]
    bandwidth_payload_shas: dict[str, str] = {}
    for prompt_id, result in sorted(suite["prompt_results"].items()):
        payload = make_bandwidth_payload(
            prompt_id,
            float(result["sigma"]),
            scaler_payload_shas[prompt_id],
            input_audit["manifest_file_sha256"],
            schema.sha256,
            bandwidth_source=result["bandwidth_source"],
            global_exclusion_mask_sha256=exclusion_payload_sha,
            global_bandwidth_payload_sha256=(
                global_payload_sha if result["bandwidth_source"] == "global_degenerate_fallback" else None
            ),
        )
        payload.update(
            {
                "artifact_status": "candidate",
                "feature_degenerate": result["feature_degenerate"],
                "positive_distance_count": result["positive_distance_count"],
                "positive_distance_fraction": result["positive_distance_fraction"],
                "stability_status": result["overall_status"],
                "prompt_specific_failure_fell_back_to_global": False,
            }
        )
        bandwidth_payload_shas[prompt_id] = save_h8_artifact(
            bandwidth_dir / f"{prompt_id}.json", "h8_prompt_bandwidth_candidate", payload
        )

    stability_payload = {
        **suite,
        "artifact_status": "candidate",
        "data_role": H8_DATA_ROLE,
        "feature_schema_sha256": schema.sha256,
        "global_exclusion_mask_payload_sha256": exclusion_payload_sha,
        "scaler_payload_sha256_by_prompt": scaler_payload_shas,
        "global_bandwidth_payload_sha256": global_payload_sha,
        "calibration_manifest_sha256": input_audit["manifest_file_sha256"],
        "responses_sha256": input_audit["responses_sha256"],
    }
    stability_path = output_dir / output_config["stability_report"]
    stability_payload_sha = save_h8_artifact(
        stability_path, "h8_bandwidth_stability_candidate", stability_payload
    )

    artifact_paths = [
        feature_path,
        row_index_path,
        schema_path,
        qa_path,
        exclusion_path,
        global_path,
        stability_path,
        *(scaler_dir / f"{prompt_id}.json" for prompt_id in sorted(values_by_prompt)),
        *(bandwidth_dir / f"{prompt_id}.json" for prompt_id in sorted(values_by_prompt)),
    ]
    artifact_hashes = {
        path.relative_to(output_dir).as_posix(): file_sha256(path) for path in artifact_paths
    }
    gate_status = suite["overall_status"]
    status = {"PASS": "PASS", "WARN": "WARN_REVIEW_REQUIRED", "FAIL": "FAIL_STOP_REVIEW"}[
        gate_status
    ]
    report = {
        "schema_version": "h8-m0gh-scaler-bandwidth-report-1.0",
        "status": status,
        "phase": "H8_M0G_H_OFFLINE_FEATURE_SCALER_BANDWIDTH",
        "candidate_artifacts_only": True,
        "input_audit": input_audit,
        "feature_qa_payload_sha256": qa_payload_sha,
        "feature_schema_sha256": schema.sha256,
        "feature_schema_candidate_payload_sha256": schema_payload_sha,
        "global_exclusion_mask_payload_sha256": exclusion_payload_sha,
        "excluded_dimension_count": int(sum(exclusion_mask)),
        "active_dimension_count": int(schema.dimension - sum(exclusion_mask)),
        "scaler_payload_sha256_by_prompt": scaler_payload_shas,
        "bandwidth_payload_sha256_by_prompt": bandwidth_payload_shas,
        "global_bandwidth_payload_sha256": global_payload_sha,
        "bandwidth_stability_payload_sha256": stability_payload_sha,
        "global_bandwidth": {
            "sigma": suite["global_fallback"]["sigma"],
            "status": suite["global_fallback"]["overall_status"],
            "positive_within_prompt_distance_count": suite["global_fallback"][
                "positive_within_prompt_distance_count"
            ],
            "cross_prompt_distances_used": False,
        },
        "fingerprints": {
            prompt_id: {
                "bandwidth_source": result["bandwidth_source"],
                "sigma": result["sigma"],
                "feature_degenerate": result["feature_degenerate"],
                "positive_distance_count": result["positive_distance_count"],
                "positive_distance_fraction": result["positive_distance_fraction"],
                "fixed_full_scaler_status": result["fixed_full_scaler"]["status"],
                "refit_subsample_scaler_status": result["refit_subsample_scaler"]["status"],
                "overall_status": result["overall_status"],
                "prompt_specific_failure_fell_back_to_global": False,
            }
            for prompt_id, result in sorted(suite["prompt_results"].items())
        },
        "artifact_file_sha256": artifact_hashes,
        "provenance": config["provenance"],
        "model_sampling_performed": False,
        "formal_reference_read_or_generated": False,
        "heldout_read_or_generated": False,
        "attack_read_or_generated": False,
        "mmd_pseudo_trials_performed": False,
        "d_to_s_fit_performed": False,
        "top_r_or_global_test_performed": False,
        "next_gate": "STOP_AND_WAIT_FOR_EXPLICIT_M0GH_REVIEW_APPROVAL",
    }
    report_path = output_dir / output_config["report"]
    atomic_json(report_path, report)
    artifact_hashes[report_path.relative_to(output_dir).as_posix()] = file_sha256(report_path)
    hashes_path = output_dir / output_config["artifact_hashes"]
    atomic_json(
        hashes_path,
        {
            "schema_version": "h8-m0gh-artifact-hashes-1.0",
            "hash_algorithm": "sha256",
            "base_directory": output_config["directory"],
            "files": dict(sorted(artifact_hashes.items())),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H8 M0-G/H offline scaler/bandwidth candidate runner")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(main(arguments.config.resolve()))
