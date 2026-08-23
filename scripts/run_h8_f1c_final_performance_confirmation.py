from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm_integrity.features import FeatureExtractor  # noqa: E402
from llm_integrity.h8_d1c import load_frozen_score_calibration  # noqa: E402
from llm_integrity.h8_f1a import (  # noqa: E402
    ALPHA,
    ATTACK_FAMILIES,
    FINAL_ATTACK_ROLE,
    FINAL_INTACT_ROLE,
    FINAL_REFERENCE_ROLE,
    FORMAL_TOTAL,
    INTACT_UNIT_COUNT,
    N_REFERENCE,
    N_TARGET,
    PERMUTATIONS,
    PROMPT_COUNT,
    TOP_R,
    exact_binomial_interval,
    file_sha256,
    load_canonical_envelope,
    load_frozen_detector,
    summarize_final_decisions,
)
from llm_integrity.h8_f1b import (  # noqa: E402
    BASE_PARTITION_ID,
    audit_formal_records,
    read_jsonl_strict,
    validate_frozen_inputs,
)
from llm_integrity.h8_f1c import (  # noqa: E402
    F1C_SCHEMA_VERSION,
    evaluate_final_unit,
    numeric_summary,
)
from llm_integrity.h8_precalibration import (  # noqa: E402
    FamilyBalancedScaler,
    build_h8_feature_schema,
    canonical_sha256,
    h8_rbf_kernel,
    load_h8_artifact,
)
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement  # noqa: E402


ARTIFACT_TYPES = {
    "generation": "h8_f1a_final_generation_manifest",
    "attack": "h8_f1a_fresh_heldout_attack_endpoint_manifest",
    "permutation": "h8_f1a_final_global_permutation_seed_manifest",
    "freshness": "h8_f1a_freshness_audit",
    "historical": "h8_f1a_historical_identity_index",
    "smoke": "h8_f1a_tiny_smoke_manifest",
}


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def canonical_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def canonical_jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    temporary.replace(path)


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"F1-C {label} SHA256 mismatch: {path}")


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def verify_git(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_branch = str(config["git"]["branch"])
    branch = git_output("branch", "--show-current")
    head = git_output("rev-parse", "HEAD")
    if branch != expected_branch:
        raise ValueError("F1-C Git branch mismatch")
    for key in ("required_f1b_terminal_archive_commit", "implementation_commit"):
        ancestor = str(config["git"][key])
        status = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, head], cwd=PROJECT_ROOT, check=False
        ).returncode
        if status != 0:
            raise ValueError(f"F1-C Git HEAD does not descend from {key}")
    dirty = git_output("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("F1-C tracked worktree is not clean")
    return {"branch": branch, "head": head, "tracked_worktree_clean": True}


def verify_f1b_terminal_index(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    frozen = config["frozen_inputs"]
    root = resolve(frozen["f1b_output_root"])
    index_path = root / "F1B_FORMAL_SAMPLING_SHA256_INDEX.json"
    require_hash(index_path, frozen["f1b_terminal_index_sha256"], "F1-B terminal index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    files = index.get("files")
    if int(index.get("file_count", -1)) != 178 or not isinstance(files, dict) or len(files) != 178:
        raise ValueError("F1-C requires the frozen 178-file F1-B terminal index")
    mismatches: list[str] = []
    response_hashes: dict[str, str] = {}
    for logical, metadata in sorted(files.items()):
        path = (root / str(logical)).resolve()
        if path != root and root not in path.parents:
            raise ValueError("F1-B terminal index path escapes output root")
        expected_hash = str(metadata["sha256"])
        expected_size = int(metadata["size_bytes"])
        if not path.is_file() or path.stat().st_size != expected_size or file_sha256(path) != expected_hash:
            mismatches.append(str(logical))
        if str(logical).startswith("records/"):
            response_hashes[str(logical)] = expected_hash
    if mismatches or len(response_hashes) != 41:
        raise ValueError(f"F1-B terminal index mismatch: {mismatches[:3]}")
    return index, response_hashes


def load_envelope(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    row = config["frozen_inputs"]["f1a_artifacts"][key]
    return load_canonical_envelope(
        resolve(row["path"]),
        expected_artifact_type=ARTIFACT_TYPES[key],
        expected_file_sha256=str(row["file_sha256"]),
        expected_payload_sha256=str(row["payload_sha256"]),
    )


def load_records(root: Path, record_hashes: Mapping[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for logical, expected in sorted(record_hashes.items()):
        path = root / logical
        require_hash(path, expected, f"formal response partition {logical}")
        records.extend(read_jsonl_strict(path))
    return sorted(records, key=lambda row: int(row["schedule_position"]))


def materialization_identities(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in sorted((root / "materialization").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        identity = str(value.get("materialization_identity_sha256", ""))
        if len(identity) != 64 or value.get("status") != "PASS":
            raise ValueError(f"F1-C invalid materialization identity: {path.name}")
        output[path.stem] = identity
    if set(output) != {BASE_PARTITION_ID} | {
        path.stem for path in (root / "records").glob("*.jsonl") if path.stem != BASE_PARTITION_ID
    }:
        raise ValueError("F1-C materialization/record partition identity set mismatch")
    return output


def fingerprint_task_rows(fingerprint: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = list(fingerprint.get("entries", []))
    if len(entries) != PROMPT_COUNT:
        raise ValueError("F1-C fingerprint does not contain MCC12")
    output: dict[str, dict[str, Any]] = {}
    for entry in entries:
        prompt_id = str(entry["prompt_id"])
        row = dict(entry.get("metadata") or {})
        row["category"] = entry.get("category")
        output[prompt_id] = row
    if len(output) != PROMPT_COUNT:
        raise ValueError("F1-C fingerprint prompt IDs are duplicated")
    return output


def verify_runtime_and_bge(config: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    feature = config["feature_extraction"]
    if os.environ.get("HF_HUB_CACHE") != feature["hf_hub_cache"]:
        raise ValueError("F1-C HF_HUB_CACHE is not the frozen offline data-disk path")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("F1-C strict offline environment is not enabled")
    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sentence_transformers": importlib.metadata.version("sentence-transformers"),
    }
    if actual != dict(feature["required_runtime"]):
        raise ValueError(f"F1-C feature runtime drift: {actual}")
    torch.set_num_threads(int(feature["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)
    revision = str(feature["semantic_model_revision"])
    snapshot = Path(feature["hf_hub_cache"]) / "models--BAAI--bge-small-zh-v1.5" / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError("F1-C frozen BGE snapshot is absent from the offline cache")
    inventory = hashlib.sha256()
    file_count = 0
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        inventory.update(relative.encode("utf-8") + b"\0")
        inventory.update(str(path.stat().st_size).encode("ascii") + b"\0")
        inventory.update(file_sha256(path).encode("ascii") + b"\n")
        file_count += 1
    if file_count == 0:
        raise ValueError("F1-C frozen BGE snapshot inventory is empty")
    return {
        **actual,
        "semantic_model": feature["semantic_model"],
        "semantic_model_revision": revision,
        "semantic_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "snapshot_file_count": file_count,
        "snapshot_listing_and_content_sha256": inventory.hexdigest(),
    }


def load_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    git = verify_git(config)
    index, response_hashes = verify_f1b_terminal_index(config)
    terminal_path = resolve(frozen["f1b_terminal_audit"])
    require_hash(terminal_path, frozen["f1b_terminal_audit_sha256"], "F1-B terminal audit")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if (
        terminal.get("status") != "PASS_TERMINAL_F1B_ARTIFACT_AUDIT"
        or int(terminal.get("formal_responses", -1)) != FORMAL_TOTAL
        or int(terminal.get("indexed_file_count", -1)) != 178
        or int(terminal.get("index_mismatch_count", -1)) != 0
        or terminal.get("original_frozen_identity_sha256") != frozen["f1b_frozen_identity_sha256"]
        or terminal.get("formal_detector_statistics_computed") is not False
    ):
        raise ValueError("F1-C F1-B terminal audit gate failed")

    artifacts = {key: load_envelope(config, key) for key in ARTIFACT_TYPES}
    f1a_validation = validate_frozen_inputs(
        artifacts["generation"], artifacts["attack"], artifacts["permutation"],
        artifacts["freshness"], artifacts["historical"], artifacts["smoke"]
    )
    if f1a_validation["status"] != "PASS" or f1a_validation["permutation_seed_count"] != 1_198_800:
        raise ValueError("F1-C F1-A frozen manifest validation failed")

    detector = load_frozen_detector(
        resolve(frozen["detector_archive"]),
        expected_manifest_sha256=str(frozen["detector_manifest_sha256"]),
    )
    selected = detector["selected_detector"]
    if canonical_sha256(selected) != frozen["selected_detector_payload_sha256"]:
        raise ValueError("F1-C selected detector payload mismatch")
    mmd = load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=str(frozen["mmd_manifest_sha256"]),
        expected_mmd_implementation_commit=str(frozen["mmd_implementation_commit"]),
    )
    score = load_frozen_score_calibration(
        resolve(frozen["score_archive"]),
        expected_manifest_sha256=str(frozen["score_manifest_sha256"]),
        binding=mmd,
        expected_score_schema_sha256=str(frozen["score_schema_sha256"]),
    )
    if canonical_sha256(score) != frozen["score_parameter_payload_sha256"]:
        raise ValueError("F1-C frozen score parameter payload mismatch")

    fingerprint_path = resolve(frozen["fingerprint"])
    require_hash(fingerprint_path, frozen["fingerprint_sha256"], "H6 MCC12 fingerprint")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    task_rows = fingerprint_task_rows(fingerprint)
    prompt_ids = tuple(sorted(task_rows))
    if prompt_ids != mmd.prompt_ids:
        raise ValueError("F1-C fingerprint/MMD prompt identity mismatch")

    root = resolve(frozen["f1b_output_root"])
    records = load_records(root, response_hashes)
    identities = materialization_identities(root)
    record_audit = audit_formal_records(
        records,
        artifacts["generation"]["requests"],
        frozen_identity_sha256=str(frozen["f1b_frozen_identity_sha256"]),
        endpoint_materialization_identities=identities,
    )
    if record_audit["status"] != "PASS" or record_audit["total_response_count"] != FORMAL_TOTAL:
        raise ValueError("F1-C 12,720-response integrity audit failed")
    runtime = verify_runtime_and_bge(config)
    return {
        "git": git,
        "terminal_audit": terminal,
        "terminal_index": index,
        "response_hashes": response_hashes,
        "records": records,
        "record_audit": record_audit,
        "artifacts": artifacts,
        "detector": detector,
        "mmd": mmd,
        "score": score,
        "fingerprint": fingerprint,
        "task_rows": task_rows,
        "prompt_ids": prompt_ids,
        "runtime": runtime,
    }


def deterministic_features(
    records: Sequence[Mapping[str, Any]],
    task_by_prompt: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    feature = config["feature_extraction"]
    extractor = FeatureExtractor(
        semantic_model_name=str(feature["semantic_model"]),
        semantic_model_revision=str(feature["semantic_model_revision"]),
        semantic_device="cpu",
        semantic_local_files_only=True,
    )
    texts = [str(record["raw_response"]) for record in records]
    unique_texts = sorted(set(texts))
    encoded_unique = extractor.transform(
        unique_texts, rows=None, include_surface=True, include_semantic=True, include_task=False
    ).astype(np.float64)
    lookup = {text: encoded_unique[index] for index, text in enumerate(unique_texts)}
    surface_semantic = np.stack([lookup[text] for text in texts]).astype(np.float64)
    task_rows = [dict(task_by_prompt[str(record["prompt_id"])]) for record in records]
    task = extractor.transform(
        texts, rows=task_rows, include_surface=False, include_semantic=False, include_task=True
    ).astype(np.float64)
    matrix = np.concatenate([surface_semantic, task], axis=1).astype("<f8")
    row_index = [
        {
            "row_index": index,
            "response_id": str(record["response_id"]),
            "schedule_position": int(record["schedule_position"]),
            "data_role": str(record["data_role"]),
            "prompt_id": str(record["prompt_id"]),
            "evaluation_unit_id": record.get("evaluation_unit_id"),
            "attack_family": record.get("attack_family"),
        }
        for index, record in enumerate(records)
    ]
    ids = [row["response_id"] for row in row_index]
    qa = {
        "schema_version": F1C_SCHEMA_VERSION,
        "status": "PASS",
        "feature_count": int(matrix.shape[0]),
        "feature_dimension": int(matrix.shape[1]),
        "dtype": "float64",
        "nan_count": int(np.isnan(matrix).sum()),
        "inf_count": int(np.isinf(matrix).sum()),
        "missing_feature_count": 0,
        "duplicate_response_id_count": len(ids) - len(set(ids)),
        "unique_raw_response_count": len(unique_texts),
        "final_data_used_for_scaler_bandwidth_or_schema_fit": False,
    }
    expected_schema = build_h8_feature_schema(512)
    if (
        matrix.shape != (FORMAL_TOTAL, 528)
        or not np.isfinite(matrix).all()
        or len(set(ids)) != FORMAL_TOTAL
        or expected_schema.sha256 != feature["expected_feature_schema_sha256"]
    ):
        raise ValueError("F1-C deterministic feature QA failed")
    return matrix, row_index, qa


def load_measurement_payloads(preflight: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[str, float]]:
    schema = build_h8_feature_schema(512)
    archive = resolve(config["frozen_inputs"]["mmd_archive"]) / "measurement"
    binding = preflight["mmd"]
    scalers: dict[str, Any] = {}
    bandwidths: dict[str, float] = {}
    for prompt_id in preflight["prompt_ids"]:
        scaler_payload = load_h8_artifact(
            archive / "scaler_candidates" / f"{prompt_id}.json",
            "h8_family_balanced_scaler_candidate",
            binding.scaler_payload_sha256_by_prompt[prompt_id],
        )
        bandwidth_payload = load_h8_artifact(
            archive / "bandwidth_candidates" / f"{prompt_id}.json",
            "h8_prompt_bandwidth_candidate",
            binding.bandwidth_payload_sha256_by_prompt[prompt_id],
        )
        scaler = FamilyBalancedScaler.from_dict(scaler_payload)
        if scaler.prompt_id != prompt_id:
            raise ValueError("F1-C scaler prompt identity mismatch")
        sigma = float(bandwidth_payload["sigma"])
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("F1-C frozen bandwidth is invalid")
        scalers[prompt_id] = scaler
        bandwidths[prompt_id] = sigma
    return schema, scalers, bandwidths


def build_membership(
    records: Sequence[Mapping[str, Any]],
    streams: Sequence[Mapping[str, Any]],
    prompt_ids: Sequence[str],
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]], list[dict[str, Any]]]:
    reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    targets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        prompt_id = str(record["prompt_id"])
        role = str(record["data_role"])
        if role == FINAL_REFERENCE_ROLE:
            reference[prompt_id].append(dict(record))
        elif role in {FINAL_INTACT_ROLE, FINAL_ATTACK_ROLE}:
            targets[str(record["evaluation_unit_id"])][prompt_id].append(dict(record))
        else:
            raise ValueError("F1-C encountered a forbidden final data role")
    reference_ids = {
        prompt_id: [str(row["response_id"]) for row in sorted(reference[prompt_id], key=lambda x: int(x["bank_position"]))]
        for prompt_id in prompt_ids
    }
    target_ids = {
        unit_id: {
            prompt_id: [
                str(row["response_id"])
                for row in sorted(targets[unit_id][prompt_id], key=lambda x: int(x["replicate_id"]))
            ]
            for prompt_id in prompt_ids
        }
        for unit_id in targets
    }
    unit_rows = [dict(row) for row in streams]
    stream_ids = [str(row["evaluation_unit_id"]) for row in unit_rows]
    if (
        len(unit_rows) != 100
        or len(set(stream_ids)) != 100
        or set(stream_ids) != set(target_ids)
        or any(len(reference_ids[prompt_id]) != N_REFERENCE for prompt_id in prompt_ids)
        or any(len(target_ids[unit_id][prompt_id]) != N_TARGET for unit_id in stream_ids for prompt_id in prompt_ids)
    ):
        raise ValueError("F1-C frozen 100-unit membership validation failed")
    return reference_ids, target_ids, unit_rows


def save_npy(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype="<f8"), allow_pickle=False)
    temporary.replace(path)
    return file_sha256(path)


def terminal_output_index(output: Path, excluded: set[str]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        logical = path.relative_to(output).as_posix()
        if logical in excluded:
            continue
        files[logical] = {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return {"schema_version": F1C_SCHEMA_VERSION, "file_count": len(files), "files": files}


def run(config: Mapping[str, Any], *, preflight_only: bool) -> None:
    output = resolve(config["output"]["result_directory"])
    output.mkdir(parents=True, exist_ok=True)
    preflight = load_preflight(config)
    preflight_report = {
        "schema_version": F1C_SCHEMA_VERSION,
        "status": "PASS",
        "mode": "preflight_only" if preflight_only else "formal_offline_confirmation",
        "git": preflight["git"],
        "f1b_terminal_audit_status": preflight["terminal_audit"]["status"],
        "f1b_formal_response_count": preflight["record_audit"]["total_response_count"],
        "f1b_terminal_index_file_count": preflight["terminal_index"]["file_count"],
        "f1b_terminal_index_mismatch_count": 0,
        "f1b_frozen_identity_sha256": config["frozen_inputs"]["f1b_frozen_identity_sha256"],
        "h6_fingerprint_sha256": config["frozen_inputs"]["fingerprint_sha256"],
        "mmd_manifest_sha256": preflight["mmd"].manifest_sha256,
        "score_manifest_sha256": config["frozen_inputs"]["score_manifest_sha256"],
        "selected_detector_payload_sha256": config["frozen_inputs"]["selected_detector_payload_sha256"],
        "detector_manifest_sha256": config["frozen_inputs"]["detector_manifest_sha256"],
        "permutation_seed_count": preflight["artifacts"]["permutation"]["derived_seed_count"],
        "runtime_and_bge_identity": preflight["runtime"],
        "new_model_responses": 0,
        "qwen_generation_imported_or_invoked": False,
        "measurement_or_score_refit": False,
    }
    canonical_write(output / config["output"]["preflight_report"], preflight_report)
    if preflight_only:
        print(json.dumps(preflight_report, ensure_ascii=False, sort_keys=True))
        return

    records = preflight["records"]
    features, feature_rows, feature_qa = deterministic_features(records, preflight["task_rows"], config)
    feature_hash = save_npy(output / config["output"]["feature_cache"], features)
    canonical_jsonl_write(output / config["output"]["feature_row_index"], feature_rows)
    feature_qa.update(
        {
            "feature_cache_sha256": feature_hash,
            "feature_row_index_sha256": file_sha256(output / config["output"]["feature_row_index"]),
            "feature_schema_sha256": config["feature_extraction"]["expected_feature_schema_sha256"],
            "semantic_identity": preflight["runtime"],
        }
    )
    canonical_write(output / config["output"]["feature_audit"], feature_qa)

    schema, scalers, bandwidths = load_measurement_payloads(preflight, config)
    score_parameters = preflight["score"]["structures"]["r60_q10"]["prompt_parameters"]
    reference_ids, target_ids, units = build_membership(
        records, preflight["artifacts"]["permutation"]["streams"], preflight["prompt_ids"]
    )
    row_by_id = {row["response_id"]: int(row["row_index"]) for row in feature_rows}

    full_kernels: dict[str, np.ndarray] = {}
    full_indices: dict[str, dict[str, list[int]]] = {}
    for prompt_id in preflight["prompt_ids"]:
        ordered_ids = list(reference_ids[prompt_id])
        unit_positions: dict[str, list[int]] = {}
        for unit in units:
            unit_id = str(unit["evaluation_unit_id"])
            start = len(ordered_ids)
            ordered_ids.extend(target_ids[unit_id][prompt_id])
            unit_positions[unit_id] = list(range(start, start + N_TARGET))
        raw = features[[row_by_id[response_id] for response_id in ordered_ids]]
        transformed = scalers[prompt_id].transform(raw, schema)
        full_kernels[prompt_id] = h8_rbf_kernel(transformed, transformed, bandwidths[prompt_id])
        full_indices[prompt_id] = unit_positions

    observed_raw = np.empty((100, PROMPT_COUNT), dtype=np.float64)
    observed_scores = np.empty((100, PROMPT_COUNT), dtype=np.float64)
    permutation_raw = np.empty((100, PROMPT_COUNT, PERMUTATIONS), dtype=np.float64)
    permutation_scores = np.empty((100, PROMPT_COUNT, PERMUTATIONS), dtype=np.float64)
    permutation_top2 = np.empty((100, PERMUTATIONS), dtype=np.float64)
    unit_results: list[dict[str, Any]] = []
    prompt_results: list[dict[str, Any]] = []
    stream_digests: dict[str, str] = {}
    for unit_position, unit in enumerate(units):
        unit_id = str(unit["evaluation_unit_id"])
        kernels: dict[str, np.ndarray] = {}
        for prompt_id in preflight["prompt_ids"]:
            selected = list(range(N_REFERENCE)) + full_indices[prompt_id][unit_id]
            kernels[prompt_id] = full_kernels[prompt_id][np.ix_(selected, selected)]
        result = evaluate_final_unit(
            kernels,
            score_parameters,
            stream_seed=int(unit["stream_seed_uint64"]),
            expected_stream_digest_sha256=str(unit["derived_prompt_permutation_seed_set_sha256"]),
        )
        stream_digests[unit_id] = result.seed_stream_digest_sha256
        for prompt_position, prompt_id in enumerate(preflight["prompt_ids"]):
            raw = float(result.observed_raw_mmd2[prompt_id])
            score = float(result.observed_scores[prompt_id])
            observed_raw[unit_position, prompt_position] = raw
            observed_scores[unit_position, prompt_position] = score
            permutation_raw[unit_position, prompt_position] = result.permutation_raw_mmd2[prompt_id]
            permutation_scores[unit_position, prompt_position] = result.permutation_scores[prompt_id]
            prompt_results.append(
                {
                    "schema_version": F1C_SCHEMA_VERSION,
                    "evaluation_unit_id": unit_id,
                    "unit_position": unit_position,
                    "data_role": unit["data_role"],
                    "attack_family": unit.get("attack_family"),
                    "prompt_id": prompt_id,
                    "raw_unbiased_mmd2": raw,
                    "raw_negative_preserved": raw < 0.0,
                    "score": score,
                    "score_rank_descending": 1 + sorted(
                        result.observed_scores, key=lambda key: (-result.observed_scores[key], key)
                    ).index(prompt_id),
                    "selected_in_dynamic_top2": prompt_id in result.top_prompt_ids,
                    "permutation_raw_mmd2_summary": numeric_summary(result.permutation_raw_mmd2[prompt_id]),
                    "permutation_score_summary": numeric_summary(result.permutation_scores[prompt_id]),
                }
            )
        permutation_top2[unit_position] = result.permutation_top2
        unit_results.append(
            {
                "schema_version": F1C_SCHEMA_VERSION,
                "evaluation_unit_id": unit_id,
                "unit_position": unit_position,
                "data_role": unit["data_role"],
                "attack_family": unit.get("attack_family"),
                "reference_bank_id": "final_reference_bank",
                "reference_count_per_prompt": N_REFERENCE,
                "target_count_per_prompt": N_TARGET,
                "observed_top2": result.observed_top2,
                "top1_prompt_id": result.top_prompt_ids[0],
                "top1_score": result.observed_scores[result.top_prompt_ids[0]],
                "top2_prompt_id": result.top_prompt_ids[1],
                "top2_score": result.observed_scores[result.top_prompt_ids[1]],
                "exceedance_count": result.exceedance_count,
                "p_global": result.p_global,
                "detected": result.alarm,
                "alarm": result.alarm,
                "p_value_grid": 0.001,
                "alarm_equivalent_to_exceedance_count_le_49": result.alarm == (result.exceedance_count <= 49),
                "permutation_seed_stream_digest_sha256": result.seed_stream_digest_sha256,
            }
        )

    if len(unit_results) != 100 or len(prompt_results) != 1200:
        raise ValueError("F1-C final result record count changed")
    decisions = summarize_final_decisions(unit_results)
    decisions["overall_attack_rate_is_secondary_only"] = True
    decisions["all_intervals_are_conditional_descriptive"] = True

    array_hashes = {
        "observed_raw_mmd2": save_npy(output / config["output"]["observed_raw_mmd2"], observed_raw),
        "observed_scores": save_npy(output / config["output"]["observed_scores"], observed_scores),
        "permutation_raw_mmd2": save_npy(output / config["output"]["permutation_raw_mmd2"], permutation_raw),
        "permutation_scores": save_npy(output / config["output"]["permutation_scores"], permutation_scores),
        "permutation_top2": save_npy(output / config["output"]["permutation_top2"], permutation_top2),
    }
    canonical_jsonl_write(output / config["output"]["unit_results"], unit_results)
    canonical_jsonl_write(output / config["output"]["prompt_statistics"], prompt_results)
    canonical_write(output / config["output"]["performance_summary"], decisions)
    permutation_audit = {
        "schema_version": F1C_SCHEMA_VERSION,
        "status": "PASS",
        "evaluation_unit_count": 100,
        "prompt_count_per_unit": 12,
        "permutations_per_unit": PERMUTATIONS,
        "frozen_prompt_level_seed_count": 1_198_800,
        "frozen_seed_set_sha256": preflight["artifacts"]["permutation"]["derived_seed_set_sha256"],
        "all_stream_digests_match": True,
        "stream_digests": stream_digests,
        "within_prompt_only": True,
        "group_sizes": [N_REFERENCE, N_TARGET],
        "cross_prompt_exchange": False,
        "measurement_or_score_refit_inside_permutation": False,
        "p_value_formula": "(1 + count(T_perm >= T_observed)) / 1000",
        "p_value_grid": 0.001,
        "p_value_grid_all_pass": all(
            abs(float(row["p_global"]) * 1000 - round(float(row["p_global"]) * 1000)) <= 1e-12
            for row in unit_results
        ),
        "alarm_equivalence_all_pass": all(row["alarm_equivalent_to_exceedance_count_le_49"] for row in unit_results),
        "array_sha256": array_hashes,
    }
    if not permutation_audit["p_value_grid_all_pass"] or not permutation_audit["alarm_equivalence_all_pass"]:
        raise ValueError("F1-C permutation p-value gate failed")
    canonical_write(output / config["output"]["permutation_audit"], permutation_audit)

    _, post_response_hashes = verify_f1b_terminal_index(config)
    if post_response_hashes != preflight["response_hashes"]:
        raise ValueError("F1-C detected a formal response bank mutation")
    report = {
        "schema_version": F1C_SCHEMA_VERSION,
        "status": "PASS",
        "phase": "H8_F1C_FINAL_PERFORMANCE_CONFIRMATION",
        "frozen_detector": "r60_q10_top2",
        "n_reference": N_REFERENCE,
        "n_target": N_TARGET,
        "top_r": TOP_R,
        "alpha": ALPHA,
        "permutations": PERMUTATIONS,
        "evaluation_unit_count": 100,
        "intact_unit_count": INTACT_UNIT_COUNT,
        "attack_family_endpoint_counts": {family: 10 for family in ATTACK_FAMILIES},
        "shared_reference_bank_count": 1,
        "feature_audit_sha256": file_sha256(output / config["output"]["feature_audit"]),
        "unit_results_sha256": file_sha256(output / config["output"]["unit_results"]),
        "prompt_statistics_sha256": file_sha256(output / config["output"]["prompt_statistics"]),
        "permutation_audit_sha256": file_sha256(output / config["output"]["permutation_audit"]),
        "performance_summary_sha256": file_sha256(output / config["output"]["performance_summary"]),
        "array_sha256": array_hashes,
        "formal_response_hashes_unchanged": True,
        "new_model_responses": 0,
        "attack_training_or_rematerialization_performed": False,
        "response_replacement_performed": False,
        "seed_rederivation_or_membership_change_performed": False,
        "measurement_or_score_refit_performed": False,
        "detector_tuning_performed": False,
        "conditional_on_single_frozen_shared_reference_bank": True,
        "primary_outcomes": decisions,
        "next_gate": "STOP_NO_AUTOMATIC_TUNING_OR_FOLLOWUP_EXPERIMENT",
    }
    canonical_write(output / config["output"]["final_report"], report)
    index_path = output / config["output"]["terminal_index"]
    terminal_index = terminal_output_index(output, {index_path.name})
    canonical_write(index_path, terminal_index)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if config.get("final_performance_confirmation_authorized") is not True:
        raise ValueError("F1-C formal offline confirmation is not authorized")
    run(config, preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
