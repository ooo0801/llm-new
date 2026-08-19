from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_integrity.h8_m0ij import (
    deterministic_uint64,
    frozen_split,
    mmd2_biased_from_kernel,
    mmd2_unbiased_from_kernel,
    numerical_summary,
    permutation_sanity,
)
from llm_integrity.h8_precalibration import (
    FamilyBalancedScaler,
    build_h8_feature_schema,
    canonical_sha256,
    h8_mmd2_biased,
    h8_mmd2_unbiased_unequal,
    h8_rbf_kernel,
    load_h8_artifact,
)


DEFAULT_CONFIG = ROOT / "configs" / "h8_m0ij_mmd_numerical_validation.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")
    temporary.replace(path)


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes repository")
    return path


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


def git_commit_and_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise ValueError("M0-I/J requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def candidate_snapshot(config: Mapping[str, Any]) -> dict[str, str]:
    source = resolve(config["candidate_input"]["directory"])
    files = [path for path in source.rglob("*") if path.is_file()]
    return {str(path.relative_to(source)).replace("\\", "/"): file_sha256(path) for path in sorted(files)}


def validate_scope(config: Mapping[str, Any]) -> None:
    for key in (
        "model_sampling_authorized",
        "formal_reference_access_authorized",
        "heldout_access_authorized",
        "attack_access_authorized",
        "parameter_modification_authorized",
    ):
        if config.get(key) is not False:
            raise ValueError(f"Forbidden authorization is not false: {key}")
    if not all(bool(value) for value in config["forbidden_operations"].values()):
        raise ValueError("Every forbidden operation must be explicitly enabled as a prohibition")


def load_candidates(config: Mapping[str, Any]) -> dict[str, Any]:
    item = config["candidate_input"]
    directory = resolve(item["directory"])
    for key, label in (
        ("report", "M0-G/H report"),
        ("artifact_hash_index", "M0-G/H hash index"),
        ("feature_matrix", "feature matrix"),
        ("feature_row_index", "feature row index"),
        ("feature_schema", "feature schema"),
        ("global_exclusion_mask", "global exclusion mask"),
        ("global_bandwidth", "global bandwidth"),
        ("bandwidth_stability", "bandwidth stability"),
    ):
        # Some artifacts expose both a byte-level file digest and a semantic
        # schema/payload digest.  File verification must prefer the explicit
        # byte-level field; the semantic digest is checked separately below.
        expected = item.get(f"{key}_file_sha256") or item.get(f"{key}_sha256")
        require_hash(directory / item[key], expected, label)

    report = json.loads((directory / item["report"]).read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("candidate_artifacts_only") is not True:
        raise ValueError("M0-G/H candidate report is not PASS/candidate-only")
    schema_payload = load_h8_artifact(
        directory / item["feature_schema"],
        "h8_feature_schema_candidate",
        item["feature_schema_payload_sha256"],
    )
    schema = build_h8_feature_schema(512)
    if schema.sha256 != item["feature_schema_sha256"] or schema_payload["feature_schema_sha256"] != schema.sha256:
        raise ValueError("Feature schema does not reconstruct to the frozen hash")
    mask_payload = load_h8_artifact(
        directory / item["global_exclusion_mask"],
        "h8_global_exclusion_mask_candidate",
        item["global_exclusion_mask_payload_sha256"],
    )
    global_payload = load_h8_artifact(
        directory / item["global_bandwidth"],
        "h8_global_bandwidth_candidate",
        item["global_bandwidth_payload_sha256"],
    )
    stability_payload = load_h8_artifact(
        directory / item["bandwidth_stability"],
        "h8_bandwidth_stability_candidate",
        item["bandwidth_stability_payload_sha256"],
    )
    if global_payload.get("overall_status") != "PASS" or stability_payload.get("overall_status") != "PASS":
        raise ValueError("M0-G/H stability is not PASS")

    matrix = np.load(directory / item["feature_matrix"], allow_pickle=False)
    if matrix.dtype != np.float64 or matrix.shape != (item["required_total_rows"], schema.dimension):
        raise ValueError("Feature matrix dtype/shape mismatch")
    rows = [json.loads(line) for line in (directory / item["feature_row_index"]).read_text(encoding="utf-8").splitlines()]
    if len(rows) != item["required_total_rows"]:
        raise ValueError("Feature row index count mismatch")
    grouped: dict[str, list[int]] = defaultdict(list)
    for expected_row, row in enumerate(rows):
        if row.get("data_role") != "mmd_precalibration_fit_only" or int(row["row_index"]) != expected_row:
            raise ValueError("Forbidden role or malformed feature row index")
        grouped[str(row["prompt_id"])].append(expected_row)
    if len(grouped) != item["required_prompt_count"] or set(map(len, grouped.values())) != {item["required_rows_per_prompt"]}:
        raise ValueError("Prompt/row counts mismatch")

    scalers: dict[str, FamilyBalancedScaler] = {}
    bandwidths: dict[str, dict[str, Any]] = {}
    for prompt_id in sorted(grouped):
        scaler_payload = load_h8_artifact(
            directory / item["scaler_directory"] / f"{prompt_id}.json",
            "h8_family_balanced_scaler_candidate",
            item["expected_scaler_payload_sha256"][prompt_id],
        )
        bandwidth_payload = load_h8_artifact(
            directory / item["bandwidth_directory"] / f"{prompt_id}.json",
            "h8_prompt_bandwidth_candidate",
            item["expected_bandwidth_payload_sha256"][prompt_id],
        )
        scaler = FamilyBalancedScaler.from_dict(scaler_payload)
        if scaler.prompt_id != prompt_id or scaler.feature_schema_sha256 != schema.sha256:
            raise ValueError("Scaler prompt/schema binding mismatch")
        if bandwidth_payload["prompt_id"] != prompt_id or bandwidth_payload["scaler_payload_sha256"] != canonical_sha256(scaler_payload):
            raise ValueError("Bandwidth/scaler binding mismatch")
        if bandwidth_payload["stability_status"] != "PASS" or float(bandwidth_payload["sigma"]) <= 0:
            raise ValueError("Bandwidth is invalid or did not PASS")
        if bandwidth_payload["bandwidth_source"] == "global_degenerate_fallback":
            if bandwidth_payload["global_bandwidth_payload_sha256"] != item["global_bandwidth_payload_sha256"]:
                raise ValueError("Global fallback hash binding mismatch")
        scalers[prompt_id] = scaler
        bandwidths[prompt_id] = bandwidth_payload
    if tuple(mask_payload["mask"]) != scalers[next(iter(scalers))].global_exclusion_mask:
        raise ValueError("Global exclusion mask/scaler mismatch")
    return {
        "directory": directory,
        "schema": schema,
        "matrix": matrix,
        "grouped": dict(grouped),
        "scalers": scalers,
        "bandwidths": bandwidths,
        "global_bandwidth": global_payload,
        "stability": stability_payload,
    }


def make_seed_split_manifest(config: Mapping[str, Any], prompt_ids: list[str]) -> dict[str, Any]:
    numerical = config["numerical_validation"]
    permutation = config["permutation_sanity"]
    trials: list[dict[str, Any]] = []
    representatives: list[dict[str, Any]] = []
    for prompt_id in prompt_ids:
        for n_reference, n_target in numerical["sample_size_structures"]:
            structure = f"r{n_reference}_q{n_target}"
            for trial_id in range(numerical["repetitions_per_structure"]):
                seed = deterministic_uint64(numerical["root_seed"], prompt_id, structure, trial_id)
                ref, target = frozen_split(100, n_reference, n_target, seed)
                entry = {
                    "prompt_id": prompt_id,
                    "structure": structure,
                    "n_reference": n_reference,
                    "n_target": n_target,
                    "trial_id": trial_id,
                    "resampling_seed_uint64": seed,
                    "reference_indices": ref.tolist(),
                    "target_indices": target.tolist(),
                }
                trials.append(entry)
                if trial_id == permutation["representative_trial_index"]:
                    representatives.append({
                        **entry,
                        "permutation_seed_uint64": deterministic_uint64(
                            permutation["root_seed"], prompt_id, structure, "permutation"
                        ),
                    })
    payload = {
        "schema_version": "h8-m0ij-1.0",
        "rng": "numpy.random.Generator(PCG64)",
        "numerical_root_seed": numerical["root_seed"],
        "permutation_root_seed": permutation["root_seed"],
        "trials": trials,
        "representative_trials": representatives,
    }
    return {"payload": payload, "payload_sha256": canonical_sha256(payload)}


def regression_gates() -> dict[str, Any]:
    x = np.asarray([[0.0], [2.0], [4.0]], dtype=np.float64)
    y = np.asarray([[1.0], [3.0]], dtype=np.float64)
    direct = h8_mmd2_unbiased_unequal(x, y, 1.1)
    reverse = h8_mmd2_unbiased_unequal(y, x, 1.1)
    kernel = h8_rbf_kernel(np.vstack([x, y]), np.vstack([x, y]), 1.1)
    indexed = mmd2_unbiased_from_kernel(kernel, [0, 1, 2], [3, 4])
    biased_direct = h8_mmd2_biased(x, y, 1.1)
    biased_indexed = mmd2_biased_from_kernel(kernel, [0, 1, 2], [3, 4])
    negative_preserved = direct < 0.0
    return {
        "symmetry": {"pass": abs(direct - reverse) <= 1e-15, "difference": direct - reverse},
        "unequal_size": {"pass": abs(direct - indexed) <= 1e-15, "direct": direct, "indexed": indexed},
        "biased_kernel_equivalence": {"pass": abs(biased_direct - biased_indexed) <= 1e-15},
        "negative_value_preservation": {"pass": negative_preserved, "value": direct},
    }


def write_frozen_archive(
    config: Mapping[str, Any],
    commit: str,
    result_dir: Path,
    candidate_dir: Path,
) -> tuple[Path, str]:
    archive = resolve(config["output"]["frozen_archive"])
    if archive.exists():
        raise ValueError("Frozen archive already exists")
    archive.mkdir(parents=True)
    copies: list[tuple[Path, Path]] = []
    for protocol in (
        config["provenance"]["m0f_protocol"],
        config["provenance"]["m0gh_protocol"],
        config["provenance"]["m0ij_protocol"],
    ):
        source = resolve(protocol)
        copies.append((source, archive / "protocols" / source.name))
    for name in (
        "feature_schema_candidate.json",
        "global_exclusion_mask_candidate.json",
        "global_bandwidth_candidate.json",
        "bandwidth_stability_report.json",
        "H8_M0GH_SCALER_BANDWIDTH_REPORT.json",
    ):
        copies.append((candidate_dir / name, archive / "measurement" / name))
    for directory in ("scaler_candidates", "bandwidth_candidates"):
        for source in sorted((candidate_dir / directory).glob("*.json")):
            copies.append((source, archive / "measurement" / directory / source.name))
    for name in (
        config["output"]["seed_manifest"],
        config["output"]["numerical_report"],
        config["output"]["permutation_report"],
        config["output"]["final_report_json"],
        config["output"]["final_report_md"],
    ):
        copies.append((result_dir / name, archive / "validation" / name))
    for source in (
        ROOT / "src" / "llm_integrity" / "h8_precalibration.py",
        ROOT / "src" / "llm_integrity" / "h8_m0ij.py",
        ROOT / "scripts" / "run_h8_m0ij_validation.py",
        ROOT / "configs" / "h8_m0ij_mmd_numerical_validation.yaml",
    ):
        copies.append((source, archive / "implementation" / source.name))
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    hashes = {
        str(path.relative_to(archive)).replace("\\", "/"): file_sha256(path)
        for path in sorted(archive.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema_version": "h8-mmd-frozen-manifest-1.0",
        "measurement_layer_status": "frozen",
        "detector_status": "not_frozen",
        "mmd_implementation_commit": commit,
        "data_scope": "mmd_precalibration_fit_only",
        "formal_reference_responses": 0,
        "heldout_responses": 0,
        "attack_responses": 0,
        "new_model_responses": 0,
        "files": hashes,
    }
    manifest_path = archive / config["output"]["frozen_manifest"]
    atomic_json(manifest_path, manifest)
    manifest_sha = file_sha256(manifest_path)
    (manifest_path.with_suffix(manifest_path.suffix + ".sha256")).write_text(
        manifest_sha + "  " + manifest_path.name + "\n", encoding="ascii"
    )
    return archive, manifest_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_scope(config)
    commit = git_commit_and_clean()
    provenance = config["provenance"]
    for key in ("m0f", "m0gh", "m0ij"):
        require_hash(resolve(provenance[f"{key}_protocol"]), provenance[f"{key}_protocol_sha256"], f"{key} protocol")
    result_dir = resolve(config["output"]["result_directory"])
    frozen_dir = resolve(config["output"]["frozen_archive"])
    if result_dir.exists() or frozen_dir.exists():
        raise ValueError("M0-I/J outputs already exist; fail closed instead of overwriting")

    before = candidate_snapshot(config)
    inputs = load_candidates(config)
    prompt_ids = sorted(inputs["grouped"])
    result_dir.mkdir(parents=True)
    manifest = make_seed_split_manifest(config, prompt_ids)
    manifest_path = result_dir / config["output"]["seed_manifest"]
    atomic_json(manifest_path, manifest)

    transformed: dict[str, np.ndarray] = {}
    kernels: dict[str, np.ndarray] = {}
    for prompt_id in prompt_ids:
        values = inputs["matrix"][inputs["grouped"][prompt_id]]
        transformed[prompt_id] = inputs["scalers"][prompt_id].transform(values, inputs["schema"])
        kernels[prompt_id] = h8_rbf_kernel(
            transformed[prompt_id], transformed[prompt_id], float(inputs["bandwidths"][prompt_id]["sigma"])
        )

    pseudo_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    degenerate_failures: list[str] = []
    tolerance = float(config["numerical_validation"]["unbiased_tolerance_degenerate"])
    grouped_trials: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest["payload"]["trials"]:
        grouped_trials[(entry["prompt_id"], entry["structure"])].append(entry)
    for (prompt_id, structure), trials in sorted(grouped_trials.items()):
        unbiased: list[float] = []
        biased: list[float] = []
        for entry in trials:
            value = mmd2_unbiased_from_kernel(kernels[prompt_id], entry["reference_indices"], entry["target_indices"])
            sensitivity = mmd2_biased_from_kernel(kernels[prompt_id], entry["reference_indices"], entry["target_indices"])
            unbiased.append(value)
            biased.append(sensitivity)
            pseudo_rows.append({
                "prompt_id": prompt_id,
                "structure": structure,
                "trial_id": entry["trial_id"],
                "n_reference": entry["n_reference"],
                "n_target": entry["n_target"],
                "seed_uint64": entry["resampling_seed_uint64"],
                "unbiased_mmd2": value,
                "biased_mmd2": sensitivity,
            })
        feature_degenerate = bool(inputs["bandwidths"][prompt_id]["feature_degenerate"])
        invariant_pass = not feature_degenerate or (
            max(map(abs, unbiased)) <= tolerance and max(map(abs, biased)) <= tolerance
        )
        if not invariant_pass:
            degenerate_failures.append(f"{prompt_id}:{structure}")
        summaries[f"{prompt_id}:{structure}"] = {
            "prompt_id": prompt_id,
            "structure": structure,
            "feature_degenerate": feature_degenerate,
            "identical_feature_numerical_invariant_pass": invariant_pass,
            "unbiased_mmd2": numerical_summary(unbiased),
            "biased_mmd2": numerical_summary(biased),
        }
    atomic_jsonl(result_dir / config["output"]["pseudo_trials"], pseudo_rows)
    numerical_nonfinite = sum(
        row["unbiased_mmd2"]["nan_count"] + row["unbiased_mmd2"]["inf_count"]
        + row["biased_mmd2"]["nan_count"] + row["biased_mmd2"]["inf_count"]
        for row in summaries.values()
    )
    numerical_report = {
        "phase": "H8_M0I",
        "status": "PASS" if not degenerate_failures and numerical_nonfinite == 0 else "FAIL",
        "sample_size_selection_performed": False,
        "scaler_or_bandwidth_refit_performed": False,
        "trial_count": len(pseudo_rows),
        "degenerate_invariant_failures": degenerate_failures,
        "nan_or_inf_count": numerical_nonfinite,
        "summaries": summaries,
    }
    atomic_json(result_dir / config["output"]["numerical_report"], numerical_report)

    permutation_rows: list[dict[str, Any]] = []
    permutation_failures: list[str] = []
    for entry in manifest["payload"]["representative_trials"]:
        prompt_id = entry["prompt_id"]
        result = permutation_sanity(
            kernels[prompt_id], entry["reference_indices"], entry["target_indices"],
            permutation_seed=entry["permutation_seed_uint64"],
            permutations=config["permutation_sanity"]["B_precheck"],
        )
        degenerate = bool(inputs["bandwidths"][prompt_id]["feature_degenerate"])
        zero_pass = not degenerate or (
            abs(result.observed) <= tolerance
            and max(map(abs, result.permutation_values)) <= tolerance
            and result.p_value == 1.0
        )
        passed = (
            result.reproducible and result.group_size_correct
            and result.observed_evaluation_count == 1 and 0.0 < result.p_value <= 1.0
            and zero_pass
        )
        if not passed:
            permutation_failures.append(f"{prompt_id}:{entry['structure']}")
        permutation_rows.append({
            "prompt_id": prompt_id,
            "structure": entry["structure"],
            "n_reference": entry["n_reference"],
            "n_target": entry["n_target"],
            "resampling_seed_uint64": entry["resampling_seed_uint64"],
            "permutation_seed_uint64": entry["permutation_seed_uint64"],
            "observed_unbiased_mmd2": result.observed,
            "permutation_summary": numerical_summary(result.permutation_values),
            "exceedance_count": result.exceedance_count,
            "p_value": result.p_value,
            "observed_evaluation_count": result.observed_evaluation_count,
            "fixed_seed_reproducible": result.reproducible,
            "group_size_correct": result.group_size_correct,
            "degenerate_zero_and_p_one_pass": zero_pass,
            "status": "PASS" if passed else "FAIL",
        })
    atomic_jsonl(result_dir / config["output"]["permutation_trials"], permutation_rows)
    regressions = regression_gates()
    regression_pass = all(item["pass"] for item in regressions.values())
    after = candidate_snapshot(config)
    immutable = before == after
    permutation_report = {
        "phase": "H8_M0J",
        "status": "PASS" if not permutation_failures and regression_pass and immutable else "FAIL",
        "representative_trial_count": len(permutation_rows),
        "permutations_per_trial": config["permutation_sanity"]["B_precheck"],
        "observed_in_permutation_loop": False,
        "scaler_or_bandwidth_refit_performed": False,
        "failures": permutation_failures,
        "regression_gates": regressions,
        "candidate_artifact_hashes_unchanged": immutable,
    }
    atomic_json(result_dir / config["output"]["permutation_report"], permutation_report)

    overall_pass = numerical_report["status"] == permutation_report["status"] == "PASS"
    structured = inputs["stability"]["prompt_results"]["structured_h6_ab6cbbad91f4"]
    final_report = {
        "phase": "H8_M0I_J",
        "status": "PASS" if overall_pass else "FAIL",
        "measurement_layer_status": "frozen" if overall_pass else "candidate",
        "detector_status": "not_frozen",
        "mmd_implementation_commit": commit,
        "sample_size_selection_performed": False,
        "model_sampling_performed": False,
        "new_model_responses": 0,
        "formal_reference_responses": 0,
        "heldout_responses": 0,
        "attack_responses": 0,
        "mmd_precalibration_fit_only_responses": 1200,
        "pseudo_mmd_trial_count": len(pseudo_rows),
        "permutation_representative_trial_count": len(permutation_rows),
        "permutation_statistic_count": len(permutation_rows) * config["permutation_sanity"]["B_precheck"],
        "candidate_snapshot_sha256": canonical_sha256(before),
        "feature_schema_sha256": config["candidate_input"]["feature_schema_sha256"],
        "global_exclusion_mask_payload_sha256": config["candidate_input"]["global_exclusion_mask_payload_sha256"],
        "global_bandwidth_payload_sha256": config["candidate_input"]["global_bandwidth_payload_sha256"],
        "scaler_payload_sha256_by_prompt": config["candidate_input"]["expected_scaler_payload_sha256"],
        "bandwidth_payload_sha256_by_prompt": config["candidate_input"]["expected_bandwidth_payload_sha256"],
        "numerical_status": numerical_report["status"],
        "permutation_status": permutation_report["status"],
        "structured_stability_preexisting_observation": {
            "gate_was_preregistered_and_passed": structured["overall_status"] == "PASS",
            "fixed_full_scaler_min_ratio": min(structured["fixed_full_scaler"]["ratios"]),
            "fixed_full_scaler_q05": structured["fixed_full_scaler"]["q05"],
            "fixed_full_scaler_q95": structured["fixed_full_scaler"]["q95"],
            "refit_scaler_min_ratio": min(structured["refit_subsample_scaler"]["ratios"]),
            "refit_scaler_q05": structured["refit_subsample_scaler"]["q05"],
            "refit_scaler_q95": structured["refit_subsample_scaler"]["q95"],
            "post_hoc_gate_change_performed": False,
        },
        "forbidden_downstream_steps_performed": False,
        "next_step": "STOP_AND_WAIT_FOR_EXPLICIT_APPROVAL",
    }
    atomic_json(result_dir / config["output"]["final_report_json"], final_report)
    markdown = (
        "# H8 MMD Precalibration Final Report\n\n"
        f"- Status: **{final_report['status']}**\n"
        f"- Measurement layer: **{final_report['measurement_layer_status']}**\n"
        "- Detector: **not_frozen**\n"
        f"- Intact-only pseudo-MMD trials: {len(pseudo_rows)}\n"
        f"- Representative permutation trials: {len(permutation_rows)} × 999\n"
        "- New model responses: 0\n"
        "- Formal Reference / held-out / attack responses: 0 / 0 / 0\n"
        "- Sample-size selection performed: false\n"
        "- D→S, Top-r/Max/Energy and global permutation: not performed\n\n"
        "The structured-prompt low individual subsample ratios were recorded without changing the preregistered PASS gate.\n"
    )
    (result_dir / config["output"]["final_report_md"]).write_text(markdown, encoding="utf-8", newline="\n")

    if not overall_pass:
        raise RuntimeError("M0-I/J gate failed; candidate measurement layer was not frozen")
    archive, manifest_sha = write_frozen_archive(config, commit, result_dir, inputs["directory"])
    hashes = {
        str(path.relative_to(result_dir)).replace("\\", "/"): file_sha256(path)
        for path in sorted(result_dir.rglob("*")) if path.is_file()
    }
    hashes["frozen_manifest_sha256"] = manifest_sha
    hashes["frozen_archive"] = str(archive.relative_to(ROOT)).replace("\\", "/")
    atomic_json(result_dir / config["output"]["result_hashes"], hashes)
    print(json.dumps({"status": "PASS", "result_directory": str(result_dir), "frozen_manifest_sha256": manifest_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
