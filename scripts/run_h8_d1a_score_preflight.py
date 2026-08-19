from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_integrity.h8_precalibration import canonical_sha256
from llm_integrity.h8_score_calibration import (
    SAMPLE_STRUCTURES,
    fit_score_calibration_parameters,
    load_frozen_mmd_measurement,
    score_schema_descriptor,
    score_schema_sha256,
    score_value,
)


DEFAULT_CONFIG = ROOT / "configs" / "h8_d1a_score_calibration_preflight.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(logical: str) -> Path:
    target = (ROOT / logical).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise ValueError("Configured path escapes repository")
    return target


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def git_commit_and_clean() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    if status.strip():
        raise ValueError("D1-A preflight requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def archive_snapshot(archive: Path) -> dict[str, str]:
    return {
        str(path.relative_to(archive)).replace("\\", "/"): file_sha256(path)
        for path in sorted(archive.rglob("*"))
        if path.is_file()
    }


def validate_scope(config: Mapping[str, Any]) -> None:
    for key in (
        "model_sampling_authorized",
        "formal_reference_access_authorized",
        "heldout_access_authorized",
        "attack_access_authorized",
    ):
        if config.get(key) is not False:
            raise ValueError(f"Forbidden authorization must be false: {key}")
    if not all(bool(value) for value in config["forbidden_operations"].values()):
        raise ValueError("Every forbidden operation must be explicitly enforced")


def load_development_nulls(path: Path, prompt_ids: tuple[str, ...]) -> dict[str, dict[str, list[float]]]:
    if not path.is_file():
        raise ValueError("Development dry-run input is unavailable")
    grouped: dict[str, dict[str, list[float]]] = {
        structure: defaultdict(list) for structure in SAMPLE_STRUCTURES
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            structure = str(row.get("structure"))
            prompt_id = str(row.get("prompt_id"))
            if structure not in grouped or prompt_id not in prompt_ids:
                raise ValueError(f"Unexpected dry-run row at line {line_number}")
            value = float(row["unbiased_mmd2"])
            if not math.isfinite(value):
                raise ValueError("Dry-run MMD contains NaN or infinity")
            grouped[structure][prompt_id].append(value)
    for structure in SAMPLE_STRUCTURES:
        if set(grouped[structure]) != set(prompt_ids):
            raise ValueError(f"Dry-run prompt set mismatch: {structure}")
        if set(map(len, grouped[structure].values())) != {200}:
            raise ValueError(f"Dry-run must contain 200 null statistics per prompt: {structure}")
    return {structure: dict(values) for structure, values in grouped.items()}


def internal_toy_gates() -> dict[str, bool]:
    nondegenerate = {
        "mapping": "nondegenerate_centered",
        "null_median_m": 2.0,
        "effective_scale": 0.5,
    }
    degenerate = {
        "mapping": "degenerate_global_scale",
        "null_median_m": 0.0,
        "effective_scale": 0.25,
    }
    negative_raw = -0.75
    scores = [
        score_value(value, nondegenerate)
        for value in (-1.0, 2.0, 2.25, 2.5, 3.0)
    ]
    degenerate_scores = [score_value(value, degenerate) for value in (0.0, 0.25, 0.5)]
    return {
        "raw_negative_value_preserved": negative_raw == -0.75,
        "center_maps_to_zero": scores[1] == 0.0,
        "below_center_maps_to_zero": scores[0] == 0.0,
        "above_center_monotone": 0.0 < scores[2] < scores[3] < scores[4],
        "degenerate_zero_maps_to_zero": degenerate_scores[0] == 0.0,
        "degenerate_positive_maps_positive": degenerate_scores[1] > 0.0,
        "all_scores_finite": all(math.isfinite(value) for value in scores + degenerate_scores),
    }


def proposed_data_manifest_schema(binding: Any) -> dict[str, Any]:
    return {
        "schema_version": "h8-score-calibration-data-manifest-1.0",
        "manifest_level_fields": [
            "data_role=score_calibration_fit_only",
            "root_seed",
            "seed_derivation_algorithm",
            "record_count",
            "sample_structures",
            "measurement_manifest_sha256",
            "feature_schema_sha256",
            "scaler_payload_sha256_by_prompt",
            "bandwidth_payload_sha256_by_prompt",
            "schedule_sha256",
        ],
        "record_fields": [
            "record_id",
            "prompt_id",
            "sample_structure",
            "calibration_replicate_id",
            "split_seed_uint64",
            "reference_response_ids",
            "query_response_ids",
            "reference_response_ids_sha256",
            "query_response_ids_sha256",
            "raw_unbiased_mmd2_float64",
            "measurement_manifest_sha256",
            "feature_schema_sha256",
            "scaler_payload_sha256",
            "bandwidth_payload_sha256",
            "eligible_for_formal_reference=false",
            "eligible_for_heldout_evaluation=false",
            "eligible_for_target_evaluation=false",
            "eligible_for_attack_evaluation=false",
        ],
        "required_prompt_ids": list(binding.prompt_ids),
        "required_sample_structures": list(SAMPLE_STRUCTURES),
        "fit_api_rejects_roles": ["formal_reference", "heldout", "target", "attack"],
    }


def run_preflight_unit_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_h8_score_calibration.py",
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    validate_scope(config)
    code_commit = git_commit_and_clean()
    protocol = resolve(config["provenance"]["protocol"])
    if file_sha256(protocol) != config["provenance"]["protocol_sha256"]:
        raise ValueError("D1-A protocol SHA256 mismatch")

    frozen = config["frozen_measurement"]
    archive = resolve(frozen["archive"])
    before = archive_snapshot(archive)
    binding = load_frozen_mmd_measurement(
        archive,
        expected_manifest_sha256=frozen["manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )
    score_config = config["score_calibration"]
    if tuple(score_config["sample_structures"]) != SAMPLE_STRUCTURES:
        raise ValueError("Configured sample structures do not match the frozen score schema")
    threshold = float(score_config["numerical_zero_threshold"])
    if float(score_config["robust_mad_multiplier"]) != 1.4826:
        raise ValueError("Robust MAD multiplier mismatch")

    dry_path = resolve(config["development_dry_run"]["input"])
    nulls = load_development_nulls(dry_path, binding.prompt_ids)
    dry_payload = fit_score_calibration_parameters(
        nulls,
        binding,
        data_role="development_dry_run",
        numerical_zero_threshold=threshold,
    )
    if dry_payload["eligible_to_freeze"] is not False or dry_payload["artifact_status"] != "development_dry_run":
        raise ValueError("Development dry-run was incorrectly made freeze-eligible")
    dry_scores: list[float] = []
    negative_raw_count = 0
    structure_diagnostics: dict[str, Any] = {}
    for structure in SAMPLE_STRUCTURES:
        parameters = dry_payload["structures"][structure]
        per_prompt: dict[str, Any] = {}
        for prompt_id in binding.prompt_ids:
            prompt_parameters = parameters["prompt_parameters"][prompt_id]
            values = nulls[structure][prompt_id]
            negative_raw_count += sum(value < 0.0 for value in values)
            scores = [score_value(value, prompt_parameters) for value in values]
            dry_scores.extend(scores)
            per_prompt[prompt_id] = {
                "feature_degenerate": prompt_parameters["feature_degenerate"],
                "mapping": prompt_parameters["mapping"],
                "null_median_m": prompt_parameters["null_median_m"],
                "own_robust_scale_a": prompt_parameters["own_robust_scale_a"],
                "effective_scale": prompt_parameters["effective_scale"],
                "score_min": min(scores),
                "score_max": max(scores),
                "score_finite": all(math.isfinite(value) for value in scores),
            }
        structure_diagnostics[structure] = {
            "a_global": parameters["a_global"],
            "prompt_diagnostics": per_prompt,
        }

    toy_gates = internal_toy_gates()
    unit_tests = run_preflight_unit_tests()
    after = archive_snapshot(archive)
    gates = {
        **toy_gates,
        "frozen_archive_hashes_unchanged": before == after,
        "all_development_scores_finite": all(math.isfinite(value) for value in dry_scores),
        "four_structures_preserved": set(dry_payload["structures"]) == set(SAMPLE_STRUCTURES),
        "development_dry_run_not_freeze_eligible": dry_payload["eligible_to_freeze"] is False,
        "sample_size_selection_not_performed": dry_payload["sample_size_selection_performed"] is False,
        "top_r_selection_not_performed": dry_payload["top_r_selection_performed"] is False,
        "preflight_unit_tests_pass": unit_tests["passed"],
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "phase": "H8_D1A_SCORE_CALIBRATION_PREFLIGHT",
        "status": status,
        "preflight_code_commit": code_commit,
        "measurement_layer_status": "frozen",
        "detector_status": "not_frozen",
        "score_calibration_status": "preflight_only_not_frozen",
        "mmd_frozen_manifest_sha256": binding.manifest_sha256,
        "mmd_implementation_commit": binding.mmd_implementation_commit,
        "feature_schema_sha256": binding.feature_schema_sha256,
        "feature_schema_payload_sha256": binding.feature_schema_payload_sha256,
        "global_exclusion_mask_payload_sha256": binding.global_exclusion_mask_payload_sha256,
        "global_bandwidth_payload_sha256": binding.global_bandwidth_payload_sha256,
        "scaler_payload_sha256_by_prompt": dict(binding.scaler_payload_sha256_by_prompt),
        "bandwidth_payload_sha256_by_prompt": dict(binding.bandwidth_payload_sha256_by_prompt),
        "score_schema": score_schema_descriptor(threshold),
        "score_schema_sha256": score_schema_sha256(threshold),
        "primary_score_formula": {
            "nondegenerate": "S_j=max(0,(D_j-m_j)/a_j), m_j=median(D_null), a_j=1.4826*MAD(D_null)",
            "degenerate": "S_j=max(0,D_j/a_global_structure)",
            "a_global": "median positive nondegenerate a_k in the same sample structure",
            "epsilon_scale_replacement": False,
            "raw_negative_unbiased_mmd2_preserved": True,
        },
        "sample_structures": list(SAMPLE_STRUCTURES),
        "sample_size_selection_performed": False,
        "top_r_selection_performed": False,
        "final_score_parameter_fit_performed": False,
        "new_model_responses": 0,
        "formal_reference_read_or_generated": False,
        "heldout_read_or_generated": False,
        "attack_read_or_generated": False,
        "development_dry_run": {
            "performed": True,
            "data_role": "development_dry_run",
            "eligible_to_freeze": False,
            "source": config["development_dry_run"]["input"],
            "raw_statistic_count": sum(len(values) for prompts in nulls.values() for values in prompts.values()),
            "negative_raw_statistic_count": negative_raw_count,
            "score_count": len(dry_scores),
            "parameter_payload_sha256_development_only": canonical_sha256(dry_payload),
            "structure_diagnostics": structure_diagnostics,
        },
        "test_results": {
            "internal_gates": gates,
            "all_pass": all(gates.values()),
            "preflight_unit_tests": unit_tests,
        },
        "proposed_new_data_manifest_schema": proposed_data_manifest_schema(binding),
        "next_step": "STOP_AND_WAIT_FOR_EXPLICIT_APPROVAL_BEFORE_NEW_SCORE_CALIBRATION_DATA",
    }
    output = resolve(config["output"]["report"])
    if output.exists():
        raise ValueError("D1-A preflight report already exists; refusing overwrite")
    atomic_json(output, report)
    print(json.dumps({"status": status, "report": str(output), "score_schema_sha256": report["score_schema_sha256"]}))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
