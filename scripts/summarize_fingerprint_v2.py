from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compact_realization(value: dict) -> dict:
    result = deepcopy(value)
    details = result.get("details")
    if isinstance(details, dict):
        adapter_path = details.get("adapter_path")
        if adapter_path:
            details["adapter_path"] = (
                "results/fingerprint_v2_global_20260731/finetuning_adapters/"
                + Path(str(adapter_path)).name
            )
        attack_details = details.get("details")
        if isinstance(attack_details, dict):
            selected = attack_details.pop("selected_structures", None)
            if isinstance(selected, dict):
                attack_details["selected_structure_counts"] = {
                    str(layer): len(indices)
                    for layer, indices in selected.items()
                }
                attack_details["selected_structures_sha256"] = canonical_sha256(selected)
    return result


def compact_fingerprint(source: Path, target: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for entry in payload.get("entries", []):
        components = entry.pop("components", [])
        metadata = entry.setdefault("metadata", {})
        metadata["activation_component_count"] = len(components)
        metadata["activation_components_sha256"] = canonical_sha256(components)
    payload.setdefault("metadata", {})["activation_components"] = (
        "Counts and canonical checksums are published per entry; runtime component "
        "profiles remain under ignored results/."
    )
    write_json(target, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish compact global-calibrated fingerprint V2 evidence")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_path(config_path)
    config = load_config(config_path)
    output_dir = project_path(config["output_dir"])
    release_dir = project_path(config["reproducibility_dir"])
    manifests = read_jsonl(project_path(config["data"]["attack_manifest"]))
    results = []
    compact_results = []
    for manifest in manifests:
        variant_id = str(manifest["variant_id"])
        path = output_dir / "verification" / f"global_mcc_v2__{variant_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing V2 endpoint result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(payload)
        compact = {key: value for key, value in payload.items() if key != "response_records"}
        compact["variant_realization"] = compact_realization(dict(compact["variant_realization"]))
        compact_results.append(compact)

    true_positive = sum(row["ground_truth_modified"] and row["predicted_modified"] for row in results)
    false_negative = sum(row["ground_truth_modified"] and not row["predicted_modified"] for row in results)
    false_positive = sum(not row["ground_truth_modified"] and row["predicted_modified"] for row in results)
    true_negative = sum(not row["ground_truth_modified"] and not row["predicted_modified"] for row in results)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    metrics = {
        "accuracy": (true_positive + true_negative) / len(results),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "variants": len(results),
    }
    selection_path = output_dir / "mcc_selection.json"
    activation_path = output_dir / "activation_audit.json"
    universe_path = output_dir / "global_component_universe.json"
    fingerprint_path = output_dir / "fingerprints" / "global_mcc_v2.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    selection["fingerprint"] = "results/fingerprint_v2_global_20260731/fingerprints/global_mcc_v2.json"
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = None

    v1_selection_path = ROOT / "reproducibility/fingerprint_v1_20260730/mcc_selection.json"
    v1_selection = json.loads(v1_selection_path.read_text(encoding="utf-8"))
    v1_ids = set(str(value) for value in v1_selection["selected_ids"])
    v2_ids = set(str(value) for value in selection["selected_ids"])
    report = {
        "schema_version": "fingerprint_v2_global_final_report_1.0",
        "experiment_version": "fingerprint_v2_global_20260731",
        "status": "complete",
        "scope": "Independent calibration224 -> empirical global observable component universe -> global unweighted MCC12 -> new reference fingerprint -> seven-state endpoint verification",
        "scientific_boundary": "The global universe is a finite calibration estimate under the frozen V1 component schema; each endpoint family still has one configured state and is not a population-level rate estimate.",
        "git_commit": git_commit,
        "v1_parent_tag": "v1-fingerprint-closed-loop",
        "calibration_prompts": int(universe["calibration_prompts"]),
        "global_component_count": int(universe["component_count"]),
        "global_component_counts_by_type": universe["component_counts_by_type"],
        "global_universe_gates": universe["gates"],
        "strict16_prompts": 16,
        "mcc_selected_prompts": len(selection["selected_ids"]),
        "selected_ids": selection["selected_ids"],
        "selected_category_counts": dict(sorted(Counter(selection["selected_categories"]).items())),
        "v1_v2_selected_jaccard": len(v1_ids & v2_ids) / len(v1_ids | v2_ids),
        "coverage_metrics": selection["coverage_metrics"],
        "per_type_coverage": selection["per_type_coverage"],
        "type_macro_final_global_coverage": selection["type_macro_final_global_coverage"],
        "candidate_relative_coverage_v1_definition": selection["candidate_relative_coverage_v1_definition"],
        "reference_task_pass_rate": selection["reference_task_pass_rate"],
        "activation_gate_passed": bool(activation["passed"]),
        "primary_selection": "global_unweighted_mcc",
        "primary_method": str(config["statistics"]["method"]),
        "pooled_mmd_role": "ablation",
        "metrics": metrics,
        "endpoint_results": compact_results,
        "total_queries": sum(int(row["queries"]) for row in results),
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in results),
        "config_sha256": file_sha256(config_path),
        "fingerprint_sha256": file_sha256(fingerprint_path),
        "global_universe_sha256": str(universe["components_sha256"]),
        "activation_profiles_sha256": {
            split: file_sha256(output_dir / "activations" / f"{split}_profiles.jsonl")
            for split in ("build", "audit", "candidate")
        },
    }
    release_dir.mkdir(parents=True, exist_ok=True)
    selection_output = release_dir / "mcc_selection.json"
    activation_output = release_dir / "activation_audit.json"
    universe_output = release_dir / "global_component_universe.json"
    verification_output = release_dir / "verification_results.json"
    fingerprint_output = release_dir / "fingerprint_global_mcc12.json"
    write_json(selection_output, selection)
    write_json(activation_output, activation)
    write_json(universe_output, universe)
    write_json(verification_output, compact_results)
    compact_fingerprint(fingerprint_path, fingerprint_output)
    published_paths = {
        "protocol.json": release_dir / "protocol.json",
        "strict16_manifest.jsonl": release_dir / "strict16_manifest.jsonl",
        "calibration_build_manifest.jsonl": release_dir / "calibration_build_manifest.jsonl",
        "calibration_audit_manifest.jsonl": release_dir / "calibration_audit_manifest.jsonl",
        "attack_manifest.jsonl": release_dir / "attack_manifest.jsonl",
        "activation_audit.json": activation_output,
        "global_component_universe.json": universe_output,
        "mcc_selection.json": selection_output,
        "fingerprint_global_mcc12.json": fingerprint_output,
        "verification_results.json": verification_output,
    }
    report["published_artifact_sha256"] = {
        name: file_sha256(path)
        for name, path in published_paths.items()
    }
    write_json(release_dir / "FINAL_REPORT.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "global_component_count": report["global_component_count"],
                "coverage_metrics": report["coverage_metrics"],
                "selected_ids": report["selected_ids"],
                "metrics": metrics,
                "total_queries": report["total_queries"],
                "total_elapsed_seconds": report["total_elapsed_seconds"],
                "output": str(release_dir / "FINAL_REPORT.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
