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
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compact_realization(value: dict) -> dict:
    result = deepcopy(value)
    details = result.get("details")
    if isinstance(details, dict):
        adapter_path = details.get("adapter_path")
        if adapter_path:
            details["adapter_path"] = (
                "results/fingerprint_v1_20260730/finetuning_adapters/"
                + Path(str(adapter_path)).name
            )
        attack_details = details.get("details")
        if isinstance(attack_details, dict):
            selected = attack_details.pop("selected_structures", None)
            if isinstance(selected, dict):
                attack_details["selected_structure_counts"] = {
                    str(layer): len(indices) for layer, indices in selected.items()
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
        "Counts and checksums are published per entry; the complete runtime "
        "profiles remain in ignored results/. Components are not needed for "
        "online verification after MCC selection is frozen."
    )
    write_json(target, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish compact fingerprint V1 evidence")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v1_qwen_7b.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    reproducibility_dir = project_path(config["reproducibility_dir"])
    manifests = read_jsonl(project_path(config["data"]["attack_manifest"]))
    results = []
    compact_results = []
    for manifest in manifests:
        variant_id = str(manifest["variant_id"])
        path = output_dir / "verification" / f"mcc_v1__{variant_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing endpoint result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(payload)
        compact = {key: value for key, value in payload.items() if key != "response_records"}
        compact["variant_realization"] = compact_realization(
            dict(compact["variant_realization"])
        )
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
    activation_audit_path = output_dir / "activation_audit.json"
    fingerprint_path = output_dir / "fingerprints" / "mcc_v1.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    activation_audit = json.loads(activation_audit_path.read_text(encoding="utf-8"))
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        git_commit = None
    report = {
        "schema_version": "fingerprint_v1_final_report_1.0",
        "experiment_version": "fingerprint_v1_20260730",
        "status": "complete",
        "scope": "Existing V6 strict16 -> activation coverage -> MCC12 -> reference fingerprint -> endpoint verification",
        "scientific_boundary": "This is a closed first version with one endpoint variant per configured state, not a high-powered estimate of per-family detection rates.",
        "git_commit": git_commit,
        "strict16_prompts": 16,
        "mcc_selected_prompts": selection["selected_prompts"],
        "selected_category_counts": dict(sorted(Counter(selection["selected_categories"]).items())),
        "activation_gate_passed": bool(activation_audit["passed"]),
        "reference_task_pass_rate": selection["reference_task_pass_rate"],
        "primary_method": str(config["statistics"]["method"]),
        "pooled_mmd_role": "ablation",
        "metrics": metrics,
        "endpoint_results": compact_results,
        "total_queries": sum(int(row["queries"]) for row in results),
        "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in results),
        "config_sha256": file_sha256(Path(args.config)),
        "fingerprint_sha256": file_sha256(fingerprint_path),
        "activation_profiles_sha256": file_sha256(output_dir / "activation_profiles.jsonl"),
    }
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    write_json(reproducibility_dir / "FINAL_REPORT.json", report)
    write_json(reproducibility_dir / "mcc_selection.json", selection)
    write_json(reproducibility_dir / "activation_audit.json", activation_audit)
    write_json(reproducibility_dir / "verification_results.json", compact_results)
    compact_fingerprint(
        fingerprint_path,
        reproducibility_dir / "fingerprint_mcc12.json",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "metrics": metrics,
                "total_queries": report["total_queries"],
                "total_elapsed_seconds": report["total_elapsed_seconds"],
                "output": str(reproducibility_dir / "FINAL_REPORT.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
