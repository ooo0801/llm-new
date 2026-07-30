from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl


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


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the complete, publishable fingerprint V1 evidence chain"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/fingerprint_v1_qwen_7b.yaml"),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_path(config_path)
    config = load_config(config_path)
    output_dir = project_path(config["output_dir"])
    release_dir = project_path(config["reproducibility_dir"])

    protocol = read_json(release_dir / "protocol.json")
    strict16 = read_jsonl(release_dir / "strict16_manifest.jsonl")
    attacks = read_jsonl(release_dir / "attack_manifest.jsonl")
    activation = read_json(release_dir / "activation_audit.json")
    selection = read_json(release_dir / "mcc_selection.json")
    fingerprint = read_json(release_dir / "fingerprint_mcc12.json")
    verification = read_json(release_dir / "verification_results.json")
    report = read_json(release_dir / "FINAL_REPORT.json")

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    strict_ids = [str(row["prompt_id"]) for row in strict16]
    attack_ids = [str(row["variant_id"]) for row in attacks]
    selected_ids = [str(value) for value in selection["selected_ids"]]
    fingerprint_ids = [str(row["prompt_id"]) for row in fingerprint["entries"]]
    result_ids = [str(row["variant_id"]) for row in verification]

    check(len(strict16) == 16, "strict16 manifest must contain exactly 16 prompts")
    check(len(set(strict_ids)) == 16, "strict16 prompt IDs must be unique")
    check(protocol["strict16_rows"] == 16, "protocol strict16 count is inconsistent")
    check(protocol["optimized_prompt_enforced"] is True, "optimized prompts are not enforced")
    check(canonical_sha256(strict16) == protocol["strict16_sha256"], "strict16 content hash differs from the frozen protocol")
    check(canonical_sha256(attacks) == protocol["attack_manifest_sha256"], "attack manifest content hash differs from the frozen protocol")
    check(file_sha256(config_path) == protocol["config_sha256"], "config hash differs from the frozen protocol")

    check(len(attacks) == 7, "attack manifest must contain exactly seven endpoint states")
    check(len(set(attack_ids)) == 7, "endpoint variant IDs must be unique")
    check(attack_ids == list(protocol["attack_variants"]), "protocol and attack manifest variant order differ")

    check(activation["passed"] is True, "activation reproducibility gate did not pass")
    check(activation["prompts"] == 16, "activation audit must cover 16 prompts")
    check(activation["repetitions"] == 2, "activation audit must use two repetitions")
    check(activation["rows"] == 32, "activation audit must contain 32 profiles")
    check(math.isclose(float(activation["minimum_jaccard"]), 1.0), "minimum activation Jaccard must equal 1.0")
    check(activation["logit_invariance"]["allclose"] is True, "activation hooks changed model logits")
    check(math.isclose(float(activation["logit_invariance"]["max_abs_difference"]), 0.0), "activation hook logit difference is non-zero")

    selected_size = int(config["fingerprint"]["selected_size"])
    reference_repetitions = int(config["fingerprint"]["reference_repetitions"])
    target_repetitions = int(config["statistics"]["repetitions"])
    permutations = int(config["statistics"]["permutations"])
    minimum_p = 1.0 / (permutations + 1)

    check(selection["passed"] is True, "MCC selection gate did not pass")
    check(selection["candidate_prompts"] == 16, "MCC candidate count must equal 16")
    check(selection["selected_prompts"] == selected_size, "MCC selected count differs from config")
    check(len(selected_ids) == selected_size, "MCC selected ID count differs from config")
    check(len(set(selected_ids)) == selected_size, "MCC selected IDs must be unique")
    check(set(selected_ids).issubset(strict_ids), "MCC selected an ID outside strict16")
    check(selection["stable_components_selected"] <= selection["stable_components_all"], "MCC covered component count exceeds the universe")
    coverage12 = selection["coverage_curves"][str(selected_size)]
    check(coverage12["selected_ids"] == selected_ids, "MCC12 trace and selected IDs differ")
    check(float(coverage12["coverage_rate"]) >= 0.95, "MCC12 stable-component coverage is below 95%")
    check(selection["reference_responses"] == selected_size * reference_repetitions, "reference response count is inconsistent")

    check(len(fingerprint["entries"]) == selected_size, "published fingerprint must contain 12 entries")
    check(fingerprint_ids == selected_ids, "fingerprint prompt order differs from frozen MCC selection")
    for entry in fingerprint["entries"]:
        prompt_id = str(entry["prompt_id"])
        check("components" not in entry, f"{prompt_id}: compact fingerprint unexpectedly contains full activation components")
        check(len(entry["reference_responses"]) == reference_repetitions, f"{prompt_id}: reference response count differs from config")
        check(len(entry["reference_response_seeds"]) == reference_repetitions, f"{prompt_id}: reference seed count differs from config")
        metadata = entry["metadata"]
        check(int(metadata["activation_component_count"]) > 0, f"{prompt_id}: activation component count is empty")
        check(len(str(metadata["activation_components_sha256"])) == 64, f"{prompt_id}: activation component checksum is malformed")

    check(isinstance(verification, list) and len(verification) == len(attacks), "verification results must contain seven endpoints")
    check(result_ids == attack_ids, "verification result order differs from the frozen attack manifest")
    expected_queries = selected_size * target_repetitions
    for row, manifest in zip(verification, attacks):
        variant_id = str(row["variant_id"])
        primary = row["primary_test"]
        diagnostics = primary["diagnostics"]
        expected_modified = manifest["family"] != "intact"
        check(row["family"] == manifest["family"], f"{variant_id}: family differs from manifest")
        check(row["ground_truth_modified"] is expected_modified, f"{variant_id}: ground truth is inconsistent")
        check(row["predicted_modified"] is primary["reject"], f"{variant_id}: prediction differs from primary test")
        check(row["correct"] is (row["predicted_modified"] == expected_modified), f"{variant_id}: correctness flag is inconsistent")
        check(primary["method"] == "prompt_stratified_mmd", f"{variant_id}: primary method is not prompt-stratified MMD")
        check(int(diagnostics["permutations"]) == permutations, f"{variant_id}: permutation count differs from config")
        corrected_p = (int(diagnostics["exceedances"]) + 1) / (permutations + 1)
        check(math.isclose(float(primary["p_value"]), corrected_p), f"{variant_id}: p-value does not use the finite-permutation correction")
        check(float(primary["p_value"]) >= minimum_p, f"{variant_id}: p-value is below the attainable minimum")
        check(int(diagnostics["strata"]) == selected_size, f"{variant_id}: stratum count differs from MCC size")
        check(row["prompts"] == selected_size, f"{variant_id}: prompt count differs from MCC size")
        check(row["repetitions"] == target_repetitions, f"{variant_id}: target repetition count differs from config")
        check(row["queries"] == expected_queries, f"{variant_id}: query count is inconsistent")
        check(len(row["target_seeds"]) == target_repetitions, f"{variant_id}: target seed count is inconsistent")

    true_positive = sum(row["ground_truth_modified"] and row["predicted_modified"] for row in verification)
    false_negative = sum(row["ground_truth_modified"] and not row["predicted_modified"] for row in verification)
    false_positive = sum(not row["ground_truth_modified"] and row["predicted_modified"] for row in verification)
    true_negative = sum(not row["ground_truth_modified"] and not row["predicted_modified"] for row in verification)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    expected_metrics = {
        "accuracy": (true_positive + true_negative) / len(verification),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": false_positive / max(false_positive + true_negative, 1),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "variants": len(verification),
    }
    for name, expected in expected_metrics.items():
        observed = report["metrics"][name]
        if isinstance(expected, float):
            check(math.isclose(float(observed), expected), f"final metric {name} is inconsistent")
        else:
            check(observed == expected, f"final metric {name} is inconsistent")

    check(report["status"] == "complete", "final report is not marked complete")
    check(report["strict16_prompts"] == 16, "final report strict16 count is inconsistent")
    check(report["mcc_selected_prompts"] == selected_size, "final report MCC count is inconsistent")
    check(report["activation_gate_passed"] is True, "final report activation gate is false")
    check(report["primary_method"] == "prompt_stratified_mmd", "final report primary method is incorrect")
    check(report["pooled_mmd_role"] == "ablation", "pooled MMD must remain an ablation")
    check(report["total_queries"] == len(attacks) * expected_queries, "final total query count is inconsistent")
    check(len(report["endpoint_results"]) == len(attacks), "final endpoint count is inconsistent")
    check(file_sha256(config_path) == report["config_sha256"], "final report config hash is stale")

    for name, expected_hash in report["published_artifact_sha256"].items():
        check(file_sha256(release_dir / name) == expected_hash, f"published checksum mismatch: {name}")

    runtime_fingerprint = output_dir / "fingerprints" / "mcc_v1.json"
    runtime_activations = output_dir / "activation_profiles.jsonl"
    if runtime_fingerprint.exists():
        check(file_sha256(runtime_fingerprint) == report["fingerprint_sha256"], "runtime fingerprint hash differs from final report")
    if runtime_activations.exists():
        check(file_sha256(runtime_activations) == report["activation_profiles_sha256"], "runtime activation hash differs from final report")

    forbidden_fragments = (
        "/root/",
        "C:\\Users\\",
        "BEGIN OPENSSH PRIVATE KEY",
        "BEGIN PRIVATE KEY",
        "id_ed25519",
        "autodl_codex_ed25519",
    )
    for path in release_dir.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            check(fragment not in text, f"published file contains forbidden private/absolute-path fragment: {path.name}")

    if failures:
        print(json.dumps({"status": "failed", "failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "status": "passed",
                "strict16_prompts": len(strict16),
                "mcc_selected_prompts": len(selected_ids),
                "endpoint_states": len(verification),
                "total_queries": report["total_queries"],
                "published_checksums": len(report["published_artifact_sha256"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
