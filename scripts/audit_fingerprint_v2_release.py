from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.global_coverage import (
    component_counts,
    coverage_metrics,
    stable_prompt_components,
)
from llm_integrity.io import read_jsonl


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete global-calibrated fingerprint V2 evidence chain")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_path(config_path)
    config = load_config(config_path)
    output_dir = project_path(config["output_dir"])
    release_dir = project_path(config["reproducibility_dir"])

    protocol = read_json(release_dir / "protocol.json")
    expansion = read_json(release_dir / "calibration_expansion_record.json")
    strict16 = read_jsonl(release_dir / "strict16_manifest.jsonl")
    calibration_build = read_jsonl(release_dir / "calibration_build_manifest.jsonl")
    calibration_audit = read_jsonl(release_dir / "calibration_audit_manifest.jsonl")
    attacks = read_jsonl(release_dir / "attack_manifest.jsonl")
    activation = read_json(release_dir / "activation_audit.json")
    universe = read_json(release_dir / "global_component_universe.json")
    selection = read_json(release_dir / "mcc_selection.json")
    fingerprint = read_json(release_dir / "fingerprint_global_mcc12.json")
    verification = read_json(release_dir / "verification_results.json")
    report = read_json(release_dir / "FINAL_REPORT.json")

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    strict_ids = [str(row["prompt_id"]) for row in strict16]
    build_ids = [str(row["id"]) for row in calibration_build]
    audit_ids = [str(row["id"]) for row in calibration_audit]
    selected_ids = [str(value) for value in selection["selected_ids"]]
    attack_ids = [str(row["variant_id"]) for row in attacks]
    result_ids = [str(row["variant_id"]) for row in verification]

    expected_families = int(config["calibration"]["task_families"])
    build_per_family = int(config["calibration"]["build_per_family"])
    audit_per_family = int(config["calibration"]["audit_per_family"])
    expected_build = expected_families * build_per_family
    expected_audit = expected_families * audit_per_family
    expected_calibration = expected_build + expected_audit
    expected_batches = build_per_family + audit_per_family

    check(protocol["schema_version"] == "fingerprint_v2_global_protocol_1.1", "unexpected V2 protocol schema")
    check(expansion["endpoint_experiments_started_before_decision"] is False, "calibration was expanded after an endpoint experiment")
    check(expansion["initial_observations"]["passed"] is False, "expansion record does not preserve the initial frozen-gate failure")
    check(expansion["revision"]["build_per_family"] == build_per_family, "expansion build count differs from config")
    check(expansion["revision"]["audit_per_family"] == audit_per_family, "expansion audit count differs from config")
    check(expansion["revision"]["calibration_prompts"] == expected_calibration, "expansion total differs from config")
    check(expansion["frozen_gates"]["maximum_last_batch_relative_gain"] == float(config["calibration"]["max_last_batch_relative_gain"]), "saturation threshold changed during expansion")
    check(expansion["frozen_gates"]["maximum_audit_novelty_rate"] == float(config["calibration"]["max_audit_novelty_rate"]), "audit-novelty threshold changed during expansion")
    check(protocol["v1_parent_tag"] == "v1-fingerprint-closed-loop", "V1 parent tag is not frozen")
    check(len(strict16) == 16 and len(set(strict_ids)) == 16, "strict16 is not a unique set of 16")
    check(len(calibration_build) == expected_build and len(set(build_ids)) == expected_build, "calibration build split count differs from config")
    check(len(calibration_audit) == expected_audit and len(set(audit_ids)) == expected_audit, "calibration audit split count differs from config")
    check(not (set(build_ids) & set(audit_ids)), "calibration build and audit IDs overlap")
    check(not (set(build_ids + audit_ids) & set(strict_ids)), "calibration and strict16 IDs overlap")
    calibration_texts = [str(row["prompt"]) for row in calibration_build + calibration_audit]
    check(len(set(calibration_texts)) == expected_calibration, "calibration prompt texts are not unique")
    check(len(set(row["category"] for row in calibration_build)) == expected_families, "build split does not cover every task family")
    check(len(set(row["category"] for row in calibration_audit)) == expected_families, "audit split does not cover every task family")
    check(set(Counter(row["category"] for row in calibration_build).values()) == {build_per_family}, "build task-family allocation is not balanced")
    check(set(Counter(row["category"] for row in calibration_audit).values()) == {audit_per_family}, "audit task-family allocation is not balanced")
    check(float(protocol["closest_strict16_trigram_jaccard"]["score"]) < float(protocol["near_duplicate_threshold"]), "calibration near-duplicate gate failed")
    check(canonical_sha256(strict16) == protocol["strict16_sha256"], "strict16 canonical hash is stale")
    check(canonical_sha256(calibration_build) == protocol["calibration_build_sha256"], "calibration build canonical hash is stale")
    check(canonical_sha256(calibration_audit) == protocol["calibration_audit_sha256"], "calibration audit canonical hash is stale")
    check(canonical_sha256(attacks) == protocol["attack_manifest_sha256"], "attack canonical hash is stale")
    check(file_sha256(config_path) == protocol["config_sha256"], "V2 config hash differs from the protocol")

    repetitions = int(config["activations"]["repetitions"])
    minimum_jaccard = float(config["activations"]["minimum_jaccard"])
    check(activation["passed"] is True, "V2 activation gate did not pass")
    check(activation["logit_invariance"]["allclose"] is True, "V2 profiler changed model logits")
    check(activation["logit_invariance"]["same_rendered_system_prompt"] is True, "logit invariance did not use the profiling system prompt")
    for split, prompts in (("build", expected_build), ("audit", expected_audit), ("candidate", 16)):
        row = activation["splits"][split]
        check(row["prompts"] == prompts, f"{split}: activation prompt count is inconsistent")
        check(row["repetitions"] == repetitions, f"{split}: activation repetitions differ from config")
        check(row["rows"] == prompts * repetitions, f"{split}: activation row count is inconsistent")
        check(float(row["minimum_jaccard"]) >= minimum_jaccard, f"{split}: activation Jaccard gate failed")
        check(row["passed"] is True, f"{split}: activation split is not marked passed")

    components = [str(value) for value in universe["components"]]
    check(universe["component_id_schema"] == "activation_observable_v1", "global universe component schema changed")
    check(universe["calibration_prompts"] == expected_calibration, "global universe must use every frozen calibration prompt")
    check(len(components) == universe["component_count"], "global universe component count is inconsistent")
    check(components == sorted(set(components)), "global universe components are not sorted and unique")
    check(component_counts(components) == universe["component_counts_by_type"], "global universe per-type counts are inconsistent")
    check(canonical_sha256(components) == universe["components_sha256"], "global universe canonical hash is stale")
    gates = universe["gates"]
    check(gates["saturation_passed"] is True, "global universe saturation gate failed")
    check(gates["audit_novelty_passed"] is True, "global universe audit novelty gate failed")
    check(gates["passed"] is True, "global universe is not publishable")
    curve = universe["cumulative_coverage_curve"]
    check(len(curve) == expected_batches, "global universe curve length differs from the frozen balanced batches")
    curve_counts = [int(row["component_count"]) for row in curve]
    check(curve_counts == sorted(curve_counts), "global universe cumulative counts are not monotonic")
    check(curve_counts[-1] == len(components), "global universe curve does not end at the published count")
    check(all(0.0 <= float(row["relative_gain"]) <= 1.0 for row in curve), "global universe curve contains an invalid relative gain")

    selected_size = int(config["fingerprint"]["selected_size"])
    check(selection["passed"] is True, "V2 MCC selection gate did not pass")
    check(selection["primary_objective"] == "global_unweighted_mcc", "V2 primary selection objective is incorrect")
    check(len(selected_ids) == selected_size and len(set(selected_ids)) == selected_size, "V2 selected IDs are not a unique MCC12")
    check(set(selected_ids).issubset(strict_ids), "V2 selected a prompt outside strict16")
    check(selection["selection_deterministic"] is True, "V2 MCC selection is not deterministic")
    check(selection["factorization_passed"] is True, "three-coverage factorization gate failed")
    coverage = selection["coverage_metrics"]
    check(0.0 <= float(coverage["candidate_reachable_coverage"]) <= 1.0, "candidate reachable coverage is invalid")
    check(0.0 <= float(coverage["selection_efficiency"]) <= 1.0, "selection efficiency is invalid")
    check(0.0 <= float(coverage["final_global_coverage"]) <= 1.0, "final global coverage is invalid")
    check(math.isclose(float(coverage["final_global_coverage"]), float(coverage["candidate_reachable_coverage"]) * float(coverage["selection_efficiency"]), abs_tol=1e-12), "three coverage values do not factorize")
    check(coverage["selected_global_components"] <= coverage["candidate_global_components"] <= coverage["global_components"], "global coverage component counts violate set inclusion")

    runtime_candidate = output_dir / "activations" / "candidate_profiles.jsonl"
    if runtime_candidate.exists():
        candidate_components = stable_prompt_components(
            read_jsonl(runtime_candidate),
            repetitions=repetitions,
            stable_frequency=float(config["activations"]["stable_component_frequency"]),
        )
        recalculated = coverage_metrics(candidate_components, selected_ids, components).as_dict()
        for key, expected in recalculated.items():
            observed = coverage[key]
            if isinstance(expected, float):
                check(math.isclose(float(observed), expected, abs_tol=1e-12), f"coverage metric {key} differs from runtime profiles")
            else:
                check(observed == expected, f"coverage metric {key} differs from runtime profiles")

    reference_repetitions = int(config["fingerprint"]["reference_repetitions"])
    fingerprint_ids = [str(row["prompt_id"]) for row in fingerprint["entries"]]
    check(fingerprint["schema_version"] == "2.0", "V2 fingerprint schema is incorrect")
    check(fingerprint["selection_method"] == "global_unweighted_mcc", "V2 fingerprint selection method is incorrect")
    check(fingerprint_ids == selected_ids, "V2 fingerprint prompt order differs from MCC selection")
    for entry in fingerprint["entries"]:
        prompt_id = str(entry["prompt_id"])
        check("components" not in entry, f"{prompt_id}: compact fingerprint contains full components")
        check(len(entry["reference_responses"]) == reference_repetitions, f"{prompt_id}: reference response count differs from config")
        check(len(entry["reference_response_seeds"]) == reference_repetitions, f"{prompt_id}: reference seed count differs from config")
        check(int(entry["metadata"]["activation_component_count"]) > 0, f"{prompt_id}: activation component count is empty")

    target_repetitions = int(config["statistics"]["repetitions"])
    permutations = int(config["statistics"]["permutations"])
    expected_queries = selected_size * target_repetitions
    check(len(attacks) == 7 and len(set(attack_ids)) == 7, "V2 attack manifest must contain seven unique endpoints")
    check(result_ids == attack_ids, "V2 verification order differs from the attack manifest")
    for row, manifest in zip(verification, attacks):
        variant_id = str(row["variant_id"])
        primary = row["primary_test"]
        diagnostics = primary["diagnostics"]
        expected_modified = manifest["family"] != "intact"
        check(row["family"] == manifest["family"], f"{variant_id}: family differs from manifest")
        check(row["ground_truth_modified"] is expected_modified, f"{variant_id}: ground truth is inconsistent")
        check(row["predicted_modified"] is primary["reject"], f"{variant_id}: prediction differs from primary test")
        check(row["correct"] is (row["predicted_modified"] == expected_modified), f"{variant_id}: correctness flag is inconsistent")
        check(primary["method"] == "prompt_stratified_mmd", f"{variant_id}: primary method is incorrect")
        check(int(diagnostics["permutations"]) == permutations, f"{variant_id}: permutation count differs from config")
        corrected_p = (int(diagnostics["exceedances"]) + 1) / (permutations + 1)
        check(math.isclose(float(primary["p_value"]), corrected_p), f"{variant_id}: finite-permutation p correction is invalid")
        check(int(diagnostics["strata"]) == selected_size, f"{variant_id}: stratum count differs from MCC size")
        check(row["prompts"] == selected_size and row["repetitions"] == target_repetitions, f"{variant_id}: query design differs from config")
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

    check(report["status"] == "complete", "V2 final report is not complete")
    check(report["calibration_prompts"] == expected_calibration, "V2 final report calibration count is inconsistent")
    check(report["global_component_count"] == len(components), "V2 final report universe count is inconsistent")
    check(report["selected_ids"] == selected_ids, "V2 final report selected IDs are inconsistent")
    check(report["total_queries"] == len(attacks) * expected_queries, "V2 total query count is inconsistent")
    check(file_sha256(config_path) == report["config_sha256"], "V2 final report config hash is stale")
    check(report["global_universe_sha256"] == universe["components_sha256"], "V2 final report universe hash is stale")

    for name, expected_hash in report["published_artifact_sha256"].items():
        check(file_sha256(release_dir / name) == expected_hash, f"published checksum mismatch: {name}")
    runtime_fingerprint = output_dir / "fingerprints" / "global_mcc_v2.json"
    if runtime_fingerprint.exists():
        check(file_sha256(runtime_fingerprint) == report["fingerprint_sha256"], "runtime V2 fingerprint hash differs from final report")
    for split, expected_hash in report["activation_profiles_sha256"].items():
        runtime = output_dir / "activations" / f"{split}_profiles.jsonl"
        if runtime.exists():
            check(file_sha256(runtime) == expected_hash, f"runtime {split} activation hash differs from final report")

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
                "calibration_prompts": len(calibration_build) + len(calibration_audit),
                "global_components": len(components),
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
