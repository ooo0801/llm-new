from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_json


MODIFIED_FAMILIES = ["unstructured_pruning", "structured_pruning", "gaussian_noise", "quantization", "finetuning"]


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_verification(row: dict[str, Any], attack: dict[str, Any], config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    variant_id = str(attack["variant_id"])
    family = str(attack["family"])
    selected_size = int(config["fingerprint"]["selected_size"])
    repetitions = int(config["statistics"]["repetitions"])
    expected_queries = selected_size * repetitions
    seed_base = int(config["statistics"]["target_seed_base"])
    expected_seeds = [seed_base + index for index in range(repetitions)]
    permutations = int(config["statistics"]["permutations"])
    alpha = float(config["statistics"]["alpha"])

    if str(row.get("variant_id")) != variant_id or str(row.get("family")) != family:
        errors.append("identity mismatch")
    if int(row.get("prompts", -1)) != selected_size:
        errors.append("prompt count mismatch")
    if int(row.get("repetitions", -1)) != repetitions:
        errors.append("repetition count mismatch")
    if int(row.get("queries", -1)) != expected_queries:
        errors.append("query mismatch")
    if row.get("target_seeds") != expected_seeds:
        errors.append("target seed mismatch")

    response_records = row.get("response_records")
    if not isinstance(response_records, list) or len(response_records) != expected_queries:
        errors.append("response record count mismatch")
    else:
        record_seeds = sorted({int(item.get("seed", -1)) for item in response_records})
        prompt_ids = {str(item.get("prompt_id")) for item in response_records}
        keys = {(str(item.get("prompt_id")), int(item.get("repetition", -1))) for item in response_records}
        if record_seeds != expected_seeds:
            errors.append("response record seed mismatch")
        if len(prompt_ids) != selected_size or len(keys) != expected_queries:
            errors.append("response prompt/repetition structure mismatch")

    primary = row.get("primary_test")
    if not isinstance(primary, dict):
        errors.append("missing primary test")
    else:
        expected_method = str(config["statistics"].get("method", "prompt_stratified_mmd"))
        if primary.get("method") != expected_method:
            errors.append("primary method mismatch")
        if not finite_number(primary.get("statistic")):
            errors.append("nonfinite primary statistic")
        if not finite_number(primary.get("effect_size")):
            errors.append("nonfinite primary effect size")
        p_value = primary.get("p_value")
        if not finite_number(p_value) or not 0.0 <= float(p_value) <= 1.0:
            errors.append("invalid primary p-value")
        if not finite_number(primary.get("alpha")) or float(primary["alpha"]) != alpha:
            errors.append("primary alpha mismatch")
        if bool(row.get("predicted_modified")) != bool(primary.get("reject")):
            errors.append("prediction/rejection mismatch")
        diagnostics = primary.get("diagnostics", {})
        if expected_method == "prompt_stratified_mmd":
            if int(diagnostics.get("permutations", -1)) != permutations:
                errors.append("permutation mismatch")
            if int(diagnostics.get("strata", -1)) != selected_size:
                errors.append("strata count mismatch")
            sample_counts = diagnostics.get("samples_per_group", {})
            if len(sample_counts) != selected_size or any(int(value) != repetitions for value in sample_counts.values()):
                errors.append("stratum sample count mismatch")
            bandwidths = diagnostics.get("bandwidths", {})
            if len(bandwidths) != selected_size or any(not finite_number(value) or float(value) <= 0.0 for value in bandwidths.values()):
                errors.append("invalid prompt bandwidth")
        elif expected_method == "paired_block_sign_flip":
            if int(diagnostics.get("blocks", -1)) != repetitions:
                errors.append("paired block count mismatch")
            if int(diagnostics.get("samples_per_block", -1)) != selected_size:
                errors.append("paired block size mismatch")
            if diagnostics.get("exchangeable_unit") != "generation_seed_block":
                errors.append("paired exchangeable unit mismatch")
            expected_patterns = 1 << repetitions
            if bool(config["statistics"].get("exact_sign_flips", True)) and int(diagnostics.get("evaluated_sign_patterns", -1)) != expected_patterns:
                errors.append("exact sign-pattern count mismatch")

    pooled = row.get("secondary_tests", {}).get("pooled_mmd")
    if not isinstance(pooled, dict) or pooled.get("method") != "pooled_mmd":
        errors.append("missing pooled MMD ablation")
    elif not finite_number(pooled.get("statistic")) or not finite_number(pooled.get("p_value")):
        errors.append("nonfinite pooled MMD ablation")

    realization = row.get("variant_realization")
    if not isinstance(realization, dict):
        errors.append("missing variant realization")
    else:
        if str(realization.get("variant_id")) != variant_id or str(realization.get("family")) != family:
            errors.append("variant realization identity mismatch")
        if realization.get("isolated_base_reload") is not True:
            errors.append("variant was not independently reloaded")
    expected_modified = family != "intact"
    if bool(row.get("ground_truth_modified")) != expected_modified:
        errors.append("ground-truth label mismatch")
    if bool(row.get("correct")) != (bool(row.get("predicted_modified")) == expected_modified):
        errors.append("correctness field mismatch")
    return errors


def proportion_interval(successes: int, total: int) -> dict[str, float | int]:
    if total <= 0:
        return {"successes": successes, "total": total, "estimate": 0.0, "low": 0.0, "high": 1.0}
    try:
        from scipy.stats import binomtest

        interval = binomtest(successes, total).proportion_ci(confidence_level=0.95, method="exact")
        low, high = float(interval.low), float(interval.high)
    except Exception:
        low, high = 0.0, 1.0
    return {"successes": successes, "total": total, "estimate": successes / total, "low": low, "high": high}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_g_qwen14b_mcc.yaml"))
    parser.add_argument("--required-modified", type=int, default=9)
    args = parser.parse_args()
    config = load_config(args.config)
    detection_label = str(
        config.get("protocol_labels", {}).get("detection", "H1")
    )
    output = project_path(config["output_dir"])
    prefix = str(config["fingerprint"].get("verification_prefix", "global_mcc_14b"))
    attacks = read_jsonl(project_path(config["data"]["attack_manifest"]))
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for attack in attacks:
        variant_id = str(attack["variant_id"])
        path = output / "verification" / f"{prefix}__{variant_id}.json"
        if not path.exists():
            errors.append(f"missing verification: {variant_id}")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        errors.extend(f"{variant_id}: {message}" for message in validate_verification(row, attack, config))
        rows.append(row)

    intact = [row for row in rows if row["family"] == "intact"]
    modified = [row for row in rows if row["family"] != "intact"]
    detected = [row for row in modified if bool(row["predicted_modified"])]
    family_totals = Counter(str(row["family"]) for row in modified)
    family_detected = Counter(str(row["family"]) for row in detected)
    family_report = {
        family: {
            "detected": family_detected[family],
            "total": family_totals[family],
            "rate": family_detected[family] / family_totals[family] if family_totals[family] else 0.0,
            "variants": [
                {"variant_id": row["variant_id"], "predicted_modified": row["predicted_modified"], "p_value": row["primary_test"]["p_value"], "statistic": row["primary_test"]["statistic"]}
                for row in modified if row["family"] == family
            ],
        }
        for family in MODIFIED_FAMILIES
    }
    expected_intact = int(config["statistics"].get("expected_intact_states", 1))
    expected_modified = int(config["statistics"].get("expected_modified_states", 11))
    technical_passed = bool(
        not errors
        and len(rows) == len(attacks) == expected_intact + expected_modified
        and len(intact) == expected_intact
        and len(modified) == expected_modified
    )
    intact_correct = bool(len(intact) == expected_intact and all(not row["predicted_modified"] for row in intact))
    family_coverage = bool(all(family_detected[family] >= 1 for family in MODIFIED_FAMILIES))
    hypothesis_supported = bool(technical_passed and intact_correct and len(detected) >= args.required_modified and family_coverage)
    accuracy = sum(bool(row["correct"]) for row in rows)
    report = {
        "schema_version": f"fingerprint_14b_{detection_label.lower()}_summary_1.0",
        "states": len(rows),
        "technical_passed": technical_passed,
        "technical_errors": errors,
        "intact_correct": intact_correct,
        "modified_detected": len(detected),
        "modified_total": len(modified),
        "required_modified_detected": args.required_modified,
        "modified_recall_exact_95ci": proportion_interval(len(detected), len(modified)),
        "correct_states": accuracy,
        "accuracy_exact_95ci": proportion_interval(accuracy, len(rows)),
        "family_coverage_passed": family_coverage,
        "by_family": family_report,
        "hypothesis": f"H-{detection_label}",
        "hypothesis_supported": hypothesis_supported,
        "claim_boundary": "Fixed registered model-state instances; not a population detection-rate estimate.",
    }
    write_json(output / "final_mmd_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if hypothesis_supported else 1)


if __name__ == "__main__":
    main()
