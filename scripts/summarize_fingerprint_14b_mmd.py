from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_json


MODIFIED_FAMILIES = ["unstructured_pruning", "structured_pruning", "gaussian_noise", "quantization", "finetuning"]


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
        if str(row.get("variant_id")) != variant_id or str(row.get("family")) != str(attack["family"]):
            errors.append(f"identity mismatch: {variant_id}")
        expected_queries = int(config["fingerprint"]["selected_size"]) * int(config["statistics"]["repetitions"])
        if int(row.get("queries", -1)) != expected_queries:
            errors.append(f"query mismatch: {variant_id}")
        primary = row.get("primary_test", {})
        if int(primary.get("diagnostics", {}).get("permutations", -1)) != int(config["statistics"]["permutations"]):
            errors.append(f"permutation mismatch: {variant_id}")
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
    technical_passed = bool(not errors and len(rows) == len(attacks) == 12 and len(intact) == 1 and len(modified) == 11)
    intact_correct = bool(len(intact) == 1 and not intact[0]["predicted_modified"])
    family_coverage = bool(all(family_detected[family] >= 1 for family in MODIFIED_FAMILIES))
    hypothesis_supported = bool(technical_passed and intact_correct and len(detected) >= args.required_modified and family_coverage)
    accuracy = sum(bool(row["correct"]) for row in rows)
    report = {
        "schema_version": "fingerprint_14b_h1_summary_1.0",
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
        "hypothesis": "H-H1",
        "hypothesis_supported": hypothesis_supported,
        "claim_boundary": "Fixed registered model-state instances; not a population detection-rate estimate.",
    }
    write_json(output / "final_mmd_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if hypothesis_supported else 1)


if __name__ == "__main__":
    main()
