from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "reproducibility"
    / "h8_qwen32b_score_calibration_20260819"
    / "d1a"
    / "H8_D1A_SCORE_CALIBRATION_PREFLIGHT_REPORT.json"
)
EXPECTED_MMD_MANIFEST_SHA256 = "df406d671dcd0867a679a095eaf52db5111590129024cb012908026a4558a063"
EXPECTED_STRUCTURES = {"r40_q10", "r40_q20", "r60_q10", "r60_q20"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    dry = report["development_dry_run"]
    diagnostics = dry["structure_diagnostics"]
    mapping_counts = {}
    all_scores_finite = True
    for structure, value in diagnostics.items():
        prompts = value["prompt_diagnostics"]
        mapping_counts[structure] = {
            "nondegenerate_centered": sum(
                item["mapping"] == "nondegenerate_centered" for item in prompts.values()
            ),
            "degenerate_global_scale": sum(
                item["mapping"] == "degenerate_global_scale" for item in prompts.values()
            ),
        }
        all_scores_finite &= all(item["score_finite"] for item in prompts.values())
    manifest_schema = report["proposed_new_data_manifest_schema"]
    gates = {
        "status_pass": report["status"] == "PASS",
        "measurement_frozen": report["measurement_layer_status"] == "frozen",
        "detector_not_frozen": report["detector_status"] == "not_frozen",
        "score_not_frozen": report["score_calibration_status"] == "preflight_only_not_frozen",
        "mmd_manifest_bound": report["mmd_frozen_manifest_sha256"] == EXPECTED_MMD_MANIFEST_SHA256,
        "four_structures": set(report["sample_structures"]) == EXPECTED_STRUCTURES == set(diagnostics),
        "six_plus_six_mappings_each": all(
            counts == {"nondegenerate_centered": 6, "degenerate_global_scale": 6}
            for counts in mapping_counts.values()
        ),
        "all_dry_scores_finite": all_scores_finite,
        "dry_run_only": dry["performed"] is True
        and dry["data_role"] == "development_dry_run"
        and dry["eligible_to_freeze"] is False,
        "dry_count_9600": dry["raw_statistic_count"] == dry["score_count"] == 9600,
        "no_model_responses": report["new_model_responses"] == 0,
        "no_forbidden_data": report["formal_reference_read_or_generated"] is False
        and report["heldout_read_or_generated"] is False
        and report["attack_read_or_generated"] is False,
        "no_sample_selection": report["sample_size_selection_performed"] is False,
        "no_top_r_selection": report["top_r_selection_performed"] is False,
        "no_final_fit": report["final_score_parameter_fit_performed"] is False,
        "all_internal_tests_pass": report["test_results"]["all_pass"] is True
        and report["test_results"]["preflight_unit_tests"]["passed"] is True,
        "proposed_manifest_rejects_forbidden_roles": set(manifest_schema["fit_api_rejects_roles"])
        == {"formal_reference", "heldout", "target", "attack"},
        "frozen_mmd_unchanged": report["test_results"]["internal_gates"]["frozen_archive_hashes_unchanged"] is True,
    }
    output = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "report_sha256": sha256(REPORT),
        "score_schema_sha256": report["score_schema_sha256"],
        "preflight_code_commit": report["preflight_code_commit"],
        "negative_raw_statistic_count": dry["negative_raw_statistic_count"],
        "mapping_counts": mapping_counts,
        "a_global_by_structure": {
            structure: value["a_global"] for structure, value in diagnostics.items()
        },
    }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
