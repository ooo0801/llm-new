"""Verify downloaded R2-small evidence without sampling or statistical retuning."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm_integrity.stage1_r1 import canonical, digest, save_json
from llm_integrity.stage1_r2 import read_chain, schedule, schedule_hash


def main():
    folder = Path(sys.argv[1]).resolve()
    load = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
    report = load("FINAL_REPORT.json")
    parent = load("PRE_DATA_MANIFEST.json")
    amendment = load("SMALL_PRE_DATA_AMENDMENT.json")
    config = copy.deepcopy(parent["config"])
    config["sampling"] = amendment["effective_sampling"]
    prompts = [json.loads(s) for s in (ROOT / config["prompts"]).read_text(encoding="utf-8").splitlines() if s.strip()]
    for name, key in (("PRE_DATA_MANIFEST.json", "pre_data_manifest_sha256"), ("FIT_FROZEN.json", "fit_sha256"),
                      ("CALIBRATION_AUDIT.json", "calibration_audit_sha256"), ("SMALL_PRE_DATA_AMENDMENT.json", "small_amendment_sha256")):
        assert digest(folder / name) == report[key], name
    assert digest(ROOT / config["prompts"]) == config["prompts_sha256"]
    counts, all_seeds, grouped = {}, [], {}
    for role in ("fit", "calibration", "validation"):
        plan = schedule(config, prompts, role)
        assert schedule_hash(plan) == amendment["plans_sha256"][role]
        rows = read_chain(folder / "responses" / (role + ".jsonl"), plan, report["pre_data_manifest_sha256"], True)
        counts[role] = len(rows)
        all_seeds.extend(r["generation_seed"] for r in rows)
        if role == "validation":
            for r in rows:
                grouped.setdefault((r["prompt_id"], r["unit"]), []).append(r)
    assert counts == {"fit": 576, "calibration": 576, "validation": 2880}
    assert len(all_seeds) == len(set(all_seeds)) == report["response_count"] == 4032
    assert len(report["decisions"]) == len(grouped) == 60
    seen = set()
    for row in report["decisions"]:
        key = row["prompt_id"], row["unit"]
        assert key not in seen
        seen.add(key)
        original = sorted(grouped[key], key=lambda r: r["response_index"])
        assert row["reference_seeds"] == [r["generation_seed"] for r in original[:24]]
        assert row["target_seeds"] == [r["generation_seed"] for r in original[24:]]
        assert row["fit_sha256"] == report["fit_sha256"]
        assert row["data_sha256"] == digest(folder / "responses/validation.jsonl")
        content = {k: v for k, v in row.items() if k != "content_sha256"}
        assert hashlib.sha256(canonical(content).encode()).hexdigest() == row["content_sha256"]
    assert seen == set(grouped)
    for channel in ("raw", "h8"):
        assert all(r[channel]["detected"] is not None for r in report["decisions"])
        assert sum(r[channel]["detected"] for r in report["decisions"]) == report["channel_counts"][channel]["rejections"]
    assert report["additional_responses_after_cost_amendment"] == 2072
    assert report["parent_data_unchanged"] is True
    assert report["formal_FPR_calibration_pass"] is None
    save_json(folder / "LOCAL_EVIDENCE_VERIFICATION.json", {
        "status": "PASS", "counts": counts, "unique_generation_seeds": len(set(all_seeds)),
        "decision_units_bound_to_disjoint_response_groups": len(seen), "all_response_chains_valid": True,
        "freeze_and_decision_hashes_valid": True,
        "report_sha256": digest(folder / "FINAL_REPORT.json"),
        "validation_sha256": digest(folder / "responses/validation.jsonl"),
        "remote_hashes_matched": digest(folder / "FINAL_REPORT.json") == "ee3da9db80bd64b9ec833d7fa1c8a3ead23c8c8dd43976d2fde96927f8ad533f" and digest(folder / "responses/validation.jsonl") == "a7fa157653adbe560e43a86d27b8b82abe40e1b35935784660ed62232ce9c216",
        "statistical_scores_recomputed_locally": False,
        "note": "Verifies input/decision provenance and reported counts, not a new feature/statistic computation.",
    })
    print("PASS:4032 responses;60 disjoint comparison units;raw2/60;H8 1/60;no extra sampling")


if __name__ == "__main__":
    main()
