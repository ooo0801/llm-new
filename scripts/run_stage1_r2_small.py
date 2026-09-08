"""User-authorized cost amendment: retain existing banks, cap validation at5 units.

Generation identity and R1 tests are unchanged. This is an exploratory smoke check,
not a pass under the abandoned60-unit engineering acceptance protocol.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

import run_stage1_r2 as runtime
from llm_integrity.stage1_r1 import canonical, digest
from llm_integrity.stage1_r2 import atomic_json, confidence_interval, freeze_json, read_chain, schedule, schedule_hash


def small_summary(rows, prompts, config):
    expected = {(p["id"], unit) for p in prompts for unit in range(5)}
    keys = [(r["prompt_id"], r["unit"]) for r in rows]
    if len(keys) != 60 or set(keys) != expected:
        raise ValueError("Small analysis requires exactly5 unique units per each of12 prompts")
    per_prompt = []
    counts = {}
    total_unknown = 0
    for channel in ("raw", "h8"):
        rejected = 0
        unknown = 0
        for p in prompts:
            values = [r[channel]["detected"] for r in rows if r["prompt_id"] == p["id"]]
            k = sum(v is True for v in values)
            missing = sum(v is None for v in values)
            rejected += k
            unknown += missing
            per_prompt.append({"prompt_id": p["id"], "channel": channel, "rejections": k,
                               "independent_units": 5, "unevaluable": missing,
                               "observed_fraction": k / 5 if not missing else None,
                               "pointwise_CP95_interval": confidence_interval(k, 5) if not missing else None})
        counts[channel] = {"rejections": rejected, "units": 60, "unevaluable": unknown,
                           "observed_fraction": rejected / 60 if not unknown else None}
        total_unknown += unknown
    return {"status": "EXPLORATORY_SMALL_CHECK_COMPLETE" if not total_unknown else "TECHNICAL_FAILURE_UNEVALUABLE",
            "technical_status": "PASS" if not total_unknown else "FAIL", "per_prompt": per_prompt,
            "channel_counts": counts, "original_large_sample_acceptance_applied": False,
            "formal_FPR_calibration_pass": None, "stage2_formal_release": False,
            "scope": "User-cost-amended5-unit-per-prompt exploration; no observed-zero-implies-FPR-zero claim.",
            "interpretation": "Report counts and pointwise uncertainty only. No macro engineering bound or hypothesis-based release gate."}


def run(args):
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or source in out.parents:
        raise ValueError("Small run must use a separate sibling output directory")
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (source / "RUN.lock").open("a") as parent_lock, (out / "RUN.lock").open("a") as child_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(child_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_manifest = runtime.read_json(source / "PRE_DATA_MANIFEST.json")
        for name, value in parent_manifest["source_sha256"].items():
            if digest(runtime.ROOT / name) != value:
                raise ValueError("Original frozen runtime source changed")
        original = parent_manifest["config"]
        for section, recorded, expected in ((original["model"], parent_manifest["target_model"], original["target_weight_sha256"]),
                                            (original["semantic"], parent_manifest["semantic_model"], original["semantic"]["weight_sha256"])):
            if runtime.snapshot_files(section["name"], section["revision"], expected) != recorded:
                raise ValueError("Parent model/tokenizer assets changed")
        config = copy.deepcopy(original)
        config["sampling"]["validation_units_per_prompt"] = 5
        config["sampling"]["response_budget"] = 4032
        prompts = runtime.json_rows(runtime.ROOT / config["prompts"])
        if digest(runtime.ROOT / config["prompts"]) != config["prompts_sha256"]:
            raise ValueError("Prompt hash mismatch")
        plans = {role: schedule(config, prompts, role) for role in ("fit", "calibration", "validation")}
        parent_hash = digest(source / "PRE_DATA_MANIFEST.json")
        initial = {}
        for role in plans:
            parent_rows = read_chain(source / "responses" / (role + ".jsonl"), schedule(original, prompts, role), parent_hash,
                                     complete=role != "validation")
            if role == "validation" and len(parent_rows) > 2880:
                raise ValueError("More than the new cap was already generated; requires explicit subset handling")
            initial[role] = len(parent_rows)
        if runtime.read_json(source / "CALIBRATION_AUDIT.json")["technical_status"] != "PASS":
            raise ValueError("Parent calibration technical check did not pass")
        if runtime.read_json(source / "FIT_FROZEN.json")["fit_responses_sha256"] != digest(source / "responses/fit.jsonl"):
            raise ValueError("Parent Fit response hash changed")
        if runtime.read_json(source / "CALIBRATION_AUDIT.json")["frozen_fit_sha256"] != digest(source / "FIT_FROZEN.json"):
            raise ValueError("Parent calibration is not bound to frozen Fit")
        source_paths = [source / name for name in ("PRE_DATA_MANIFEST.json", "FIT_FROZEN.json", "CALIBRATION_AUDIT.json")]
        source_paths += [source / "responses" / (role + ".jsonl") for role in plans]
        amendment = {"version": "r2-small-user-cost-amendment-v1", "reason": "user_explicitly_approved_pause_and5_units_per_prompt",
                     "parent_directory": str(source), "parent_manifest_sha256": parent_hash,
                     "effective_sampling": config["sampling"], "old_acceptance_abandoned_for_this_run": True,
                     "no_p_value_based_subset_selection": True, "unit_selection": "first5 pre-existing scheduled units per prompt",
                     "initial_inherited_counts": initial, "maximum_additional_model_responses": 2880 - initial["validation"],
                     "plans_sha256": {r: schedule_hash(p) for r, p in plans.items()},
                     "source_artifacts_sha256": {str(p.relative_to(source)): digest(p) for p in source_paths},
                     "wrapper_sha256": digest(Path(__file__)),
                     "record_binding": "Inherited generation manifest identifies unchanged model/sampling; this amendment supplies the reduced execution cap.",
                     "statistical_scope": "small independent exploratory negative check_not_formal_FPR_validation"}
        freeze_json(out / "SMALL_PRE_DATA_AMENDMENT.json", amendment)
        for p in source_paths:
            target = out / p.relative_to(source)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
            elif p.parent.name != "responses" and digest(target) != digest(p):
                raise ValueError("Copied parent freeze mismatch")
        # Independent copies: new cache/response writes cannot affect the paused run.
        if not (out / "semantic_cache").exists():
            shutil.copytree(source / "semantic_cache", out / "semantic_cache")
        pause = {"state": "PAUSED_BY_USER_COST_LIMIT", "updated_utc": runtime.utc(),
                 "retained_counts": initial, "replacement_directory": str(out),
                 "large_sample_result": "NOT_COMPLETED_NOT_PASSED"}
        atomic_json(source / "USER_PAUSE_RECORD.json", pause)
        atomic_json(source / "STATUS.json", pause)
        runtime.status(out, phase="small_resume", state="RUNNING", inherited_validation=initial["validation"], target_validation=2880)
        # Existing generator and per-unit test are retained; only summary is replaced
        # in this wrapper process. No mutation to parent runtime or original gates.
        runtime.generate_role(config, prompts, plans, out, "validation")
        runtime.summarize_validation = small_summary
        runtime.analyze_role(config, prompts, plans, out, parent_manifest, "validation")
        final_path = out / "FINAL_REPORT.json"
        final = runtime.read_json(final_path)
        final.update({"small_amendment_sha256": digest(out / "SMALL_PRE_DATA_AMENDMENT.json"),
                      "reused_fresh_R2_responses": sum(initial.values()),
                      "additional_responses_after_cost_amendment": 2880 - initial["validation"],
                      "old_32B_or_stage1_response_reuse": 0,
                      "parent_data_unchanged": all(digest(source / name) == value for name, value in amendment["source_artifacts_sha256"].items())})
        atomic_json(final_path, final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        atomic_json(args.output / "STATUS.json", {"state": "FAILED", "error": str(exc), "updated_utc": runtime.utc()})
        traceback.print_exc()
        raise
