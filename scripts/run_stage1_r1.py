"""Local offline R1 reanalysis of the immutable Stage1 audit snapshot."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_precalibration import FamilyBalancedScaler, build_h8_feature_schema
from llm_integrity.stage1_r1 import (
    bind_existing_cache, canonical, digest, evaluate_task_r1, majority, permutation_test, save_json,
)


def read_rows(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(("stage1-r1-v1/" + key).encode()).digest()[:8], "little")


def task_summary(texts, prompt):
    rows = [evaluate_task_r1(t, prompt) for t in texts]
    checked = [r["passed"] for r in rows if r["passed"] is not None]
    proxies = [r["proxy_passed"] for r in rows if r.get("proxy_passed") is not None]
    return {"statuses": dict(Counter(r["status"] for r in rows)), "total": len(rows),
            "evaluable": len(checked), "pass_rate": float(np.mean(checked)) if checked else None,
            "proxy_rate": float(np.mean(proxies)) if proxies else None, "reason": rows[0]["reason"]}


def run(args):
    source = args.input.resolve()
    out = args.output.resolve()
    if out == source or source in out.parents:
        raise ValueError("Output must be outside immutable input snapshot")
    audit = source / "results/stage1_audit_20260905"
    prompt_path = ROOT / "experiments/h9-stage1/inputs/stage1_prompts_v1.jsonl"
    prompts = read_rows(prompt_path)
    prompt_map = {p["id"]: {**p, "category": p["task_family"]} for p in prompts}
    primary = source / "results/h9_stage1_qwen05b_attack_stat_calibration_20260904"
    extension = source / "results/h9_stage1_qwen05b_gaussian_escalation_20260904"
    report_path = audit / "primary/analysis/FINAL_REPORT.json"
    frozen = json.loads(report_path.read_text(encoding="utf-8"))
    schema = build_h8_feature_schema(512)
    if frozen["feature_calibration"]["schema_sha256"] != schema.sha256:
        raise ValueError("Frozen feature schema mismatch")
    provenance = audit / "EXECUTION_PROVENANCE.json"
    # Historical truncation wasn't recorded explicitly: bind actual bank bytes and
    # do not pretend to reconstruct encoder settings or encode any unseen text.
    identity = {
        "name": "BAAI/bge-small-zh-v1.5", "revision": "7999e1d3359715c523056ef9478215996d62a620",
        "normalize_embeddings": True, "text_preprocessing": "exact_response_no_added_prompt",
        "encoder_max_seq_length": "historical_default_not_explicitly_recorded",
        "encoding_policy": "import_frozen_bank_only_no_miss_encoding",
        "source_provenance_sha256": digest(provenance),
        "source_audit_script_sha256": digest(source / "scripts/audit_h9_stage1_20260905.py"),
    }
    out.mkdir(parents=True, exist_ok=True)
    cache = bind_existing_cache(out / "semantic_cache_manifest.json", audit / "canonical_texts.json",
                                audit / "canonical_semantic_embeddings.npy", identity)
    extractor = FeatureExtractor(semantic_model_name=identity["name"],
                                 semantic_model_revision=identity["revision"], semantic_cache=cache)
    inventory = []
    banks = {}
    reused = []
    for label, parent in (("primary", primary), ("extension", extension)):
        for path in sorted((parent / "responses").glob("*.jsonl")):
            rows = read_rows(path)
            endpoint = path.stem
            n = 48 if endpoint == "intact" else 24
            expected = {(p["id"], i) for p in prompts for i in range(n)}
            keys = [(r["prompt_id"], r["response_index"]) for r in rows]
            if len(keys) != len(expected) or len(keys) != len(set(keys)) or set(keys) != expected:
                raise ValueError(f"Invalid response membership: {path}")
            for r in rows:
                if r["endpoint_id"] != endpoint or r["response_sha256"] != hashlib.sha256(r["response"].encode()).hexdigest():
                    raise ValueError(f"Response identity/hash mismatch: {path}")
                expected_index = next(i for i, p in enumerate(prompts) if p["id"] == r["prompt_id"])
                if r["prompt_index"] != expected_index or r["generation_seed"] != 910000 + expected_index * 1000 + r["response_index"]:
                    raise ValueError(f"Generation schedule mismatch: {path}")
            inventory.append({"file": str(path), "sha256": digest(path), "rows": len(rows)})
            key = "intact" if endpoint == "intact" else f"{label}/{endpoint}"
            if label == "extension" and (endpoint == "intact" or endpoint.startswith("lora_")):
                original = primary / "responses" / path.name
                if digest(path) != digest(original):
                    raise ValueError("Reused intact/LoRA banks differ")
                reused.append(str(path))
                continue
            banks[key] = rows
    if len(inventory) != 38 or len(banks) != 28 or sum(map(len, banks.values())) != 8352:
        raise ValueError("Incomplete historical snapshot")
    protocol = {
        "version": "stage1-r1-v1", "interpretation": "exploratory_old_data_reanalysis_not_R2_validation",
        "new_model_responses": 0, "permutations": args.permutations, "alpha": .05, "tie_atol": 1e-12,
        "raw_statistic": "max(prefix10_tv, bigram_JS_with_empty_sentinel, half_centroid_cosine_distance)",
        "h8_statistic": "unstandardized_unbiased_MMD2",
        "decision_policy": "two_separate_channels_no_OR_no_strength_selection",
        "fit_policy": "reuse_frozen_canonical_audit_scaler_bandwidth_from_48_intact_exploratory_only",
        "reference_indices": [24, 47], "attack_indices": [0, 23],
        "prompt_sha256": digest(prompt_path), "frozen_analysis_sha256": digest(report_path),
        "cache_manifest_sha256": digest(out / "semantic_cache_manifest.json"),
        "source_files": {str(p.relative_to(ROOT)): digest(p) for p in
                         [Path(__file__), ROOT / "src/llm_integrity/stage1_r1.py",
                          ROOT / "src/llm_integrity/features.py", ROOT / "src/llm_integrity/h8_precalibration.py"]},
        "inputs": inventory, "reused_bank_files_not_double_counted": reused,
    }
    manifest = out / "RUN_MANIFEST.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Resume manifest mismatch; choose a fresh output directory")
    save_json(manifest, protocol)
    run_hash = digest(manifest)
    scalers = {pid: FamilyBalancedScaler.from_dict(s) for pid, s in frozen["feature_calibration"]["scalers"].items()}
    groups = {}
    for key, rows in banks.items():
        for pid in prompt_map:
            group = sorted([r for r in rows if r["prompt_id"] == pid], key=lambda r: r["response_index"])
            texts = [r["response"] for r in group]
            feat = extractor.transform(texts, [prompt_map[pid]] * len(texts))
            groups[key, pid] = (group, feat, scalers[pid].transform(feat, schema))
    # Cache reload, reordered requests, and split batches must give exact rows.
    from llm_integrity.stage1_r1 import FrozenSemanticCache
    selected = list(cache.index)[:80]
    original = cache.transform(selected)
    reloaded = FrozenSemanticCache(out / "semantic_cache_manifest.json", identity)
    invariant = (np.array_equal(original, cache.transform(list(reversed(selected)))[::-1]) and
                 np.array_equal(original, np.concatenate([cache.transform(selected[:17]), cache.transform(selected[17:])])) and
                 np.array_equal(original, reloaded.transform(selected)))
    if not invariant:
        raise RuntimeError("Frozen cache invariance failure")
    save_json(out / "cache_checks.json", {"status": "PASS", "unique_texts": len(cache.index),
              "reload_reorder_split_batch_exact_equal": invariant, "encoder_loaded": False})
    results = []
    for key in sorted(banks):
        if key == "intact":
            continue
        for pid, prompt in prompt_map.items():
            unit_id = key + "/" + pid
            unit_path = out / "units" / (hashlib.sha256(unit_id.encode()).hexdigest() + ".json")
            if unit_path.exists():
                unit = json.loads(unit_path.read_text(encoding="utf-8"))
                content = {k: v for k, v in unit.items() if k != "content_sha256"}
                if unit["run_manifest_sha256"] != run_hash or unit["unit_id"] != unit_id or unit["content_sha256"] != hashlib.sha256(canonical(content).encode()).hexdigest():
                    raise ValueError("Resume unit identity/content mismatch")
                results.append(unit)
                continue
            intact, ifeat, ix = groups["intact", pid]
            attack, afeat, ax = groups[key, pid]
            refs = intact[24:48]
            if {r["generation_seed"] for r in refs} & {r["generation_seed"] for r in attack}:
                raise ValueError("Reference/attack random-seed overlap")
            texts = [r["response"] for r in refs + attack]
            outcome = permutation_test(texts, np.concatenate([ifeat[24:48, 11:523], afeat[:, 11:523]]),
                                       np.concatenate([ix[24:48], ax]), 24,
                                       frozen["feature_calibration"]["bandwidths"][pid]["value"],
                                       stable_seed(unit_id), args.permutations)
            itask = task_summary([r["response"] for r in intact], prompt)
            atask = task_summary([r["response"] for r in attack], prompt)
            unit = {"unit_id": unit_id, "run_manifest_sha256": run_hash, "bank": key,
                    "prompt_id": pid, "task_family": prompt["task_family"], "variant_role": prompt["variant_role"],
                    "family": attack[0]["family"], "strength": attack[0]["strength"], "attack_seed": attack[0]["attack_seed"],
                    "reference_count": 24, "attack_count": 24, "seed": stable_seed(unit_id),
                    **outcome, "intact_task": itask, "attack_task": atask,
                    "task_pass_drop": itask["pass_rate"] - atask["pass_rate"] if itask["pass_rate"] is not None else None}
            unit["content_sha256"] = hashlib.sha256(canonical(unit).encode()).hexdigest()
            save_json(unit_path, unit)
            results.append(unit)
        print(f"completed {key}: {len(results)}/324 units", flush=True)
    # One old-bank intact split per prompt is a smoke check, NEVER an FPR estimate.
    smoke = []
    for pid in prompt_map:
        rows, feat, transformed = groups["intact", pid]
        smoke.append({"prompt_id": pid, **permutation_test([r["response"] for r in rows], feat[:, 11:523],
                     transformed, 24, frozen["feature_calibration"]["bandwidths"][pid]["value"],
                     stable_seed("intact_smoke/" + pid), args.permutations)})
    save_json(out / "intact_old_bank_smoke.json", {"interpretation": "12 dependent-fit exploratory checks_not_independent_FPR", "rows": smoke})
    summaries = []
    for source_label, family, strength in sorted({(r["bank"].split("/")[0], r["family"], r["strength"]) for r in results}):
        subset = [r for r in results if (r["bank"].split("/")[0], r["family"], r["strength"]) == (source_label, family, strength)]
        per_prompt = []
        for pid in prompt_map:
            rs = [r for r in subset if r["prompt_id"] == pid]
            drops = [r["task_pass_drop"] for r in rs if r["task_pass_drop"] is not None]
            per_prompt.append({"prompt_id": pid, **{channel: majority(r[channel]["detected"] for r in rs) for channel in ("raw", "h8")},
                               "task_pass_drop_seed_median": float(np.median(drops)) if drops else None,
                               "task_pass_drop_worst_seed": max(drops) if drops else None})
        summary = {"source": source_label, "family": family, "strength": strength, "prompts": per_prompt}
        for channel in ("raw", "h8"):
            status = Counter(p[channel]["status"] for p in per_prompt)
            summary[channel] = {s: status[s] for s in ("detected", "not_detected", "unevaluable")}
            summary[channel]["total"] = len(per_prompt)
        strict = [p["task_pass_drop_seed_median"] for p in per_prompt if p["task_pass_drop_seed_median"] is not None]
        summary["strict_task_evaluable_prompts"] = len(strict)
        summary["strict_task_drop_mean"] = float(np.mean(strict)) if strict else None
        summaries.append(summary)
    # Record old/new decisions explicitly; new raw max is not legacy max-z.
    comparison = []
    for label in ("primary", "extension"):
        legacy = read_rows(audit / label / "analysis/prompt_endpoint_results.jsonl")
        new = {(r["bank"].split("/", 1)[1], r["prompt_id"]): r for r in results if r["bank"].startswith(label + "/")}
        for old in legacy:
            if (old["endpoint_id"], old["prompt_id"]) not in new:
                continue
            r = new[old["endpoint_id"], old["prompt_id"]]
            comparison.append({"unit_id": r["unit_id"], "legacy_raw": old["raw_detected"], "r1_raw": r["raw"]["detected"],
                               "legacy_h8": old["h8_mmd_detected"], "r1_h8": r["h8"]["detected"],
                               "legacy_task_drop": old["task_pass_drop"], "r1_strict_task_drop": r["task_pass_drop"]})
    save_json(out / "legacy_comparison.json", comparison)
    save_json(out / "prompt_endpoint_results.json", results)
    summary = {
        "execution_status": "COMPLETE", "R1_measurement_reanalysis": "COMPLETE",
        "stage1_overall_release": "NOT_PASSED_PENDING_R2_R3", "new_model_responses": 0,
        "distinct_existing_responses": 8352, "attack_prompt_units": len(results),
        "cache_checks": "PASS", "channel_status_counts": {ch: dict(Counter(r[ch]["status"] for r in results)) for ch in ("raw", "h8")},
        "strength_summaries": summaries,
        "limitations": ["Post hoc exploratory reanalysis; no independent FPR or power claim.",
                        "Historical intact-fitted transforms reuse the reference bank; permutation p-values are exploratory.",
                        "No raw/H8 OR and no multiple-prompt/model-level detector claim.",
                        "Raw bounded max replaces legacy max-z; scores and old eligibility thresholds are not comparable.",
                        "Code, safety, summary and term-presence evaluators are proxies, excluded from strict task pass rate.",
                        "Historical encoder truncation was not explicitly recorded; bank is byte-frozen and cache misses fail closed.",
                        "No H6/MCC changes, strength selection, new attack training or model generation."],
        "input_hashes_unchanged_after_run": all(digest(i["file"]) == i["sha256"] for i in inventory),
        "run_manifest_sha256": run_hash,
    }
    save_json(out / "R1_REPORT.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("strength_summaries", "limitations")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=999)
    run(parser.parse_args())
