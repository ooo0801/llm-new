"""Stages 3 and 5 for the three-proxy Stage2 mechanism experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical, digest, raw_statistics, save_json
from stage2_three_proxy_search import CAL, OLD, PROXIES, normalize, read, sha_value

OUT = project_path(
    os.environ.get(
        "STAGE2_THREE_PROXY_OUT",
        "results/stage2_three_proxy_ceiling_v2_20260906",
    )
)


PILOT = OUT / "pilot_behavior"
FORMAL = OUT / "formal_behavior"


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Refusing to change frozen artifact: {path}")
    else:
        save_json(path, value)


def status(phase, **fields):
    payload = {"phase": phase, "utc_unix": time.time(), **fields}
    save_json(OUT / "STATUS.json", payload)
    print(canonical(payload), flush=True)


def configure_base(folder, repetitions):
    base = importlib.import_module("stage2_behavior_confirmation_26")
    base.CAL = CAL
    base.OUT = folder
    base.EVAL = folder / "evaluation"
    base.REPETITIONS = int(repetitions)
    base.EVAL.mkdir(parents=True, exist_ok=True)
    return base


def selected_variants(strengths, seeds):
    report = read(CAL / "CALIBRATION_REPORT.json")
    design = read(CAL / "DESIGN.json")
    lookup = {row["variant_id"]: row for row in design["variants"]}
    variants = []
    for family in ("gaussian_noise", "finetuning"):
        groups = {group["selected_strength"]: group for group in report["selected_three_strength_groups"][family]}
        for strength in strengths:
            group = groups[strength]
            for endpoint in group["endpoint_ids"][:seeds]:
                row = dict(lookup[endpoint])
                row["strength"] = strength
                row["calibration_group_intensity"] = group["intensity"]
                variants.append(row)
    return variants


def candidate_union(stage):
    preflight = read(OUT / "PREFLIGHT_REPORT.json")
    sources = preflight["pilot_sources"] if stage == "pilot" else preflight["formal_sources"]
    proxies = PROXIES if stage == "pilot" else tuple(read(OUT / "PROMOTION.json")["promoted_proxies"])
    root = OUT / ("pilot_search" if stage == "pilot" else "formal_search")
    prompts = []
    aliases = []
    by_text = {}
    # Source controls are intentionally included to quantify search lift.
    source_limit = 16 if stage == "formal" else len(sources)
    for row in sources[:source_limit]:
        prompt_id = "u_" + hashlib.sha256(row["prompt"].encode()).hexdigest()[:16]
        if row["prompt"] not in by_text:
            item = dict(row); item["id"] = prompt_id
            by_text[row["prompt"]] = item; prompts.append(item)
        aliases.append({"prompt_id": prompt_id, "method": "source", "source_id": row["id"], "proxy_gain": 0.0})
    candidate_rows = []
    for proxy in proxies:
        for row in sources:
            path = root / proxy / f"{row['id']}.json"
            if not path.exists():
                continue
            result = read(path)
            if not result.get("accepted") or not result.get("optimized_prompt"):
                continue
            candidate_rows.append({"proxy": proxy, "source": row, "result": result})
    if stage == "formal":
        reduced = []
        for proxy in proxies:
            rows = [item for item in candidate_rows if item["proxy"] == proxy]
            rows.sort(key=lambda item: (-float(item["result"]["proxy_objective_gain"]), item["source"]["id"]))
            reduced.extend(rows[:16])
        candidate_rows = reduced
    for item in candidate_rows:
        text = item["result"]["optimized_prompt"]
        prompt_id = "u_" + hashlib.sha256(text.encode()).hexdigest()[:16]
        if text not in by_text:
            prompt = dict(item["source"]); prompt["id"] = prompt_id; prompt["prompt"] = text
            by_text[text] = prompt; prompts.append(prompt)
        aliases.append({"prompt_id": prompt_id, "method": item["proxy"], "source_id": item["source"]["id"],
                        "proxy_gain": float(item["result"]["proxy_objective_gain"])})
    return prompts, aliases


def behavior_plan(stage):
    search_plan = read(OUT / "PLAN.json")
    old_plan = read(OLD / "evaluation" / "PLAN.json")
    prompts, aliases = candidate_union(stage)
    if stage == "pilot":
        repetitions = 4
        variants = selected_variants(("medium",), 2)
        endpoints = ["intact_reference", "intact_null"] + [row["variant_id"] for row in variants]
        generation_seed = 750_000_000
    else:
        repetitions = 8
        variants = selected_variants(("weak", "medium", "strong"), 3)
        endpoints = ["intact_fit", "intact_reference", "intact_null"] + [row["variant_id"] for row in variants]
        generation_seed = 780_000_000
    # Reimplement the bank schedule with a disjoint seed domain.
    banks = {}
    seed = generation_seed
    for endpoint in endpoints:
        rows = []
        for response_index in range(repetitions):
            for prompt_index in range(len(prompts)):
                rows.append({"endpoint": endpoint, "prompt_index": prompt_index,
                             "response_index": response_index, "generation_seed": seed})
                seed += 1
        banks[endpoint] = rows
    return {
        "schema": f"stage2-three-proxy-{stage}-behavior-v1",
        "model": search_plan["model"], "prompts": prompts, "aliases": aliases,
        "variants": variants, "responses_per_prompt_endpoint": repetitions,
        "generation": {"max_input_tokens": 512, "max_new_tokens": 128, "do_sample": True,
                       "temperature": .7, "top_p": .9, "top_k": 50},
        "statistics": {"permutations": 999 if stage == "formal" else 199,
                       "alpha": .05, "tie_atol": 1e-12},
        "semantic": old_plan["semantic"], "semantic_model": old_plan["semantic_model"],
        "packages": old_plan["packages"],
        "banks": {name: {"count": len(rows), "sha256": sha_value(rows)} for name, rows in banks.items()},
        "response_budget": sum(len(rows) for rows in banks.values()),
        "search_plan_sha256": digest(OUT / "PLAN.json"),
        "preflight_sha256": digest(OUT / "PREFLIGHT_REPORT.json"),
        "promotion_sha256": digest(OUT / "PROMOTION.json") if stage == "formal" else None,
        "validity": "intact response modal count; attacked local task correctness is not a gate",
        "channels": "Raw primary, H8 separately reported in formal stage; no OR rule",
    }, banks


def freeze_behavior(stage):
    folder = PILOT if stage == "pilot" else FORMAL
    evaluation = folder / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    plan, banks = behavior_plan(stage)
    freeze(evaluation / "PLAN.json", plan)
    freeze(evaluation / "BANKS.json", banks)
    status(f"{stage.upper()}_BEHAVIOR_PLAN_FROZEN", prompts=len(plan["prompts"]),
           endpoints=len(banks), responses=plan["response_budget"])


def load_behavior(stage):
    folder = PILOT if stage == "pilot" else FORMAL
    plan = read(folder / "evaluation" / "PLAN.json")
    banks = read(folder / "evaluation" / "BANKS.json")
    actual = {name: {"count": len(rows), "sha256": sha_value(rows)} for name, rows in banks.items()}
    if actual != plan["banks"]:
        raise RuntimeError("Behavior bank schedule changed")
    if digest(OUT / "PLAN.json") != plan["search_plan_sha256"]:
        raise RuntimeError("Search plan changed")
    return plan, banks


def generate(stage):
    folder = PILOT if stage == "pilot" else FORMAL
    repetitions = 4 if stage == "pilot" else 8
    base = configure_base(folder, repetitions)
    plan, banks = load_behavior(stage)
    base.generate_all(plan, banks)
    status(f"{stage.upper()}_BEHAVIOR_GENERATION_COMPLETE", responses=plan["response_budget"])


def empirical_tv(left, right):
    keys = set(left) | set(right)
    a, b = Counter(normalize(value) for value in left), Counter(normalize(value) for value in right)
    return .5 * sum(abs(a[key] / len(left) - b[key] / len(right)) for key in keys)


def permutation_raw(texts, semantic, n_left, seed, permutations):
    rng = np.random.default_rng(seed)
    membership = np.zeros((permutations + 1, len(texts)))
    membership[0, :n_left] = 1
    for index in range(1, permutations + 1):
        membership[index, rng.permutation(len(texts))[:n_left]] = 1
    maximum, components = raw_statistics(texts, semantic, membership, n_left)
    null = maximum[1:]
    std = float(null.std())
    return {"statistic": float(maximum[0]),
            "p_value": float((1 + np.count_nonzero(null >= maximum[0] - 1e-12)) / (permutations + 1)),
            "standardized_effect": None if std <= 1e-15 else float((maximum[0] - null.mean()) / std),
            "components": {name: float(components[0, i]) for i, name in enumerate(("prefix10_tv", "bigram_js", "semantic"))}}


def analyze_pilot():
    base = configure_base(PILOT, 4)
    plan, banks = load_behavior("pilot")
    cache = base.semantic_cache(plan)
    from llm_integrity.features import FeatureExtractor
    extractor = FeatureExtractor(semantic_cache=cache)
    reference = base.bank(plan, banks, "intact_reference", complete=True)
    records = []
    try:
        for variant in plan["variants"]:
            right = base.bank(plan, banks, variant["variant_id"], complete=True)
            for prompt in plan["prompts"]:
                left_texts = [row["response"] for row in reference if row["prompt_id"] == prompt["id"]]
                right_texts = [row["response"] for row in right if row["prompt_id"] == prompt["id"]]
                texts = left_texts + right_texts
                features = extractor.transform(texts, [prompt] * len(texts))
                semantic = features[:, 11:523]
                seed = int(hashlib.sha256(f"pilot:{variant['variant_id']}:{prompt['id']}".encode()).hexdigest()[:15], 16)
                records.append({"prompt_id": prompt["id"], "endpoint": variant["variant_id"],
                                "family": variant["family"], "attack_seed": variant["seed"],
                                "text_tv": empirical_tv(left_texts, right_texts),
                                "raw": permutation_raw(texts, semantic, 4, seed, plan["statistics"]["permutations"]),
                                "intact_modal": Counter(normalize(value) for value in left_texts).most_common(1)[0][1],
                                "nonempty": all(normalize(value) for value in texts)})
        save_json(PILOT / "evaluation" / "PILOT_MATRIX.json", {"records": records})
    finally:
        cache.close(); base.cleanup()
    aliases = [alias for alias in plan["aliases"] if alias["method"] in PROXIES]
    alias_lookup = defaultdict(list)
    for alias in aliases: alias_lookup[alias["method"]].append(alias)
    summary = {}
    for proxy in PROXIES:
        prompt_ids = {alias["prompt_id"] for alias in alias_lookup[proxy]}
        family_values = {}
        all_values = []
        for family in ("gaussian_noise", "finetuning"):
            means = []
            for prompt_id in prompt_ids:
                rows = [row for row in records if row["prompt_id"] == prompt_id and row["family"] == family]
                if len(rows) == 2 and min(row["intact_modal"] for row in rows) >= 3 and all(row["nonempty"] for row in rows):
                    means.append(float(np.mean([row["text_tv"] for row in rows])))
            family_values[family] = {"prompt_count": len(means), "mean_text_tv": float(np.mean(means)) if means else 0.0,
                                     "median_text_tv": float(np.median(means)) if means else 0.0,
                                     "sensitive_prompts": sum(value >= .25 for value in means)}
            all_values.extend(means)
        raw_z = [row["raw"]["standardized_effect"] for row in records if row["prompt_id"] in prompt_ids
                 and row["raw"]["standardized_effect"] is not None]
        qualifies = all(family_values[family]["sensitive_prompts"] >= 2 for family in family_values)
        score = .5 * sum(family_values[family]["median_text_tv"] for family in family_values)
        summary[proxy] = {"families": family_values, "median_raw_standardized_effect": float(np.median(raw_z)) if raw_z else None,
                          "promotion_score": score, "qualifies": qualifies}
    ranking = sorted(PROXIES, key=lambda proxy: (-summary[proxy]["promotion_score"], proxy))
    can_continue = any(summary[proxy]["qualifies"] for proxy in PROXIES)
    promotion = {"schema": "stage2-three-proxy-promotion-v1", "can_continue": can_continue,
                 "promoted_proxies": ranking[:2] if can_continue else [], "ranking": ranking,
                 "proxy_summaries": summary, "pilot_matrix_sha256": digest(PILOT / "evaluation" / "PILOT_MATRIX.json"),
                 "decision": "CONTINUE_FORMAL" if can_continue else "STOP_NO_MEASURABLE_GAUSSIAN_LORA_SIGNAL"}
    freeze(OUT / "PROMOTION.json", promotion)
    status("PILOT_BEHAVIOR_COMPLETE", decision=promotion["decision"], ranking=ranking)


def formal_generate_and_analyze():
    base = configure_base(FORMAL, 8)
    plan, banks = load_behavior("formal")
    base.generate_all(plan, banks)
    base.fit_h8(plan, banks)
    base.analyze(plan, banks)
    status("FORMAL_BEHAVIOR_ANALYSIS_COMPLETE", responses=plan["response_budget"])


def summarize_formal():
    base = configure_base(FORMAL, 8)
    plan, banks = load_behavior("formal")
    decisions = []
    for variant in plan["variants"]:
        decisions.extend(read(FORMAL / "evaluation" / "decisions" / f"{variant['variant_id']}.json")["records"])
    null_rows = read(FORMAL / "evaluation" / "decisions" / "intact_null.json")["records"]
    fit = base.bank(plan, banks, "intact_fit", complete=True)
    ref = base.bank(plan, banks, "intact_reference", complete=True)
    null_bank = base.bank(plan, banks, "intact_null", complete=True)
    aliases_by_method = defaultdict(list)
    for alias in plan["aliases"]: aliases_by_method[alias["method"]].append(alias)
    prompt_valid = {}
    for prompt in plan["prompts"]:
        groups = [[row["response"] for row in bank if row["prompt_id"] == prompt["id"]] for bank in (fit, ref, null_bank)]
        prompt_valid[prompt["id"]] = all(len(group) == 8 and all(normalize(value) for value in group)
                                             and Counter(normalize(value) for value in group).most_common(1)[0][1] >= 6
                                             for group in groups)
    methods = ["source"] + list(read(OUT / "PROMOTION.json")["promoted_proxies"])
    summaries = {}
    for method in methods:
        prompt_ids = {alias["prompt_id"] for alias in aliases_by_method[method]}
        strength_summary = []
        stable_family_prompts = defaultdict(set)
        for family in ("gaussian_noise", "finetuning"):
            for strength in ("weak", "medium", "strong"):
                stable_raw = 0; stable_h8 = 0; raw_values = []; tv_values = []
                variants = [row for row in plan["variants"] if row["family"] == family and row["strength"] == strength]
                for prompt_id in prompt_ids:
                    rows = [row for row in decisions if row["prompt_id"] == prompt_id and row["family"] == family
                            and row["strength"] == strength]
                    raw_seeds = sum(prompt_valid[prompt_id] and row["raw"]["detected"] is True
                                    and row["raw"]["statistic"] > 0 for row in rows)
                    h8_seeds = sum(prompt_valid[prompt_id] and row["h8"]["detected"] is True
                                   and row["h8"]["statistic"] is not None and row["h8"]["statistic"] > 0 for row in rows)
                    stable_raw += raw_seeds >= 2; stable_h8 += h8_seeds >= 2
                    if raw_seeds >= 2: stable_family_prompts[family].add(prompt_id)
                    raw_values.extend(float(row["raw"]["statistic"]) for row in rows)
                    for variant in variants:
                        left = [record["response"] for record in ref if record["prompt_id"] == prompt_id]
                        right_bank = base.bank(plan, banks, variant["variant_id"], complete=True)
                        right = [record["response"] for record in right_bank if record["prompt_id"] == prompt_id]
                        tv_values.append(empirical_tv(left, right))
                strength_summary.append({"family": family, "strength": strength,
                                         "stable_raw_prompts": int(stable_raw), "stable_h8_prompts": int(stable_h8),
                                         "median_raw_distance": float(np.median(raw_values)) if raw_values else 0.0,
                                         "median_text_tv": float(np.median(tv_values)) if tv_values else 0.0})
        both = stable_family_prompts["gaussian_noise"] & stable_family_prompts["finetuning"]
        summaries[method] = {"aliases": len(aliases_by_method[method]),
                             "unique_prompts": len(prompt_ids), "valid_intact_prompts": sum(prompt_valid[p] for p in prompt_ids),
                             "stable_any_gaussian": len(stable_family_prompts["gaussian_noise"]),
                             "stable_any_lora": len(stable_family_prompts["finetuning"]),
                             "stable_both_families": len(both), "strengths": strength_summary}
    promoted = read(OUT / "PROMOTION.json")["promoted_proxies"]
    ranking = sorted(promoted, key=lambda method: (-summaries[method]["stable_both_families"],
                                                    -summaries[method]["stable_any_gaussian"],
                                                    -summaries[method]["stable_any_lora"], method))
    best = ranking[0]
    pass_to_constrained = summaries[best]["stable_any_gaussian"] >= 4 and summaries[best]["stable_any_lora"] >= 4
    summary = {"schema": "stage2-three-proxy-formal-summary-v1",
               "status": "PASS_MECHANISM_CEILING" if pass_to_constrained else "FAIL_MECHANISM_CEILING",
               "best_proxy": best, "ranking": ranking, "methods": summaries,
               "null": {"raw_detections": sum(row["raw"]["detected"] is True for row in null_rows),
                        "h8_detections": sum(row["h8"]["detected"] is True for row in null_rows),
                        "units": len(null_rows), "formal_fpr_certified": False},
               "policy": "Raw and H8 separate; local attacked task changes are sensitivity, not invalidity; intact modal>=6/8 required.",
               "next": "restore semantic/task constraints and measure Pareto frontier" if pass_to_constrained
                       else "do not expand search; revise H6 objective or attack representation"}
    freeze(FORMAL / "evaluation" / "SUMMARY_THREE_PROXY.json", summary)
    status("ALL_FIVE_STAGES_COMPLETE", result=summary["status"], best_proxy=best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["pilot-freeze", "pilot-generate", "pilot-analyze",
                                          "formal-freeze", "formal-run", "formal-report"])
    args = parser.parse_args()
    if args.phase == "pilot-freeze": freeze_behavior("pilot")
    elif args.phase == "pilot-generate": generate("pilot")
    elif args.phase == "pilot-analyze": analyze_pilot()
    elif args.phase == "formal-freeze": freeze_behavior("formal")
    elif args.phase == "formal-run": formal_generate_and_analyze()
    elif args.phase == "formal-report": summarize_formal()


if __name__ == "__main__":
    main()
