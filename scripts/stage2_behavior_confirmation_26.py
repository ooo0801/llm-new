"""Fresh 26-prompt behavioral confirmation using utility-calibrated attacks.

Raw behavioral distance and H8 are always reported as separate channels. The
program never accepts an OR across them and never uses these responses to pick
attack strengths.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical, digest, evaluate_task_r1, permutation_test, raw_statistics
from llm_integrity.stage1_r2 import atomic_json, freeze_json, read_chain


CAL = project_path("results/stage2_attack_utility_calibration_v2_20260906")
OLD = project_path("results/stage2_six_source_pilot_20260905")
OUT = project_path("results/stage2_behavior_confirmation_26_v1_20260906")
EVAL = OUT / "evaluation"
REPETITIONS = 8


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status(phase, **fields):
    value = {"phase": phase, "utc_unix": time.time(), **fields}
    atomic_json(EVAL / "STATUS.json", value)
    print(canonical(value), flush=True)


def setup(strict=False):
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    return torch


def cleanup():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def sha_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def build_schedule(prompt_count, endpoints):
    banks = {}
    seed = 610_000_000
    for endpoint in endpoints:
        rows = []
        for response_index in range(REPETITIONS):
            for prompt_index in range(prompt_count):
                rows.append({
                    "endpoint": endpoint,
                    "prompt_index": prompt_index,
                    "response_index": response_index,
                    "generation_seed": seed,
                })
                seed += 1
        banks[endpoint] = rows
    return banks


def freeze_plan():
    calibration = read(CAL / "CALIBRATION_REPORT.json")
    if not calibration["can_proceed_stage3"] or calibration["status"] != "PASS_FOR_STAGE3":
        raise RuntimeError("Utility calibration did not authorize Stage3")
    cal_design = read(CAL / "DESIGN.json")
    old_plan = read(OLD / "evaluation" / "PLAN.json")
    prompts = old_plan["candidates"]["prompts"]
    if len(prompts) != 26 or len({row["id"] for row in prompts}) != 26:
        raise RuntimeError("Expected exactly 26 unique frozen prompts")
    variant_lookup = {row["variant_id"]: row for row in cal_design["variants"]}
    variants = []
    for family in ("gaussian_noise", "finetuning"):
        groups = calibration["selected_three_strength_groups"][family]
        if [row["selected_strength"] for row in groups] != ["weak", "medium", "strong"]:
            raise RuntimeError(f"Invalid selected strength labels for {family}")
        for group in groups:
            for endpoint in group["endpoint_ids"]:
                variant = dict(variant_lookup[endpoint])
                variant["strength"] = group["selected_strength"]
                variant["calibration_group_intensity"] = group["intensity"]
                variants.append(variant)
    if len(variants) != 18:
        raise RuntimeError("Stage3 needs 2 families x 3 strengths x 3 seeds")
    endpoints = ["intact_fit", "intact_reference", "intact_null"] + [row["variant_id"] for row in variants]
    banks = build_schedule(len(prompts), endpoints)
    source_files = [
        Path(__file__),
        project_path("src/llm_integrity/modeling.py"),
        project_path("src/llm_integrity/stage1_r1.py"),
        project_path("src/llm_integrity/stage1_r2.py"),
        project_path("src/llm_integrity/features.py"),
        project_path("src/llm_integrity/h8_precalibration.py"),
        project_path("src/llm_integrity/paper_variant_executor.py"),
        project_path("scripts/run_stage1_r2.py"),
    ]
    payload = {
        "schema": "stage2-26-prompt-behavior-confirmation-v1",
        "purpose": "test whether fresh utility-qualified Gaussian/LoRA attacks have measurable behavior on existing prompts",
        "calibration_report_sha256": digest(CAL / "CALIBRATION_REPORT.json"),
        "calibration_design_sha256": digest(CAL / "DESIGN.json"),
        "old_candidate_plan_sha256": digest(OLD / "evaluation" / "PLAN.json"),
        "model": cal_design["model"],
        "generation": {"max_input_tokens": 512, "max_new_tokens": 128, "do_sample": True,
                       "temperature": .7, "top_p": .9, "top_k": 50},
        "prompts": prompts,
        "aliases": old_plan["candidates"]["aliases"],
        "variants": variants,
        "responses_per_prompt_endpoint": REPETITIONS,
        "banks": {name: {"count": len(rows), "sha256": sha_value(rows)} for name, rows in banks.items()},
        "response_budget": sum(len(rows) for rows in banks.values()),
        "statistics": {"permutations": 999, "alpha": .05, "tie_atol": 1e-12},
        "behavior_channels": {
            "primary": "raw max-calibrated response distance",
            "components": ["prefix10_tv", "bigram_js_with_empty_sentinel", "half_semantic_centroid_cosine"],
            "secondary_separate": "H8 MMD",
            "combination": "no Raw/H8 OR rule",
        },
        "task_stability": {"minimum_intact_correct_of_8": 6, "maximum_attack_loss_from_intact": 1},
        "stage4_entry_rule": {
            "per_endpoint_detection": "raw p<=0.05 and positive raw distance and task-stable endpoint",
            "stable_strength": "at least 2 of 3 seeds detected",
            "family_measurable": "at least 2 unique prompts stable at any strength and at least 1 prompt stable at two strengths",
            "stage4": "both Gaussian and LoRA must be family-measurable",
        },
        "freshness": "No old response is reused; all generation seeds start at 610000000",
        "semantic": old_plan["semantic"],
        "semantic_model": old_plan["semantic_model"],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "sentence-transformers", "numpy", "scipy", "peft", "bitsandbytes")},
        "source_hashes": {str(path.relative_to(project_path("."))).replace("\\", "/"): digest(path)
                          for path in source_files},
    }
    freeze_json(EVAL / "PLAN.json", payload)
    status("PLAN_FROZEN", prompts=len(prompts), endpoints=len(endpoints), responses=payload["response_budget"])


def context():
    plan = read(EVAL / "PLAN.json")
    if digest(CAL / "CALIBRATION_REPORT.json") != plan["calibration_report_sha256"]:
        raise RuntimeError("Calibration report changed")
    if digest(CAL / "DESIGN.json") != plan["calibration_design_sha256"]:
        raise RuntimeError("Calibration design changed")
    if digest(OLD / "evaluation" / "PLAN.json") != plan["old_candidate_plan_sha256"]:
        raise RuntimeError("Candidate source plan changed")
    for name, expected in plan["source_hashes"].items():
        if digest(project_path(name)) != expected:
            raise RuntimeError(f"Stage3 code changed after freeze: {name}")
    for package, expected in plan["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"Package version changed: {package}")
    endpoints = ["intact_fit", "intact_reference", "intact_null"] + [row["variant_id"] for row in plan["variants"]]
    banks = build_schedule(len(plan["prompts"]), endpoints)
    actual = {name: {"count": len(rows), "sha256": sha_value(rows)} for name, rows in banks.items()}
    if actual != plan["banks"]:
        raise RuntimeError("Generation schedule changed")
    return plan, banks


def bank(plan, banks, endpoint, complete=True):
    return read_chain(EVAL / "responses" / f"{endpoint}.jsonl", banks[endpoint],
                      digest(EVAL / "PLAN.json"), complete=complete)


def generate_bank(bundle, plan, banks, endpoint):
    from llm_integrity.modeling import render_prompt

    torch = setup(False)
    expected = banks[endpoint]
    done = bank(plan, banks, endpoint, complete=False)
    if len(done) == len(expected):
        return
    encodings = {}
    for index, prompt in enumerate(plan["prompts"]):
        rendered = render_prompt(bundle.tokenizer, prompt["prompt"])
        encoded = bundle.tokenizer(rendered, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > plan["generation"]["max_input_tokens"]:
            raise RuntimeError(f"Prompt exceeds input budget: {prompt['id']}")
        encodings[index] = {key: value.to(bundle.device) for key, value in encoded.items()}
    path = EVAL / "responses" / f"{endpoint}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = done[-1]["record_sha256"] if done else "0" * 64
    first = len(done)
    started = time.time()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for index in range(first, len(expected)):
            spec = expected[index]
            seed = spec["generation_seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            encoded = encodings[spec["prompt_index"]]
            generation = plan["generation"]
            with torch.inference_mode():
                tokens = bundle.model.generate(
                    **encoded,
                    max_new_tokens=generation["max_new_tokens"],
                    do_sample=True,
                    temperature=generation["temperature"],
                    top_p=generation["top_p"],
                    top_k=generation["top_k"],
                    pad_token_id=bundle.tokenizer.pad_token_id,
                    eos_token_id=bundle.tokenizer.eos_token_id,
                )
            ids = tokens[0, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
            text = bundle.tokenizer.decode(ids, skip_special_tokens=True)
            eos = bundle.model.generation_config.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            eos.add(bundle.tokenizer.eos_token_id)
            truncated = len(ids) >= generation["max_new_tokens"] and (not ids or ids[-1] not in eos)
            prompt = plan["prompts"][spec["prompt_index"]]
            task = evaluate_task_r1(text, prompt)
            row = {
                **spec,
                "prompt_id": prompt["id"],
                "manifest_sha256": digest(EVAL / "PLAN.json"),
                "previous_sha256": previous,
                "response": text,
                "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "completion_token_ids": ids,
                "completion_token_count": len(ids),
                "truncated": truncated,
                "task_evaluation": task,
                "task_passed": task.get("passed") is True and not truncated,
                "utc_unix": time.time(),
            }
            row["record_sha256"] = sha_value(row)
            handle.write(canonical(row) + "\n")
            handle.flush()
            previous = row["record_sha256"]
            if (index + 1) % 24 == 0 or index + 1 == len(expected):
                os.fsync(handle.fileno())
                elapsed = time.time() - started
                rate = (index + 1 - first) / max(elapsed, 1e-9)
                status("generate", endpoint=endpoint, completed=index + 1, total=len(expected),
                       responses_per_second=rate,
                       estimated_remaining_seconds=(len(expected) - index - 1) / max(rate, 1e-9))
    bank(plan, banks, endpoint, complete=True)


def loaded_variant(plan, variant):
    from llm_integrity.paper_variant_executor import load_manifest_variant

    adapter = CAL / "adapters" / variant["variant_id"] if variant["family"] == "finetuning" else None
    return load_manifest_variant(plan["model"], variant,
                                 adapter_path=str(adapter) if adapter is not None else None)


def generate_all(plan, banks):
    from llm_integrity.modeling import load_model

    bundle = load_model(plan["model"])
    try:
        for endpoint in ("intact_fit", "intact_reference", "intact_null"):
            if endpoint not in banks:
                continue
            generate_bank(bundle, plan, banks, endpoint)
    finally:
        bundle.close()
        cleanup()
    for index, variant in enumerate(plan["variants"]):
        endpoint = variant["variant_id"]
        if len(bank(plan, banks, endpoint, complete=False)) == len(banks[endpoint]):
            continue
        status("load_behavior_endpoint", completed=index, total=len(plan["variants"]), endpoint=endpoint)
        loaded = loaded_variant(plan, variant)
        try:
            realization = dataclasses.asdict(loaded.report)
            adapter_hashes = {}
            if variant["family"] == "finetuning":
                folder = CAL / "adapters" / endpoint
                adapter_hashes = {path.name: digest(path) for path in sorted(folder.iterdir()) if path.is_file()}
            freeze_json(EVAL / "realizations" / f"{endpoint}.json", {
                "plan_sha256": digest(EVAL / "PLAN.json"),
                "manifest": variant,
                "realization": realization,
                "adapter_hashes": adapter_hashes,
            })
            generate_bank(loaded.bundle, plan, banks, endpoint)
        finally:
            loaded.close()
            cleanup()


def semantic_cache(plan):
    from run_stage1_r2 import NewSemanticCache

    manifest = {"semantic_model": plan["semantic_model"], "packages": plan["packages"]}
    return NewSemanticCache({"semantic": plan["semantic"]}, EVAL, manifest)


def fit_h8(plan, banks):
    target = EVAL / "FIT.json"
    fit_rows = bank(plan, banks, "intact_fit", complete=True)
    fit_sha = digest(EVAL / "responses" / "intact_fit.jsonl")
    if target.exists():
        frozen = read(target)
        if frozen["fit_response_sha256"] != fit_sha:
            raise RuntimeError("H8 fit responses changed")
        return
    from llm_integrity.features import FeatureExtractor
    from llm_integrity.h8_precalibration import (
        build_h8_feature_schema,
        fit_family_balanced_scaler,
        global_continuous_exclusion_mask,
        h8_global_degenerate_bandwidth,
        h8_median_positive_pairwise_distance,
    )

    cache = semantic_cache(plan)
    extractor = FeatureExtractor(semantic_cache=cache)
    schema = build_h8_feature_schema(512)
    matrices = {}
    try:
        for prompt in plan["prompts"]:
            rows = [row for row in fit_rows if row["prompt_id"] == prompt["id"]]
            matrices[prompt["id"]] = extractor.transform(
                [row["response"] for row in rows], [prompt] * len(rows))
        pooled = np.concatenate(list(matrices.values()))
        exclusion = global_continuous_exclusion_mask(pooled, schema)
        scalers = {prompt_id: fit_family_balanced_scaler(prompt_id, values, pooled, schema,
                                                        exclusion_mask=exclusion)
                   for prompt_id, values in matrices.items()}
        transformed = {prompt_id: scalers[prompt_id].transform(values, schema)
                       for prompt_id, values in matrices.items()}
        global_bandwidth = h8_global_degenerate_bandwidth(transformed)
        bandwidths = {}
        for prompt_id, values in transformed.items():
            try:
                bandwidths[prompt_id] = {"value": h8_median_positive_pairwise_distance(values),
                                         "source": "prompt_fit"}
            except ValueError:
                bandwidths[prompt_id] = {"value": global_bandwidth["sigma"],
                                         "source": "global_fit_fallback"}
        freeze_json(target, {
            "plan_sha256": digest(EVAL / "PLAN.json"),
            "fit_response_sha256": fit_sha,
            "schema_sha256": schema.sha256,
            "semantic_identity_sha256": cache.identity_hash,
            "scalers": {prompt_id: scaler.as_dict() for prompt_id, scaler in scalers.items()},
            "bandwidths": bandwidths,
            "global_bandwidth": global_bandwidth,
            "attack_responses_used": False,
        })
    finally:
        cache.close()
        cleanup()


def component_tests(texts, semantic, n_left, seed, permutations=999, alpha=.05, tie_atol=1e-12):
    rng = np.random.default_rng(seed)
    memberships = np.zeros((permutations + 1, len(texts)))
    memberships[0, :n_left] = 1
    for index in range(1, permutations + 1):
        memberships[index, rng.permutation(len(texts))[:n_left]] = 1
    _, components = raw_statistics(texts, semantic, memberships, n_left)
    names = ["prefix10_tv", "bigram_js_with_empty_sentinel", "half_semantic_centroid_cosine"]
    output = {}
    for index, name in enumerate(names):
        values = components[:, index]
        p_value = (1 + np.count_nonzero(values[1:] >= values[0] - tie_atol)) / (permutations + 1)
        null_std = float(values[1:].std())
        output[name] = {
            "statistic": float(values[0]),
            "p_value": float(p_value),
            "detected": bool(p_value <= alpha),
            "null_mean": float(values[1:].mean()),
            "null_std": null_std,
            "standardized_effect": None if null_std <= 1e-15 else float((values[0] - values[1:].mean()) / null_std),
        }
    return output


def analyze(plan, banks):
    from llm_integrity.features import FeatureExtractor
    from llm_integrity.h8_precalibration import build_h8_feature_schema, FamilyBalancedScaler

    frozen = read(EVAL / "FIT.json")
    if frozen["fit_response_sha256"] != digest(EVAL / "responses" / "intact_fit.jsonl"):
        raise RuntimeError("H8 fit bank changed")
    schema = build_h8_feature_schema(512)
    scalers = {prompt_id: FamilyBalancedScaler.from_dict(value)
               for prompt_id, value in frozen["scalers"].items()}
    reference = bank(plan, banks, "intact_reference", complete=True)
    endpoint_manifests = {"intact_null": {"family": "intact", "strength": "null", "seed": 0}}
    endpoint_manifests.update({row["variant_id"]: row for row in plan["variants"]})
    cache = semantic_cache(plan)
    extractor = FeatureExtractor(semantic_cache=cache)
    try:
        for endpoint, manifest in endpoint_manifests.items():
            target = EVAL / "decisions" / f"{endpoint}.json"
            right = bank(plan, banks, endpoint, complete=True)
            binding = {
                "plan_sha256": digest(EVAL / "PLAN.json"),
                "fit_sha256": digest(EVAL / "FIT.json"),
                "reference_sha256": digest(EVAL / "responses" / "intact_reference.jsonl"),
                "target_sha256": digest(EVAL / "responses" / f"{endpoint}.jsonl"),
            }
            if target.exists():
                old = read(target)
                if any(old[key] != value for key, value in binding.items()):
                    raise RuntimeError(f"Decision binding changed: {endpoint}")
                continue
            records = []
            for prompt in plan["prompts"]:
                left_rows = [row for row in reference if row["prompt_id"] == prompt["id"]]
                right_rows = [row for row in right if row["prompt_id"] == prompt["id"]]
                if len(left_rows) != 8 or len(right_rows) != 8:
                    raise RuntimeError("Incomplete response group")
                texts = [row["response"] for row in left_rows + right_rows]
                features = extractor.transform(texts, [prompt] * 16)
                transformed = scalers[prompt["id"]].transform(features, schema)
                semantic = features[:, 11:523]
                seed = int(sha_value({"endpoint": endpoint, "prompt_id": prompt["id"]})[:15], 16)
                tested = permutation_test(
                    texts, semantic, transformed, 8, frozen["bandwidths"][prompt["id"]]["value"], seed,
                    permutations=plan["statistics"]["permutations"],
                    alpha=plan["statistics"]["alpha"], tie_atol=plan["statistics"]["tie_atol"])
                components = component_tests(
                    texts, semantic, 8, seed,
                    permutations=plan["statistics"]["permutations"],
                    alpha=plan["statistics"]["alpha"], tie_atol=plan["statistics"]["tie_atol"])
                records.append({
                    "prompt_id": prompt["id"],
                    "endpoint": endpoint,
                    "family": manifest["family"],
                    "strength": manifest.get("strength", "null"),
                    "attack_seed": manifest.get("seed", 0),
                    "intact_correct": sum(row["task_passed"] for row in left_rows),
                    "attack_correct": sum(row["task_passed"] for row in right_rows),
                    "components": components,
                    **tested,
                })
            freeze_json(target, {**binding, "records": records})
            status("analyze", endpoint=endpoint, records=len(records))
    finally:
        cache.close()
        cleanup()


def summarize(plan, banks):
    decisions = []
    for variant in plan["variants"]:
        decisions.extend(read(EVAL / "decisions" / f"{variant['variant_id']}.json")["records"])
    null_rows = read(EVAL / "decisions" / "intact_null.json")["records"]
    task_gate = plan["task_stability"]
    by_prompt = defaultdict(list)
    for row in decisions:
        row["task_stable"] = (row["intact_correct"] >= task_gate["minimum_intact_correct_of_8"]
                              and row["attack_correct"] >= row["intact_correct"] - task_gate["maximum_attack_loss_from_intact"])
        row["primary_endpoint_pass"] = row["task_stable"] and row["raw"]["detected"] is True and row["raw"]["statistic"] > 0
        by_prompt[row["prompt_id"]].append(row)

    aliases = plan["aliases"]
    alias_by_prompt = defaultdict(list)
    for alias in aliases:
        alias_by_prompt[alias["prompt_id"]].append(alias)
    family_summaries = {}
    strength_rows = []
    prompt_results = []
    component_names = plan["behavior_channels"]["components"]
    for family in ("gaussian_noise", "finetuning"):
        stable_by_prompt = defaultdict(set)
        for strength in ("weak", "medium", "strong"):
            group = [row for row in decisions if row["family"] == family and row["strength"] == strength]
            endpoint_ids = sorted({row["endpoint"] for row in group})
            stable_count = 0
            component_stable = {name: 0 for name in component_names}
            h8_stable = 0
            distances = []
            for prompt in plan["prompts"]:
                rows = [row for row in group if row["prompt_id"] == prompt["id"]]
                primary_seeds = sum(row["primary_endpoint_pass"] for row in rows)
                h8_seeds = sum(row["task_stable"] and row["h8"]["detected"] is True
                               and row["h8"]["statistic"] is not None and row["h8"]["statistic"] > 0 for row in rows)
                stable = primary_seeds >= 2
                if stable:
                    stable_count += 1
                    stable_by_prompt[prompt["id"]].add(strength)
                if h8_seeds >= 2:
                    h8_stable += 1
                component_seed_counts = {}
                for name in component_names:
                    count = sum(row["task_stable"] and row["components"][name]["detected"]
                                and row["components"][name]["statistic"] > 0 for row in rows)
                    component_seed_counts[name] = count
                    if count >= 2:
                        component_stable[name] += 1
                distances.extend(row["raw"]["statistic"] for row in rows)
                prompt_results.append({
                    "prompt_id": prompt["id"], "family": family, "strength": strength,
                    "primary_detected_seeds": primary_seeds, "raw_stable": stable,
                    "h8_detected_seeds": h8_seeds, "h8_stable": h8_seeds >= 2,
                    "component_detected_seeds": component_seed_counts,
                })
            strength_rows.append({
                "family": family, "strength": strength, "endpoint_ids": endpoint_ids,
                "raw_stable_prompts": stable_count, "h8_stable_prompts": h8_stable,
                "component_stable_prompts": component_stable,
                "median_raw_distance": float(np.median(distances)),
            })
        stable_any = {prompt_id for prompt_id, strengths in stable_by_prompt.items() if strengths}
        multi = {prompt_id for prompt_id, strengths in stable_by_prompt.items() if len(strengths) >= 2}
        non_source = {prompt_id for prompt_id in stable_any
                      if any(alias["method"] != "source" for alias in alias_by_prompt[prompt_id])}
        source_ids = {alias["source_id"] for prompt_id in non_source for alias in alias_by_prompt[prompt_id]
                      if alias["method"] != "source"}
        measurable = len(stable_any) >= 2 and len(multi) >= 1
        family_summaries[family] = {
            "stable_any_strength_unique_prompts": len(stable_any),
            "stable_at_two_or_more_strengths_unique_prompts": len(multi),
            "stable_non_source_candidate_prompts": len(non_source),
            "represented_candidate_source_ids": len(source_ids),
            "family_measurable": measurable,
            "stable_prompt_ids": sorted(stable_any),
            "multi_strength_prompt_ids": sorted(multi),
        }
    can_enter_stage4 = all(value["family_measurable"] for value in family_summaries.values())
    null = {
        "raw_detections": sum(row["raw"]["detected"] is True for row in null_rows),
        "h8_detections": sum(row["h8"]["detected"] is True for row in null_rows),
        "units": len(null_rows),
        "formal_fpr_certified": False,
    }
    evidence = {str(path.relative_to(EVAL)): digest(path)
                for folder in ("responses", "decisions", "realizations")
                for path in sorted((EVAL / folder).glob("*")) if path.is_file()}
    summary = {
        "schema": "stage2-26-prompt-behavior-confirmation-summary-v1",
        "status": "ENTER_STAGE4" if can_enter_stage4 else "BLOCK_STAGE4_RECALIBRATE_BEHAVIOR",
        "can_enter_stage4": can_enter_stage4,
        "family_summaries": family_summaries,
        "strength_summaries": strength_rows,
        "null": null,
        "channel_policy": "Raw primary; H8 separate; no OR",
        "prompt_strength_results": prompt_results,
        "response_rows": sum(len(bank(plan, banks, endpoint)) for endpoint in banks),
        "evidence_sha256": evidence,
    }
    freeze_json(EVAL / "SUMMARY.json", summary)
    with (EVAL / "behavior_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["prompt_id", "endpoint", "family", "strength", "attack_seed", "task_stable",
                  "intact_correct", "attack_correct", "raw_distance", "raw_p", "raw_detected",
                  "prefix_distance", "prefix_p", "bigram_distance", "bigram_p", "semantic_distance", "semantic_p",
                  "h8_mmd2", "h8_p", "h8_detected"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in decisions:
            components = row["components"]
            writer.writerow({
                "prompt_id": row["prompt_id"], "endpoint": row["endpoint"], "family": row["family"],
                "strength": row["strength"], "attack_seed": row["attack_seed"], "task_stable": row["task_stable"],
                "intact_correct": row["intact_correct"], "attack_correct": row["attack_correct"],
                "raw_distance": row["raw"]["statistic"], "raw_p": row["raw"]["p_value"],
                "raw_detected": row["raw"]["detected"],
                "prefix_distance": components["prefix10_tv"]["statistic"], "prefix_p": components["prefix10_tv"]["p_value"],
                "bigram_distance": components["bigram_js_with_empty_sentinel"]["statistic"],
                "bigram_p": components["bigram_js_with_empty_sentinel"]["p_value"],
                "semantic_distance": components["half_semantic_centroid_cosine"]["statistic"],
                "semantic_p": components["half_semantic_centroid_cosine"]["p_value"],
                "h8_mmd2": row["h8"]["statistic"], "h8_p": row["h8"]["p_value"],
                "h8_detected": row["h8"]["detected"],
            })
    response_bank_count = len(list((EVAL / "responses").glob("*.jsonl")))
    decision_file_count = len(list((EVAL / "decisions").glob("*.json")))
    realization_file_count = len(list((EVAL / "realizations").glob("*.json")))
    verification = {
        "complete": response_bank_count == 21 and decision_file_count == 19 and realization_file_count == 18,
        "response_banks": response_bank_count,
        "decision_files": decision_file_count,
        "realization_files": realization_file_count,
        "summary_sha256": digest(EVAL / "SUMMARY.json"),
    }
    atomic_json(EVAL / "VERIFICATION.json", verification)
    if verification["response_banks"] != 21 or verification["decision_files"] != 19 or verification["realization_files"] != 18:
        raise RuntimeError(f"Stage3 artifact count mismatch: {verification}")
    status("BEHAVIOR_CONFIRMATION_COMPLETE", result=summary["status"], responses=summary["response_rows"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "generate", "fit", "analyze", "report", "all"])
    args = parser.parse_args()
    EVAL.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (OUT / "RUN.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.phase in {"freeze", "all"} and not (EVAL / "PLAN.json").exists():
                freeze_plan()
            if args.phase == "freeze":
                return
            plan, banks = context()
            if args.phase in {"generate", "all"}:
                generate_all(plan, banks)
                if args.phase == "generate":
                    status("GENERATION_COMPLETE")
                    return
            if args.phase in {"fit", "all"}:
                fit_h8(plan, banks)
                if args.phase == "fit":
                    status("FIT_COMPLETE")
                    return
            if args.phase in {"analyze", "all"}:
                analyze(plan, banks)
                if args.phase == "analyze":
                    status("ANALYSIS_COMPLETE")
                    return
            if args.phase in {"report", "all"}:
                summarize(plan, banks)
        except Exception as exc:
            status("FAILED", error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == "__main__":
    main()
