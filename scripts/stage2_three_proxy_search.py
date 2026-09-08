"""Stages 1, 2 and 4 of the Stage2 three-proxy mechanism experiment.

The experiment is intentionally versioned away from all frozen 2026-09-06
evidence.  Every phase is resumable at a response, calibration-record, or
source/proxy boundary.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import project_path
from llm_integrity.inner_variant_sampler import FAMILIES, StratifiedVariantSampler
from llm_integrity.stage1_r1 import canonical, digest, save_json


CAL = project_path("results/stage2_attack_utility_calibration_v2_20260906")
OLD = project_path("results/stage2_six_source_pilot_20260905")
OUT = project_path("results/stage2_three_proxy_ceiling_v1_20260906")
PROXIES = ("js", "topk_continuous", "raw_logit_l2")
CATEGORIES = ("logic", "math", "extraction", "classification", "instruction", "structured")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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


def setup():
    import torch
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def cleanup():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sha_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def prompt_pool():
    rows = []
    labels = list("甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥天地")
    for i in range(24):
        a, b, c = labels[i], labels[(i + 7) % 24], labels[(i + 13) % 24]
        ask_high = i % 2 == 0
        relation = "高" if i % 4 < 2 else "重"
        target = a if ask_high else c
        question = "最高" if ask_high and relation == "高" else "最重" if ask_high else "最矮" if relation == "高" else "最轻"
        rows.append({"id": f"pool_logic_{i:02d}", "category": "logic", "template": f"chain_{i%4}",
                     "prompt": f"{a}比{b}{relation}，{b}比{c}{relation}。谁{question}？只输出其一个汉字代号。",
                     "expected_answer": target, "evaluator": "token"})
    operations = []
    for i in range(24):
        if i % 3 == 0:
            x, y, op, ans = 17 + i, 23 + 2 * i, "+", 17 + i + 23 + 2 * i
        elif i % 3 == 1:
            x, y, op, ans = 80 + 2 * i, 11 + i, "-", 80 + 2 * i - 11 - i
        else:
            x, y, op, ans = 6 + i // 3, 4 + i % 5, "×", (6 + i // 3) * (4 + i % 5)
        operations.append((x, y, op, ans))
    for i, (x, y, op, ans) in enumerate(operations):
        rows.append({"id": f"pool_math_{i:02d}", "category": "math", "template": f"arithmetic_{i%4}",
                     "prompt": f"计算 {x}{op}{y}。只输出最终整数，不写过程。",
                     "expected_answer": str(ans), "evaluator": "number"})
    words = ["松涛", "远帆", "星河", "青岚", "暮雪", "晨曦", "竹影", "秋水",
             "云海", "霜叶", "山月", "江潮", "石径", "兰舟", "风铃", "雨巷",
             "白鹭", "清泉", "落霞", "长亭", "孤城", "春野", "寒梅", "苍穹"]
    for i, word in enumerate(words):
        marker = ["方括号", "尖括号", "花括号", "双井号"][i % 4]
        wrapped = [f"[{word}]", f"<{word}>", f"{{{word}}}", f"##{word}##"][i % 4]
        rows.append({"id": f"pool_extract_{i:02d}", "category": "extraction", "template": marker,
                     "prompt": f"从字符串“左段{wrapped}右段”中提取{marker}里的两个汉字，只输出提取结果。",
                     "expected_answer": word, "evaluator": "token"})
    for i in range(24):
        n = 31 + i * 3
        answer = "偶" if n % 2 == 0 else "奇"
        rows.append({"id": f"pool_class_{i:02d}", "category": "classification", "template": f"parity_{i%4}",
                     "prompt": f"判断整数 {n} 的奇偶性。偶数只输出“偶”，奇数只输出“奇”。",
                     "expected_answer": answer, "evaluator": "token"})
    phrases = ["春风", "夏雨", "秋月", "冬雪", "青山", "绿水", "朝霞", "晚钟",
               "海棠", "梧桐", "飞鸟", "游鱼", "明灯", "古桥", "新竹", "远山",
               "清茶", "淡墨", "长河", "短歌", "白石", "红叶", "蓝天", "金桂"]
    for i, phrase in enumerate(phrases):
        repeated = phrase * (2 if i % 2 == 0 else 1)
        rows.append({"id": f"pool_instr_{i:02d}", "category": "instruction", "template": f"copy_{i%4}",
                     "prompt": f"严格原样输出“{repeated}”，不要引号、标点或解释。",
                     "expected_answer": repeated, "evaluator": "compact"})
    for i in range(24):
        start = 2 + i
        if i % 2 == 0:
            answer = f"[{start},{start+1},{start+2}]"
            prompt = f"严格输出JSON数组，内容为从{start}开始的三个连续整数。不要代码块或解释。"
            evaluator = "json"
        else:
            key, value = f"k{i}", i + 10
            answer = json.dumps({key: value}, ensure_ascii=False, separators=(",", ":"))
            prompt = f"严格输出一个JSON对象，唯一键为“{key}”，整数值为{value}。不要代码块或解释。"
            evaluator = "json"
        rows.append({"id": f"pool_struct_{i:02d}", "category": "structured", "template": f"json_{i%4}",
                     "prompt": prompt, "expected_answer": answer, "evaluator": evaluator})
    if len(rows) != 144 or len({row["prompt"] for row in rows}) != 144:
        raise RuntimeError("Prompt pool construction is not 144 unique prompts")
    return rows


def normalize(text):
    return re.sub(r"\s+", "", str(text).strip()).replace("，", ",")


def task_pass(text, row):
    expected = str(row["expected_answer"])
    evaluator = row["evaluator"]
    value = str(text).strip()
    if evaluator == "number":
        match = re.search(r"[-+]?\d+", value)
        return bool(match and match.group(0) == expected)
    if evaluator == "json":
        try:
            return json.loads(value) == json.loads(expected)
        except Exception:
            return False
    if evaluator == "token":
        return normalize(value).strip("。.!！\"'“”") == normalize(expected)
    return normalize(value).strip("。.!！\"'“”") == normalize(expected)


def selected_search_variants(calibration, design):
    lookup = {row["variant_id"]: row for row in design["variants"]}
    chosen = []
    # Medium, seed-0 endpoints are used only as development proxies.
    for family in ("gaussian_noise", "finetuning"):
        group = calibration["selected_three_strength_groups"][family][1]
        endpoint = group["endpoint_ids"][0]
        row = dict(lookup[endpoint])
        row["split"] = "development"
        row["strength"] = "medium"
        chosen.append(row)
    for family in ("unstructured_pruning", "structured_pruning", "quantization"):
        endpoint = calibration["selected_coverage"][family]["variant_id"]
        row = dict(lookup[endpoint])
        row["split"] = "development"
        row["strength"] = "coverage"
        chosen.append(row)
    if {row["family"] for row in chosen} != set(FAMILIES):
        raise RuntimeError("Missing calibrated family")
    return chosen


def freeze_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = read(CAL / "CALIBRATION_REPORT.json")
    design = read(CAL / "DESIGN.json")
    if calibration["status"] != "PASS_FOR_STAGE3":
        raise RuntimeError("Attack utility calibration is not valid")
    pool = prompt_pool()
    lora_prompts = {row["prompt"] for row in design["lora_training"]}
    utility_prompts = {row["prompt"] for row in read(CAL / "UTILITY_SET.json")["prompts"]}
    if {row["prompt"] for row in pool} & (lora_prompts | utility_prompts):
        raise RuntimeError("Pool overlaps LoRA or independent utility data")
    source_files = [Path(__file__), project_path("src/llm_integrity/macro_proxy.py"),
                    project_path("src/llm_integrity/discrete_joint_inner_optimizer.py"),
                    project_path("src/llm_integrity/inner_micro_proxy.py"),
                    project_path("src/llm_integrity/inner_variant_sampler.py")]
    payload = {
        "schema": "stage2-three-proxy-ceiling-v1",
        "model": design["model"],
        "proxies": list(PROXIES),
        "pool": pool,
        "pool_size": 144,
        "categories": list(CATEGORIES),
        "preflight": {"repetitions": 4, "minimum_correct": 3, "minimum_modal_response": 3,
                      "retain_per_category": 10, "minimum_per_category": 8,
                      "temperature": .2, "top_p": .9, "top_k": 50,
                      "max_input_tokens": 256, "max_new_tokens": 48},
        "pilot_search": {"sources": 16, "rounds": 5, "probes": 8,
                         "block_types": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         "representative_layers": 4, "candidate_positions": 10,
                         "candidates_per_position": 48, "rerank_candidates": 24,
                         "variants_per_family": 1, "minimum_nondegraded_families": 5,
                         "max_edit_ratio": .40, "semantic_constraints": False,
                         "surface_constraint": False, "perplexity_constraint": False,
                         "task_preservation_during_search": False, "top_k_proxy": 10},
        "formal_search": {"sources": 32, "rounds": 10, "probes": 8,
                          "candidate_positions": 10, "candidates_per_position": 48,
                          "rerank_candidates": 24, "variants_per_family": 1,
                          "minimum_nondegraded_families": 5, "max_edit_ratio": .40},
        "search_variants": selected_search_variants(calibration, design),
        "calibration_report_sha256": digest(CAL / "CALIBRATION_REPORT.json"),
        "calibration_design_sha256": digest(CAL / "DESIGN.json"),
        "old_micro_calibration_sha256": digest(OLD / "CALIBRATION.json"),
        "promotion_rule": {"rank": "mean family-balanced empirical text TV then raw standardized effect",
                           "advance": 2, "minimum_prompts_each_gaussian_lora": 2,
                           "minimum_two_seed_mean_text_tv": .25},
        "separation": "Pool has zero exact overlap with utility and LoRA training prompts; pilot/formal source IDs are frozen before behavior data.",
        "scope": "Mechanism-ceiling arm: semantic/PPL/task preservation disabled during token search; intact stability remains mandatory.",
        "source_hashes": {str(path.relative_to(project_path('.'))).replace('\\', '/'): digest(path) for path in source_files},
    }
    freeze(OUT / "PLAN.json", payload)
    status("PLAN_FROZEN", pool=144, proxies=len(PROXIES))


def context():
    plan = read(OUT / "PLAN.json")
    if digest(CAL / "CALIBRATION_REPORT.json") != plan["calibration_report_sha256"]:
        raise RuntimeError("Calibration report changed")
    if digest(CAL / "DESIGN.json") != plan["calibration_design_sha256"]:
        raise RuntimeError("Calibration design changed")
    for name, expected in plan["source_hashes"].items():
        if digest(project_path(name)) != expected:
            raise RuntimeError(f"Code changed after plan freeze: {name}")
    return plan


def preflight(plan):
    from llm_integrity.modeling import load_model, render_prompt
    torch = setup()
    schedule = [{"prompt_index": i, "response_index": r, "generation_seed": 720_000_000 + r * 144 + i}
                for r in range(plan["preflight"]["repetitions"]) for i in range(144)]
    path = OUT / "preflight_responses.jsonl"
    done = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if len(done) > len(schedule):
        raise RuntimeError("Preflight response overflow")
    bundle = load_model(plan["model"])
    try:
        encodings = {}
        for i, row in enumerate(plan["pool"]):
            rendered = render_prompt(bundle.tokenizer, row["prompt"])
            encoded = bundle.tokenizer(rendered, return_tensors="pt", truncation=False)
            if encoded["input_ids"].shape[1] > plan["preflight"]["max_input_tokens"]:
                raise RuntimeError(f"Pool prompt too long: {row['id']}")
            encodings[i] = {key: value.to(bundle.device) for key, value in encoded.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for index in range(len(done), len(schedule)):
                spec = schedule[index]
                seed = spec["generation_seed"]
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                encoded = encodings[spec["prompt_index"]]
                cfg = plan["preflight"]
                with torch.inference_mode():
                    tokens = bundle.model.generate(**encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=True,
                                                   temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                                                   pad_token_id=bundle.tokenizer.pad_token_id,
                                                   eos_token_id=bundle.tokenizer.eos_token_id)
                ids = tokens[0, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
                text = bundle.tokenizer.decode(ids, skip_special_tokens=True)
                row = plan["pool"][spec["prompt_index"]]
                record = {**spec, "prompt_id": row["id"], "response": text,
                          "task_passed": task_pass(text, row), "plan_sha256": digest(OUT / "PLAN.json")}
                record["record_sha256"] = sha_value(record)
                handle.write(canonical(record) + "\n"); handle.flush()
                if (index + 1) % 48 == 0:
                    os.fsync(handle.fileno())
                    status("PREFLIGHT_GENERATE", completed=index + 1, total=len(schedule))
    finally:
        bundle.close(); cleanup()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["prompt_id"]].append(record)
    audits = []
    by_category = defaultdict(list)
    for row in plan["pool"]:
        values = grouped[row["id"]]
        modal = Counter(normalize(item["response"]) for item in values).most_common(1)[0][1]
        correct = sum(item["task_passed"] for item in values)
        valid = correct >= plan["preflight"]["minimum_correct"] and modal >= plan["preflight"]["minimum_modal_response"]
        item = {**row, "correct": correct, "modal_response_count": modal, "valid": valid}
        audits.append(item)
        if valid:
            by_category[row["category"]].append(item)
    selected = []
    for category in CATEGORIES:
        candidates = by_category[category]
        if len(candidates) < plan["preflight"]["minimum_per_category"]:
            freeze(OUT / "PREFLIGHT_REPORT.json", {"status": "INSUFFICIENT_STABLE_POOL", "audit": audits,
                                                   "valid_by_category": {key: len(value) for key, value in by_category.items()}})
            raise RuntimeError(f"Insufficient stable prompts in {category}: {len(candidates)}")
        candidates.sort(key=lambda x: (-x["correct"], -x["modal_response_count"], x["id"]))
        selected.extend(candidates[:plan["preflight"]["retain_per_category"]])
    pilot_counts = {category: (3 if index < 4 else 2) for index, category in enumerate(CATEGORIES)}
    pilot = []
    formal = []
    for category in CATEGORIES:
        rows = [row for row in selected if row["category"] == category]
        pilot.extend(rows[:pilot_counts[category]])
        formal.extend(rows[:min(len(rows), 6 if category in CATEGORIES[:2] else 5)])
    formal = formal[:32]
    if len(pilot) != 16 or len(formal) != 32:
        raise RuntimeError("Frozen source split has wrong size")
    freeze(OUT / "PREFLIGHT_REPORT.json", {"status": "PASS", "audit": audits,
                                           "valid_by_category": {key: len(value) for key, value in by_category.items()},
                                           "selected": selected, "pilot_sources": pilot, "formal_sources": formal,
                                           "response_sha256": digest(path)})
    status("PREFLIGHT_COMPLETE", stable=len(selected), pilot=16, formal=32)


def loaded_variant(plan, variant):
    from llm_integrity.paper_variant_executor import load_manifest_variant
    adapter = CAL / "adapters" / variant["variant_id"] if variant["family"] == "finetuning" else None
    return load_manifest_variant(plan["model"], variant, adapter_path=str(adapter) if adapter else None)


def calibrate(plan):
    target = OUT / "PROXY_CALIBRATION.json"
    if target.exists():
        return
    torch = setup()
    from llm_integrity.modeling import load_model, tokenize_prompts
    from llm_integrity.macro_proxy import differentiable_macro_proxy
    preflight_report = read(OUT / "PREFLIGHT_REPORT.json")
    texts = [row["prompt"] for row in preflight_report["selected"][:8]]
    reference = load_model(plan["model"])
    records = []
    try:
        for variant in plan["search_variants"]:
            loaded = loaded_variant(plan, variant)
            try:
                for text_index, text in enumerate(texts):
                    encoded = tokenize_prompts(reference, [text], 256)
                    ids = encoded["input_ids"]
                    mask = encoded["attention_mask"]
                    embeddings = reference.model.get_input_embeddings()(ids).detach().requires_grad_(True)
                    a = reference.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits[0, -1].float()
                    b = loaded.bundle.model(inputs_embeds=embeddings, attention_mask=mask, use_cache=False).logits[0, -1].float()
                    for proxy in PROXIES:
                        score = differentiable_macro_proxy(a, b, proxy=proxy, top_k=10)
                        gradient = torch.autograd.grad(score, embeddings, retain_graph=True)[0]
                        records.append({"family": variant["family"], "variant_id": variant["variant_id"],
                                        "text_index": text_index, "proxy": proxy, "score": float(score.item()),
                                        "gradient_norm": float(gradient.float().norm().item()),
                                        "finite": bool(torch.isfinite(gradient).all().item())})
                    del encoded, ids, mask, embeddings, a, b, score, gradient
                    cleanup()
            finally:
                loaded.close(); cleanup()
            save_json(OUT / "proxy_calibration_records.json", records)
            status("PROXY_CALIBRATION", families=len({row['family'] for row in records}), total=5)
    finally:
        reference.close(); cleanup()
    scales = {}
    for proxy in PROXIES:
        scales[proxy] = {}
        for family in FAMILIES:
            values = [row["score"] for row in records if row["proxy"] == proxy and row["family"] == family and row["finite"]]
            positive = [value for value in values if value > 1e-30 and np.isfinite(value)]
            if not positive:
                raise RuntimeError(f"Degenerate calibration: {proxy}/{family}")
            scales[proxy][family] = float(np.median(positive))
    old_micro = read(OLD / "CALIBRATION.json")
    freeze(target, {"schema": "three-proxy-calibration-v1", "records_sha256": digest(OUT / "proxy_calibration_records.json"),
                    "macro_scales": scales,
                    "micro_scales": {key: value["scale"] for key, value in old_micro["micro"].items()},
                    "micro_clips": {key: value["normalized_gradient_norm"]["clip_threshold_log_median_plus_5_mad"]
                                    for key, value in old_micro["micro"].items()},
                    "recommended_micro_weight": old_micro["recommended_micro_weight"]})
    status("PROXY_CALIBRATION_COMPLETE")


def search(plan, formal=False):
    setup()
    from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer
    from llm_integrity.modeling import load_model
    from llm_integrity.stage2_contract import BLOCK_TYPES
    report = read(OUT / "PREFLIGHT_REPORT.json")
    sources = report["formal_sources"] if formal else report["pilot_sources"]
    if formal:
        promotion = read(OUT / "PROMOTION.json")
        proxies = tuple(promotion["promoted_proxies"])
        cfg = plan["formal_search"]
        folder_name = "formal_search"
    else:
        proxies = PROXIES
        cfg = plan["pilot_search"]
        folder_name = "pilot_search"
    calibration = read(OUT / "PROXY_CALIBRATION.json")
    registry = {row["variant_id"]: str(CAL / "adapters" / row["variant_id"])
                for row in plan["search_variants"] if row["family"] == "finetuning"}
    for proxy_index, proxy in enumerate(proxies):
        folder = OUT / folder_name / proxy
        folder.mkdir(parents=True, exist_ok=True)
        sampler = StratifiedVariantSampler(plan["search_variants"], family_weights={family: .2 for family in FAMILIES},
                                           adapter_registry=registry, seed=730_000_000 + proxy_index * 10000)
        for source_index, row in enumerate(sources):
            target = folder / f"{row['id']}.json"
            if target.exists():
                continue
            bundle = load_model(plan["model"])
            optimizer = None
            started = time.time()
            try:
                optimizer = DiscreteJointInnerOptimizer(
                    reference=bundle, sampler=sampler, model_config=plan["model"],
                    micro_scales={name: calibration["micro_scales"][name] for name in BLOCK_TYPES},
                    macro_scales=calibration["macro_scales"][proxy],
                    micro_component_clips={name: calibration["micro_clips"][name] for name in BLOCK_TYPES},
                    macro_component_clips={}, micro_weight=calibration["recommended_micro_weight"], macro_weight=1.0,
                    rounds=cfg["rounds"], candidate_positions=cfg["candidate_positions"],
                    candidates_per_position=cfg["candidates_per_position"], rerank_candidates=cfg["rerank_candidates"],
                    probes=cfg["probes"], seed=740_000_000 + proxy_index * 1_000_000 + source_index * 10000,
                    max_length=128, max_edit_ratio=cfg["max_edit_ratio"], ppl_ratio_limit=1e12,
                    minimum_nondegraded_families=cfg["minimum_nondegraded_families"], family_relative_tolerance=.01,
                    variants_per_family=cfg["variants_per_family"], anchor_variants_per_family=0,
                    gradient_restarts=1, family_gate_aggregation="mean", require_task_preservation=False,
                    sequential_model_execution=False, block_types=tuple(BLOCK_TYPES), representative_layer_count=4,
                    block_schedule="balanced", task_validation_mode="strict_r1", macro_proxy=proxy, macro_top_k=10,
                    enforce_surface_compatibility=False, enforce_perplexity=False)
                status("FORMAL_SEARCH" if formal else "PILOT_SEARCH", proxy=proxy,
                       source=source_index + 1, total=len(sources))
                result = optimizer.optimize(row)
                save_json(target, {**result.payload(), "proxy": proxy, "source": row,
                                   "plan_sha256": digest(OUT / "PLAN.json"), "seconds": time.time() - started})
                status("SEARCH_RESULT", stage="formal" if formal else "pilot", proxy=proxy,
                       source=source_index + 1, accepted=result.accepted, seconds=time.time() - started,
                       failure=result.failure)
                if result.failure:
                    raise RuntimeError(f"Search technical failure: {result.failure}")
            finally:
                (optimizer.reference if optimizer is not None else bundle).close(); cleanup()
    status("FORMAL_SEARCH_COMPLETE" if formal else "PILOT_SEARCH_COMPLETE",
           proxies=len(proxies), sources=len(sources))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "preflight", "calibrate", "pilot-search", "formal-search"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (OUT / "SEARCH.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.phase == "freeze":
                freeze_plan(); return
            plan = context()
            if args.phase == "preflight": preflight(plan)
            elif args.phase == "calibrate": calibrate(plan)
            elif args.phase == "pilot-search": search(plan, False)
            elif args.phase == "formal-search": search(plan, True)
        except Exception as exc:
            status("FAILED", requested_phase=args.phase, error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == "__main__":
    main()
