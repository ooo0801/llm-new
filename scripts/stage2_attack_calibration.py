"""Stage2 attack-strength calibration on a frozen independent 32-task utility set.

This program deliberately never loads H6 candidate responses when selecting attacks.
It is resumable at the preflight repetition, LoRA adapter, and attack endpoint levels.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical, digest, evaluate_task_r1, save_json
from stage2_prepare import freeze, training_data


OUT = project_path("results/stage2_attack_utility_calibration_v2_20260906")
MODEL_CONFIG_PATH = project_path("configs/stage2_qwen15b.json")
OLD_STAGE2 = project_path("results/stage2_six_source_pilot_20260905")
PREFLIGHT_REPETITIONS = 4
SELECT_PER_CATEGORY = 8


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def status(phase: str, **fields):
    payload = {"phase": phase, "utc_unix": time.time(), **fields}
    save_json(OUT / "STATUS.json", payload)
    print(canonical(payload), flush=True)


def setup():
    import torch

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def cleanup():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def utility_pool():
    rows = []

    logic = [
        ("甲比乙高，乙比丙高。谁最矮？只输出一个汉字。", "丙"),
        ("丁比戊重，戊比己重。谁最重？只输出一个汉字。", "丁"),
        ("春比夏早，夏比秋早。谁最晚？只输出一个汉字。", "秋"),
        ("东比南远，南比西远。谁最近？只输出一个汉字。", "西"),
        ("红盒比蓝盒轻，蓝盒比绿盒轻。哪个盒子最重？只输出颜色。", "绿"),
        ("松树比柏树高，柏树比柳树高。哪种树最矮？只输出树名。", "柳树"),
        ("一号比二号快，二号比三号快。哪个号码最慢？只输出中文号码。", "三号"),
        ("小安比小北年长，小北比小辰年长。谁最年轻？只输出名字。", "小辰"),
        ("铜球比铁球轻，铁球比木球重，木球比铜球重。哪个球最轻？只输出名称。", "铜球"),
        ("A排在B前面，B排在C前面。谁排在最后？只输出字母。", "C"),
        ("猫比狗轻，狗比羊轻。哪种动物最重？只输出动物名。", "羊"),
        ("书架甲比书架乙宽，书架乙比书架丙宽。最窄的是哪个？只输出代号。", "丙"),
        ("河流甲比河流乙长，河流乙比河流丙长。最长的是哪条？只输出代号。", "甲"),
        ("灯A比灯B亮，灯B比灯C亮。最暗的是哪盏？只输出字母。", "C"),
        ("班级一比班级二人数多，班级二比班级三人数多。人数最少的是？只输出中文数字。", "三"),
        ("温度甲低于乙，乙低于丙。温度最高的是谁？只输出代号。", "丙"),
    ]
    math_rows = [
        ("计算1+2，只输出整数。", "3"),
        ("计算3+4，只输出整数。", "7"),
        ("计算5+3，只输出整数。", "8"),
        ("计算6+7，只输出整数。", "13"),
        ("计算9-3，只输出整数。", "6"),
        ("计算8-2，只输出整数。", "6"),
        ("计算12-5，只输出整数。", "7"),
        ("计算15-6，只输出整数。", "9"),
        ("计算2×3，只输出整数。", "6"),
        ("计算3×4，只输出整数。", "12"),
        ("计算5×4，只输出整数。", "20"),
        ("计算6×5，只输出整数。", "30"),
        ("计算8÷2，只输出整数。", "4"),
        ("计算9÷3，只输出整数。", "3"),
        ("计算12÷4，只输出整数。", "3"),
        ("计算15÷5，只输出整数。", "3"),
    ]
    instruction = [
        ("请原样输出：竹影清风。不要添加标点或解释。", "竹影清风"),
        ("请原样输出：远山近水。不要添加标点或解释。", "远山近水"),
        ("只输出三个汉字：松竹梅。", "松竹梅"),
        ("只输出四个汉字：琴棋书画。", "琴棋书画"),
        ("严格输出从4开始的4个连续整数，用英文逗号分隔。", "4,5,6,7"),
        ("严格输出从7开始的3个连续整数，用英文逗号分隔。", "7,8,9"),
        ("把“蓝色天空”原样输出一次，不添加其他内容。", "蓝色天空"),
        ("把“实验记录”原样输出一次，不添加其他内容。", "实验记录"),
        ("只输出字母XYZ，保持大写。", "XYZ"),
        ("只输出小写字母abc。", "abc"),
        ("按此顺序输出甲乙丙，中间不加空格。", "甲乙丙"),
        ("按此顺序输出春秋冬，中间不加空格。", "春秋冬"),
        ("将数字5重复三次，不加空格或标点。", "555"),
        ("将汉字“云”重复两次，不加空格或标点。", "云云"),
        ("只回答“同意”两个汉字。", "同意"),
        ("只回答“完成”两个汉字。", "完成"),
    ]
    structured = [
        ('输出一个JSON对象：name为"cat"、level为1、enabled为true。不要解释。', '{"name":"cat","level":1,"enabled":true}'),
        ('输出一个JSON对象：name为"dog"、level为2、enabled为false。不要解释。', '{"name":"dog","level":2,"enabled":false}'),
        ('输出一个JSON对象：name为"red"、level为3、enabled为true。不要解释。', '{"name":"red","level":3,"enabled":true}'),
        ('输出一个JSON对象：name为"blue"、level为4、enabled为false。不要解释。', '{"name":"blue","level":4,"enabled":false}'),
        ('输出一个JSON对象：name为"east"、level为5、enabled为true。不要解释。', '{"name":"east","level":5,"enabled":true}'),
        ('输出一个JSON对象：name为"west"、level为6、enabled为false。不要解释。', '{"name":"west","level":6,"enabled":false}'),
        ('输出一个JSON对象：name为"sun"、level为7、enabled为true。不要解释。', '{"name":"sun","level":7,"enabled":true}'),
        ('输出一个JSON对象：name为"moon"、level为8、enabled为false。不要解释。', '{"name":"moon","level":8,"enabled":false}'),
        ('输出一个JSON对象：name为"alpha"、level为9、enabled为true。不要解释。', '{"name":"alpha","level":9,"enabled":true}'),
        ('输出一个JSON对象：name为"beta"、level为10、enabled为false。不要解释。', '{"name":"beta","level":10,"enabled":false}'),
        ('输出一个JSON对象：name为"north"、level为11、enabled为true。不要解释。', '{"name":"north","level":11,"enabled":true}'),
        ('输出一个JSON对象：name为"south"、level为12、enabled为false。不要解释。', '{"name":"south","level":12,"enabled":false}'),
        ('输出一个JSON对象：name为"one"、level为13、enabled为true。不要解释。', '{"name":"one","level":13,"enabled":true}'),
        ('输出一个JSON对象：name为"two"、level为14、enabled为false。不要解释。', '{"name":"two","level":14,"enabled":false}'),
        ('输出一个JSON对象：name为"green"、level为15、enabled为true。不要解释。', '{"name":"green","level":15,"enabled":true}'),
        ('输出一个JSON对象：name为"white"、level为16、enabled为false。不要解释。', '{"name":"white","level":16,"enabled":false}'),
    ]

    for category, pairs in (
        ("logic", logic),
        ("math", math_rows),
        ("instruction", instruction),
        ("structured", structured),
    ):
        for index, (prompt, answer) in enumerate(pairs):
            rows.append({
                "id": f"s2cal_{category}_{index:02d}",
                "prompt": prompt,
                "expected_answer": answer,
                "category": category,
                "evaluator": "json_semantic" if category == "structured" else ("numeric" if category == "math" else "exact"),
            })
    return rows


def attack_grid():
    variants = []

    gaussian_levels = [.001, .002, .003, .004, .005, .006, .008]
    for level_index, sigma in enumerate(gaussian_levels):
        for seed_index in range(3):
            variants.append({
                "variant_id": f"cal_gaussian_g{level_index:02d}_s{seed_index}",
                "family": "gaussian_noise",
                "grid_index": level_index,
                "intensity": sigma,
                "seed": 20269100 + level_index * 10 + seed_index,
                "configuration": {
                    "method": "relative_gaussian",
                    "std_ratio": sigma,
                    "target_scope": "attention_ffn",
                },
            })

    lora_levels = [
        (20, 1.0e-4),
        (40, 1.0e-4),
        (60, 1.0e-4),
        (40, 1.5e-4),
        (60, 1.5e-4),
    ]
    for level_index, (steps, learning_rate) in enumerate(lora_levels):
        for seed_index in range(3):
            variants.append({
                "variant_id": f"cal_lora_g{level_index:02d}_s{seed_index}",
                "family": "finetuning",
                "grid_index": level_index,
                "intensity": steps * learning_rate,
                "seed": 20269200 + level_index * 10 + seed_index,
                "configuration": {
                    "method": "lora",
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "learning_rate": learning_rate,
                    "steps": steps,
                    "target_scope": "attention_ffn",
                },
            })

    for level_index, ratio in enumerate([.08, .10, .12, .14, .16, .18, .20]):
        variants.append({
            "variant_id": f"cal_unstructured_g{level_index:02d}",
            "family": "unstructured_pruning",
            "grid_index": level_index,
            "intensity": ratio,
            "seed": 20269300 + level_index,
            "configuration": {
                "method": "layerwise_magnitude",
                "ratio": ratio,
                "target_scope": "attention_ffn",
            },
        })

    for level_index, ratio in enumerate([.005, .01, .02, .03, .04, .05]):
        variants.append({
            "variant_id": f"cal_structured_g{level_index:02d}",
            "family": "structured_pruning",
            "grid_index": level_index,
            "intensity": ratio,
            "seed": 20269400 + level_index,
            "configuration": {
                "structure": "ffn_channels",
                "ratio": ratio,
                "selection": "magnitude",
                "layer_scope": "all_layers",
                "implementation": "mask",
            },
        })

    for level_index, threshold in enumerate([4.0, 6.0, 8.0]):
        variants.append({
            "variant_id": f"cal_int8_t{int(threshold)}",
            "family": "quantization",
            "grid_index": level_index,
            "intensity": threshold,
            "seed": 20269500 + level_index,
            "configuration": {
                "method": "int8",
                "llm_int8_threshold": threshold,
                "compute_dtype": "bfloat16",
                "target_scope": "full_model",
                "double_quant": False,
            },
        })
    variants.append({
        "variant_id": "cal_nf4_destructive_control",
        "family": "quantization",
        "grid_index": 99,
        "intensity": 99.0,
        "selection_role": "destructive_control_only",
        "seed": 20269599,
        "configuration": {
            "method": "nf4",
            "compute_dtype": "bfloat16",
            "target_scope": "full_model",
            "double_quant": True,
        },
    })
    return variants


def freeze_design():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = read(MODEL_CONFIG_PATH)
    pool = utility_pool()
    variants = attack_grid()
    lora_rows = training_data()
    old_design = read(OLD_STAGE2 / "DESIGN.json")
    excluded = (
        [row["prompt"] for row in old_design["prompts"]]
        + [row["prompt"] for row in old_design["lora_training"]]
        + [row["prompt"] for row in old_design["utility"]]
    )
    pool_prompts = [row["prompt"] for row in pool]
    if len(pool_prompts) != len(set(pool_prompts)):
        raise RuntimeError("Duplicate prompt in utility candidate pool")
    overlap = sorted(set(pool_prompts) & set(excluded))
    if overlap:
        raise RuntimeError(f"Utility role overlap: {overlap}")
    payload = {
        "schema": "stage2-attack-utility-calibration-v2",
        "purpose": "attack selection from independent task utility only; H6 behavior is unavailable to the selector",
        "model": cfg["model"],
        "generation": {
            "preflight": {"do_sample": True, "temperature": .2, "top_p": .9, "top_k": 50,
                          "max_input_tokens": 256, "max_new_tokens": 48},
            "utility": {"do_sample": False, "max_input_tokens": 256, "max_new_tokens": 48},
        },
        "utility_pool": pool,
        "preflight_repetitions": PREFLIGHT_REPETITIONS,
        "preflight_minimum_passes": 3,
        "select_per_category": SELECT_PER_CATEGORY,
        "final_utility_size": 32,
        "utility_categories": ["logic", "math", "instruction", "structured"],
        "utility_gate": {
            "minimum_intact_correct": 24,
            "maximum_absolute_loss_from_intact": 2,
            "gaussian_lora_group_policy": "all three independently realized seeds must pass",
            "coverage_policy": "each deterministic configuration checked independently",
        },
        "selection": {
            "gaussian_noise": "three highest sigma groups for which all three seeds pass utility",
            "finetuning": "three highest predeclared update-budget groups for which all three seeds pass utility",
            "unstructured_pruning": "highest ratio passing utility",
            "structured_pruning": "highest ratio passing utility",
            "quantization": "highest INT8 threshold passing utility; NF4 is never selectable and remains destructive control",
        },
        "variants": variants,
        "lora_training": lora_rows,
        "exclusion": {
            "old_stage2_design_sha256": digest(OLD_STAGE2 / "DESIGN.json"),
            "excluded_prompt_count": len(set(excluded)),
            "exact_prompt_overlap_count": 0,
            "roles": ["H6 source prompts", "LoRA training prompts", "old utility prompts"],
        },
        "code_sha256": {
            str(path.relative_to(project_path("."))).replace("\\", "/"): digest(path)
            for path in [
                project_path("src/llm_integrity/modeling.py"),
                project_path("src/llm_integrity/stage1_r1.py"),
                project_path("src/llm_integrity/paper_variant_executor.py"),
                project_path("src/llm_integrity/paper_in_memory_attacks.py"),
                project_path("src/llm_integrity/paper_quantization.py"),
                project_path("src/llm_integrity/paper_finetuning.py"),
            ]
        },
        "driver_sha256": digest(__file__),
    }
    freeze(OUT / "DESIGN.json", payload)
    train_path = OUT / "lora_train.jsonl"
    train_text = "".join(canonical(row) + "\n" for row in lora_rows)
    if train_path.exists() and train_path.read_text(encoding="utf-8") != train_text:
        raise RuntimeError("LoRA training data changed")
    if not train_path.exists():
        train_path.write_text(train_text, encoding="utf-8")
    return payload


def get_design():
    design = read(OUT / "DESIGN.json")
    if digest(__file__) != design["driver_sha256"]:
        raise RuntimeError("Calibration driver changed after design freeze")
    for name, expected in design["code_sha256"].items():
        if digest(project_path(name)) != expected:
            raise RuntimeError(f"Code changed after design freeze: {name}")
    return design


def batches(rows, size=8):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def evaluate_utility(text, row):
    if row.get("evaluator") != "json_semantic":
        return evaluate_task_r1(text, row)

    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.S | re.I)
    if fenced:
        stripped = fenced.group(1).strip()

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        actual = json.loads(stripped, object_pairs_hook=unique_object,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        expected = json.loads(row["expected_answer"], object_pairs_hook=unique_object)
        valid = actual == expected
    except (ValueError, TypeError, json.JSONDecodeError):
        valid = False
    return {
        "status": "pass" if valid else "fail",
        "passed": valid,
        "proxy_passed": None,
        "evaluation_scope": "parsed_exact_json_semantics",
        "reason": "exact_keys_values_types; whitespace_and_full_json_fence_ignored",
        "evaluator": "json_semantic",
    }


def generate_rows(bundle, prompts, generation, seed, endpoint, repetition=None):
    from llm_integrity.modeling import generate_texts

    output = []
    for batch_index, batch in enumerate(batches(prompts, 8)):
        metadata = generate_texts(
            bundle,
            [row["prompt"] for row in batch],
            generation,
            seed=seed + batch_index,
            return_metadata=True,
        )
        for row, generated in zip(batch, metadata, strict=True):
            evaluation = evaluate_utility(generated["text"], row)
            output.append({
                "prompt_id": row["id"],
                "category": row["category"],
                "endpoint": endpoint,
                "repetition": repetition,
                "seed": seed + batch_index,
                "response": generated["text"],
                "token_count": generated["token_count"],
                "truncated": generated["truncated"],
                "task_passed": evaluation.get("passed"),
                "evaluation": evaluation,
            })
    return output


def run_preflight(design):
    target = OUT / "UTILITY_SET.json"
    if target.exists():
        return read(target)
    setup()
    from llm_integrity.modeling import load_model

    record_path = OUT / "preflight_records.json"
    records = read(record_path) if record_path.exists() else []
    finished = {int(row["repetition"]) for row in records}
    bundle = load_model(design["model"])
    try:
        for repetition in range(design["preflight_repetitions"]):
            if repetition in finished:
                continue
            new_rows = generate_rows(
                bundle,
                design["utility_pool"],
                design["generation"]["preflight"],
                seed=20269600 + repetition * 100,
                endpoint="intact_preflight",
                repetition=repetition,
            )
            records.extend(new_rows)
            save_json(record_path, records)
            status("utility_preflight", repetition=repetition + 1,
                   total=design["preflight_repetitions"], completed_rows=len(records))
    finally:
        bundle.close()
        cleanup()

    by_id = defaultdict(list)
    row_by_id = {row["id"]: row for row in design["utility_pool"]}
    for row in records:
        by_id[row["prompt_id"]].append(row)
    selected = []
    diagnostics = []
    for prompt_id, rows in by_id.items():
        passed = sum(item["task_passed"] is True for item in rows)
        diagnostics.append({
            "prompt_id": prompt_id,
            "category": row_by_id[prompt_id]["category"],
            "passes": passed,
            "repetitions": len(rows),
            "truncations": sum(bool(item["truncated"]) for item in rows),
            "mean_tokens": sum(item["token_count"] for item in rows) / len(rows),
        })
    shortages = {}
    for category in design["utility_categories"]:
        eligible = [item for item in diagnostics
                    if item["category"] == category
                    and item["passes"] >= design["preflight_minimum_passes"]
                    and item["truncations"] == 0]
        eligible.sort(key=lambda item: (-item["passes"], item["mean_tokens"], item["prompt_id"]))
        if len(eligible) < design["select_per_category"]:
            shortages[category] = len(eligible)
            continue
        selected.extend(row_by_id[item["prompt_id"]] for item in eligible[:design["select_per_category"]])
    if shortages:
        report = {"status": "INSUFFICIENT_STABLE_UTILITY_PROMPTS", "shortages": shortages,
                  "diagnostics": diagnostics, "design_sha256": digest(OUT / "DESIGN.json")}
        freeze(OUT / "PREFLIGHT_FAILURE.json", report)
        raise RuntimeError(f"Utility preflight shortage: {shortages}")
    selected.sort(key=lambda row: (design["utility_categories"].index(row["category"]), row["id"]))
    payload = {
        "status": "FROZEN",
        "prompts": selected,
        "count": len(selected),
        "category_counts": {category: sum(row["category"] == category for row in selected)
                            for category in design["utility_categories"]},
        "selection_diagnostics": diagnostics,
        "preflight_records_sha256": digest(record_path),
        "design_sha256": digest(OUT / "DESIGN.json"),
    }
    if len(selected) != design["final_utility_size"]:
        raise RuntimeError("Frozen utility set has wrong size")
    freeze(target, payload)
    return payload


def train_adapters(design):
    setup()
    from llm_integrity.paper_finetuning import train_lora_manifest_variant

    (OUT / "adapters").mkdir(exist_ok=True)
    data_path = OUT / "lora_train.jsonl"
    variants = [row for row in design["variants"] if row["family"] == "finetuning"]
    for index, variant in enumerate(variants):
        folder = OUT / "adapters" / variant["variant_id"]
        report_path = folder / "training_report.json"
        if report_path.exists():
            report = read(report_path)
            if report["data_sha256"] != digest(data_path):
                raise RuntimeError("LoRA adapter data hash mismatch")
            if report["completed_steps"] != variant["configuration"]["steps"]:
                raise RuntimeError("LoRA adapter step mismatch")
            continue
        status("training_lora", completed=index, total=len(variants),
               variant_id=variant["variant_id"], steps=variant["configuration"]["steps"])
        train_lora_manifest_variant(
            design["model"], variant, data_path=data_path,
            output_root=OUT / "adapters", max_length=128,
            batch_size=1, gradient_accumulation_steps=4,
        )
        cleanup()


def loaded_variant(design, variant):
    from llm_integrity.paper_variant_executor import load_manifest_variant

    adapter = OUT / "adapters" / variant["variant_id"] if variant["family"] == "finetuning" else None
    return load_manifest_variant(design["model"], variant,
                                 adapter_path=str(adapter) if adapter is not None else None)


def endpoint_result_path(endpoint):
    return OUT / "utility_endpoints" / f"{endpoint}.json"


def run_utility(design, utility_set):
    setup()
    from llm_integrity.modeling import load_model

    endpoint_root = OUT / "utility_endpoints"
    endpoint_root.mkdir(exist_ok=True)
    baseline_path = endpoint_result_path("intact")
    if not baseline_path.exists():
        bundle = load_model(design["model"])
        try:
            rows = generate_rows(bundle, utility_set["prompts"], design["generation"]["utility"],
                                 seed=20269700, endpoint="intact")
            freeze(baseline_path, {"endpoint": "intact", "rows": rows,
                                   "design_sha256": digest(OUT / "DESIGN.json"),
                                   "utility_set_sha256": digest(OUT / "UTILITY_SET.json")})
        finally:
            bundle.close()
            cleanup()
    baseline = read(baseline_path)
    intact_correct = sum(row["task_passed"] is True for row in baseline["rows"])
    if intact_correct < design["utility_gate"]["minimum_intact_correct"]:
        raise RuntimeError(f"Frozen intact utility failed: {intact_correct}/32")

    variants = design["variants"]
    for index, variant in enumerate(variants):
        target = endpoint_result_path(variant["variant_id"])
        if target.exists():
            continue
        status("attack_utility", completed=index, total=len(variants),
               variant_id=variant["variant_id"], family=variant["family"])
        loaded = loaded_variant(design, variant)
        try:
            rows = generate_rows(loaded.bundle, utility_set["prompts"], design["generation"]["utility"],
                                 seed=20269800 + index * 10, endpoint=variant["variant_id"])
            freeze(target, {
                "endpoint": variant["variant_id"],
                "variant": variant,
                "realization": asdict(loaded.report),
                "rows": rows,
                "design_sha256": digest(OUT / "DESIGN.json"),
                "utility_set_sha256": digest(OUT / "UTILITY_SET.json"),
            })
        finally:
            loaded.close()
            cleanup()


def choose_group(eligible_groups, count=3):
    ordered = sorted(eligible_groups, key=lambda row: (row["intensity"], row["grid_index"]))
    return ordered[-count:] if len(ordered) >= count else []


def evaluate_and_select(design, utility_set):
    target = OUT / "CALIBRATION_REPORT.json"
    if target.exists():
        return read(target)
    baseline = read(endpoint_result_path("intact"))
    intact_by_id = {row["prompt_id"]: row["task_passed"] is True for row in baseline["rows"]}
    intact_correct = sum(intact_by_id.values())
    endpoint_results = []
    by_variant = {variant["variant_id"]: variant for variant in design["variants"]}
    for variant_id, variant in by_variant.items():
        endpoint = read(endpoint_result_path(variant_id))
        attack_by_id = {row["prompt_id"]: row["task_passed"] is True for row in endpoint["rows"]}
        attacked_correct = sum(attack_by_id.values())
        retained = sum(attack_by_id[prompt_id] for prompt_id, passed in intact_by_id.items() if passed)
        loss = intact_correct - retained
        category = {}
        for name in design["utility_categories"]:
            ids = [row["id"] for row in utility_set["prompts"] if row["category"] == name]
            category[name] = {
                "intact_correct": sum(intact_by_id[item] for item in ids),
                "attack_correct": sum(attack_by_id[item] for item in ids),
                "retained_intact_correct": sum(attack_by_id[item] for item in ids if intact_by_id[item]),
            }
        endpoint_results.append({
            "variant_id": variant_id,
            "family": variant["family"],
            "grid_index": variant["grid_index"],
            "intensity": variant["intensity"],
            "seed": variant["seed"],
            "intact_correct": intact_correct,
            "attack_correct": attacked_correct,
            "retained_intact_correct": retained,
            "absolute_loss_from_intact": loss,
            "qualified": loss <= design["utility_gate"]["maximum_absolute_loss_from_intact"],
            "category": category,
            "configuration": variant["configuration"],
            "selection_role": variant.get("selection_role", "candidate"),
        })

    grouped = defaultdict(list)
    for row in endpoint_results:
        if row["family"] in {"gaussian_noise", "finetuning"}:
            grouped[(row["family"], row["grid_index"])].append(row)
    group_results = []
    for (family, grid_index), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["seed"])
        group_results.append({
            "family": family,
            "grid_index": grid_index,
            "intensity": rows[0]["intensity"],
            "endpoint_ids": [row["variant_id"] for row in rows],
            "qualified_seeds": sum(row["qualified"] for row in rows),
            "all_three_seeds_qualified": len(rows) == 3 and all(row["qualified"] for row in rows),
            "correct_counts": [row["attack_correct"] for row in rows],
            "losses": [row["absolute_loss_from_intact"] for row in rows],
            "configuration": rows[0]["configuration"],
        })

    selected_groups = {}
    for family in ("gaussian_noise", "finetuning"):
        eligible = [row for row in group_results if row["family"] == family and row["all_three_seeds_qualified"]]
        chosen = choose_group(eligible, 3)
        labels = ["weak", "medium", "strong"]
        selected_groups[family] = [
            {**row, "selected_strength": labels[index]}
            for index, row in enumerate(chosen)
        ]

    selected_coverage = {}
    for family in ("unstructured_pruning", "structured_pruning"):
        eligible = [row for row in endpoint_results if row["family"] == family and row["qualified"]]
        selected_coverage[family] = max(eligible, key=lambda row: row["intensity"], default=None)
    eligible_int8 = [row for row in endpoint_results
                     if row["family"] == "quantization" and row["qualified"]
                     and row["configuration"]["method"] == "int8"]
    selected_coverage["quantization"] = max(eligible_int8, key=lambda row: row["intensity"], default=None)
    nf4 = next(row for row in endpoint_results if row["variant_id"] == "cal_nf4_destructive_control")

    can_proceed = (
        intact_correct >= design["utility_gate"]["minimum_intact_correct"]
        and len(selected_groups["gaussian_noise"]) == 3
        and len(selected_groups["finetuning"]) == 3
    )
    payload = {
        "schema": "stage2-attack-utility-calibration-report-v2",
        "status": "PASS_FOR_STAGE3" if can_proceed else "BLOCK_STAGE3",
        "can_proceed_stage3": can_proceed,
        "intact_correct": intact_correct,
        "utility_total": len(utility_set["prompts"]),
        "category_counts": utility_set["category_counts"],
        "endpoint_results": endpoint_results,
        "three_seed_group_results": group_results,
        "selected_three_strength_groups": selected_groups,
        "selected_coverage": selected_coverage,
        "nf4_destructive_control": nf4,
        "selection_used_h6_behavior": False,
        "selection_contract": design["selection"],
        "design_sha256": digest(OUT / "DESIGN.json"),
        "utility_set_sha256": digest(OUT / "UTILITY_SET.json"),
    }
    freeze(target, payload)
    return payload


def verify(design, utility_set, report):
    hashes = {}
    for path in sorted(OUT.rglob("*.json")):
        if path.name in {"STATUS.json", "VERIFICATION.json"}:
            continue
        hashes[str(path.relative_to(OUT))] = digest(path)
    expected_endpoints = {"intact"} | {row["variant_id"] for row in design["variants"]}
    actual_endpoints = {path.stem for path in (OUT / "utility_endpoints").glob("*.json")}
    payload = {
        "schema": "stage2-attack-utility-calibration-verification-v2",
        "complete": expected_endpoints == actual_endpoints,
        "expected_endpoints": len(expected_endpoints),
        "actual_endpoints": len(actual_endpoints),
        "missing_endpoints": sorted(expected_endpoints - actual_endpoints),
        "utility_count": utility_set["count"],
        "report_status": report["status"],
        "artifact_sha256": hashes,
    }
    save_json(OUT / "VERIFICATION.json", payload)
    if not payload["complete"]:
        raise RuntimeError(f"Endpoint verification failed: {payload['missing_endpoints']}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "preflight", "train", "utility", "report", "all"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (OUT / "RUN.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            design = freeze_design() if args.phase in {"freeze", "all"} and not (OUT / "DESIGN.json").exists() else get_design()
            if args.phase == "freeze":
                status("DESIGN_FROZEN")
                return
            utility_set = run_preflight(design) if args.phase in {"preflight", "all"} else read(OUT / "UTILITY_SET.json")
            if args.phase == "preflight":
                status("UTILITY_SET_FROZEN", count=utility_set["count"])
                return
            if args.phase in {"train", "all"}:
                train_adapters(design)
                if args.phase == "train":
                    status("LORA_TRAINING_COMPLETE")
                    return
            if args.phase in {"utility", "all"}:
                run_utility(design, utility_set)
                if args.phase == "utility":
                    status("UTILITY_ENDPOINTS_COMPLETE")
                    return
            if args.phase in {"report", "all"}:
                report = evaluate_and_select(design, utility_set)
                verification = verify(design, utility_set, report)
                status("CALIBRATION_COMPLETE", report_status=report["status"],
                       can_proceed_stage3=report["can_proceed_stage3"],
                       endpoints=verification["actual_endpoints"])
        except Exception as exc:
            status("FAILED", error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == "__main__":
    main()
