"""Stage2 three-proxy search v2: repair the v1 prompt-pool preflight only.

The 120 unchanged pool prompts reuse their already generated v1 responses with
explicit provenance.  Only 24 replacement logic prompts receive new model
calls.  Search, calibration and optimizer code remain the v1 implementation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import stage2_three_proxy_search as base
from _bootstrap import project_path
from llm_integrity.stage1_r1 import digest, save_json


V1 = project_path("results/stage2_three_proxy_ceiling_v1_20260906")
OUT = project_path("results/stage2_three_proxy_ceiling_v2_20260906")
base.OUT = OUT
ORIGINAL_PROMPT_POOL = base.prompt_pool


def logic_rows():
    triples = [
        ("小明", "小红", "小刚"), ("小林", "小王", "小周"),
        ("小李", "小陈", "小赵"), ("小杨", "小刘", "小孙"),
        ("甲队", "乙队", "丙队"), ("红盒", "蓝盒", "绿盒"),
        ("松树", "柏树", "柳树"), ("东站", "西站", "南站"),
    ]
    rows = []
    relations = [
        ("高", "最高", "最矮"), ("重", "最重", "最轻"),
        ("快", "最快", "最慢"),
    ]
    index = 0
    for relation, upper, lower in relations:
        for triple_index, (a, b, c) in enumerate(triples):
            ask_upper = (triple_index + index) % 2 == 0
            question = upper if ask_upper else lower
            answer = a if ask_upper else c
            rows.append({"id": f"pool_logic_v2_{index:02d}", "category": "logic",
                         "template": f"familiar_chain_{relation}_{triple_index%4}",
                         "prompt": f"{a}比{b}{relation}，{b}比{c}{relation}。谁{question}？只回答名称，不解释。",
                         "expected_answer": answer, "evaluator": "token"})
            index += 1
    return rows


def prompt_pool():
    rows = [row for row in ORIGINAL_PROMPT_POOL() if row["category"] != "logic"]
    rows = logic_rows() + rows
    if len(rows) != 144 or len({row["prompt"] for row in rows}) != 144:
        raise RuntimeError("v2 pool is not 144 unique prompts")
    return rows


def strip_code_fence(text):
    value = str(text).strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    return value.strip()


def task_pass(text, row):
    value = str(text).strip()
    expected = str(row["expected_answer"])
    evaluator = row["evaluator"]
    if evaluator == "number":
        values = re.findall(r"[-+]?\d+", value)
        return bool(values and values[-1] == expected)
    if evaluator == "json":
        try:
            return json.loads(strip_code_fence(value)) == json.loads(expected)
        except Exception:
            return False
    compact = base.normalize(value).strip("。.!！\"'“”")
    target = base.normalize(expected)
    if row["category"] == "classification":
        return compact.startswith(target)
    if evaluator == "token":
        return compact == target or compact.startswith(target) or target in compact
    return compact == target


base.prompt_pool = prompt_pool
base.task_pass = task_pass


def freeze_plan():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = base.read(base.CAL / "CALIBRATION_REPORT.json")
    design = base.read(base.CAL / "DESIGN.json")
    pool = prompt_pool()
    lora_prompts = {row["prompt"] for row in design["lora_training"]}
    utility_prompts = {row["prompt"] for row in base.read(base.CAL / "UTILITY_SET.json")["prompts"]}
    if {row["prompt"] for row in pool} & (lora_prompts | utility_prompts):
        raise RuntimeError("v2 pool overlaps LoRA or utility prompts")
    # Start from the frozen v1 design so search budgets and promotion rules do
    # not change in response to v1 outcomes.
    v1 = base.read(V1 / "PLAN.json")
    payload = dict(v1)
    payload.update({
        "schema": "stage2-three-proxy-ceiling-v2",
        "pool": pool,
        "repair": "replace ambiguous uncommon-label logic prompts; robust task parser for classification/JSON/math",
        "reuse": "120 unchanged prompts reuse v1 response text; only 24 logic prompts are newly generated",
        "v1_plan_sha256": digest(V1 / "PLAN.json"),
        "v1_preflight_response_sha256": digest(V1 / "preflight_responses.jsonl"),
        "v1_failed_report_sha256": digest(V1 / "PREFLIGHT_REPORT.json"),
        "v2_driver_sha256": digest(__file__),
    })
    # Replace the v1 source hashes because the original search driver is now a
    # library dependency and this v2 driver defines pool/evaluation semantics.
    sources = [Path(__file__), project_path("scripts/stage2_three_proxy_search.py"),
               project_path("src/llm_integrity/macro_proxy.py"),
               project_path("src/llm_integrity/discrete_joint_inner_optimizer.py"),
               project_path("src/llm_integrity/inner_micro_proxy.py"),
               project_path("src/llm_integrity/inner_variant_sampler.py")]
    payload["source_hashes"] = {str(path.relative_to(project_path('.'))).replace('\\', '/'): digest(path) for path in sources}
    base.freeze(OUT / "PLAN.json", payload)
    base.status("V2_PLAN_FROZEN", pool=144, reused_prompts=120, new_logic_prompts=24)


def context():
    plan = base.read(OUT / "PLAN.json")
    if digest(V1 / "PLAN.json") != plan["v1_plan_sha256"]:
        raise RuntimeError("v1 plan changed")
    if digest(V1 / "preflight_responses.jsonl") != plan["v1_preflight_response_sha256"]:
        raise RuntimeError("v1 preflight bank changed")
    for name, expected in plan["source_hashes"].items():
        if digest(project_path(name)) != expected:
            raise RuntimeError(f"v2 source changed after freeze: {name}")
    return plan


def preflight(plan):
    from llm_integrity.modeling import load_model, render_prompt
    torch = base.setup()
    path = OUT / "preflight_responses.jsonl"
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        v1_plan = base.read(V1 / "PLAN.json")
        v1_lookup = {row["id"]: row for row in v1_plan["pool"]}
        unchanged = {row["id"] for row in plan["pool"] if row["category"] != "logic"}
        with (V1 / "preflight_responses.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                old = json.loads(line)
                if old["prompt_id"] not in unchanged:
                    continue
                pool_row = v1_lookup[old["prompt_id"]]
                record = dict(old)
                record.update({"task_passed": task_pass(old["response"], pool_row),
                               "reused_from_v1": True, "source_record_sha256": old["record_sha256"],
                               "plan_sha256": digest(OUT / "PLAN.json")})
                record["record_sha256"] = base.sha_value({key: value for key, value in record.items() if key != "record_sha256"})
                records.append(record)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records: handle.write(base.canonical(record) + "\n")
    logic = [row for row in plan["pool"] if row["category"] == "logic"]
    logic_by_id = {row["id"]: row for row in logic}
    completed = {(row["prompt_id"], row["response_index"]) for row in records if row["prompt_id"] in logic_by_id}
    schedule = [(row, response_index, 760_000_000 + response_index * 24 + index)
                for response_index in range(4) for index, row in enumerate(logic)
                if (row["id"], response_index) not in completed]
    bundle = load_model(plan["model"])
    try:
        encodings = {}
        for row in logic:
            rendered = render_prompt(bundle.tokenizer, row["prompt"])
            encodings[row["id"]] = {key: value.to(bundle.device) for key, value in
                                     bundle.tokenizer(rendered, return_tensors="pt", truncation=False).items()}
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            for index, (row, response_index, seed) in enumerate(schedule):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                encoded = encodings[row["id"]]
                with torch.inference_mode():
                    tokens = bundle.model.generate(**encoded, max_new_tokens=48, do_sample=True, temperature=.2,
                                                   top_p=.9, top_k=50, pad_token_id=bundle.tokenizer.pad_token_id,
                                                   eos_token_id=bundle.tokenizer.eos_token_id)
                ids = tokens[0, encoded["input_ids"].shape[1]:].detach().cpu().tolist()
                text = bundle.tokenizer.decode(ids, skip_special_tokens=True)
                record = {"prompt_index": plan["pool"].index(row), "response_index": response_index,
                          "generation_seed": seed, "prompt_id": row["id"], "response": text,
                          "task_passed": task_pass(text, row), "reused_from_v1": False,
                          "plan_sha256": digest(OUT / "PLAN.json")}
                record["record_sha256"] = base.sha_value(record)
                handle.write(base.canonical(record) + "\n"); handle.flush()
                records.append(record)
                if (index + 1) % 24 == 0:
                    os.fsync(handle.fileno()); base.status("V2_LOGIC_PREFLIGHT", completed=index + 1, total=len(schedule))
    finally:
        bundle.close(); base.cleanup()
    grouped = defaultdict(list)
    for record in records: grouped[record["prompt_id"]].append(record)
    audits = []; by_category = defaultdict(list)
    for row in plan["pool"]:
        values = grouped[row["id"]]
        if len(values) != 4: raise RuntimeError(f"Expected four responses: {row['id']}")
        modal = Counter(base.normalize(item["response"]) for item in values).most_common(1)[0][1]
        correct = sum(item["task_passed"] for item in values)
        valid = correct >= 3 and modal >= 3
        item = {**row, "correct": correct, "modal_response_count": modal, "valid": valid}
        audits.append(item)
        if valid: by_category[row["category"]].append(item)
    counts = {category: len(by_category[category]) for category in base.CATEGORIES}
    if any(counts[category] < 8 for category in base.CATEGORIES):
        base.freeze(OUT / "PREFLIGHT_REPORT.json", {"status": "INSUFFICIENT_STABLE_POOL", "audit": audits,
                                                    "valid_by_category": counts, "response_sha256": digest(path)})
        raise RuntimeError(f"v2 stable pool still insufficient: {counts}")
    selected = []
    for category in base.CATEGORIES:
        candidates = sorted(by_category[category], key=lambda x: (-x["correct"], -x["modal_response_count"], x["id"]))
        selected.extend(candidates[:10])
    pilot_counts = {category: (3 if index < 4 else 2) for index, category in enumerate(base.CATEGORIES)}
    pilot = []; formal = []
    for category in base.CATEGORIES:
        rows = [row for row in selected if row["category"] == category]
        pilot.extend(rows[:pilot_counts[category]])
        formal.extend(rows[:6 if category in base.CATEGORIES[:2] else 5])
    formal = formal[:32]
    base.freeze(OUT / "PREFLIGHT_REPORT.json", {"status": "PASS", "audit": audits,
                                                "valid_by_category": counts, "selected": selected,
                                                "pilot_sources": pilot, "formal_sources": formal,
                                                "response_sha256": digest(path), "reused_response_rows": 480,
                                                "new_response_rows": 96})
    base.status("V2_PREFLIGHT_COMPLETE", stable=len(selected), valid_by_category=counts, pilot=16, formal=32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["freeze", "preflight", "calibrate", "pilot-search", "formal-search"])
    args = parser.parse_args()
    if args.phase == "freeze": freeze_plan(); return
    plan = context()
    if args.phase == "preflight": preflight(plan)
    elif args.phase == "calibrate": base.calibrate(plan)
    elif args.phase == "pilot-search": base.search(plan, False)
    elif args.phase == "formal-search": base.search(plan, True)


if __name__ == "__main__":
    main()
