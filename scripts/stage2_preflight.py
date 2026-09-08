"""Bounded, resumable intact feasibility and higher-order gradient preflight."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from collections import Counter
from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical, digest, save_json


def task_check(row, text, truncated=False):
    import re
    if truncated:
        return False
    if row["category"] == "structured":
        try:
            def pairs(items):
                d = dict(items)
                if len(d) != len(items):
                    raise ValueError("duplicate JSON keys")
                return d
            obj = json.loads(text.strip(), object_pairs_hook=pairs)
            return (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                    and type(obj.get("level")) is int and type(obj.get("enabled")) is bool)
        except (ValueError, TypeError):
            return False
    return re.sub(r"\s+", "", text.strip()) == re.sub(r"\s+", "", str(row["expected_answer"]))


def generate(bundle, row, seed, max_new_tokens=128, do_sample=True):
    import torch
    from transformers import set_seed
    set_seed(seed)
    ids = bundle.tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt").to(bundle.device)
    if ids.shape[1] > 512:
        raise ValueError("No silent prompt truncation allowed")
    kw = {"do_sample": do_sample}
    if do_sample:
        kw.update(temperature=.7, top_p=.9, top_k=50)
    with torch.inference_mode():
        result = bundle.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens, pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id, **kw)[0, ids.shape[1]:]
    tokens = result.tolist()
    eos = bundle.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    truncated = len(tokens) >= max_new_tokens and (not tokens or tokens[-1] not in eos)
    response = bundle.tokenizer.decode(tokens, skip_special_tokens=True)
    return {"prompt_id": row["id"], "response_seed": seed, "response": response,
            "generated_token_ids": tokens, "input_tokens": int(ids.shape[1]),
            "truncated": truncated, "task_correct": task_check(row, response, truncated)}


def main():
    import torch
    from llm_integrity.modeling import load_model, tokenize_prompts
    from llm_integrity.inner_micro_proxy import discover_micro_blocks, representative_layers, select_micro_block, differentiable_block_micro_proxy
    from llm_integrity.stage2_contract import BLOCK_TYPES
    cfg_path = project_path("configs/stage2_qwen15b.json")
    cfg = json.loads(cfg_path.read_text())
    out = project_path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    asset = json.loads((out/"ASSETS.json").read_text())
    if asset["revision"] != cfg["model"]["revision"]:
        raise RuntimeError("Unexpected model revision")
    source = project_path(cfg["source_prompts"])
    rows = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines() if s.strip()]
    categories = cfg["preflight"]["categories"]
    groups = {c: [r for r in rows if r["category"] == c][:3] for c in categories}
    pool = [groups[c][i] for i in range(3) for c in categories]
    manifest = {"config_sha256": digest(cfg_path), "source_sha256": digest(source),
                "asset_sha256": digest(out/"ASSETS.json"), "preflight_script_sha256": digest(__file__),
                "pool": pool, "selection": "round-robin first eight with >=3/4 exact successes; no attack information"}
    frozen = out/"PREFLIGHT_FROZEN.json"
    if frozen.exists() and json.loads(frozen.read_text()) != manifest:
        raise RuntimeError("Preflight identity changed; use a new version")
    if not frozen.exists():
        save_json(frozen, manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    bank = out/"preflight_responses.jsonl"
    existing = [json.loads(s) for s in bank.read_text(encoding="utf-8").splitlines()] if bank.exists() else []
    done = {(r["prompt_id"], r["response_seed"]) for r in existing}
    bundle = load_model(cfg["model"])
    started = time.time()
    try:
        with bank.open("a", encoding="utf-8") as f:
            for i,row in enumerate(pool):
                for repetition in range(4):
                    seed = 2026091000+i*10+repetition
                    if (row["id"],seed) in done:
                        continue
                    result = generate(bundle,row,seed)
                    result["manifest_sha256"] = digest(frozen)
                    f.write(canonical(result)+"\n"); f.flush()
                    existing.append(result)
                    print("PREFLIGHT",len(existing),48,row["category"],result["task_correct"],flush=True)
        passing = []
        summaries = []
        for row in pool:
            records = [r for r in existing if r["prompt_id"] == row["id"]]
            count = sum(r["task_correct"] for r in records)
            fits = all(r["input_tokens"] <= 128 for r in records)
            summaries.append({"id":row["id"],"correct":count,"total":len(records),"fits_search":fits})
            if len(records)==4 and count>=3 and fits:
                passing.append(row)
        report = {"status":"PASS" if len(passing)>=8 else "INSUFFICIENT_VALID_PROMPTS",
                  "summaries":summaries,"selected":passing[:8],"elapsed_seconds":time.time()-started}
        save_json(out/"PREFLIGHT_REPORT.json",report)
        if len(passing)<8:
            print(json.dumps(report,ensure_ascii=False),flush=True)
            return
        save_json(out/"SELECTED_PROMPTS.json",passing[:8])
        # Smoke all seven module types at one representative layer, including true second derivatives.
        blocks=discover_micro_blocks(bundle.model,BLOCK_TYPES)
        layers=representative_layers(blocks,4)
        checks=[]
        for j,t in enumerate(BLOCK_TYPES):
            encoded=tokenize_prompts(bundle,[passing[0]["prompt"]],128)
            emb=bundle.model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
            torch.cuda.reset_peak_memory_stats()
            tick=time.time()
            result=differentiable_block_micro_proxy(bundle,emb,encoded["attention_mask"],
                block=select_micro_block(blocks,block_type=t,layer_id=layers[j%4]),probes=4,seed=2026092000+j*10)
            checks.append({**result.metadata(),"seconds":time.time()-tick,
                           "peak_allocated_bytes":torch.cuda.max_memory_allocated()})
            save_json(out/"MICRO_SMOKE.json",{"checks":checks,"complete":len(checks)==7})
            print("MICRO",t,checks[-1]["seconds"],checks[-1]["gradient_finite"],flush=True)
            if not result.gradient_finite or not result.gradient_nonzero:
                raise RuntimeError("Invalid higher-order gradient")
            del result,emb,encoded
            torch.cuda.empty_cache()
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
