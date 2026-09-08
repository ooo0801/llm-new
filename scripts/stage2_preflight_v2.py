"""User-authorized equivalent-wording retest. Exactly 12*4 rows maximum; no adaptive expansion."""
import argparse
import json
import time
from pathlib import Path
from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical, digest, save_json
from stage2_preflight import generate, task_check


def revised_pool(source_rows, changes):
    categories=["logic","math","instruction","structured"]
    groups={c:[r for r in source_rows if r["category"]==c][:3] for c in categories}
    pool=[groups[c][i] for i in range(3) for c in categories]
    ids={r["id"] for r in pool}
    if set(changes)-ids:
        raise ValueError("Unexpected source id")
    result=[]
    for row in pool:
        original=row["prompt"]
        text=changes.get(row["id"],original)
        result.append({**row,"id":row["id"]+"__wording_v2","prompt_id":row["id"]+"__wording_v2",
                       "source_prompt_id":row["id"],"original_prompt":original,"prompt":text,
                       "wording_changed":text!=original,"source":"stage2_equivalent_wording_v2"})
    if len(result)!=12:
        raise ValueError("Expected twelve source prompts")
    return result


def freeze(path,value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8"))!=value:
            raise RuntimeError(f"Frozen file changed: {path}")
    else:
        save_json(path,value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--assets",required=True)
    args=parser.parse_args()
    cfg_path=project_path("configs/stage2_qwen15b.json")
    cfg=json.loads(cfg_path.read_text())
    out=project_path("results/stage2_qwen15b_wording_v2_20260905")
    out.mkdir(parents=True,exist_ok=True)
    source=project_path(cfg["source_prompts"])
    changes_path=project_path("experiments/h9-stage2/prompt_revisions_v2.json")
    changes=json.loads(changes_path.read_text(encoding="utf-8"))
    pool=revised_pool([json.loads(s) for s in source.read_text(encoding="utf-8").splitlines() if s.strip()],changes["replacements"])
    assets_path=Path(args.assets)
    assets=json.loads(assets_path.read_text())
    if assets["revision"]!=cfg["model"]["revision"]:
        raise RuntimeError("Model revision mismatch")
    manifest={"schema":"stage2-preflight-wording-v2","pool":pool,"maximum_responses":48,
      "repetitions":4,"minimum_correct":3,"selected_count":8,"response_seed_base":2026094000,
      "selection":"first eight in frozen round-robin order with >=3/4 strict correct and <=128 input tokens",
      "config_sha256":digest(cfg_path),"source_sha256":digest(source),"changes_sha256":digest(changes_path),
      "assets_sha256":digest(assets_path),"code_sha256":{name:digest(project_path(name)) for name in
        ["scripts/stage2_preflight_v2.py","scripts/stage2_preflight.py","src/llm_integrity/modeling.py","src/llm_integrity/stage1_r1.py","src/llm_integrity/inner_micro_proxy.py"]},
      "authorization":"User agreed to one extra <=48 response equivalent-wording retest; no answer or task threshold change",
      "analysis_scope":"source instances are not independent task templates; instruction instances share one task"}
    frozen=out/"PREFLIGHT_FROZEN.json"
    freeze(frozen,manifest)
    freeze(out/"ASSETS.json",assets)
    for name,info in assets["files"].items():
        if digest(Path(assets["path"])/name)!=info["sha256"]:
            raise RuntimeError(f"Corrupt asset: {name}")
    import torch
    from llm_integrity.modeling import load_model,tokenize_prompts
    from llm_integrity.inner_micro_proxy import discover_micro_blocks,representative_layers,select_micro_block,differentiable_block_micro_proxy
    from llm_integrity.stage2_contract import BLOCK_TYPES
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    bank=out/"preflight_responses.jsonl"
    records=[]
    if bank.exists():
        raw=bank.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise RuntimeError("Partial response row; do not silently truncate")
        records=[json.loads(s) for s in raw.decode().splitlines()]
    plan=[(row,2026094000+i*10+j) for i,row in enumerate(pool) for j in range(4)]
    if len(records)>len(plan):
        raise RuntimeError("Budget exceeded")
    for r,(row,seed) in zip(records,plan):
        if (r["prompt_id"],r["response_seed"],r["manifest_sha256"])!=(row["id"],seed,digest(frozen)):
            raise RuntimeError("Response prefix or identity mismatch")
        if r["task_correct"]!=task_check(row,r["response"],r["truncated"]):
            raise RuntimeError("Task result mismatch")
    started=time.time()
    bundle=load_model(cfg["model"])
    try:
        with bank.open("a",encoding="utf-8") as f:
            for row,seed in plan[len(records):]:
                record=generate(bundle,row,seed)
                record["manifest_sha256"]=digest(frozen)
                f.write(canonical(record)+"\n"); f.flush()
                records.append(record)
                print("PREFLIGHT_V2",len(records),48,row["category"],record["task_correct"],flush=True)
        passing=[]; summaries=[]
        for row in pool:
            rr=[r for r in records if r["prompt_id"]==row["id"]]
            count=sum(r["task_correct"] for r in rr)
            fits=all(r["input_tokens"]<=128 for r in rr)
            summaries.append({"id":row["id"],"source_prompt_id":row["source_prompt_id"],"category":row["category"],
                              "correct":count,"total":len(rr),"fits_search":fits,"wording_changed":row["wording_changed"]})
            if count>=3 and fits:
                passing.append(row)
        report={"status":"PASS" if len(passing)>=8 else "INSUFFICIENT_VALID_PROMPTS","summaries":summaries,
                "selected":passing[:8],"passing_count":len(passing),"response_count":len(records),
                "responses_sha256":digest(bank),"manifest_sha256":digest(frozen),"elapsed_seconds":time.time()-started}
        save_json(out/"PREFLIGHT_REPORT.json",report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
        if len(passing)<8:
            return
        freeze(out/"SELECTED_PROMPTS.json",passing[:8])
        blocks=discover_micro_blocks(bundle.model,BLOCK_TYPES)
        layers=representative_layers(blocks,4)
        checks=[]
        for j,t in enumerate(BLOCK_TYPES):
            encoded=tokenize_prompts(bundle,[passing[0]["prompt"]],128)
            emb=bundle.model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
            torch.cuda.reset_peak_memory_stats(); tick=time.time()
            result=differentiable_block_micro_proxy(bundle,emb,encoded["attention_mask"],
                block=select_micro_block(blocks,block_type=t,layer_id=layers[j%4]),probes=4,seed=2026095000+j*10)
            checks.append({**result.metadata(),"seconds":time.time()-tick,"peak_allocated_bytes":torch.cuda.max_memory_allocated()})
            save_json(out/"MICRO_SMOKE.json",{"checks":checks,"complete":len(checks)==7})
            print("MICRO_V2",t,checks[-1]["seconds"],result.gradient_finite,flush=True)
            if not result.gradient_finite or not result.gradient_nonzero:
                raise RuntimeError("Invalid higher-order gradient")
            del result,emb,encoded
            torch.cuda.empty_cache()
    finally:
        bundle.close()


if __name__=="__main__":
    main()
