"""Bounded six-source Stage2 pilot. Each phase checks a frozen, hash-bound design."""
import argparse
import gc
import json
import os
import time
import math
from pathlib import Path
from dataclasses import asdict
from _bootstrap import project_path
from llm_integrity.stage1_r1 import canonical,digest,save_json
from stage2_prepare import design,training_data,freeze

OUT=project_path("results/stage2_six_source_pilot_20260905")
V2=Path("/root/autodl-tmp/llm/stage2_wording_v2_execution_20260905/results/stage2_qwen15b_wording_v2_20260905")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_payload(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:finite_payload(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [finite_payload(v) for v in value]
    return value


def status(phase,**kw):
    save_json(OUT/"STATUS.json",{"phase":phase,"utc_unix":time.time(),**kw})
    print(canonical({"phase":phase,**kw}),flush=True)


def setup():
    import torch
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    return torch


def cleanup():
    import torch
    gc.collect(); torch.cuda.empty_cache()


def pilot_utility():
    # Independent instances in the pilot's two admitted task categories; no math/JSON proxy for this pilot.
    pairs=[("甲比乙高，乙比丙高。只用一个汉字回答最矮的人，不解释。","丙","logic"),
           ("丁比戊高，戊比己高。只用一个汉字回答最矮的人，不解释。","己","logic"),
           ("红比蓝重，蓝比绿重。哪个最重？只回答颜色的一个汉字。","红","logic"),
           ("东比西远，西比南远。哪个最近？只回答方向的一个汉字。","南","logic"),
           ("严格输出从2开始的4个连续整数，英文逗号分隔，不添加其他内容。","2,3,4,5","instruction"),
           ("严格输出从3开始的3个连续整数，英文逗号分隔，不添加其他内容。","3,4,5","instruction"),
           ("请原样输出汉字：春夏秋冬。不要标点或其他文字。","春夏秋冬","instruction"),
           ("请原样输出汉字：东南西北。不要标点或其他文字。","东南西北","instruction")]
    return [{"id":f"pilot_utility_{i}","prompt":p,"expected_answer":a,"category":c,"evaluator":"exact"} for i,(p,a,c) in enumerate(pairs)]


def freeze_design():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg_path=project_path("configs/stage2_qwen15b.json")
    cfg=read(cfg_path); previous=read(V2/"PREFLIGHT_REPORT.json")
    assert previous["passing_count"]==6 and previous["status"]=="INSUFFICIENT_VALID_PROMPTS"
    prompts=previous["selected"]
    assert len(prompts)==6
    variants=design()
    assets=read(V2/"ASSETS.json")
    p={"schema":"stage2-six-source-pilot-v1","authorization":"User explicitly agreed to six-source limited pilot; old eight-source gate remains failed",
       "config":cfg,"config_sha256":digest(cfg_path),"prompts":prompts,"assets":assets,
       "v2_report_sha256":digest(V2/"PREFLIGHT_REPORT.json"),"v2_bank_sha256":digest(V2/"preflight_responses.jsonl"),
       "variants":variants,"lora_training":training_data(),"utility":pilot_utility(),
       "methods":["legacy","enhanced"],"maximum_candidates_per_source_per_method":2,
       "maximum_unique_response_prompts":30,"response_repetitions":8,"intact_roles":["fit","confirmation","heldout","null"],
       "maximum_behavior_response_rows":11040,"maximum_utility_generations":464,
       "scope":"six source instances, two task categories, repeated templates; no full-Stage2 or multi-category generalization claim",
       "statistics":{"permutations":999,"alpha":.05,"tie_atol":1e-12,"channels_separate":True},
       "behavior_gate":{"intact_minimum_correct_of_8":6,"max_intact_drop_from_source":.125,
                        "gaussian_lora_each_strength_minimum_seeds_raw_detected":2,
                        "require_positive_h8_mmd_each_detected_seed":True,"no_strong_for_weak_compensation":True,
                        "coverage_non_degradation_absolute_mmd_tolerance":.01,
                        "utility_maximum_drop_on_intact_correct_subset":.15},
       "code_sha256":{str(f.relative_to(project_path('.'))).replace('\\','/'):digest(f)
          for f in sorted(project_path('src/llm_integrity').glob('*.py'))},
       "driver_sha256":digest(__file__)}
    freeze(OUT/"DESIGN.json",p)
    (OUT/"adapters").mkdir(exist_ok=True)
    train_path=OUT/"lora_train.jsonl"
    text=''.join(canonical(r)+'\n' for r in p["lora_training"])
    if train_path.exists() and train_path.read_text(encoding="utf-8")!=text:
        raise RuntimeError("Changed SFT data")
    if not train_path.exists(): train_path.write_text(text,encoding="utf-8")
    return p


def get_design():
    p=read(OUT/"DESIGN.json")
    if digest(__file__)!=p["driver_sha256"]: raise RuntimeError("Driver changed after freeze")
    for name,sha in p["code_sha256"].items():
        if digest(project_path(name))!=sha: raise RuntimeError(f"Code changed: {name}")
    return p


def micro_smoke(p):
    torch=setup()
    from llm_integrity.modeling import load_model,tokenize_prompts
    from llm_integrity.inner_micro_proxy import discover_micro_blocks,representative_layers,select_micro_block,differentiable_block_micro_proxy
    from llm_integrity.stage2_contract import BLOCK_TYPES
    done=OUT/"MICRO_SMOKE.json"
    if done.exists() and read(done).get("complete"): return
    bundle=load_model(p["config"]["model"])
    checks=[]
    try:
        blocks=discover_micro_blocks(bundle.model,BLOCK_TYPES); layers=representative_layers(blocks,4)
        for j,t in enumerate(BLOCK_TYPES):
            encoded=tokenize_prompts(bundle,[p["prompts"][0]["prompt"]],128)
            emb=bundle.model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
            torch.cuda.reset_peak_memory_stats(); start=time.time()
            result=differentiable_block_micro_proxy(bundle,emb,encoded["attention_mask"],
                block=select_micro_block(blocks,block_type=t,layer_id=layers[j%4]),probes=4,seed=20264000+j*10)
            checks.append({**result.metadata(),"seconds":time.time()-start,"peak_bytes":torch.cuda.max_memory_allocated()})
            if not result.gradient_finite or not result.gradient_nonzero: raise RuntimeError("Invalid micro gradient")
            save_json(done,{"complete":len(checks)==7,"checks":checks})
            status("micro_smoke",module=t,complete_modules=len(checks),seconds=checks[-1]["seconds"],peak_bytes=checks[-1]["peak_bytes"])
            del result,emb,encoded; cleanup()
    finally: bundle.close()


def loaded_variant(p,v):
    from llm_integrity.paper_variant_executor import load_manifest_variant
    return load_manifest_variant(p["config"]["model"],v,
        adapter_path=str(OUT/"adapters"/v["variant_id"]) if v["family"]=="finetuning" else None)


def attacks(p,split="train"):
    setup()
    from llm_integrity.paper_finetuning import train_lora_manifest_variant
    for v in p["variants"]:
        if v["family"]!="finetuning" or v["split"]!=split: continue
        folder=OUT/"adapters"/v["variant_id"]
        report=folder/"training_report.json"
        if report.exists():
            r=read(report)
            if r["data_sha256"]!=digest(OUT/"lora_train.jsonl") or r["completed_steps"]!=v["configuration"]["steps"]:
                raise RuntimeError("Inconsistent trained adapter")
            continue
        status("training_adapter",variant=v["variant_id"],steps=v["configuration"]["steps"])
        train_lora_manifest_variant(p["config"]["model"],v,data_path=OUT/"lora_train.jsonl",output_root=OUT/"adapters",
                                    max_length=128,batch_size=1,gradient_accumulation_steps=4)
        cleanup()


def calibrate(p):
    torch=setup()
    from llm_integrity.modeling import load_model,tokenize_prompts
    from llm_integrity.inner_micro_proxy import discover_micro_blocks,representative_layers,select_micro_block,differentiable_block_micro_proxy
    from llm_integrity.stage2_contract import BLOCK_TYPES
    from llm_integrity.inner_variant_sampler import FAMILIES
    from build_discrete_calibration import grouped_calibration
    import statistics
    target=OUT/"CALIBRATION.json"
    if target.exists(): return
    # Independent tiny calibration texts, not candidate/evaluation responses.
    texts=["请原样输出：青山绿水。不要添加其他文字。","丙比丁高，丁比戊高。只回答最高者的一个汉字名字。",
           "请原样输出：日月星辰。不要添加其他文字。","乙比丙高，丙比丁高。只回答最矮者的一个汉字名字。"]
    bundle=load_model(p["config"]["model"])
    for par in bundle.model.parameters(): par.requires_grad_(False)
    micro=[]; macro=[]
    try:
        blocks=discover_micro_blocks(bundle.model,BLOCK_TYPES); layers=representative_layers(blocks,4)
        for i,text in enumerate(texts):
            for j,t in enumerate(BLOCK_TYPES):
                encoded=tokenize_prompts(bundle,[text],128)
                emb=bundle.model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
                result=differentiable_block_micro_proxy(bundle,emb,encoded["attention_mask"],
                    block=select_micro_block(blocks,block_type=t,layer_id=layers[i]),probes=4,seed=20265000+i*100+j*10)
                micro.append({**result.metadata(),"block_type":t,"valid":result.gradient_finite and result.gradient_nonzero})
                save_json(OUT/"calibration_micro_records.json",micro)
                status("micro_calibration",completed=len(micro),total=28)
                del result,emb,encoded; cleanup()
        for v in [v for v in p["variants"] if v["split"]=="train"]:
            loaded=loaded_variant(p,v)
            try:
                for par in loaded.bundle.model.parameters(): par.requires_grad_(False)
                for i,text in enumerate(texts):
                    encoded=tokenize_prompts(bundle,[text],128)
                    emb=bundle.model.get_input_embeddings()(encoded["input_ids"]).detach().requires_grad_(True)
                    a=bundle.model(inputs_embeds=emb,attention_mask=encoded["attention_mask"],use_cache=False).logits[0,-1].float()
                    b=loaded.bundle.model(inputs_embeds=emb,attention_mask=encoded["attention_mask"],use_cache=False).logits[0,-1].float()
                    score=(b-a).square().sum(); grad=torch.autograd.grad(score,emb)[0]
                    macro.append({"family":v["family"],"variant_id":v["variant_id"],"raw_macro_score":float(score.item()),
                        "raw_embedding_gradient_norm":float(grad.float().norm().item()),"valid":bool(torch.isfinite(grad).all() and grad.norm()>0)})
                    del encoded,emb,a,b,score,grad
                save_json(OUT/"calibration_macro_records.json",macro)
                status("macro_calibration",variant=v["variant_id"],completed=len(macro),total=60)
            finally: loaded.close(); cleanup()
        mic=grouped_calibration(micro,group_key="block_type",groups=BLOCK_TYPES,score_key="raw_micro_score")
        mac=grouped_calibration(macro,group_key="family",groups=FAMILIES,score_key="raw_macro_score")
        macro_target=sum(mac[f]["normalized_gradient_norm"]["median"] for f in FAMILIES)/5
        mw=macro_target/statistics.median(mic[b]["normalized_gradient_norm"]["median"] for b in BLOCK_TYPES)
        freeze(target,{"version":2,"micro":mic,"macro":mac,"recommended_micro_weight":mw,
                       "calibration_texts":texts,"design_sha256":digest(OUT/"DESIGN.json")})
    finally: bundle.close()


def search(p):
    setup()
    from llm_integrity.modeling import load_model
    from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer
    from llm_integrity.inner_variant_sampler import StratifiedVariantSampler,FAMILIES
    cal=read(OUT/"CALIBRATION.json")
    registry={v["variant_id"]:str(OUT/"adapters"/v["variant_id"]) for v in p["variants"] if v["family"]=="finetuning" and v["split"]=="train"}
    for method in p["methods"]:
        folder=OUT/"search"/method; folder.mkdir(parents=True,exist_ok=True)
        s=dict(p["config"]["search"])
        if method=="legacy": s.update(p["config"]["legacy_overrides"])
        sampler=StratifiedVariantSampler([v for v in p["variants"] if v["split"]=="train"],family_weights={f:.2 for f in FAMILIES},adapter_registry=registry,seed=20266000)
        for i,row in enumerate(p["prompts"]):
            target=folder/(row["id"]+".json")
            if target.exists():
                if read(target)["design_sha256"]!=digest(OUT/"DESIGN.json"): raise RuntimeError("Search resume mismatch")
                continue
            bundle=load_model(p["config"]["model"])
            opt=None; tick=time.time()
            try:
                opt=DiscreteJointInnerOptimizer(reference=bundle,sampler=sampler,model_config=p["config"]["model"],
                    micro_scales={b:cal["micro"][b]["scale"] for b in s["block_types"]},
                    macro_scales={f:cal["macro"][f]["scale"] for f in FAMILIES},
                    micro_component_clips={b:cal["micro"][b]["normalized_gradient_norm"]["clip_threshold_log_median_plus_5_mad"] for b in s["block_types"]},
                    macro_component_clips={f:cal["macro"][f]["normalized_gradient_norm"]["clip_threshold_log_median_plus_5_mad"] for f in FAMILIES},
                    micro_weight=cal["recommended_micro_weight"],macro_weight=1.,rounds=s["rounds"],
                    candidate_positions=s["candidate_positions"],candidates_per_position=s["candidates_per_position"],rerank_candidates=s["rerank_candidates"],
                    probes=s["probes_inner"],seed=20267000+i*100000,max_length=128,max_edit_ratio=.25,ppl_ratio_limit=2.,
                    minimum_nondegraded_families=s["minimum_nondegraded_families"],family_relative_tolerance=.01,
                    variants_per_family=s["variants_per_family"],anchor_variants_per_family=1,gradient_restarts=1,
                    family_gate_aggregation="all_anchor_variants",require_task_preservation=True,task_max_input_tokens=512,
                    task_max_new_tokens=128,sequential_model_execution=False,block_types=tuple(s["block_types"]),
                    representative_layer_count=4,block_schedule=s["block_schedule"],task_validation_mode="strict_r1")
                status("search",method=method,source=i+1,total=6,rounds=s["rounds"])
                result=opt.optimize(row)
                payload={**result.payload(),"method":method,"source":row,"effective_parameters":s,
                         "micro_weight":cal["recommended_micro_weight"],"design_sha256":digest(OUT/"DESIGN.json"),"seconds":time.time()-tick}
                # Legacy optimizer can carry NaN diagnostics for empty candidate pools; preserve explicitly as null.
                payload=finite_payload(payload)
                save_json(target,payload)
                status("search_result",method=method,source=i+1,accepted=result.accepted,seconds=payload["seconds"],failure=result.failure)
                if result.failure: raise RuntimeError(f"Search technical failure: {result.failure}")
            finally:
                (opt.reference if opt is not None else bundle).close(); cleanup()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=["prepare","search","evaluate"])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/"RUN.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.phase=="prepare":
                p=freeze_design(); micro_smoke(p); attacks(p,"train"); calibrate(p)
                status("PREPARATION_COMPLETE")
            elif args.phase=="search":
                p=get_design(); search(p); status("SEARCH_COMPLETE")
            else:
                raise NotImplementedError("Evaluation driver must be verified before launch")
        except Exception as exc:
            status("FAILED",error_type=type(exc).__name__,error=str(exc)); raise


if __name__=="__main__": main()
