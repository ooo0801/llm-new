"""Fresh-server bounded pilot: new dual-level prompts, RESF-inspired detection.

Every phase is explicit; no historical result paths, no semantic model, no shutdown.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
from dataclasses import asdict

import _bootstrap
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_identity():
    repo = Path(__file__).resolve().parents[1]
    files = sorted((repo/'src'/'llm_integrity').glob('*.py')) + sorted((repo/'scripts').glob('*.py'))
    return identity({str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def file_identity(folder):
    return identity({str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(folder.rglob('*')) if p.is_file()})


def freeze(path, value):
    if Path(path).exists() and read(path) != value:
        raise RuntimeError(f"Frozen artifact changed: {path}")
    save(path, value)


def clean():
    import torch
    gc.collect(); torch.cuda.empty_cache()


def progress(phase, **fields):
    print(json.dumps(dict(phase=phase,time=time.time(),**fields),ensure_ascii=False),flush=True)


def variant(family, label, seed, config, split):
    return dict(family=family, variant_id=label, seed=seed, configuration=config, split=split)


def prepare(root, model, utility_mode='gate'):
    from stage2_prepare import training_data
    from stage2_attack_calibration import utility_pool
    from stage2_three_proxy_search import prompt_pool
    families = []
    for i in range(4):
        split = "search" if i == 0 else "unseen_attack_seed_evaluation"
        families.append(variant("gaussian_noise", f"gaussian_s{i}", 260917100+i,
            dict(method="relative_gaussian", std_ratio=.006, target_scope="attention_ffn", measure_realized=True), split))
        families.append(variant("finetuning", f"lora_s{i}", 260917200+i,
            dict(method="lora", rank=8, alpha=16, dropout=0., learning_rate=1e-4,
                 steps=40, lr_scheduler_type="constant", target_scope="attention_ffn"), split))
    families += [
        variant("unstructured_pruning", "pruning", 260917300, dict(method="layerwise_magnitude", ratio=.2, target_scope="attention_ffn"), "search"),
        variant("structured_pruning", "structured", 260917301, dict(structure="ffn_channels", ratio=.005, selection="magnitude", layer_scope="all_layers", implementation="mask"), "search"),
        variant("quantization", "int8", 260917302, dict(method="int8", bits=8, target_scope="full_model", compute_dtype="bfloat16", llm_int8_threshold=8.), "search"),
    ]
    utility = [r for c in ("logic","math","instruction","structured") for r in [x for x in utility_pool() if x['category']==c][:8]]
    sources = [r for c in ("extraction", "classification") for r in [x for x in prompt_pool() if x['category']==c][:4]]
    train = training_data()
    assert not {r['prompt'] for r in sources} & {r['prompt'] for r in utility+train}
    plan = dict(schema="resf-inspired-fresh-pilot-v1", model=dict(name=str(Path(model).resolve()), dtype="bfloat16", device_map="auto",
        local_files_only=True, trust_remote_code=False, eager_attention=True), model_revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        variants=families, utility=utility, source_pool=sources, train_data=train,
        proxies=["js","topk_continuous","raw_logit_l2"], max_sources=2, max_candidates=6,
        search=dict(rounds=5, probes=8, candidate_positions=4, candidates_per_position=16, rerank_candidates=8),
        temperatures=[.5,.7,.9], actual_temperature=.7, top_k=50, top_p=.9, looks=[30,60,100], simulations=10000,
        panel_alpha=.05, e_fraction=.2, preflight_repetitions=4, utility_mode=utility_mode,
        budget=dict(lora_optimizer_steps=160, search_jobs=6, search_rounds=30,
                    utility_responses=384, preflight_responses=32, token_responses=4800,
                    reference_smoke_requests=6, total_generation_requests=5222, utility_max_new_tokens=64),
        source_selection="first valid per category; report-only mode uses first source if none is task-valid",
        scope="new prompt optimization; two source categories; same-strength unseen attack seeds, not configuration/data held-out",
        detector="RESF-inspired: union post-decoding support; finite temperature grid; exact sparse multinomial deviance MC; alpha split across prompts/rules/looks",
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256=code_identity(),
        model_manifest_sha256=hashlib.sha256((Path(model)/'DOWNLOAD_MANIFEST.json').read_bytes()).hexdigest())
    freeze(root/'PLAN.json', plan)
    training_text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train)
    path=root/'train.jsonl'
    if path.exists() and path.read_text(encoding='utf-8')!=training_text:raise RuntimeError('Training data changed')
    path.write_text(training_text,encoding='utf-8')


def load_variant(plan, row, root):
    from llm_integrity.paper_variant_executor import load_manifest_variant
    path = root/'adapters'/row['variant_id'] if row['family']=='finetuning' else None
    if path is not None:
        report=read(root/'training'/f"{row['variant_id']}.json")
        if file_identity(path)!=report['adapter_files_sha256']:raise RuntimeError('Adapter files changed')
    return load_manifest_variant(plan['model'],row,adapter_path=path)


def exact_task(row, text):
    text=text.strip()
    expected=str(row['expected_answer']).strip()
    if row.get('evaluator')=='json_semantic':
        try:
            fence=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',text,re.S|re.I)
            if fence:text=fence[1].strip()
            def pairs(items):
                d={}
                for k,v in items:
                    if k in d: raise ValueError('duplicate key')
                    d[k]=v
                return d
            a=json.loads(text,object_pairs_hook=pairs);b=json.loads(expected)
            return json.dumps(a,sort_keys=True,separators=(',',':'))==json.dumps(b,sort_keys=True,separators=(',',':'))
        except (ValueError,TypeError):return False
    return text==expected


def generate(bundle, row, seed, *, first=False, temperature=.7):
    import torch
    from transformers import set_seed
    ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
    if ids.shape[1]>256:raise RuntimeError('Prompt length exceeds budget')
    set_seed(seed)
    options=dict(max_new_tokens=1 if first else 64,do_sample=first,pad_token_id=bundle.tokenizer.pad_token_id)
    if first:options.update(temperature=temperature,top_k=50,top_p=.9)
    with torch.inference_mode(): out=bundle.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**options)
    tokens=out[0,ids.shape[1]:].tolist()
    if not tokens:raise RuntimeError('No generated token IDs')
    return dict(token_ids=tokens,response=bundle.tokenizer.decode(tokens,skip_special_tokens=True),generation_seed=seed)


def train(root, plan):
    from llm_integrity.paper_finetuning import train_lora_manifest_variant
    for row in plan['variants']:
        if row['family']!='finetuning':continue
        target=root/'training'/f"{row['variant_id']}.json"
        if target.exists():
            if read(target)['adapter_files_sha256']!=file_identity(root/'adapters'/row['variant_id']):
                raise RuntimeError('Adapter files changed')
            continue
        progress('training',endpoint=row['variant_id'])
        report=train_lora_manifest_variant(model_config=plan['model'],variant=row,data_path=root/'train.jsonl',output_root=root/'adapters',
             max_length=128,batch_size=1,gradient_accumulation_steps=4)
        if report.completed_steps!=40:raise RuntimeError('Incomplete LoRA training')
        save(target,dict(**asdict(report),adapter_files_sha256=file_identity(root/'adapters'/row['variant_id'])));clean()


def utility(root, plan):
    from llm_integrity.modeling import load_model
    reports={}
    for j,row in enumerate([None]+plan['variants']):
        label='intact' if row is None else row['variant_id'];target=root/'utility'/f'{label}.json'
        if target.exists():
            report=read(target)
            questions={r['id']:r for r in plan['utility']}
            for item in report['records']:item['passed']=exact_task(questions[item['id']],item['response'])
            save(target,report);reports[label]=report;continue
        progress('utility',endpoint=label)
        loaded=None
        bundle=load_model(plan['model']) if row is None else (loaded:=load_variant(plan,row,root)).bundle
        records=[]
        try:
            for i,question in enumerate(plan['utility']):
                result=generate(bundle,question,270000000+j*100+i)
                records.append(dict(id=question['id'],passed=exact_task(question,result['response']),**result))
            save(target,dict(records=records,execution=None if loaded is None else asdict(loaded.report)))
            reports[label]=read(target)
        finally:
            (loaded.close() if loaded else bundle.close());clean()
    baseline={r['id'] for r in reports['intact']['records'] if r['passed']}
    summary={}
    for label,report in reports.items():
        lost=[r['id'] for r in report['records'] if r['id'] in baseline and not r['passed']]
        summary[label]=dict(correct=sum(r['passed'] for r in report['records']),lost=lost,
                            qualifies=len(baseline)>=24 and len(lost)/len(baseline)<=.0625)
    save(root/'UTILITY.json',summary)
    if not all(x['qualifies'] for x in summary.values()) and plan['utility_mode']=='gate':
        raise RuntimeError('Utility gate failed; no automatic attack escalation')


def preflight(root, plan):
    from llm_integrity.modeling import load_model
    if (root/'PREFLIGHT.json').exists():return
    bundle=load_model(plan['model']);records=[];selected=[]
    try:
        for i,row in enumerate(plan['source_pool']):
            results=[generate(bundle,row,271000000+i*10+j) for j in range(4)]
            passed=sum(exact_task(row,r['response']) for r in results)
            records.append(dict(source=row,results=results,correct=passed))
        for category in ['extraction','classification']:
            valid=[r['source'] for r in records if r['source']['category']==category and r['correct']>=3]
            if not valid:
                save(root/'PREFLIGHT_AUDIT.json',dict(records=records,failed_category=category))
                if plan['utility_mode']=='gate':raise RuntimeError('No valid source in '+category)
                valid=[r for r in plan['source_pool'] if r['category']==category]
            selected.append(valid[0])
        freeze(root/'PREFLIGHT.json',dict(records=records,selected=selected))
    finally:bundle.close();clean()


def calibrate(root, plan):
    import torch
    from llm_integrity.modeling import load_model
    from llm_integrity.inner_micro_proxy import discover_micro_blocks,representative_layers,differentiable_block_micro_proxy
    from llm_integrity.macro_proxy import differentiable_macro_proxy
    from llm_integrity.stage2_contract import BLOCK_TYPES
    if (root/'CALIBRATION.json').exists():return
    bundle=load_model(plan['model'])
    for p in bundle.model.parameters():p.requires_grad_(False)
    texts=read(root/'PREFLIGHT.json')['selected'];micro=[];macro=[]
    try:
        blocks=discover_micro_blocks(bundle.model,BLOCK_TYPES);layers=representative_layers(blocks,4)
        for bi,kind in enumerate(BLOCK_TYPES):
            progress('micro_calibration',block=kind)
            block=next(b for b in blocks if b.block_type==kind and b.layer_id==layers[bi%len(layers)])
            row=texts[bi%len(texts)]
            ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
            embeds=bundle.model.get_input_embeddings()(ids).detach().requires_grad_(True)
            score=differentiable_block_micro_proxy(bundle,embeds,torch.ones_like(ids),block=block,probes=8,seed=272000000+bi)
            micro.append(score.metadata());del score,embeds;clean()
        for row in [r for r in plan['variants'] if r['split']=='search']:
            progress('macro_calibration',family=row['family'])
            loaded=load_variant(plan,row,root)
            for p in loaded.bundle.model.parameters():p.requires_grad_(False)
            try:
                for source in texts:
                    ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=source['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
                    e=bundle.model.get_input_embeddings()(ids).detach().requires_grad_(True);mask=torch.ones_like(ids)
                    a=bundle.model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits[0,-1].float()
                    b=loaded.bundle.model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits[0,-1].float()
                    for proxy in plan['proxies']:
                        score=differentiable_macro_proxy(a,b,proxy=proxy,top_k=10)
                        grad=torch.autograd.grad(score,e,retain_graph=True)[0]
                        if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite macro gradient')
                        macro.append(dict(family=row['family'],proxy=proxy,score=float(score),gradient=float(grad.float().norm())))
                    del a,b,e,grad,score;clean()
            finally:loaded.close();clean()
        micro_scales={r['block']['block_type']:r['raw_micro_score'] for r in micro}
        if any(v<=0 or not np.isfinite(v) for v in micro_scales.values()):raise RuntimeError('Invalid micro scale')
        scales={};weights={}
        for proxy in plan['proxies']:
            scales[proxy]={}
            for family in {r['family'] for r in macro}:
                values=[r['score'] for r in macro if r['proxy']==proxy and r['family']==family and r['score']>1e-20]
                # A completely zero Top-K component is explicit, not divide-by-epsilon.
                scales[proxy][family]=float(np.median(values)) if values else 1.
            mg=np.median([r['raw_embedding_gradient_norm']/r['raw_micro_score'] for r in micro])
            ag=np.median([r['gradient']/scales[proxy][r['family']] for r in macro if r['proxy']==proxy])
            weights[proxy]=float(np.clip(ag/max(mg,1e-20),.1,10))
        freeze(root/'CALIBRATION.json',dict(micro=micro,macro=macro,micro_scales=micro_scales,macro_scales=scales,micro_weights=weights))
    finally:bundle.close();clean()


def search(root, plan):
    from llm_integrity.modeling import load_model
    from llm_integrity.inner_variant_sampler import FAMILIES,StratifiedVariantSampler
    from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer
    from llm_integrity.stage2_contract import BLOCK_TYPES
    sources=read(root/'PREFLIGHT.json')['selected'];cal=read(root/'CALIBRATION.json')
    variants=[r for r in plan['variants'] if r['split']=='search']
    registry={r['variant_id']:str(root/'adapters'/r['variant_id']) for r in variants if r['family']=='finetuning'}
    candidates=[]
    for pi,proxy in enumerate(plan['proxies']):
        for si,source in enumerate(sources):
            target=root/'search'/f"{proxy}_{source['id']}.json"
            if not target.exists():
                progress('search',proxy=proxy,source=source['id'])
                sampler=StratifiedVariantSampler(variants,family_weights={f:.2 for f in FAMILIES},adapter_registry=registry,seed=273000000+pi)
                bundle=load_model(plan['model']);optimizer=None;t=time.time()
                try:
                    optimizer=DiscreteJointInnerOptimizer(reference=bundle,sampler=sampler,model_config=plan['model'],micro_scales=cal['micro_scales'],
                        macro_scales=cal['macro_scales'][proxy],micro_weight=cal['micro_weights'][proxy],macro_weight=1.,
                        rounds=5,probes=8,candidate_positions=4,candidates_per_position=16,rerank_candidates=8,seed=274000000+pi*10000+si*100,
                        max_length=128,max_edit_ratio=.5,ppl_ratio_limit=1e12,minimum_nondegraded_families=5,family_relative_tolerance=.01,
                        variants_per_family=1,anchor_variants_per_family=0,gradient_restarts=1,family_gate_aggregation='mean',
                        require_task_preservation=False,sequential_model_execution=False,block_types=tuple(BLOCK_TYPES),representative_layer_count=4,
                        block_schedule='balanced',task_validation_mode='strict_r1',macro_proxy=proxy,macro_top_k=10,
                        enforce_surface_compatibility=False,enforce_perplexity=False)
                    result=optimizer.optimize(source)
                    save(target,dict(**result.payload(),proxy=proxy,source=source,seconds=time.time()-t))
                    progress('search_result',proxy=proxy,source=source['id'],accepted=result.accepted,seconds=time.time()-t)
                    if result.failure:raise RuntimeError(str(result.failure))
                finally:
                    (optimizer.reference if optimizer else bundle).close();clean()
            record=read(target)
            if record.get('failure'):raise RuntimeError('Prior search technical failure: '+str(target))
            if record['accepted']:
                candidates.append(dict(id=proxy+'_'+source['id'],prompt=record['optimized_prompt'],proxy=proxy,source_id=source['id']))
    unique={}
    for r in candidates:
        unique.setdefault(r['prompt'],dict(id='p_'+hashlib.sha256(r['prompt'].encode()).hexdigest()[:16],prompt=r['prompt'],aliases=[]))['aliases'].append(r['id'])
    freeze(root/'CANDIDATES.json',dict(prompts=list(unique.values()),aliases=candidates))
    if not unique:raise RuntimeError('NO_ACCEPTED_CANDIDATES: stop, do not substitute source prompts')


def reference(root, plan):
    import torch
    from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper,RepetitionPenaltyLogitsProcessor
    from llm_integrity.modeling import load_model
    from llm_integrity.resf_token import calibration
    if (root/'REFERENCE.json').exists():return
    prompts=read(root/'CANDIDATES.json')['prompts'];bundle=load_model(plan['model']);records=[]
    try:
        for i,row in enumerate(prompts):
            ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
            with torch.inference_mode():logits=bundle.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1].float()
            penalty=float(bundle.model.generation_config.repetition_penalty)
            if penalty!=1.:
                logits=RepetitionPenaltyLogitsProcessor(penalty)(ids,logits)
            probs=[]
            for temperature in plan['temperatures']:
                scores=TemperatureLogitsWarper(temperature)(ids,logits)
                scores=TopKLogitsWarper(50)(ids,scores);scores=TopPLogitsWarper(.9)(ids,scores)
                probs.append(torch.softmax(scores,-1)[0].double().cpu().numpy())
            q=np.stack(probs);support=np.flatnonzero(q.max(axis=0)>0);q=q[:,support];q/=q.sum(axis=1,keepdims=True)
            # Compare against generate's actual processed first-step distribution.
            with torch.inference_mode():check=bundle.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=1,
                do_sample=True,temperature=.7,top_k=50,top_p=.9,return_dict_in_generate=True,output_scores=True,pad_token_id=bundle.tokenizer.pad_token_id)
            actual=torch.softmax(check.scores[0].float(),-1)[0].double().cpu().numpy()
            if not np.allclose(actual[support],q[1],atol=1e-6,rtol=1e-5) or actual.sum()-actual[support].sum()>1e-6:
                raise RuntimeError('Reference/generate distribution mismatch')
            null=calibration(q,looks=tuple(plan['looks']),simulations=plan['simulations'],seed=275000000+i)
            cache=root/'null_mc'/f"{row['id']}.npz";cache.parent.mkdir(exist_ok=True)
            np.savez_compressed(cache,**{str(k):v for k,v in null.items()})
            records.append(dict(**row,token_ids=support.tolist(),probabilities=q.tolist(),temperatures=plan['temperatures'],repetition_penalty=penalty,
                input_ids_sha256=identity(ids[0].tolist()),null_mc_sha256=hashlib.sha256(cache.read_bytes()).hexdigest()))
        freeze(root/'REFERENCE.json',dict(records=records,plan_sha256=identity(plan)))
    finally:bundle.close();clean()


def evaluate(root, plan):
    from llm_integrity.modeling import load_model
    from llm_integrity.resf_token import detect
    refs=read(root/'REFERENCE.json')['records'];reports={}
    endpoints=[('intact_null_1',None),('intact_null_2',None)]+[(r['variant_id'],r) for r in plan['variants'] if r['split']!='search']
    for ei,(label,variant_row) in enumerate(endpoints):
        progress('evaluate',endpoint=label)
        path=root/'responses'/f'{label}.jsonl';path.parent.mkdir(exist_ok=True)
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        existing={(r['prompt_id'],r['response_index']):r for r in rows}
        if len(existing)!=len(rows):raise RuntimeError('Duplicate response identities')
        expected={(r['id'],j) for r in refs for j in range(100)}
        prompt_indices={r['id']:i for i,r in enumerate(refs)}
        if not set(existing)<=expected:raise RuntimeError('Unexpected response identity')
        for item in rows:
            if item['endpoint']!=label or item['plan_sha256']!=identity(plan) or len(item['token_ids'])!=1:
                raise RuntimeError('Response binding mismatch')
            if item['generation_seed']!=276000000+ei*100000+prompt_indices[item['prompt_id']]*1000+item['response_index']:
                raise RuntimeError('Response seed mismatch')
        loaded=None;bundle=None
        if len(rows)<len(refs)*100:
            bundle=load_model(plan['model']) if variant_row is None else (loaded:=load_variant(plan,variant_row,root)).bundle
        try:
            for pi,row in enumerate(refs):
                for j in range(100):
                    if (row['id'],j) in existing:continue
                    seed=276000000+ei*100000+pi*1000+j
                    response=generate(bundle,row,seed,first=True)
                    item=dict(prompt_id=row['id'],response_index=j,endpoint=label,plan_sha256=identity(plan),**response)
                    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
                    existing[(row['id'],j)]=item
            decisions=[]
            for row in refs:
                cache=root/'null_mc'/f"{row['id']}.npz"
                if hashlib.sha256(cache.read_bytes()).hexdigest()!=row['null_mc_sha256']:raise RuntimeError('Calibration cache changed')
                with np.load(cache) as data:null={int(k):data[k] for k in data.files}
                tokens=[existing[(row['id'],j)]['token_ids'][0] for j in range(100)]
                if any(existing[(row['id'],j)]['plan_sha256']!=identity(plan) for j in range(100)):raise RuntimeError('Plan mismatch')
                decisions.append(dict(prompt_id=row['id'],**detect(tokens,row['token_ids'],row['probabilities'],null,panel_size=len(refs),
                    panel_alpha=plan['panel_alpha'],e_fraction=plan['e_fraction'],looks=tuple(plan['looks']))))
            reports[label]=dict(panel_alarm=any(r['alarm'] for r in decisions),prompts=decisions)
            save(root/'decisions'/f'{label}.json',reports[label])
        finally:
            if bundle is not None:(loaded.close() if loaded else bundle.close())
            clean()
    freeze(root/'RESULTS.json',dict(endpoints=reports,scope=plan['scope'],status='EXPLORATORY_COMPLETE',
        utility=read(root/'UTILITY.json'),utility_mode=plan['utility_mode'],
        caveat='Only two real intact-null panels; not a population FPR certification. Three unseen attack seeds at one strength per family.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','train','utility','preflight','calibrate','search','reference','evaluate'])
    parser.add_argument('--utility-mode',choices=['gate','report-only'],default='gate')
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--model',default='/root/autodl-tmp/token-integrity/models/qwen15b')
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            save(root/'STATUS.json',dict(phase=args.phase,status='RUNNING',time=time.time()))
            if args.phase=='prepare':prepare(root,args.model,args.utility_mode)
            else:
                import torch
                torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
                os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
                plan=read(root/'PLAN.json')
                if plan['code_sha256']!=code_identity():raise RuntimeError('Code changed; create a new run instead of resuming')
                if identity(plan['train_data'])!=identity([json.loads(x) for x in (root/'train.jsonl').read_text(encoding='utf-8').splitlines()]):
                    raise RuntimeError('Training data changed')
                required={'train':[], 'utility':[], 'preflight':['UTILITY.json'], 'calibrate':['UTILITY.json','PREFLIGHT.json'],
                          'search':['UTILITY.json','PREFLIGHT.json','CALIBRATION.json'],
                          'reference':['CANDIDATES.json'], 'evaluate':['CANDIDATES.json','REFERENCE.json']}
                for name in required[args.phase]:
                    if not (root/name).exists():raise RuntimeError('Missing prerequisite '+name)
                if args.phase in ('preflight','calibrate','search','reference','evaluate') and plan['utility_mode']=='gate':
                    if not all(r['qualifies'] for r in read(root/'UTILITY.json').values()):raise RuntimeError('Utility gate not passed')
                globals()[args.phase](root,plan)
            save(root/'STATUS.json',dict(phase=args.phase,status='TECHNICAL_COMPLETE',time=time.time()))
        except Exception as exc:
            save(root/'STATUS.json',dict(phase=args.phase,status='FAILED',error=str(exc),time=time.time()));raise


if __name__=='__main__':main()
