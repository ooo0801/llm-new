"""Scaling validation: four strengths, five seeds, twenty prompts and panels."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, time
from pathlib import Path
import numpy as np

import _bootstrap

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); t.replace(p)
def ident(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def load_responses(path):
    if not path.exists(): return []
    text=path.read_text(encoding='utf-8')
    if text.lstrip().startswith('['):return json.loads(text)
    return [json.loads(x) for x in text.splitlines() if x.strip()]
def check_responses(rows,p,label,ei):
    indices={r['id']:i for i,r in enumerate(p['prompts'])}; seen=set()
    for r in rows:
        key=(r['prompt_id'],r['response_index'])
        if key in seen or key[0] not in indices or not 0<=key[1]<p['responses_per_prompt']:raise ValueError('Invalid response identity')
        if r['endpoint']!=label or r['seed']!=284000000+ei*100000+indices[key[0]]*1000+key[1]:raise ValueError('Response binding mismatch')
        if type(r['token_id']) is not int or r['token_id']<0:raise ValueError('Invalid token')
        seen.add(key)
    return seen
def clean():
    gc.collect()
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
def log(**x): print(json.dumps(dict(time=time.time(),**x),ensure_ascii=False),flush=True)

def plan(root,model):
    from stage2_three_proxy_search import prompt_pool
    pool=prompt_pool(); by={c:[x for x in pool if x['category']==c] for c in ['logic','math','extraction','classification']}
    # Search used extraction/classification pool_00; these held-out rows never enter search.
    prompts=[x for c in ['logic','math','extraction','classification'] for x in by[c][10:15]]
    variants=[]
    gs=[.001,.0025,.004,.006]; ls=[5,10,20,40]
    for si,s in enumerate(gs):
        for seed in range(5): variants.append(dict(family='gaussian_noise',variant_id=f'gaussian_g{si}_s{seed}',seed=281000000+si*100+seed,configuration=dict(method='relative_gaussian',std_ratio=s,target_scope='attention_ffn',measure_realized=True),strength=s))
    for si,steps in enumerate(ls):
        for seed in range(5): variants.append(dict(family='finetuning',variant_id=f'lora_g{si}_s{seed}',seed=282000000+si*100+seed,configuration=dict(method='lora',rank=8,alpha=16,dropout=0.,learning_rate=1e-4,steps=steps,lr_scheduler_type='constant',target_scope='attention_ffn'),strength=steps))
    p=dict(schema='resf-token-scaling-v1',model=dict(name=str(Path(model).resolve()),dtype='bfloat16',device_map='auto',local_files_only=True,trust_remote_code=False,eager_attention=True),model_revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',prompts=prompts,variants=variants,normal_panels=20,responses_per_prompt=100,temperatures=[.5,.7,.9],temperature=.7,top_k=50,top_p=.9,looks=[30,60,100],simulations=10000,panel_alpha=.05,e_fraction=.2,budgets=dict(attack_endpoints=40,normal_panels=20,prompts=20,responses_per_prompt=100,total_first_token_requests=120000,lora_optimizer_steps=sum(ls)*5))
    save(root/'MATRIX_PLAN.json',p); return p

def generate(bundle,row,seed):
    import torch
    from transformers import set_seed
    ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
    set_seed(seed)
    with torch.inference_mode(): out=bundle.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=1,do_sample=True,temperature=.7,top_k=50,top_p=.9,pad_token_id=bundle.tokenizer.pad_token_id)
    return int(out[0,ids.shape[1]].item())

def train(root,p):
    from llm_integrity.paper_finetuning import train_lora_manifest_variant
    for row in p['variants']:
        if row['family']!='finetuning': continue
        target=root/'matrix_training'/f"{row['variant_id']}.json"; adapter=root/'matrix_adapters'/row['variant_id']
        if target.exists() and adapter.exists(): continue
        log(phase='train',endpoint=row['variant_id'])
        r=train_lora_manifest_variant(model_config=p['model'],variant=row,data_path=root/'train.jsonl',output_root=root/'matrix_adapters',max_length=128,batch_size=1,gradient_accumulation_steps=4)
        if r.completed_steps!=row['configuration']['steps']: raise RuntimeError(f'incomplete {row["variant_id"]}')
        save(target,dict(variant=row,completed_steps=r.completed_steps,training_loss=r.training_loss))
        clean()

def reference(root,p):
    import torch
    from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper,RepetitionPenaltyLogitsProcessor
    from llm_integrity.modeling import load_model
    from llm_integrity.resf_token import calibration
    if (root/'MATRIX_REFERENCE.json').exists(): return read(root/'MATRIX_REFERENCE.json')
    b=load_model(p['model']); out=[]
    try:
        for i,row in enumerate(p['prompts']):
            ids=b.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(b.device)
            with torch.inference_mode(): logits=b.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1].float()
            penalty=float(b.model.generation_config.repetition_penalty)
            if penalty!=1.: logits=RepetitionPenaltyLogitsProcessor(penalty)(ids,logits)
            probs=[]
            for temp in p['temperatures']:
                s=TemperatureLogitsWarper(temp)(ids,logits); s=TopKLogitsWarper(50)(ids,s); s=TopPLogitsWarper(.9)(ids,s); probs.append(torch.softmax(s,-1)[0].double().cpu().numpy())
            q=np.stack(probs); support=np.flatnonzero(q.max(axis=0)>0); q=q[:,support]; q/=q.sum(1,keepdims=True)
            null=calibration(q,looks=tuple(p['looks']),simulations=p['simulations'],seed=283000000+i)
            cache=root/'matrix_null_mc'/f'{i}.npz'; cache.parent.mkdir(exist_ok=True); np.savez_compressed(cache,**{str(k):v for k,v in null.items()})
            out.append(dict(prompt_id=row['id'],token_ids=support.tolist(),probabilities=q.tolist(),null_mc_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),input_ids_sha256=ident(ids[0].tolist())))
            log(phase='reference',prompt=i)
    finally: b.close(); clean()
    save(root/'MATRIX_REFERENCE.json',out); return out

def endpoint_bundle(p,row,root):
    from llm_integrity.modeling import load_model
    if row is None:return None,load_model(p['model'])
    from llm_integrity.paper_variant_executor import load_manifest_variant
    ap=root/'matrix_adapters'/row['variant_id'] if row['family']=='finetuning' else None
    loaded=load_manifest_variant(p['model'],row,adapter_path=ap)
    return loaded,loaded.bundle

def evaluate(root,p,refs):
    from llm_integrity.resf_token import detect
    from llm_integrity.modeling import load_model
    endpoints=[dict(label=f'normal_panel_{i:02d}',row=None,panel=i) for i in range(20)] + [dict(label=x['variant_id'],row=x,panel=0) for x in p['variants']]
    reports={}
    for ei,e in enumerate(endpoints):
        path=root/'matrix_responses'/f"{e['label']}.jsonl"; path.parent.mkdir(exist_ok=True)
        rows=load_responses(path)
        done=check_responses(rows,p,e['label'],ei)
        if path.exists() and path.read_text(encoding='utf-8').lstrip().startswith('['):
            tmp=path.with_suffix('.migrating');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');tmp.replace(path)
        loaded=None;b=None
        if len(rows)<len(p['prompts'])*p['responses_per_prompt']: loaded,b=endpoint_bundle(p,e['row'],root)
        log(phase='evaluate_start',endpoint=e['label'],existing=len(rows))
        try:
            for pi,prompt in enumerate(p['prompts']):
                for j in range(p['responses_per_prompt']):
                    if (prompt['id'],j) in done: continue
                    seed=284000000+ei*100000+pi*1000+j
                    rows.append(dict(endpoint=e['label'],prompt_id=prompt['id'],response_index=j,seed=seed,token_id=generate(b,prompt,seed)))
                    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rows[-1],ensure_ascii=False)+'\n')
            decisions=[]
            for pi,ref in enumerate(refs):
                tokens=[x['token_id'] for x in sorted(rows,key=lambda r:r['response_index']) if x['prompt_id']==ref['prompt_id']]
                cache=root/'matrix_null_mc'/f'{pi}.npz'
                if hashlib.sha256(cache.read_bytes()).hexdigest()!=ref['null_mc_sha256']:raise RuntimeError('Null cache changed')
                with np.load(cache) as data:null={int(k):data[k] for k in data.files}
                decisions.append(dict(prompt_id=ref['prompt_id'],**detect(tokens,ref['token_ids'],ref['probabilities'],null,panel_size=20,panel_alpha=.05,e_fraction=.2,looks=tuple(p['looks']))))
            reports[e['label']]=dict(panel_alarm=any(x['alarm'] for x in decisions),prompts=decisions)
            save(root/'matrix_decisions'/f"{e['label']}.json",reports[e['label']]); log(phase='evaluate_done',endpoint=e['label'],alarm=reports[e['label']]['panel_alarm'])
        finally:
            if loaded:loaded.close()
            elif b is not None:b.close()
            clean()
    save(root/'MATRIX_RESULTS.json',dict(status='EXPLORATORY_COMPLETE',plan=p,reports=reports)); return reports

def main():
    a=argparse.ArgumentParser(); a.add_argument('--root',type=Path,required=True); a.add_argument('--model',required=True); a.add_argument('phase',choices=['plan','train','reference','evaluate','all']); x=a.parse_args(); root=x.root.resolve();root.mkdir(parents=True,exist_ok=True); os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    p=read(root/'MATRIX_PLAN.json') if (root/'MATRIX_PLAN.json').exists() and x.phase!='plan' else plan(root,x.model)
    if x.phase in ('train','all'): train(root,p)
    if x.phase in ('reference','all'): refs=reference(root,p)
    elif x.phase=='evaluate': refs=read(root/'MATRIX_REFERENCE.json')
    if x.phase in ('evaluate','all'): evaluate(root,p,refs)
if __name__=='__main__': main()
