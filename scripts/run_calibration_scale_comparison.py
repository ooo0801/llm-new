"""Calibration only: frozen 20-text scales and faithful two-text reproduction."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tarfile
import time
from dataclasses import asdict

import _bootstrap
import numpy as np
import torch
from run_resf_small import read, save, freeze, identity, file_identity, clean, load_variant, progress
from stage2_three_proxy_search import prompt_pool
from llm_integrity.modeling import load_model
from llm_integrity.inner_micro_proxy import discover_micro_blocks, differentiable_block_micro_proxy
from llm_integrity.macro_proxy import differentiable_macro_proxy
from llm_integrity.stage2_contract import BLOCK_TYPES

CATEGORIES = ['logic', 'math', 'extraction', 'classification']
LAYERS = [0, 9, 18, 27, 0, 9, 18]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(root, base):
    previous = base/'runs/prompt-budget-v1'
    old_plan = read(previous/'PLAN.json')
    pilot = Path(old_plan['pilot'])
    historical = read(previous/'CALIBRATION.json')
    preflight = read(pilot/'PREFLIGHT.json')
    excluded = {r['prompt'] for r in old_plan['sources'][:20]}
    search_dirs = [previous/'search', base/'runs/generation-ablation-v1/generation/search']
    for folder in search_dirs:
        for path in sorted(folder.glob('*.json')):
            row = read(path)
            excluded.update(row[k] for k in ['initial_prompt', 'optimized_prompt'] if isinstance(row.get(k), str))
    pool = prompt_pool()
    candidates, selected = {}, []
    for ci, cat in enumerate(CATEGORIES):
        rows = sorted([r for r in pool if r['category'] == cat and 15 <= int(r['id'].rsplit('_',1)[1]) <= 23
                       and r['prompt'] not in excluded], key=lambda r:r['id'])
        assert len(rows) >= 5, (cat, 'insufficient eligible calibration texts')
        candidates[cat] = rows
        selected.extend(random.Random(926010000+ci).sample(rows, 5))
    model_dir = Path(old_plan['model']['name'])
    manifest = read(model_dir/'DOWNLOAD_MANIFEST.json')
    verified = []
    for item in manifest['files']:
        path = model_dir/item['path']
        digest = sha(path)
        assert digest == item['sha256'], ('model identity mismatch', str(path))
        verified.append(dict(path=item['path'], sha256=digest, bytes=path.stat().st_size))
    assert manifest['revision'] == old_plan['model_revision']
    adapter = pilot/'adapters/lora_s0'
    adapter_hash = file_identity(adapter)
    assert adapter_hash == read(pilot/'training/lora_s0.json')['adapter_files_sha256']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    texts = selected + preflight['selected']
    tokenized = {}
    for row in texts:
        ids = tokenizer.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=True, add_generation_prompt=True)
        assert len(ids) <= 128, ('overlength', row['id'])
        tokenized[row['id']] = ids
    plan = dict(schema='calibration-scale-comparison-v1', model=old_plan['model'], model_revision=old_plan['model_revision'],
        variants=old_plan['search_variants'], pilot=str(pilot), categories=CATEGORIES, eligible_pool=candidates,
        selected=selected, old_selected=preflight['selected'], token_ids=tokenized, excluded_texts=sorted(excluded),
        blocks=list(BLOCK_TYPES), layers=LAYERS, probes=8, micro_seed_formula='926020000 + 10000*b + 100*i + probe_index',
        old_seed_formula='272000000 + b + probe_index', sample_seed=926010000, bootstrap_seed=926030000,
        bootstrap_repetitions=1000, macro_proxy='js', temperature=1., top_k=None, top_p=None,
        minimum_valid_scale=1e-20, new_median_includes_zeros=True, historical_macro_positive_filter=True,
        lambda_clip=[.1,10.], macro_weight=1., family_weights={v['family']:.2 for v in old_plan['search_variants']},
        model_files=verified, adapter_sha256=adapter_hash, historical_identity=identity(historical),
        scope='calibration only; no generation, MCC, detector, or new training')
    freeze(root/'CALIBRATION_PLAN.json', plan)
    freeze(root/'HISTORICAL.json', historical)
    freeze(root/'OLD_PREFLIGHT.json', preflight)
    freeze(root/'MODEL_DOWNLOAD_MANIFEST.json', manifest)
    freeze(root/'ADAPTER_TRAINING.json', read(pilot/'training/lora_s0.json'))
    repo = Path(__file__).resolve().parents[1]
    code = {}
    for folder in ['scripts', 'src/llm_integrity']:
        for path in sorted((repo/folder).rglob('*.py')):
            rel = str(path.relative_to(repo)); code[rel] = sha(path)
            dest = root/'code_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists(): assert sha(dest) == code[rel], ('code changed', rel)
            else: shutil.copy2(path, dest)
    freeze(root/'CODE_MANIFEST.json', code)
    packages = {name:importlib.metadata.version(name) for name in ['torch','transformers','peft','numpy','bitsandbytes','accelerate']}
    env = dict(python=platform.python_version(), packages=packages, cuda=torch.version.cuda,
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip(),
        git_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32)
    freeze(root/'ENVIRONMENT.json', env)
    return plan


def compute(root, plan):
    binding = identity(plan)
    bundle = load_model(plan['model']); bundle.model.eval()
    for p in bundle.model.parameters(): p.requires_grad_(False)
    blocks = discover_micro_blocks(bundle.model, plan['blocks'])
    def inputs(row):
        ids = torch.tensor([plan['token_ids'][row['id']]], device=bundle.device)
        e = bundle.model.get_input_embeddings()(ids).detach().requires_grad_(True)
        return e, torch.ones_like(ids)
    def existing(path):
        if not path.exists(): return False
        assert read(path)['plan_identity'] == binding
        return True
    def record(path, data):
        data.update(plan_identity=binding, completed_at=time.time())
        save(path,data)
        progress('record_complete',file=str(path.relative_to(root)))
        save(root/'STATUS.json',dict(phase='calculating',last_record=str(path.relative_to(root)),time=time.time()))
    try:
        for mode in ['old', 'new']:
            for bi, kind in enumerate(plan['blocks']):
                block = next(b for b in blocks if b.block_type==kind and b.layer_id==plan['layers'][bi])
                rows = [(bi%2,plan['old_selected'][bi%2])] if mode=='old' else list(enumerate(plan['selected']))
                for i,row in rows:
                    path = root/'records'/mode/f'micro_{bi}_{i:02d}.json'
                    if existing(path): continue
                    seed = 272000000+bi if mode=='old' else 926020000+10000*bi+100*i
                    e,mask = inputs(row)
                    result = differentiable_block_micro_proxy(bundle,e,mask,block=block,probes=8,seed=seed)
                    data = result.metadata()
                    assert math.isfinite(data['raw_micro_score']) and math.isfinite(data['raw_embedding_gradient_norm'])
                    assert data['gradient_finite'] and data['parameter_gradient_finite']
                    record(path,dict(mode=mode,kind='micro',source_id=row['id'],category=row['category'],source_index=i,**data))
                    del result,e,mask; clean()
        for variant in plan['variants']:
            family = variant['family']
            pending = any(not (root/'records'/mode/f'macro_{family}_{i:02d}.json').exists()
                for mode in ['old','new'] for i in range(2 if mode=='old' else 20))
            if not pending: continue
            loaded = load_variant(plan,variant,Path(plan['pilot']))
            loaded.bundle.model.eval()
            freeze(root/'variant_reports'/f'{family}.json',asdict(loaded.report))
            for p in loaded.bundle.model.parameters(): p.requires_grad_(False)
            try:
                for mode in ['old','new']:
                    for i,row in enumerate(plan['old_selected'] if mode=='old' else plan['selected']):
                        path=root/'records'/mode/f'macro_{family}_{i:02d}.json'
                        if existing(path): continue
                        e,mask=inputs(row)
                        a=bundle.model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits[0,-1].float()
                        b=loaded.bundle.model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits[0,-1].float()
                        score=differentiable_macro_proxy(a,b,proxy='js',top_k=10)
                        grad=torch.autograd.grad(score,e)[0]
                        assert torch.isfinite(grad).all() and torch.isfinite(score)
                        record(path,dict(mode=mode,kind='macro',family=family,variant_id=variant['variant_id'],
                            source_id=row['id'],category=row['category'],source_index=i,score=float(score),
                            gradient=float(grad.float().norm()),proxy='js'))
                        del e,mask,a,b,score,grad;clean()
            finally: loaded.close();clean()
    finally: bundle.close();clean()


def summarize(micro, macro, blocks, families, indices=None, old=False):
    if indices is not None:
        micro=[r for i in indices for r in micro if r['source_index']==i]
        macro=[r for i in indices for r in macro if r['source_index']==i]
    ms={b:float(np.median([r['raw_micro_score'] for r in micro if r['block']['block_type']==b])) for b in blocks}
    ass={}
    for f in families:
        values=[r['score'] for r in macro if r['family']==f and (not old or r['score']>1e-20)]
        ass[f]=float(np.median(values)) if values else 1.
    valid=all(np.isfinite(v) and v>1e-20 for v in list(ms.values())+list(ass.values()))
    if not valid: return dict(valid=False,micro_scales=ms,macro_scales=ass)
    mg=float(np.median([r['raw_embedding_gradient_norm']/ms[r['block']['block_type']] for r in micro]))
    ag=float(np.median([r['gradient']/ass[r['family']] for r in macro]))
    ratio=ag/max(mg,1e-20)
    return dict(valid=True,micro_scales=ms,macro_scales=ass,gradient_micro_median=mg,gradient_macro_median=ag,
        micro_weight_unclipped=ratio,micro_weight=float(np.clip(ratio,.1,10)),weight_clipped=ratio<.1 or ratio>10,macro_weight=1.)


def aggregate(root,plan):
    families=[v['family'] for v in plan['variants']]
    records={mode:[read(p) for p in sorted((root/'records'/mode).glob('*.json'))] for mode in ['old','new']}
    summaries={}
    for mode,rows in records.items():
        micro=[r for r in rows if r['kind']=='micro'];macro=[r for r in rows if r['kind']=='macro']
        assert (len(micro),len(macro)) == ((7,10) if mode=='old' else (140,100))
        assert all(r['plan_identity']==identity(plan) for r in rows)
        summaries[mode]=summarize(micro,macro,plan['blocks'],families,old=mode=='old')
        with (root/f'{mode.upper()}_RAW.jsonl').open('w',encoding='utf8') as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    save(root/'CALIBRATION_OLD_RECOMPUTED.json',summaries['old'])
    assert summaries['new']['valid'], 'Invalid new scale: results retained, calibration not frozen'
    freeze(root/'CALIBRATION.json',summaries['new'])
    micro=[r for r in records['new'] if r['kind']=='micro'];macro=[r for r in records['new'] if r['kind']=='macro']
    rng=np.random.default_rng(926030000);boots=[]
    for _ in range(1000):
        indices=np.concatenate([rng.choice(np.arange(c*5,(c+1)*5),5,replace=True) for c in range(4)]).tolist()
        boots.append(summarize(micro,macro,plan['blocks'],families,indices))
    valid=[b for b in boots if b['valid']]
    ci={}
    for kind,keys in [('micro_scales',plan['blocks']),('macro_scales',families)]:
        for key in keys: ci[key]=np.percentile([b[kind][key] for b in valid],[2.5,50,97.5]).tolist()
    ci['micro_weight']=np.percentile([b['micro_weight'] for b in valid],[2.5,50,97.5]).tolist()
    ci['micro_weight_unclipped']=np.percentile([b['micro_weight_unclipped'] for b in valid],[2.5,50,97.5]).tolist()
    save(root/'BOOTSTRAP.json',dict(seed=926030000,repetitions=1000,valid=len(valid),invalid=1000-len(valid),percentiles=ci))
    stats={}
    for name,subset,field,grad in [(b,[r for r in micro if r['block']['block_type']==b],'raw_micro_score','raw_embedding_gradient_norm') for b in plan['blocks']]+[(f,[r for r in macro if r['family']==f],'score','gradient') for f in families]:
        stats[name]={}
        for cat in ['all']+CATEGORIES:
            rows=subset if cat=='all' else [r for r in subset if r['category']==cat]
            stats[name][cat]=dict(count=len(rows),percentiles=np.percentile([r[field] for r in rows],[0,25,50,75,100]).tolist(),
                zero_scores=sum(r[field]==0 for r in rows),zero_gradients=sum(r[grad]==0 for r in rows))
    save(root/'DESCRIPTIVE_STATS.json',stats)
    old=read(root/'HISTORICAL.json');comparison=[]
    for kind,keys in [('micro_scales',plan['blocks']),('macro_scales',families)]:
        hist=old[kind] if kind=='micro_scales' else old[kind]['js']
        for key in keys:
            a,b,c=hist[key],summaries['old'][kind][key],summaries['new'][kind][key]
            comparison.append(dict(kind=kind,name=key,historical=a,current_old=b,current_new=c,
                new_over_current_old=c/b,new_relative_change_percent=100*(c/b-1),current_old_over_historical=b/a,
                bootstrap_95=[ci[key][0],ci[key][2]]))
    save(root/'COMPARISON.json',dict(scales=comparison,weights=dict(historical=old['micro_weights']['js'],
        current_old=summaries['old']['micro_weight'],current_new=summaries['new']['micro_weight'])))
    save(root/'CALIBRATION_AUDIT.json',dict(status='PASS',counts={k:len(v) for k,v in records.items()},new_micro=140,new_macro=100,
        disjoint=not set(r['prompt'] for r in plan['selected'])&set(plan['excluded_texts']),
        unique_sources=len(set(r['id'] for r in plan['selected'])),bootstrap_valid=len(valid),
        scope='calibration only',technical_failures_converted_to_data=False))
    save(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
    progress('COMPLETE',root=str(root))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--base',type=Path,default=Path('/root/autodl-tmp/token-integrity'))
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    with (args.root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(args.root,args.base)
        compute(args.root,plan)
        aggregate(args.root,plan)


if __name__=='__main__': main()
