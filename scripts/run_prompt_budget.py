"""Frozen prompt-generation and query-budget comparison; resumable phases."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import _bootstrap
import numpy as np
from run_resf_small import read, save, freeze, identity, clean, progress


def prepare(root, base, pilot):
    from stage2_three_proxy_search import prompt_pool
    matrix = read(base/'MATRIX_PLAN.json')
    old = read(pilot/'PLAN.json')
    categories = ['logic', 'math', 'extraction', 'classification']
    sources = [r for i in range(10) for c in categories
               for r in prompt_pool() if r['category'] == c and int(r['id'].rsplit('_', 1)[1]) == i]
    p = dict(schema='prompt-budget-v1', base=str(base), pilot=str(pilot),
             model=matrix['model'], model_revision=matrix['model_revision'],
             ordinary=matrix['prompts'], sources=sources, proxies=old['proxies'],
             search_variants=[r for r in old['variants'] if r['split']=='search'],
             development_variants=[r for r in old['variants'] if r['split']!='search'],
             test_variants=matrix['variants'], sizes=[1,2,5,10,20], queries=[10,25,50,100],
             simulations=10000, normal_panels=20, alpha=.05, e_fraction=.2,
             random_panels=100, seed=319000000,
             generation='60 jobs on first 20 sources; if fewer than 20 unique accepted prompts, next 60 jobs on next 20 sources',
             ranking='development attack detection AUC over four query budgets at panel_size=20; then coverage; ID tie break',
             detector='same E/P statistic, one terminal P look per fixed-budget configuration, union support E at every query',
             calibration='independent multinomial MC from intact reference, separate from 20 normal evaluation panels',
             acceptance=dict(attack_hits=38, normal_false_alarms=1),
             caveat='40 test attacks already examined for ordinary panel; budget minima are exploratory, not independent confirmatory guarantees',
             costs='online query budget excludes separately reported offline generation/reference/calibration cost')
    assert not {r['seed'] for r in p['test_variants']} & {r['seed'] for r in p['search_variants']+p['development_variants']}
    assert not {r['prompt'] for r in p['ordinary']} & {r['prompt'] for r in sources}
    freeze(root/'PLAN.json',p)
    freeze(root/'CALIBRATION.json',read(pilot/'CALIBRATION.json'))
    return p


def search(root,p):
    from llm_integrity.modeling import load_model
    from llm_integrity.inner_variant_sampler import FAMILIES, StratifiedVariantSampler
    from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer
    from llm_integrity.stage2_contract import BLOCK_TYPES
    if (root/'CANDIDATES.json').exists(): return read(root/'CANDIDATES.json')
    cal=read(root/'CALIBRATION.json'); pilot=Path(p['pilot']); rows=[]
    registry={r['variant_id']:str(pilot/'adapters'/r['variant_id']) for r in p['search_variants'] if r['family']=='finetuning'}
    started=time.time()
    for si,source in enumerate(p['sources']):
        if si==20 and len({r['prompt'] for r in rows})>=20: break
        for pi,proxy in enumerate(p['proxies']):
            target=root/'search'/f"{proxy}_{source['id']}.json"
            if not target.exists():
                progress('search_start',source=source['id'],proxy=proxy,accepted=len(rows))
                sampler=StratifiedVariantSampler(p['search_variants'],family_weights={f:.2 for f in FAMILIES},adapter_registry=registry,seed=p['seed']+pi)
                bundle=load_model(p['model']); optimizer=None; t=time.time()
                try:
                    optimizer=DiscreteJointInnerOptimizer(reference=bundle,sampler=sampler,model_config=p['model'],
                        micro_scales=cal['micro_scales'],macro_scales=cal['macro_scales'][proxy],micro_weight=cal['micro_weights'][proxy],macro_weight=1.,
                        rounds=5,probes=8,candidate_positions=4,candidates_per_position=16,rerank_candidates=8,seed=p['seed']+10000+si*100+pi,
                        max_length=128,max_edit_ratio=.5,ppl_ratio_limit=1e12,minimum_nondegraded_families=5,family_relative_tolerance=.01,
                        variants_per_family=1,anchor_variants_per_family=0,gradient_restarts=1,family_gate_aggregation='mean',
                        require_task_preservation=False,sequential_model_execution=False,block_types=tuple(BLOCK_TYPES),representative_layer_count=4,
                        block_schedule='balanced',task_validation_mode='strict_r1',macro_proxy=proxy,macro_top_k=10,
                        enforce_surface_compatibility=False,enforce_perplexity=False)
                    result=optimizer.optimize(source)
                    record=dict(**result.payload(),proxy=proxy,source=source,seconds=time.time()-t)
                    save(target,record)
                finally:
                    (optimizer.reference if optimizer else bundle).close(); clean()
            record=read(target)
            if record.get('failure'): raise RuntimeError(f'Technical search failure: {target}: {record["failure"]}')
            if record['accepted']:
                text=record['optimized_prompt']
                rows.append(dict(id='s_'+hashlib.sha256(text.encode()).hexdigest()[:16],prompt=text,
                                 source_id=source['id'],category=source['category'],proxy=proxy,
                                 proxy_gain=record['proxy_objective_gain']))
            progress('search_done',source=source['id'],proxy=proxy,accepted=record['accepted'],seconds=record.get('seconds'))
            save(root/'STATUS.json',dict(phase='search',jobs=len(list((root/'search').glob('*.json'))),unique=len({r['prompt'] for r in rows}),time=time.time()))
    unique={}
    for r in rows: unique.setdefault(r['prompt'],r)
    if len(unique)<20: raise RuntimeError(f'Only {len(unique)} accepted unique prompts after fixed search budget; cannot substitute rejected prompts')
    result=dict(prompts=list(unique.values()),aliases=rows,elapsed_this_session=time.time()-started)
    freeze(root/'CANDIDATES.json',result)
    return result


def main():
    a=argparse.ArgumentParser();a.add_argument('phase',choices=['prepare','search']);a.add_argument('--root',type=Path,required=True)
    a.add_argument('--base',type=Path,required=True);a.add_argument('--pilot',type=Path,required=True);x=a.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4)
    root=x.root.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=read(root/'PLAN.json') if (root/'PLAN.json').exists() else prepare(root,x.base.resolve(),x.pilot.resolve())
            if x.phase=='search': search(root,p)
        except Exception as exc:
            save(root/'STATUS.json',dict(phase='FAILED',error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
