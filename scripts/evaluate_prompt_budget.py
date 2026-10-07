"""Budget sweeps using frozen rankings; no test-dependent prompt selection."""
from __future__ import annotations
import argparse
import hashlib
import os
import time
from collections import defaultdict
from pathlib import Path
import _bootstrap
import numpy as np
from run_resf_small import read, save, freeze, identity, clean, progress, generate, load_variant
from run_scaling_matrix import load_responses, endpoint_bundle, check_responses
from llm_integrity.resf_token import calibration, detect


def reference(root,p,prompts):
    import torch
    from transformers import TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper,RepetitionPenaltyLogitsProcessor
    from llm_integrity.modeling import load_model
    refs={};bundle=None
    try:
        for pi,row in enumerate(prompts):
            target=root/'references'/f"{row['id']}.json"
            cache=root/'null_mc'/f"{row['id']}.npz"
            if target.exists():
                ref=read(target)
                if ref['prompt']!=row['prompt'] or ref['null_sha256']!=hashlib.sha256(cache.read_bytes()).hexdigest():
                    raise RuntimeError('Reference/cache binding changed')
                refs[row['id']]=ref;continue
            if bundle is None: bundle=load_model(p['model'])
            ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
            with torch.inference_mode(): logits=bundle.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[:,-1].float()
            penalty=float(bundle.model.generation_config.repetition_penalty)
            if penalty!=1:logits=RepetitionPenaltyLogitsProcessor(penalty)(ids,logits)
            probs=[];checks=[]
            for temp in [.5,.7,.9]:
                scores=TemperatureLogitsWarper(temp)(ids,logits)
                scores=TopKLogitsWarper(50)(ids,scores);scores=TopPLogitsWarper(.9)(ids,scores)
                prob=torch.softmax(scores,-1)[0].double().cpu().numpy()
                with torch.inference_mode(): check=bundle.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=1,
                    do_sample=True,temperature=temp,top_k=50,top_p=.9,return_dict_in_generate=True,output_scores=True,pad_token_id=bundle.tokenizer.pad_token_id)
                actual=torch.softmax(check.scores[0].float(),-1)[0].double().cpu().numpy()
                if not np.allclose(actual,prob,atol=1e-6,rtol=1e-5):raise RuntimeError('Generate distribution mismatch')
                probs.append(prob);checks.append(dict(temperature=temp,max_error=float(np.max(np.abs(actual-prob)))))
            q=np.stack(probs);support=np.flatnonzero(q.max(axis=0)>0);q=q[:,support];q/=q.sum(1,keepdims=True)
            null=calibration(q,looks=tuple(p['queries']),simulations=p['simulations'],seed=p['seed']+20000+pi)
            cache.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache,**{str(n):v for n,v in null.items()})
            ref=dict(**row,token_ids=support.tolist(),probabilities=q.tolist(),decoding_checks=checks,
                     null_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),prompt_ids_sha256=identity(ids[0].tolist()))
            freeze(target,ref);refs[row['id']]=ref
            progress('reference',prompt=row['id'],index=pi)
    finally:
        if bundle:bundle.close();clean()
    return refs


def response_seed(p,split,ei,pi,j):
    return p['seed']+1000000+(100000000 if split=='test' else 0)+ei*100000+pi*1000+j


def validate_rows(rows,p,prompts,split,label,ei):
    indices={r['id']:i for i,r in enumerate(prompts)};seen=set()
    for r in rows:
        key=(r['prompt_id'],r['response_index'])
        if key in seen or key[0] not in indices or not 0<=key[1]<100:raise RuntimeError('Response identity mismatch')
        if r['endpoint']!=label or r['split']!=split or r['plan_sha256']!=identity(p):raise RuntimeError('Response binding mismatch')
        if r['seed']!=response_seed(p,split,ei,indices[key[0]],key[1]):raise RuntimeError('Response seed mismatch')
        if type(r['token_id']) is not int or r['token_id']<0:raise RuntimeError('Invalid token')
        seen.add(key)
    return seen


def collect(root,p,prompts,split):
    from llm_integrity.modeling import load_model
    import json
    if split=='development':
        endpoints=[dict(label=r['variant_id'],row=r) for r in p['development_variants']]
    else:
        endpoints=[dict(label=f'normal_panel_{i:02d}',row=None) for i in range(p['normal_panels'])]+[dict(label=r['variant_id'],row=r) for r in p['test_variants']]
    for ei,e in enumerate(endpoints):
        path=root/(split+'_responses')/(e['label']+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
        rows=load_responses(path);seen=validate_rows(rows,p,prompts,split,e['label'],ei)
        loaded=None;bundle=None
        if len(seen)==len(prompts)*100:continue
        try:
            if split=='development':
                loaded=load_variant(read(Path(p['pilot'])/'PLAN.json'),e['row'],Path(p['pilot']));bundle=loaded.bundle
            else:loaded,bundle=endpoint_bundle(p,e['row'],Path(p['base']))
            progress('sampling',split=split,endpoint=e['label'],existing=len(rows))
            with path.open('a',encoding='utf-8') as output:
                for pi,row in enumerate(prompts):
                    for j in range(100):
                        if (row['id'],j) in seen:continue
                        seed=response_seed(p,split,ei,pi,j)
                        token=generate(bundle,row,seed,first=True)['token_ids'][0]
                        record=dict(endpoint=e['label'],split=split,prompt_id=row['id'],response_index=j,seed=seed,token_id=int(token),plan_sha256=identity(p))
                        output.write(json.dumps(record,ensure_ascii=False)+'\n');output.flush()
                    save(root/'STATUS.json',dict(phase=split,endpoint=e['label'],prompt=pi+1,total_prompts=len(prompts),time=time.time()))
        finally:
            if loaded:loaded.close()
            elif bundle:bundle.close()
            clean()


def token_arrays(rows):
    out=defaultdict(list)
    for r in sorted(rows,key=lambda r:r['response_index']):out[r['prompt_id']].append(r['token_id'])
    return out


def nulls(root,refs):
    out={}
    for pid,ref in refs.items():
        path=root/'null_mc'/f'{pid}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['null_sha256']:raise RuntimeError('Null cache changed')
        with np.load(path) as d:out[pid]={int(k):d[k] for k in d.files}
    return out


def decision(tokens,ref,mc,m,n,p):
    return detect(tokens[:n],ref['token_ids'],ref['probabilities'],mc,panel_size=m,
                  panel_alpha=p['alpha'],e_fraction=p['e_fraction'],looks=(n,))


def ranking(root,p,prompts,refs):
    path=root/'RANKING.json'
    if path.exists():return read(path)
    mc=nulls(root,refs);scores={r['id']:dict(hits=0,coverage=0) for r in prompts}
    records=[]
    for row in p['development_variants']:
        samples=load_responses(root/'development_responses'/f"{row['variant_id']}.jsonl")
        tokens=token_arrays(samples)
        for r in prompts:
            hits=[]
            for n in p['queries']:
                d=decision(tokens[r['id']],refs[r['id']],mc[r['id']],20,n,p)
                hits.append(d['alarm']);records.append(dict(endpoint=row['variant_id'],prompt=r['id'],queries=n,**d))
            scores[r['id']]['hits']+=sum(hits);scores[r['id']]['coverage']+=bool(hits[-1])
    def ordered(ids):return sorted(ids,key=lambda pid:(-scores[pid]['hits'],-scores[pid]['coverage'],pid))
    ordinary=ordered([r['id'] for r in p['ordinary']]);sensitive=ordered([r['id'] for r in read(root/'CANDIDATES.json')['prompts']])[:20]
    freeze(path,dict(ordinary=ordinary,sensitive=sensitive,scores=scores,development_decisions=records,
                    independent_test_seeds=True,created_before_test_responses=not (root/'test_responses').exists()))
    return read(path)


def summarize(root,p,refs,ranks):
    path=root/'BUDGET_RESULTS.json'
    if path.exists():return read(path)
    endpoints=[f'normal_panel_{i:02d}' for i in range(20)]+[r['variant_id'] for r in p['test_variants']]
    all_tokens={};audit=[]
    base=Path(p['base']);old_plan=read(base/'MATRIX_PLAN.json');mc=nulls(root,refs)
    sensitive=[refs[pid] for pid in ranks['sensitive']]
    for ei,label in enumerate(endpoints):
        ordinary_rows=load_responses(base/'matrix_responses'/f'{label}.jsonl')
        if len(check_responses(ordinary_rows,old_plan,label,ei))!=2000:raise RuntimeError('Incomplete ordinary responses')
        new_rows=load_responses(root/'test_responses'/f'{label}.jsonl')
        if len(validate_rows(new_rows,p,sensitive,'test',label,ei))!=2000:raise RuntimeError('Incomplete sensitive responses')
        all_tokens[label]={**token_arrays(ordinary_rows),**token_arrays(new_rows)}
        audit.append(dict(endpoint=label,ordinary=len(ordinary_rows),sensitive=len(new_rows)))
    freeze(root/'RESPONSE_AUDIT.json',dict(status='PASS',endpoints=audit,total_records=sum(r['ordinary']+r['sensitive'] for r in audit)))
    rng=np.random.default_rng(p['seed']+30000)
    permutations=[rng.permutation([r['id'] for r in p['ordinary']]).tolist() for _ in range(p['random_panels'])]
    freeze(root/'RANDOM_PANELS.json',permutations)
    groups=defaultdict(list)
    for i,r in enumerate(p['test_variants']):groups[f"{r['family']}:{r['strength']}"] .append(i+20)
    results=[];prompt_records=[]
    for m in p['sizes']:
        for n in p['queries']:
            decisions={}
            for pid in ranks['ordinary']+ranks['sensitive']:
                decisions[pid]=[decision(all_tokens[e][pid],refs[pid],mc[pid],m,n,p) for e in endpoints]
                for e,d in zip(endpoints,decisions[pid]):prompt_records.append(dict(prompt=pid,endpoint=e,prompts=m,queries_per_prompt=n,**d))
            for method,panels in [('sensitive_ranked',[ranks['sensitive'][:m]]),('ordinary_ranked',[ranks['ordinary'][:m]]),
                                  ('ordinary_random',[r[:m] for r in permutations])]:
                for ri,panel in enumerate(panels):
                    alarms=[any(decisions[pid][ei]['alarm'] for pid in panel) for ei in range(60)]
                    k=sum(alarms[20:]);fp=sum(alarms[:20])
                    # Actual requests under prompt-major execution; only E can stop before terminal n.
                    requests=[]
                    for ei in range(60):
                        spent=0
                        for pid in panel:
                            d=decisions[pid][ei];spent+=d['query']
                            if d['alarm']:break
                        requests.append(spent)
                    results.append(dict(method=method,replicate=ri,prompts=m,queries_per_prompt=n,budget=m*n,
                        prompt_ids=panel,hits=k,false_alarms=fp,tpr=k/40,fpr=fp/20,meets_target=k>=38 and fp<=1,
                        matches_observed_baseline=k==40 and fp==0,alarms=alarms,actual_queries=requests,
                        groups={g:dict(hits=sum(alarms[i] for i in indices),n=len(indices)) for g,indices in groups.items()}))
            progress('budget_scored',prompts=m,queries=n)
    def minimum(method,strict=False):
        rows=[r for r in results if r['method']==method and r['replicate']==0 and r['matches_observed_baseline' if strict else 'meets_target']]
        return min(rows,key=lambda r:(r['budget'],r['prompts'])) if rows else None
    random_summary=[]
    for m in p['sizes']:
        for n in p['queries']:
            rows=[r for r in results if r['method']=='ordinary_random' and r['prompts']==m and r['queries_per_prompt']==n]
            random_summary.append(dict(prompts=m,queries_per_prompt=n,budget=m*n,mean_tpr=float(np.mean([r['tpr'] for r in rows])),
                tpr_q05_q95=np.quantile([r['tpr'] for r in rows],[.05,.95]).tolist(),mean_fpr=float(np.mean([r['fpr'] for r in rows])),
                fraction_meeting_target=float(np.mean([r['meets_target'] for r in rows])),
                fraction_matching_baseline=float(np.mean([r['matches_observed_baseline'] for r in rows]))))
    # Paired endpoint-stratified bootstrap preserves five seeds in each attack-strength group.
    comparisons=[];bootstrap=np.random.default_rng(p['seed']+40000)
    for m in p['sizes']:
        for n in p['queries']:
            a=next(r for r in results if r['method']=='sensitive_ranked' and r['prompts']==m and r['queries_per_prompt']==n)
            b=next(r for r in results if r['method']=='ordinary_ranked' and r['prompts']==m and r['queries_per_prompt']==n)
            difference=np.array(a['alarms'],int)-np.array(b['alarms'],int)
            samples=np.concatenate([bootstrap.choice(indices,size=(5000,len(indices)),replace=True) for indices in groups.values()],axis=1)
            comparisons.append(dict(prompts=m,queries_per_prompt=n,paired_tpr_difference=float(difference[20:].mean()),
                bootstrap95=np.quantile(difference[samples].mean(1),[.025,.975]).tolist(),
                note='exploratory pointwise interval; not adjusted for selecting budget or multiple comparisons'))
    output=dict(status='COMPLETE',results=results,ordinary_random_summary=random_summary,paired_comparisons=comparisons,
        minima={method:dict(target95=minimum(method),observed100=minimum(method,True)) for method in ['sensitive_ranked','ordinary_ranked']},
        endpoint_order=endpoints,normal_panels_are_independent_sampling_not_distinct_models=True)
    save(root/'PROMPT_DECISIONS.json',prompt_records);save(path,output)
    return output


def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);x=a.parse_args();root=x.root.resolve()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,fcntl
    torch.set_num_threads(4)
    with (root/'RUN.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=read(root/'PLAN.json');prompts=p['ordinary']+read(root/'CANDIDATES.json')['prompts']
            refs=reference(root,p,prompts)
            collect(root,p,prompts,'development')
            ranks=ranking(root,p,prompts,refs)
            collect(root,p,[refs[pid] for pid in ranks['sensitive']],'test')
            summarize(root,p,refs,ranks)
            save(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
        except Exception as exc:
            save(root/'STATUS.json',dict(phase='FAILED',error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
