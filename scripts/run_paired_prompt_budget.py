"""Same-source paired prompt experiment with immutable search outputs."""
import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
import numpy as np
import _bootstrap
from run_resf_small import read,save,freeze,identity,clean,progress,generate,load_variant
from run_scaling_matrix import load_responses,endpoint_bundle
from evaluate_prompt_budget import reference,nulls,decision,token_arrays,validate_rows


def prepare(root,old):
    previous=read(old/'PLAN.json');sources=previous['sources'][:20]
    groups={'ordinary':{s['id']:s['id'] for s in sources}}
    unique={s['id']:s for s in sources};mapping=[];hashes={}
    for proxy in previous['proxies']:
        groups[proxy]={}
        for source in sources:
            file=old/'search'/f"{proxy}_{source['id']}.json";r=read(file)
            assert r['source']==source and not r.get('failure')
            hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            if r['accepted']:
                text=r['optimized_prompt'];pid='s_'+hashlib.sha256(text.encode()).hexdigest()[:16]
                unique.setdefault(pid,dict(id=pid,prompt=text,source_id=source['id'],category=source['category']))
            else:pid=source['id']
            groups[proxy][source['id']]=pid
            mapping.append(dict(source_id=source['id'],proxy=proxy,prompt_id=pid,accepted=r['accepted'],fallback_to_original=not r['accepted']))
    p=dict(previous,schema='paired-prompt-budget-v1',seed=320000000,queries=[25,50,100],
           old=str(old),ordinary=sources,groups=groups,prompts=list(unique.values()),source_order=[s['id'] for s in sources],
           reuse_old_responses=False,sampling_policy='all responses recollected on cloned server; prior reference mismatch audit retained in paired-prompt-budget-v1',
           mapping=mapping,search_sha256=hashes,
           ranking='within each arm: development hits at n=25,50,100 with panel_size20, then source ID',
           caveat='same source repair; existing attack models and some existing samples reused; exploratory not independent confirmation',
           generation='reuse frozen 60 search outputs; unsuccessful optimizations retain original source',
           random_panels='same source subsets across four arms; all 20 singletons, 100 unique subsets for m=2/5/10, one full panel')
    rng=np.random.default_rng(320000000);subsets={}
    for m in p['sizes']:
        if m==1:sets=[(i,) for i in range(20)]
        elif m==20:sets=[tuple(range(20))]
        else:
            collected=set()
            while len(collected)<100:collected.add(tuple(sorted(rng.choice(20,m,replace=False).tolist())))
            sets=sorted(collected)
        subsets[str(m)]=[[p['source_order'][i] for i in indices] for indices in sets]
    freeze(root/'PLAN.json',p);freeze(root/'SUBSETS.json',subsets)
    return p


def expected_seed(p,split,ei,pi,j):
    return p['seed']+1000000+(200000000 if split=='test' else 0)+ei*100000+pi*1000+j


def reuse_rows(p,split,label,ei):
    if not p.get('reuse_old_responses',True):return {}
    old=Path(p['old']);op=read(old/'PLAN.json')
    if split=='development':prompts=op['ordinary']+read(old/'CANDIDATES.json')['prompts']
    else:
        r=read(old/'RANKING.json');prompts=[read(old/'references'/f'{pid}.json') for pid in r['sensitive']]
    rows=load_responses(old/(split+'_responses')/f'{label}.jsonl')
    assert len(validate_rows(rows,op,prompts,split,label,ei))==len(prompts)*100
    allowed={r['id'] for r in p['prompts']}
    return {(r['prompt_id'],r['response_index']):r for r in rows if r['prompt_id'] in allowed}


def check_rows(rows,p,split,label,ei,reused):
    seen=set();indices={r['id']:i for i,r in enumerate(p['prompts'])}
    for row in rows:
        key=(row['prompt_id'],row['response_index'])
        assert key not in seen and key[0] in indices and 0<=key[1]<100
        assert row['plan_sha256']==identity(p) and row['split']==split and row['endpoint']==label
        assert type(row['token_id']) is int and row['token_id']>=0
        if row['origin']=='reused':
            old=reused[key];assert row['seed']==old['seed'] and row['token_id']==old['token_id']
        else:
            assert row['origin']=='new' and key not in reused
            assert row['seed']==expected_seed(p,split,ei,indices[key[0]],key[1])
        seen.add(key)
    return seen


def endpoints(p,split):
    variants=p['development_variants'] if split=='development' else p['test_variants']
    normal=[] if split=='development' else [(f'normal_panel_{i:02d}',None) for i in range(20)]
    return normal+[(r['variant_id'],r) for r in variants]


def collect(root,p,split):
    for ei,(label,variant) in enumerate(endpoints(p,split)):
        reuse=reuse_rows(p,split,label,ei)
        path=root/(split+'_responses')/f'{label}.jsonl';path.parent.mkdir(exist_ok=True)
        rows=load_responses(path);seen=check_rows(rows,p,split,label,ei,reuse)
        if len(seen)==len(p['prompts'])*100:continue
        loaded=None;bundle=None
        try:
            with path.open('a',encoding='utf-8') as output:
                for pi,prompt in enumerate(p['prompts']):
                    for j in range(100):
                        key=(prompt['id'],j)
                        if key in seen:continue
                        if key in reuse:
                            old=reuse[key];token=old['token_id'];seed=old['seed'];origin='reused'
                        else:
                            if bundle is None:
                                if split=='development':
                                    loaded=load_variant(read(Path(p['pilot'])/'PLAN.json'),variant,Path(p['pilot']));bundle=loaded.bundle
                                else:loaded,bundle=endpoint_bundle(p,variant,Path(p['base']))
                            seed=expected_seed(p,split,ei,pi,j);origin='new'
                            token=generate(bundle,prompt,seed,first=True)['token_ids'][0]
                        row=dict(prompt_id=prompt['id'],response_index=j,endpoint=label,split=split,seed=seed,
                                 token_id=int(token),origin=origin,plan_sha256=identity(p))
                        output.write(json.dumps(row)+'\n');output.flush()
                    save(root/'STATUS.json',dict(phase=split,endpoint=label,endpoint_index=ei,prompt=pi+1,prompts=len(p['prompts']),time=time.time()))
            progress('endpoint_complete',split=split,endpoint=label)
        finally:
            if loaded:loaded.close()
            elif bundle:bundle.close()
            clean()


def rank(root,p,refs,mc):
    if (root/'RANKING.json').exists():return read(root/'RANKING.json')
    scores={pid:0 for pid in refs}
    for label,_ in endpoints(p,'development'):
        tokens=token_arrays(load_responses(root/'development_responses'/f'{label}.jsonl'))
        for pid in refs:
            scores[pid]+=sum(decision(tokens[pid],refs[pid],mc[pid],20,n,p)['alarm'] for n in p['queries'])
    orders={group:sorted(p['source_order'],key=lambda s:(-scores[lookup[s]],s)) for group,lookup in p['groups'].items()}
    r=dict(scores=scores,orders=orders,before_test_sampling=not (root/'test_responses').exists())
    assert r['before_test_sampling'];freeze(root/'RANKING.json',r);return r


def summarize(root,p,refs,mc,ranks):
    if (root/'RESULTS.json').exists():return
    labels=[x[0] for x in endpoints(p,'test')];tokens={};audit=[]
    for split in ['development','test']:
        for ei,(label,_) in enumerate(endpoints(p,split)):
            rows=load_responses(root/(split+'_responses')/f'{label}.jsonl')
            assert len(check_rows(rows,p,split,label,ei,reuse_rows(p,split,label,ei)))==len(p['prompts'])*100
            audit.append(dict(split=split,endpoint=label,rows=len(rows),new=sum(r['origin']=='new' for r in rows)))
            if split=='test':tokens[label]=token_arrays(rows)
    group_indices={}
    for i,v in enumerate(p['test_variants']):group_indices.setdefault(f"{v['family']}:{v['strength']}",[]).append(i+20)
    all_decisions=[];panels=[];paired=[];sets=read(root/'SUBSETS.json')
    for m in p['sizes']:
        for n in p['queries']:
            ds={pid:[decision(tokens[e][pid],refs[pid],mc[pid],m,n,p) for e in labels] for pid in refs}
            for pid,values in ds.items():
                for label,d in zip(labels,values):all_decisions.append(dict(prompt_id=pid,endpoint=label,m=m,n=n,**d))
            if m==1:
                for source in p['source_order']:
                    base=ds[source]
                    for group in p['proxies']:
                        values=ds[p['groups'][group][source]]
                        paired.append(dict(source_id=source,group=group,n=n,accepted=p['groups'][group][source]!=source,
                            baseline_hits=sum(d['alarm'] for d in base[20:]),optimized_hits=sum(d['alarm'] for d in values[20:]),
                            baseline_fp=sum(d['alarm'] for d in base[:20]),optimized_fp=sum(d['alarm'] for d in values[:20]),
                            gained=sum(not a['alarm'] and b['alarm'] for a,b in zip(base[20:],values[20:])),
                            lost=sum(a['alarm'] and not b['alarm'] for a,b in zip(base[20:],values[20:]))))
            for group,lookup in p['groups'].items():
                configs=[('paired',i,s) for i,s in enumerate(sets[str(m)])]+[('ranked',0,ranks['orders'][group][:m])]
                for kind,rep,sources in configs:
                    ids=[lookup[s] for s in sources];alarms=[];cost=[]
                    for ei in range(60):
                        alarm=False;spent=0
                        for pid in ids:
                            d=ds[pid][ei];spent+=d['query']
                            if d['alarm']:alarm=True;break
                        alarms.append(alarm);cost.append(spent)
                    hits=sum(alarms[20:]);fp=sum(alarms[:20])
                    panels.append(dict(group=group,kind=kind,replicate=rep,m=m,n=n,budget=m*n,sources=sources,prompt_ids=ids,
                        hits=hits,false_alarms=fp,tpr=hits/40,fpr=fp/20,meets_target=hits>=38 and fp<=1,
                        strict=hits==40 and fp==0,alarms=alarms,actual_queries=cost,
                        attack_groups={g:sum(alarms[i] for i in indices) for g,indices in group_indices.items()}))
    summaries=[]
    for group in p['groups']:
        for m in p['sizes']:
            for n in p['queries']:
                rows=[r for r in panels if r['group']==group and r['kind']=='paired' and r['m']==m and r['n']==n]
                summaries.append(dict(group=group,m=m,n=n,budget=m*n,subsets=len(rows),
                    mean_tpr=float(np.mean([r['tpr'] for r in rows])),mean_fpr=float(np.mean([r['fpr'] for r in rows])),
                    tpr_q05_q95=np.quantile([r['tpr'] for r in rows],[.05,.95]).tolist(),
                    target_fraction=float(np.mean([r['meets_target'] for r in rows])),strict_fraction=float(np.mean([r['strict'] for r in rows]))))
    save(root/'RESPONSE_AUDIT.json',dict(status='PASS',records=audit))
    save(root/'PROMPT_DECISIONS.json',all_decisions)
    save(root/'RESULTS.json',dict(status='COMPLETE',endpoint_order=labels,panels=panels,paired_prompts=paired,paired_panel_summary=summaries))


def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--old',type=Path,required=True);x=a.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,fcntl,tarfile
    torch.set_num_threads(4);root=x.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p=read(root/'PLAN.json') if (root/'PLAN.json').exists() else prepare(root,x.old.resolve())
            refs=reference(root,p,p['prompts'])
            # Different GPU: permit only unchanged support and numerically matched reference distributions.
            checks=[]
            for pid,r in refs.items():
                old=Path(p['old'])/'references'/f'{pid}.json'
                if old.exists():
                    prev=read(old);same_support=r['token_ids']==prev['token_ids']
                    same=same_support and np.allclose(r['probabilities'],prev['probabilities'],atol=1e-6,rtol=1e-5)
                    checks.append(dict(prompt_id=pid,same_support=same_support,matched=bool(same)))
                    if p.get('reuse_old_responses',True):assert same,'Reference changed on clone; old responses cannot be reused'
            freeze(root/'CLONE_REFERENCE_AUDIT.json',dict(status='ALL_FRESH_SAMPLING' if not p.get('reuse_old_responses',True) else 'PASS',checks=checks))
            collect(root,p,'development');mc=nulls(root,refs);ranks=rank(root,p,refs,mc)
            collect(root,p,'test');summarize(root,p,refs,mc,ranks)
            save(root/'STATUS.json',dict(phase='COMPLETE',time=time.time()))
            archive=root.parent/(root.name+'-results.tar.gz')
            with tarfile.open(archive,'w:gz') as t:
                t.add(root,arcname=root.name)
                for name in ['run_paired_prompt_budget.py','evaluate_prompt_budget.py','run_resf_small.py','run_scaling_matrix.py']:
                    t.add(Path(__file__).parent/name,arcname='code/'+name)
            save(root/'ARCHIVE.json',dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size))
        except Exception as exc:
            save(root/'STATUS.json',dict(phase='FAILED',error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
