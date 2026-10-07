import json,hashlib,sys
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data-dir', type=Path, required=True, help='Local experiment backup directory')
_data_dir = _parser.parse_args().data_dir.resolve()
import numpy as np
H=_data_dir;B=H/'generation-ablation-v1';R=B/'evaluation'
sys.path.insert(0,str(B/'code'/'scripts'))
from run_paired_prompt_budget import check_rows,endpoints
from evaluate_prompt_budget import decision,nulls,token_arrays
from run_scaling_matrix import load_responses
def read(p):return json.loads(p.read_text(encoding='utf-8'))
p=read(R/'PLAN.json');d=read(R/'RESULTS.json');ranks=read(R/'RANKING.json');sub=read(R/'SUBSETS.json')
assert hashlib.sha256((H/'generation-ablation-v1-results.tar.gz').read_bytes()).hexdigest()==read(H/'ARCHIVE.json')['sha256']
assert not p['reuse_old_responses'] and ranks['before_test_sampling']
assert not {r['seed'] for r in p['test_variants']} & {r['seed'] for r in p['development_variants']+p['search_variants']}
old=B/'generation'
generation_plan=read(old/'PLAN.json')
assert len(generation_plan['sources'])==20 and len(p['mapping'])==60
assert set(p['groups'])=={'ordinary','micro_only','macro_only','joint_js'}
assert p['queries']==[25,50,100] and p['sizes']==[1,2,5,10,20]
assert p['seed']==600000000
assert read(B/'ENVIRONMENT.json')['model_sha256']=='dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
for name,sha in read(B/'code'/'SHA256.json').items():
    assert hashlib.sha256((B/'code'/name).read_bytes()).hexdigest()==sha
for item in p['mapping']:
    file=old/'search'/f"{item['proxy']}_{item['source_id']}.json";s=read(file)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==p['search_sha256'][file.name]
    assert item['accepted']==s['accepted']
    assert not s.get('failure') and s['rounds_completed']==5
    assert s['source']==next(v for v in generation_plan['sources'] if v['id']==item['source_id'])
    arm=item['proxy'];params=generation_plan['arm_parameters'][arm]
    assert s['arm_parameters']==params
    assert s['micro_weight']==params['micro_weight'] and s['macro_weight']==params['macro_weight']
    si=p['source_order'].index(item['source_id'])
    assert s['search_seed']==generation_plan['search_seed']+si*100
    if arm=='micro_only':
        assert params['minimum_nondegraded_families']==0 and params['macro_weight']==0
        for h in s['history']:
            assert not h['macro'] and not h['gradient_variant_ids'] and not h['anchor_variant_ids']
            assert all(c['macro_gate_disabled'] and not c['macro_scores'] for c in h['reranked_candidates'])
    elif arm=='macro_only':
        assert params['micro_weight']==0 and params['minimum_nondegraded_families']==5
        assert all(h['micro']['disabled'] for h in s['history'])
    else:assert params['micro_weight']>0 and params['minimum_nondegraded_families']==5
    pid='s_'+hashlib.sha256(s['optimized_prompt'].encode()).hexdigest()[:16] if s['accepted'] else item['source_id']
    assert pid==item['prompt_id']==p['groups'][item['proxy']][item['source_id']]
refs={f.stem:read(f) for f in (R/'references').glob('*.json')};mc=nulls(R,refs)
assert set(refs)=={r['id'] for r in p['prompts']}
for prompt in p['prompts']:assert refs[prompt['id']]['prompt']==prompt['prompt']
tokens={};counts={};scores={pid:0 for pid in refs}
for split in ['development','test']:
    count=0
    for ei,(label,_) in enumerate(endpoints(p,split)):
        rows=load_responses(R/(split+'_responses')/f'{label}.jsonl')
        assert len(check_rows(rows,p,split,label,ei,{}))==len(p['prompts'])*100
        assert all(r['origin']=='new' for r in rows)
        t=token_arrays(rows);count+=len(rows)
        if split=='test':tokens[label]=t
        else:
            for pid in refs:scores[pid]+=sum(decision(t[pid],refs[pid],mc[pid],20,n,p)['alarm'] for n in p['queries'])
    counts[split]=count
assert scores==ranks['scores']
for group,lookup in p['groups'].items():
    assert ranks['orders'][group]==sorted(p['source_order'],key=lambda s:(-scores[lookup[s]],s))
decisions={}
for saved in read(R/'PROMPT_DECISIONS.json'):
    pid=saved['prompt_id'];label=saved['endpoint'];m=saved['m'];n=saved['n']
    fresh=decision(tokens[label][pid],refs[pid],mc[pid],m,n,p)
    assert all(fresh[k]==saved[k] for k in ['alarm','rule','query']),(pid,label,m,n)
    decisions[(pid,label,m,n)]=fresh
for panel in d['panels']:
    assert panel['budget']==panel['m']*panel['n']
    sources=sub[str(panel['m'])][panel['replicate']] if panel['kind']=='paired' else ranks['orders'][panel['group']][:panel['m']]
    assert panel['sources']==sources
    assert panel['prompt_ids']==[p['groups'][panel['group']][s] for s in sources]
    alarms=[];costs=[]
    for label in d['endpoint_order']:
        alarm=False;cost=0
        for pid in panel['prompt_ids']:
            v=decisions[(pid,label,panel['m'],panel['n'])];cost+=v['query']
            if v['alarm']:alarm=True;break
        alarms.append(alarm);costs.append(cost)
    assert alarms==panel['alarms'] and costs==panel['actual_queries']
    assert sum(alarms[20:])==panel['hits'] and sum(alarms[:20])==panel['false_alarms']
    assert panel['tpr']==panel['hits']/40 and panel['fpr']==panel['false_alarms']/20
    for label,total in panel['attack_groups'].items():
        assert total==sum(alarms[i+20] for i,v in enumerate(p['test_variants']) if f"{v['family']}:{v['strength']}"==label)
    assert panel['meets_target']==(panel['hits']>=38 and panel['false_alarms']<=1)
    assert panel['strict']==(panel['hits']==40 and panel['false_alarms']==0)
assert counts=={'development':len(p['prompts'])*600,'test':len(p['prompts'])*6000}
assert len(decisions)==len(p['prompts'])*900 and len(d['panels'])==3912
for summary in d['paired_panel_summary']:
    matching=[r for r in d['panels'] if r['kind']=='paired' and all(r[k]==summary[k] for k in ['group','m','n'])]
    assert len(matching)==summary['subsets']
    for target,source in [('mean_tpr','tpr'),('mean_fpr','fpr'),('target_fraction','meets_target'),('strict_fraction','strict')]:
        assert abs(summary[target]-float(np.mean([r[source] for r in matching])))<1e-12
out=dict(status='PASS',response_counts=counts,decisions_recomputed=len(decisions),panels_recomputed=len(d['panels']),
         source_mapping_verified=True,development_ranking_recomputed=True,all_responses_fresh=True)
(H/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))
for group in p['groups']:
    print(group)
    for key in ['meets_target','strict']:
        rows=[r for r in d['panels'] if r['kind']=='ranked' and r['group']==group and r[key]]
        v=min(rows,key=lambda r:(r['budget'],r['m'])) if rows else None
        print(key,{k:v[k] for k in ['m','n','budget','hits','false_alarms']} if v else None)
    for n in p['queries']:
        rows=[r for r in d['paired_prompts'] if r['group']==group and r['n']==n]
        if rows:print(n,'improved/tied/worse',sum(r['optimized_hits']>r['baseline_hits'] for r in rows),sum(r['optimized_hits']==r['baseline_hits'] for r in rows),sum(r['optimized_hits']<r['baseline_hits'] for r in rows),'mean_delta',sum(r['optimized_hits']-r['baseline_hits'] for r in rows)/800)
