import json,hashlib,sys
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data-dir', type=Path, required=True, help='Local experiment backup directory')
_data_dir = _parser.parse_args().data_dir.resolve()
import numpy as np
H=_data_dir;R=H/'paired-prompt-budget-v2'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run_paired_prompt_budget import check_rows,endpoints
from evaluate_prompt_budget import decision,nulls,token_arrays
from run_scaling_matrix import load_responses
def read(p):return json.loads(p.read_text(encoding='utf-8'))
p=read(R/'PLAN.json');d=read(R/'RESULTS.json');ranks=read(R/'RANKING.json');sub=read(R/'SUBSETS.json')
assert hashlib.sha256((H/'paired-prompt-budget-v2-results.tar.gz').read_bytes()).hexdigest()==read(H/'ARCHIVE.json')['sha256']
assert not p['reuse_old_responses'] and ranks['before_test_sampling']
assert not {r['seed'] for r in p['test_variants']} & {r['seed'] for r in p['development_variants']+p['search_variants']}
old=H.parent/'prompt_budget_20260919/prompt-budget-v1'
for item in p['mapping']:
    file=old/'search'/f"{item['proxy']}_{item['source_id']}.json";s=read(file)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==p['search_sha256'][file.name]
    assert item['accepted']==s['accepted']
    pid='s_'+hashlib.sha256(s['optimized_prompt'].encode()).hexdigest()[:16] if s['accepted'] else item['source_id']
    assert pid==item['prompt_id']==p['groups'][item['proxy']][item['source_id']]
refs={f.stem:read(f) for f in (R/'references').glob('*.json')};mc=nulls(R,refs)
tokens={};counts={};scores={pid:0 for pid in refs}
for split in ['development','test']:
    count=0
    for ei,(label,_) in enumerate(endpoints(p,split)):
        rows=load_responses(R/(split+'_responses')/f'{label}.jsonl')
        assert len(check_rows(rows,p,split,label,ei,{}))==5600
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
    assert panel['meets_target']==(panel['hits']>=38 and panel['false_alarms']<=1)
    assert panel['strict']==(panel['hits']==40 and panel['false_alarms']==0)
assert counts=={'development':33600,'test':336000}
assert len(decisions)==50400 and len(d['panels'])==3912
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
