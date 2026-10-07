import json,sys,hashlib
from pathlib import Path
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--data-dir', type=Path, required=True, help='Local experiment backup directory')
_data_dir = _parser.parse_args().data_dir.resolve()
from collections import Counter
import numpy as np

HERE=_data_dir
ROOT=HERE/'prompt-budget-v1'
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts'))
from evaluate_prompt_budget import validate_rows, token_arrays, decision, nulls
from run_scaling_matrix import check_responses,load_responses
def read(p):return json.loads(p.read_text(encoding='utf-8'))

p=read(ROOT/'PLAN.json');ranks=read(ROOT/'RANKING.json');result=read(ROOT/'BUDGET_RESULTS.json')
archive=HERE/'prompt-budget-v1-results.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='aa9ae6017b781e5fb4bac8f81b8cf5f83adbc18ca3ea4acd0c40a7adeaba11d9'
refs={f.stem:read(f) for f in (ROOT/'references').glob('*.json')};mc=nulls(ROOT,refs)
assert ranks['created_before_test_responses']
assert not {r['seed'] for r in p['test_variants']} & {r['seed'] for r in p['search_variants']+p['development_variants']}
assert len(set(ranks['sensitive']))==20
ordinary_base=HERE.parent/'scaling_validation_20260919'/'verified_results'/'scaling-v1'
old=read(ordinary_base/'MATRIX_PLAN.json');tokens={};total=0
for ei,label in enumerate(result['endpoint_order']):
    rows=load_responses(ROOT/'test_responses'/f'{label}.jsonl')
    assert len(validate_rows(rows,p,[refs[i] for i in ranks['sensitive']],'test',label,ei))==2000
    ordinary=load_responses(ordinary_base/'matrix_responses'/f'{label}.jsonl')
    assert len(check_responses(ordinary,old,label,ei))==2000
    tokens[label]={**token_arrays(rows),**token_arrays(ordinary)};total+=len(rows)+len(ordinary)
scores={pid:dict(hits=0,coverage=0) for pid in refs};dev_count=0
dev_prompts=p['ordinary']+read(ROOT/'CANDIDATES.json')['prompts']
for ei,row in enumerate(p['development_variants']):
    rows=load_responses(ROOT/'development_responses'/f"{row['variant_id']}.jsonl")
    assert len(validate_rows(rows,p,dev_prompts,'development',row['variant_id'],ei))==len(dev_prompts)*100
    t=token_arrays(rows);dev_count+=len(rows)
    for pid in refs:
        alarms=[decision(t[pid],refs[pid],mc[pid],20,n,p)['alarm'] for n in p['queries']]
        scores[pid]['hits']+=sum(alarms);scores[pid]['coverage']+=alarms[-1]
assert scores==ranks['scores']
def ordered(ids):return sorted(ids,key=lambda pid:(-scores[pid]['hits'],-scores[pid]['coverage'],pid))
assert ranks['ordinary']==ordered([r['id'] for r in p['ordinary']])
assert ranks['sensitive']==ordered([r['id'] for r in read(ROOT/'CANDIDATES.json')['prompts']])[:20]
decisions={};traces=read(ROOT/'PROMPT_DECISIONS.json')
for saved in traces:
    pid=saved['prompt'];label=saved['endpoint'];m=saved['prompts'];n=saved['queries_per_prompt']
    fresh=decision(tokens[label][pid],refs[pid],mc[pid],m,n,p)
    assert all(fresh[k]==saved[k] for k in ['alarm','rule','query']), (pid,label,m,n)
    decisions[(pid,label,m,n)]=fresh
for row in result['results']:
    alarms=[];spent=[]
    for label in result['endpoint_order']:
        ds=[decisions[(pid,label,row['prompts'],row['queries_per_prompt'])] for pid in row['prompt_ids']]
        alarms.append(any(d['alarm'] for d in ds));cost=0
        for d in ds:
            cost+=d['query']
            if d['alarm']:break
        spent.append(cost)
    assert alarms==row['alarms'] and spent==row['actual_queries']
    assert sum(alarms[20:])==row['hits'] and sum(alarms[:20])==row['false_alarms']
    assert row['budget']==row['prompts']*row['queries_per_prompt']
    assert row['meets_target']==(row['hits']>=38 and row['false_alarms']<=1)
assert len(result['results'])==2040 and len(decisions)==48000
for method in ['sensitive_ranked','ordinary_ranked']:
    for name,key in [('target95','meets_target'),('observed100','matches_observed_baseline')]:
        eligible=[r for r in result['results'] if r['method']==method and r[key]]
        assert min(eligible,key=lambda r:(r['budget'],r['prompts']))==result['minima'][method][name]
out=dict(status='PASS',archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
         formal_responses=total,development_responses=dev_count,prompt_decisions=len(decisions),
         configurations_including_random_replicates=len(result['results']),
         ranking_independently_recomputed=True,seed_separation=True,
         all_alarms_and_actual_query_counts_recomputed=True)
(HERE/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
print(json.dumps({m:{k:{x:v[x] for x in ['prompts','queries_per_prompt','budget','hits','false_alarms']} for k,v in a.items()} for m,a in result['minima'].items()}))
