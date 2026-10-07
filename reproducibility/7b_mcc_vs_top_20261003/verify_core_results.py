"""Check published evidence; raw-response/full-backup audit is a separate artifact."""
import pathlib,json,math
import numpy as np
B=pathlib.Path(__file__).resolve().parent
read=lambda name:json.loads((B/name).read_text(encoding='utf8'))
p=read('PLAN.json');fp=read('FINGERPRINTS.json');scores=read('SCORES.json');rows=read('DECISIONS.json');res=read('RESULTS.json')
assert len(rows)==720 and len(res['configurations'])==12
idx={(r['endpoint'],r['panel'],r['n']):r for r in rows};assert len(idx)==720
for method in ['mcc','top_sensitivity']:
 assert fp['panels'][method+'_4']['prompt_ids']==fp['panels'][method+'_8']['prompt_ids'][:4]
assert sorted(scores,key=lambda k:(-scores[k]['score'],k))[:8]==fp['panels']['top_sensitivity_8']['prompt_ids']
for r in rows:
 panel=fp['panels'][r['panel']];assert set(r['prompts'])==set(panel['prompt_ids'])
 assert r['query_cap']==panel['size']*r['n'] and r['alarm']==any(d['alarm'] for d in r['prompts'].values())
 for d in r['prompts'].values():
  assert math.isclose(d['alpha_e'],.05*.2/panel['size']) and math.isclose(d['alpha_p_per_look'],.05*.8/panel['size'])
for r in res['configurations']:
 key=f"{r['method']}_{r['m']}";n=r['n']
 hit=sum(idx[v['variant_id'],key,n]['alarm'] for v in p['test_variants']);fa=sum(idx[f'normal_{i:02d}',key,n]['alarm'] for i in range(20))
 assert (hit,fa)==(r['hits'],r['false_alarms']) and r['tpr']==hit/40 and r['fpr']==fa/20
 assert r['meets_empirical_target']==(hit>=38 and fa<=1)
assert len(read('GENERATION_SUMMARY.json'))==80 and len(read('CANDIDATES.json')['prompts'])==61
print('PASS: 80 sources, 61 candidates, 4 panels, 720 panel decisions, 12 summaries. Full raw-response audit remains separate.')
