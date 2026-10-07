"""Independent CPU audit; no execution of archived project code."""
import json,hashlib,pathlib,tarfile,collections,math,random
import numpy as np
B=pathlib.Path(__file__).resolve().parent
OUT=B/'detection80_backup';R=OUT/'runs/fingerprint-7b-mcc-vs-top-v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ident(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
manifest=read(B/'fingerprint-detection80-manifest.json');archive=B/'fingerprint-detection80-final.tar.gz'
assert sha(archive)==manifest['archive_sha256'];OUT.mkdir(exist_ok=True)
if not (OUT/'EXTRACTED').exists():
 with tarfile.open(archive) as t:
  for x in t.getmembers():
   assert (OUT/x.name).resolve().is_relative_to(OUT.resolve()) and not x.issym() and not x.islnk()
  t.extractall(OUT)
 (OUT/'EXTRACTED').write_text('complete')
for rel,v in manifest['files'].items():assert sha(OUT/rel)==v['sha256'],rel
C=B.parent/'calibration_7b_100_geo/calibration-7b-100-geo-v1'
for rel,v in manifest['calibration_files'].items():assert sha(C/rel)==v['sha256'],rel
p=read(R/'PLAN.json');binding=ident(p);scope=read(R/'SCOPE_80.json');ids=scope['selected_original_indices']
assert len(set(ids))==80 and sha(R/'PLAN.json')==scope['original_plan_sha256']
assert sha(R/'SCOPE_100.json')==scope['parent_scope_sha256']
parent=read(R/'SCOPE_100.json');assert set(ids)<=set(parent['selected_original_indices'])
for row in scope['strata']:
 eligible=sorted([i for i in parent['selected_original_indices'] if p['sources'][i]['category']==row['category'] and p['sources'][i]['subtask']==row['subtask']],key=lambda i:p['sources'][i]['id'])
 assert sorted(random.Random(row['seed']).sample(eligible,4))==row['selected_indices']
assert set(collections.Counter(p['sources'][i]['category'] for i in ids).values())=={20}
assert not {p['sources'][i]['prompt'] for i in ids}&{r['prompt'] for r in read(C/'CALIBRATION_PLAN.json')['selected']}
for rel,h in read(R/'CODE_MANIFEST.json').items():assert sha(R/'code_snapshot'/rel)==h
for name,key in [('CALIBRATION_PLAN.json','calibration_plan_sha256'),('CALIBRATION.json','calibration_sha256')]:assert sha(C/name)==p[key]
assert sha(R/'test_train.jsonl')==p['test_training_sha256']
gen=[read(R/'generation'/f'{i:03d}.json') for i in ids]
for i,g in zip(ids,gen):assert g['source_index']==i and g['source']==p['sources'][i] and g['plan_identity']==binding and not g.get('failure')
cand=read(R/'CANDIDATES.json');pool=cand['prompts'];assert len(pool)==61 and cand['fallbacks_in_pool']==0 and cand['scope_sha256']==sha(R/'SCOPE_80.json')
texts={g['optimized_prompt'] for g in gen if g['accepted']};assert texts=={x['prompt'] for x in pool}
assert len(texts)==sum(bool(g['accepted']) for g in gen)==61
for x in pool:
 assert x['source_index'] in ids and x['prompt_sha256']==hashlib.sha256(x['prompt'].encode()).hexdigest()
 assert x['id']=='s_'+x['prompt_sha256'][:20]
scores=read(R/'SCORES.json');profiles={}
for x in pool:
 pid=x['id'];mi=read(R/'ranking_micro'/f'{pid}.json');assert mi['plan_identity']==binding and mi['token_count']<=128
 micro=np.mean([mi['scores'][k]/p['calibration']['micro_scales'][k] for k in p['blocks']])
 fam=collections.defaultdict(list)
 for v in p['search_variants']:
  a=read(R/'ranking_macro'/v['variant_id']/f'{pid}.json');assert a['plan_identity']==binding and math.isfinite(a['score']) and a['score']>=0;fam[v['family']].append(a['score'])
 macro=np.mean([np.mean(v)/p['calibration']['macro_scales'][k] for k,v in fam.items()])
 assert np.isclose(scores[pid]['score'],p['calibration']['micro_weight']*micro+macro,rtol=1e-12)
 cv=read(R/'coverage'/f'{pid}.json');assert cv['plan_identity']==binding and cv['repeat_jaccard']>=.999;profiles[pid]=set(cv['components'])
fp=read(R/'FINGERPRINTS.json');assert fp==read(B/'FINGERPRINTS.json') and fp['plan_identity']==binding
events=[]
for line in (OUT/'setup/single-postprocess-v1.log').read_text(encoding='utf8',errors='replace').splitlines():
 try:
  event=json.loads(line)
  if isinstance(event,dict) and event.get('phase') in ['test_training','detection_sampling']:events.append(event)
 except (ValueError,TypeError):pass
assert events and min(e['time'] for e in events)>fp['frozen_at']
weight=lambda c:p['mcc']['weights'].get(c.split(':',1)[0],1.)
top=sorted(scores,key=lambda x:(-scores[x]['score'],x))[:8];assert top==fp['panels']['top_sensitivity_8']['prompt_ids']
covered=set();selected=[]
for target in fp['panels']['mcc_8']['prompt_ids']:
 gains={pid:math.fsum(weight(c) for c in cs-covered) for pid,cs in profiles.items() if pid not in selected}
 assert np.isclose(gains[target],max(gains.values()),rtol=0,atol=1e-7)
 selected.append(target);covered|=profiles[target]
universe=set().union(*profiles.values());den=math.fsum(weight(c) for c in universe)
for method in ['mcc','top_sensitivity']:
 assert fp['panels'][method+'_4']['prompt_ids']==fp['panels'][method+'_8']['prompt_ids'][:4]
for panel in fp['panels'].values():
 cov=set().union(*(profiles[x] for x in panel['prompt_ids']))
 assert np.isclose(math.fsum(weight(c) for c in cov)/den,panel['weighted_candidate_coverage'])
needed={i for x in fp['panels'].values() for i in x['prompt_ids']};prompts=[x for x in pool if x['id'] in needed];assert len(prompts)==15
assert len(list((R/'training').glob('*.json')))==20
for v in p['test_variants']:
 if v['family']=='finetuning':
  record=read(R/'training'/f"{v['variant_id']}.json");folder=R/'adapters'/v['variant_id']
  h=ident({f.relative_to(folder).as_posix():sha(f) for f in sorted(folder.rglob('*')) if f.is_file()})
  assert h==record['adapter_files_sha256'] and record['completed_steps']==v['configuration']['steps']
assert not {v['seed'] for v in p['test_variants']}&{v['seed'] for v in p['search_variants']}
def dev(counts,q):
 c=np.atleast_2d(np.asarray(counts,float));n=c.sum(1)
 with np.errstate(divide='ignore',invalid='ignore'):sat=np.where(c>0,c*np.log(c/n[:,None]),0).sum(1)
 vals=[]
 for row in q:
  a=np.maximum(0,2*(sat-c@np.log(np.where(row>0,row,1))));a[(c[:,row==0]>0).any(1)]=np.inf;vals.append(a)
 return np.min(vals,axis=0)
refs={};nulls={}
for pi,x in enumerate(prompts):
 pid=x['id'];ref=read(R/'references'/f'{pid}.json');q=np.array(ref['probabilities']);cache=R/'null_mc'/f'{pid}.npz'
 assert ref['prompt']==x['prompt'] and ref['null_sha256']==sha(cache) and np.isfinite(q).all() and (q>=0).all() and np.allclose(q.sum(1),1)
 assert len(ref['decoding_checks'])==3 and all(c['max_error']<=1e-6 for c in ref['decoding_checks'])
 rng=np.random.default_rng(p['seed']+20000+pi)
 with np.load(cache) as z:
  nulls[pid]={int(k):z[k] for k in z.files}
  for n in p['queries']:
   recal=np.stack([np.sort(dev(rng.multinomial(n,row,p['simulations']),q)) for row in q])
   assert np.allclose(recal,nulls[pid][n],atol=1e-10,rtol=1e-10)
 refs[pid]=ref
def decide(tokens,pid,m,n):
 ref=refs[pid];q=np.array(ref['probabilities']);index={t:i for i,t in enumerate(ref['token_ids'])};cnt=np.zeros(q.shape[1],int)
 for j,t in enumerate(tokens[:n],1):
  if t not in index:return True,'E',j
  cnt[index[t]]+=1
 stat=dev(cnt,q)[0];mc=nulls[pid][n];pv=max((1+len(row)-np.searchsorted(row,stat-1e-12,side='left'))/(len(row)+1) for row in mc)
 hit=pv<=p['alpha']*(1-p['e_fraction'])/m
 return hit,'P' if hit else None,n
endpoints=[f'normal_{i:02d}' for i in range(20)]+[v['variant_id'] for v in p['test_variants']]
stored=read(R/'DECISIONS.json');assert len(stored)==720;lookup={(x['endpoint'],x['panel'],x['n']):x for x in stored};assert len(lookup)==720
total=0;counts={};seeds=set()
for ei,endpoint in enumerate(endpoints):
 rows=[json.loads(l) for l in (R/'responses'/f'{endpoint}.jsonl').read_text().splitlines()];assert len(rows)==1200
 by={};indices={x['id']:i for i,x in enumerate(prompts)}
 for row in rows:
  pid=row['prompt_id'];j=row['response_index'];key=(pid,j);assert key not in by and pid in indices and 0<=j<80
  assert row['plan_identity']==binding and row['endpoint']==endpoint
  assert row['seed']==100700000+ei*100000+indices[pid]*1000+j and row['seed'] not in seeds;seeds.add(row['seed'])
  assert isinstance(row['token_id'],int) and 0<=row['token_id']<152064;by[key]=row['token_id']
 total+=len(rows)
 for name,panel in fp['panels'].items():
  for n in p['queries']:
   r=lookup[endpoint,name,n];assert r['query_cap']==n*panel['size'];alarms=[]
   assert set(r['prompts'])==set(panel['prompt_ids'])
   for pid in panel['prompt_ids']:
    hit,rule,query=decide([by[pid,j] for j in range(80)],pid,panel['size'],n);d=r['prompts'][pid]
    assert bool(hit)==d['alarm'] and rule==d['rule'] and query==d['query'];alarms.append(hit)
   assert bool(any(alarms))==r['alarm'];counts[endpoint,name,n]=int(any(alarms))
res=read(R/'RESULTS.json')
def wilson(k,n):
 z=1.959963984540054;rate=k/n;den=1+z*z/n;mid=(rate+z*z/(2*n))/den;rad=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
 return [max(0,mid-rad),min(1,mid+rad)]
for x in res['configurations']:
 name=f"{x['method']}_{x['m']}";n=x['n'];k=sum(counts[e,name,n] for e in endpoints[20:]);f=sum(counts[e,name,n] for e in endpoints[:20])
 assert x['hits']==k and x['false_alarms']==f and x['tpr']==k/40 and x['fpr']==f/20
 assert x['query_cap']==x['m']*n and x['meets_empirical_target']==(k>=38 and f<=1)
 assert np.allclose(x['tpr_wilson'],wilson(k,40)) and np.allclose(x['fpr_wilson'],wilson(f,20))
 for sk,sv in x['strata'].items():
  vs=[v for v in p['test_variants'] if f"{v['family']}:{v['strength']}"==sk];assert len(vs)==5
  assert sv['hits']==sum(counts[v['variant_id'],name,n] for v in vs)
for x in res['paired_comparisons']:
 m=x['m'];n=x['n'];aa=np.array([counts[e,f'mcc_{m}',n] for e in endpoints[20:]]);bb=np.array([counts[e,f'top_sensitivity_{m}',n] for e in endpoints[20:]])
 w=int(((aa==1)&(bb==0)).sum());l=int(((aa==0)&(bb==1)).sum());d=w+l
 exact=min(1.,2*sum(math.comb(d,i) for i in range(min(w,l)+1))/2**d) if d else 1.
 assert x['mcc_only_hits']==w and x['top_only_hits']==l and x['mcnemar_exact_unadjusted']==exact and x['tpr_difference']==float(np.mean(aa-bb))
 rng=np.random.default_rng(100900000+m*100+n)
 groups=[[i for i,v in enumerate(p['test_variants']) if v['family']==family and v['strength']==strength] for family in ['gaussian_noise','finetuning'] for strength in sorted({v['strength'] for v in p['test_variants'] if v['family']==family})]
 boot=[float(np.mean((aa-bb)[np.concatenate([rng.choice(g,5,replace=True) for g in groups])])) for _ in range(2000)]
 assert np.allclose(np.percentile(boot,[2.5,50,97.5]),x['paired_bootstrap_percentiles'])
assert total==72000
audit=dict(status='PASS',archive_sha256=manifest['archive_sha256'],files=len(manifest['files']),calibration_files=len(manifest['calibration_files']),sources=80,candidates=61,micro_records=61,macro_records=61*25,coverage_profiles=61,panels=4,unique_test_prompts=15,test_lora=20,responses=total,independent_decisions=720,configurations=12,null_caches_recomputed=15,model_identity_remote_verified=manifest['model_verified'],limitations=['GPU logits/activation values not rerun locally; stored values, identities and deterministic downstream computations checked','20 normal panels reuse one model; attack families known during search; no cross-family or cross-model confirmation'])
(B/'DETECTION80_LOCAL_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
