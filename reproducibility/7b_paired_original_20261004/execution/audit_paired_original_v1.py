import json,hashlib,pathlib,tarfile,math
import numpy as np
B=pathlib.Path(__file__).resolve().parent
OUT=B/'paired_original_backup';R=OUT/'runs/fingerprint-7b-paired-original-v1'
O=B/'detection80_backup/runs/fingerprint-7b-mcc-vs-top-v1'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ident(x):return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
manifest=read(B/'fingerprint-paired-original-v1-manifest.json')
archive=B/'fingerprint-paired-original-v1.tar.gz'
assert sha(archive)==manifest['archive_sha256'];OUT.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
 for x in t.getmembers():assert (OUT/x.name).resolve().is_relative_to(OUT.resolve()) and x.isfile() and not x.issym() and not x.islnk()
 t.extractall(OUT)
for rel,v in manifest['files'].items():assert sha(OUT/rel)==v['sha256'],rel
for rel,v in manifest['parent_files'].items():assert sha(B/'detection80_backup'/rel)==v['sha256'],rel
assert read(B/'DETECTION80_LOCAL_AUDIT.json')['status']=='PASS'
p=read(R/'PLAN.json');old=read(O/'PLAN.json');binding=ident(p);prompts=read(R/'PROMPTS.json');fp=read(R/'FINGERPRINTS.json');ofp=read(O/'FINGERPRINTS.json')
assert p['parent_plan_sha256']==sha(O/'PLAN.json') and p['parent_fingerprints_sha256']==sha(O/'FINGERPRINTS.json')
for key in ['model','test_variants','queries','sizes','alpha','e_fraction','simulations','normal_panels']:assert p[key]==old[key]
assert len(prompts)==len({x['id'] for x in prompts})==15
assert fp['plan_identity']==binding
pool={x['id']:x for x in read(O/'CANDIDATES.json')['prompts']}
for x in prompts:
 s=old['sources'][x['source_index']];sx=pool[x['sensitive_id']]
 assert x['source_id']==s['id']==sx['source_id'] and x['prompt']==s['prompt'] and x['prompt']!=sx['prompt']
 assert x['prompt_sha256']==hashlib.sha256(x['prompt'].encode()).hexdigest() and x['id']=='o_'+x['prompt_sha256'][:20]
 assert p['source_mapping'][sx['id']]==x['id']
for name,v in fp['panels'].items():
 assert v['size']==ofp['panels'][name]['size'] and v['prompt_ids']==[p['source_mapping'][pid] for pid in ofp['panels'][name]['prompt_ids']]
assert read(R/'IDENTITY_AUDIT.json')['status']==read(R/'REFERENCE_REUSE_AUDIT.json')['status']=='PASS'
for x in prompts:
 a=read(O/'references'/f"{x['sensitive_id']}.json");b=read(R/'sensitive_reference_check/references'/f"{x['sensitive_id']}.json")
 assert a['token_ids']==b['token_ids'] and a['prompt_ids_sha256']==b['prompt_ids_sha256'] and np.allclose(a['probabilities'],b['probabilities'],atol=1e-8,rtol=1e-6)
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
  assert row['seed']==1004300000+ei*100000+indices[pid]*1000+j and row['seed'] not in seeds;seeds.add(row['seed'])
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
audit=dict(status='PASS',archive_sha256=manifest['archive_sha256'],files=len(manifest['files']),parent_files=len(manifest['parent_files']),prompts=15,responses=total,decisions=720,configurations=12,null_caches_recomputed=15,source_pairs_verified=True,reference_reuse_verified=True)
(B/'PAIRED_ORIGINAL_LOCAL_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
