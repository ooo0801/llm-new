"""Generate an independent audit from the previously reviewed numerical audit."""
from pathlib import Path
B=Path(__file__).resolve().parent
prefix='''import json,hashlib,pathlib,tarfile,math
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
'''
source=(B/'audit_detection80.py').read_text(encoding='utf8')
tail=source[source.index('def dev('):source.index('audit=dict(status=')].replace('100700000','1004300000')
ending='''audit=dict(status='PASS',archive_sha256=manifest['archive_sha256'],files=len(manifest['files']),parent_files=len(manifest['parent_files']),prompts=15,responses=total,decisions=720,configurations=12,null_caches_recomputed=15,source_pairs_verified=True,reference_reuse_verified=True)
(B/'PAIRED_ORIGINAL_LOCAL_AUDIT.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
'''
(B/'audit_paired_original_v1.py').write_text(prefix+tail+ending,encoding='utf8')
