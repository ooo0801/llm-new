"""Frozen selected-source original-text control; no search, selection or training."""
import os,sys,json,hashlib,time,fcntl,traceback,inspect,importlib.metadata
from pathlib import Path
BASE=Path('/root/autodl-tmp/token-integrity')
OLD=BASE/'runs/fingerprint-7b-mcc-vs-top-v1'
ROOT=BASE/'runs/fingerprint-7b-paired-original-v1'
CAL=BASE/'runs/calibration-7b-100-geo-v1'
sys.path.insert(0,str(OLD/'code_snapshot/scripts'))
import run_7b_fingerprint_comparison as f
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def setup():
    p=f.read(OLD/'PLAN.json');fp=f.read(OLD/'FINGERPRINTS.json')
    assert fp['plan_identity']==f.identity(p)
    needed={pid for v in fp['panels'].values() for pid in v['prompt_ids']}
    sensitive=[x for x in f.read(OLD/'CANDIDATES.json')['prompts'] if x['id'] in needed]
    prompts=[];mapping={}
    for x in sensitive:
        s=p['sources'][x['source_index']]
        r=f.read(OLD/'generation'/f"{x['source_index']:03d}.json")
        assert s['id']==x['source_id'] and r['source']==s and r['accepted'] and r['optimized_prompt']==x['prompt']
        assert s['prompt']!=x['prompt']
        h=hashlib.sha256(s['prompt'].encode()).hexdigest();pid='o_'+h[:20]
        mapping[x['id']]=pid
        prompts.append(dict(id=pid,prompt=s['prompt'],category=s['category'],source_id=s['id'],source_index=x['source_index'],prompt_sha256=h,sensitive_id=x['id']))
    assert len(prompts)==len({x['id'] for x in prompts})==15
    p=dict(p,schema='7b-paired-original-v1',seed=1004200000,parent_plan_sha256=sha(OLD/'PLAN.json'),
        parent_fingerprints_sha256=sha(OLD/'FINGERPRINTS.json'),response_seed_base=1004300000,
        purpose='same selected sources original-text paired control; post-hoc known-test diagnostic',
        original_prompts=prompts,source_mapping=mapping,reference_reuse_gate=dict(exact_support=True,atol=1e-8,rtol=1e-6),
        reuse_sensitive_responses=True,bootstrap_seed=1004500000)
    f.freeze(ROOT/'PLAN.json',p)
    panels={name:dict(method=v['method'],size=v['size'],prompt_ids=[mapping[pid] for pid in v['prompt_ids']]) for name,v in fp['panels'].items()}
    fp=dict(plan_identity=f.identity(p),panels=panels,parent_sha256=sha(OLD/'FINGERPRINTS.json'))
    f.freeze(ROOT/'FINGERPRINTS.json',fp)
    f.freeze(ROOT/'PROMPTS.json',prompts)
    return p,fp,prompts,sensitive

def audit(p):
    for rel,h in f.read(OLD/'CODE_MANIFEST.json').items():assert sha(OLD/'code_snapshot'/rel)==h,rel
    for item in f.read(CAL/'MODEL_MANIFEST.json')['files']:
        assert sha(Path(p['model']['name'])/item['path'])==item['sha256'],item['path']
    old_packages=f.read(OLD/'IDENTITY_AUDIT.json')['packages']
    packages={k:importlib.metadata.version(k) for k in old_packages}
    assert packages==old_packages,(packages,old_packages)
    for v in p['test_variants']:
        if v['family']=='finetuning':assert f.file_identity(OLD/'adapters'/v['variant_id'])==f.read(OLD/'training'/f"{v['variant_id']}.json")['adapter_files_sha256']
    f.save(ROOT/'IDENTITY_AUDIT.json',dict(status='PASS',time=time.time(),packages=packages,model_verified=True,adapters_verified=20,script_sha256=sha(__file__)))

def check_environment(p,sensitive):
    f.status(ROOT,'reference_reuse_gate')
    fresh=f.reference(ROOT/'sensitive_reference_check',p,sensitive);checks=[]
    for row in sensitive:
        a=f.read(OLD/'references'/f"{row['id']}.json");b=fresh[row['id']]
        same_support=a['token_ids']==b['token_ids']
        delta=float(np.max(np.abs(np.array(a['probabilities'])-np.array(b['probabilities'])))) if same_support else None
        ok=same_support and a['prompt_ids_sha256']==b['prompt_ids_sha256'] and np.allclose(a['probabilities'],b['probabilities'],atol=1e-8,rtol=1e-6)
        checks.append(dict(id=row['id'],same_support=same_support,max_probability_error=delta,passed=bool(ok)))
    passed=all(x['passed'] for x in checks)
    f.save(ROOT/'REFERENCE_REUSE_AUDIT.json',dict(status='PASS' if passed else 'FAIL',checks=checks))
    assert passed,'Sensitive reference drift: stop before sampling; do not pool environments'

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';f.torch.set_num_threads(4)
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'RUN.lock').open('a+') as lock, (OLD/'RUN.lock').open('a+') as parent_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            p,fp,prompts,sensitive=setup();f.status(ROOT,'identity_audit');audit(p);check_environment(p,sensitive)
            refs=f.reference(ROOT,p,prompts)
            # Preserve the original collector, changing only its seed namespace and adapter lookup root.
            source=inspect.getsource(f.collect).replace('100700000','1004300000')
            source=source.replace("load_variant(p,e['variant'],root)","load_variant(p,e['variant'],OLD)")
            ns=dict(f.__dict__,OLD=OLD);exec(compile(source,'frozen_collect_original_control','exec'),ns)
            f.freeze(ROOT/'COLLECTOR_PATCH.json',dict(original_sha256=hashlib.sha256(inspect.getsource(f.collect).encode()).hexdigest(),adapted_sha256=hashlib.sha256(source.encode()).hexdigest(),changes=['new response seed base 1004300000','reuse parent test adapters']))
            ns['collect'](ROOT,p,prompts);f.evaluate(ROOT,p,fp,prompts,refs)
        except BaseException as exc:
            f.save(ROOT/'STATUS.json',dict(phase='ERROR',time=time.time(),error=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
