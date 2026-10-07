"""Execution-only migration overlay. Imports the original frozen code snapshot.

The supervisor owns RUN.lock. Workers process disjoint source indices with
per-source locks, original seeds, and original plan identity. No plan edits.
"""
import argparse,fcntl,hashlib,json,os,platform,signal,subprocess,sys,time,traceback
from pathlib import Path

BASE=Path('/root/autodl-tmp/token-integrity')
ROOT=BASE/'runs/fingerprint-7b-mcc-vs-top-v1'
CAL=BASE/'runs/calibration-7b-100-geo-v1'
EXEC=ROOT/'execution_triple_gpu_80_v4'

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def identity(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def load_frozen():
    sys.path.insert(0,str(ROOT/'code_snapshot/scripts'))
    import run_7b_fingerprint_comparison as f
    assert Path(f.__file__).resolve().is_relative_to((ROOT/'code_snapshot').resolve())
    return f
def check_record(path,p,i):
    r=read(path);assert r['plan_identity']==identity(p) and r['source_index']==i and r['source']==p['sources'][i]
    assert not r.get('failure'),('Technical failure; do not skip',str(path),r.get('failure'))
    return r
def completed(p):
    records={}
    for path in sorted((ROOT/'generation').glob('*.json')):
        i=int(path.stem);assert 0<=i<len(p['sources']);records[i]=check_record(path,p,i)
    return records
def audit():
    p=read(ROOT/'PLAN.json');assert len(p['sources'])==300
    expected3=read(EXEC/'TRIPLE_EXPECTATIONS.json')
    assert sha(ROOT/'SCOPE_80.json')==expected3['scope_sha256']
    for name,h in expected3['records_sha256'].items():assert sha(ROOT/'generation'/name)==h
    model_manifest=read(CAL/'MODEL_MANIFEST.json');cp=read(CAL/'CALIBRATION_PLAN.json')
    assert sha(ROOT/'PLAN.json')==read(EXEC/'LOCAL_EXPECTATIONS.json')['plan_sha256']
    expected=read(EXEC/'LOCAL_EXPECTATIONS.json')
    for rel,h in expected['paused_files'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in read(ROOT/'CODE_MANIFEST.json').items():assert sha(ROOT/'code_snapshot'/rel)==h,rel
    assert sha(CAL/'CALIBRATION_PLAN.json')==p['calibration_plan_sha256']
    assert sha(CAL/'CALIBRATION.json')==p['calibration_sha256']
    assert p['sources']==read(CAL/'OPTIMIZATION_POOL.json')
    for item in model_manifest['files']:assert sha(Path(p['model']['name'])/item['path'])==item['sha256']
    f=load_frozen()
    for v in p['search_variants']:
        if v['family']=='finetuning':assert f.file_identity(CAL/'adapters'/v['variant_id'])==cp['adapters_sha256'][v['variant_id']]
    import importlib.metadata
    packages={k:importlib.metadata.version(k) for k in read(ROOT/'IDENTITY_AUDIT.json')['packages']}
    assert packages==read(ROOT/'IDENTITY_AUDIT.json')['packages'],packages
    records=completed(p)
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True)
    assert len(gpus.strip().splitlines())==3,gpus
    result=dict(status='PASS',time=time.time(),packages=packages,python=platform.python_version(),gpus=gpus,
        plan_identity=identity(p),preserved_completed_indices=sorted(records),records_sha256={f'generation/{i:03d}.json':sha(ROOT/'generation'/f'{i:03d}.json') for i in records},
        implementation_sha256=sha(__file__),code_source='original code_snapshot',strategy='one process per visible GPU; selected position % 3; same source seeds and frozen objectives')
    save(EXEC/'MIGRATION_AUDIT.json',result)
    print(json.dumps(result),flush=True)

def smoke(gpu):
    f=load_frozen();torch=f.torch;torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    from run_calibration_7b_80 import inputs
    from llm_integrity.inner_micro_proxy import differentiable_block_micro_proxy
    cp=read(CAL/'CALIBRATION_PLAN.json');b=f.load_model(cp['model']);b.model.eval()
    for x in b.model.parameters():x.requires_grad_(False)
    records=[]
    try:
        block=next(x for x in f.discover_micro_blocks(b.model,cp['blocks']) if x.block_type==cp['blocks'][0] and x.layer_id==cp['layers'][0])
        for i in [0,25]:
            e,mask=inputs(b,cp,cp['selected'][i])
            s=differentiable_block_micro_proxy(b,e,mask,block=block,probes=8,seed=930020000+100*i)
            old=read(CAL/'micro'/f'0_{i:02d}.json');m=s.metadata()
            for key in ['raw_micro_score','raw_embedding_gradient_norm']:
                assert abs(m[key]-old[key])/max(abs(old[key]),1e-20)<=.001,(gpu,i,key,m[key],old[key])
            records.append(dict(source_index=i,score=m['raw_micro_score'],gradient=m['raw_embedding_gradient_norm']))
            del e,mask,s;f.clean()
    finally:b.close();f.clean()
    save(EXEC/f'SMOKE_{gpu}.json',dict(status='PASS',gpu=gpu,records=records))


def selection():
    a=read(ROOT/'SCOPE_80.json')
    assert a['original_plan_sha256']==sha(ROOT/'PLAN.json')
    ids=a['selected_original_indices']
    assert len(ids)==len(set(ids))==80
    from collections import Counter
    p=read(ROOT/'PLAN.json');parent=read(ROOT/'SCOPE_100.json')
    assert set(ids).issubset(parent['selected_original_indices'])
    assert sha(ROOT/'SCOPE_100.json')==a['parent_scope_sha256']
    assert set(Counter(p['sources'][i]['category'] for i in ids).values())=={20}
    assert len(Counter((p['sources'][i]['category'],p['sources'][i]['subtask']) for i in ids))==20
    assert set(Counter((p['sources'][i]['category'],p['sources'][i]['subtask']) for i in ids).values())=={4}
    return ids

def selected_candidates(f,p):
    chosen={};mapping=[]
    for i in sorted(selection()):
        r=check_record(ROOT/'generation'/f'{i:03d}.json',p,i)
        if not r['accepted']:continue
        text=r['optimized_prompt'];assert text!=r['source']['prompt']
        h=hashlib.sha256(text.encode()).hexdigest();pid='s_'+h[:20]
        if h not in chosen:chosen[h]=dict(id=pid,prompt=text,category=r['source']['category'],source_id=r['source']['id'],source_index=i,prompt_sha256=h)
        mapping.append(dict(source_id=r['source']['id'],candidate_id=chosen[h]['id']))
    pool=sorted(chosen.values(),key=lambda r:r['id'])
    f.freeze(ROOT/'CANDIDATES.json',dict(prompts=pool,mapping=mapping,accepted_sources=len(mapping),unique_texts=len(pool),fallbacks_in_pool=0,scope_sha256=sha(ROOT/'SCOPE_80.json')))
    if len(pool)<8:
        f.status(ROOT,'INSUFFICIENT_CANDIDATES',count=len(pool));raise RuntimeError('Fewer than 8 unique accepted candidates')
    return pool

def worker(gpu):
    f=load_frozen();torch=f.torch;torch.set_num_threads(4)
    assert torch.cuda.device_count()==1
    p=read(ROOT/'PLAN.json');binding=identity(p)
    # Never let independent workers overwrite shared STATUS.json.
    def status(root,phase,**fields):
        save(EXEC/f'worker_{gpu}.json',dict(phase=phase,gpu=gpu,pid=os.getpid(),time=time.time(),**fields))
    registry={v['variant_id']:str(CAL/'adapters'/v['variant_id']) for v in p['search_variants'] if v['family']=='finetuning'}
    for position,i in enumerate(selection()):
        if position%3!=gpu:continue
        source=p['sources'][i]
        out=ROOT/'generation'/f'{i:03d}.json'
        locks=EXEC/'source_locks';locks.mkdir(parents=True,exist_ok=True)
        with (locks/f'{i:03d}.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if out.exists():check_record(out,p,i);continue
            status(ROOT,'generation',source_index=i,completed_before=len(completed(p)),started_at=time.time())
            sampler=f.StratifiedVariantSampler(p['search_variants'],family_weights={family:.2 for family in p['calibration']['macro_scales']},adapter_registry=registry,seed=p['sampler_seed'])
            b=f.load_model(p['model']);optimizer=None;start=time.time()
            try:
                optimizer=f.DiscreteJointInnerOptimizer(reference=b,sampler=sampler,model_config=p['model'],
                    micro_scales=p['calibration']['micro_scales'],macro_scales=p['calibration']['macro_scales'],
                    micro_weight=p['calibration']['micro_weight'],macro_weight=1.,seed=p['search_seed']+i*100,
                    block_types=tuple(p['blocks']),**p['search_settings'])
                result=optimizer.optimize(source)
                record=dict(**result.payload(),source=source,source_index=i,plan_identity=binding,seconds=time.time()-start)
                f.save(out,record)
                if result.failure:raise RuntimeError(f'Technical search failure at {i}: {result.failure}')
            finally:(optimizer.reference if optimizer else b).close();f.clean()
            save(EXEC/'source_execution'/f'{i:03d}.json',dict(gpu=gpu,pid=os.getpid(),record_sha256=sha(out),seconds=record['seconds'],finished_at=time.time()))
    status(ROOT,'GENERATION_SHARD_COMPLETE')

def postprocess():
    # After parallel generation, execute original downstream code on one GPU.
    # This keeps shared fingerprints/reference calibration atomic and unchanged.
    f=load_frozen();f.torch.set_num_threads(4);p=read(ROOT/'PLAN.json')
    assert set(selection()).issubset(completed(p))
    pool=selected_candidates(f,p);scores=f.score_candidates(ROOT,p,pool);fp=f.select(ROOT,p,pool,scores)
    needed={pid for panel in fp['panels'].values() for pid in panel['prompt_ids']};prompts=[r for r in pool if r['id'] in needed]
    f.train_tests(ROOT,p);refs=f.reference(ROOT,p,prompts);f.collect(ROOT,p,prompts);f.evaluate(ROOT,p,fp,prompts,refs)

def supervisor():
    EXEC.mkdir(parents=True,exist_ok=True);children=[];streams=[]
    def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def start(mode,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        stream=(EXEC/f'{mode}_{gpu}.log').open('a',encoding='utf8');streams.append(stream)
        child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--mode',mode,'--gpu',str(gpu)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child);return child
    def wait_all(procs,generation=False):
        while True:
            codes=[p.poll() for p in procs]
            if any(c is not None and c!=0 for c in codes):raise RuntimeError(f'Worker failure: {codes}')
            if generation:
                records=completed(read(ROOT/'PLAN.json'))
                states={str(i):read(EXEC/f'worker_{i}.json') for i in [0,1,2] if (EXEC/f'worker_{i}.json').exists()}
                save(ROOT/'STATUS.json',dict(phase='generation_triple_gpu',completed=len(set(records).intersection(selection())),total=80,archived_completed=len(records),time=time.time(),workers=states))
            if all(c is not None for c in codes):return
            time.sleep(10)
    with (ROOT/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            audit()
            wait_all([start('smoke',i) for i in range(3)])
            a=read(EXEC/'SMOKE_0.json');b=read(EXEC/'SMOKE_1.json')
            for x,y in zip(a['records'],b['records']):
                for key in ['score','gradient']:assert abs(x[key]-y[key])/max(abs(x[key]),1e-20)<=.001
            c=read(EXEC/'SMOKE_2.json')
            for x,y in zip(a['records'],c['records']):
                for key in ['score','gradient']:assert abs(x[key]-y[key])/max(abs(x[key]),1e-20)<=.001
            save(EXEC/'TRIPLE_NUMERIC_AUDIT.json',dict(status='PASS',relative_tolerance=.001,smokes=[a,b,c]))
            save(EXEC/'SUPERVISOR.json',dict(phase='generation',pid=os.getpid(),time=time.time()))
            wait_all([start('worker',i) for i in range(3)],True)
            save(EXEC/'SUPERVISOR.json',dict(phase='postprocess',pid=os.getpid(),time=time.time()))
            wait_all([start('postprocess',0)])
            save(EXEC/'SUPERVISOR.json',dict(phase='COMPLETE',pid=os.getpid(),time=time.time()))
        except BaseException as exc:
            save(EXEC/'SUPERVISOR.json',dict(phase='STOPPED' if isinstance(exc,KeyboardInterrupt) else 'ERROR',time=time.time(),error=str(exc),traceback=traceback.format_exc()))
            raise
        finally:
            for p in children:
                if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            for p in children:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            for stream in streams:stream.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['supervisor','smoke','worker','postprocess'],default='supervisor');parser.add_argument('--gpu',type=int,default=0);a=parser.parse_args()
    if a.mode=='supervisor':supervisor()
    elif a.mode=='smoke':smoke(a.gpu)
    elif a.mode=='worker':worker(a.gpu)
    else:postprocess()
