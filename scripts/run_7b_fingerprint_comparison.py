"""Frozen shared generation pool -> MCC/TopSensitivity -> independent RESF tests.

Run phases are resumable; only completed atomic records can be reused.
"""
import argparse,fcntl,hashlib,json,math,os,platform,shutil,time
from pathlib import Path
from dataclasses import asdict
import _bootstrap
import numpy as np
import torch
from run_resf_small import read,save,freeze,identity,file_identity,clean,load_variant
from run_calibration_scale_comparison import sha
from run_calibration_7b_80 import status
from llm_integrity.modeling import load_model
from llm_integrity.inner_variant_sampler import StratifiedVariantSampler
from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer
from llm_integrity.inner_micro_proxy import discover_micro_blocks,score_block_micro_proxy
from llm_integrity.macro_proxy import differentiable_macro_proxy
from llm_integrity.activations import TransformerActivationProfiler
from llm_integrity.mcc import greedy_mcc,top_sensitivity,component_weight
from evaluate_prompt_budget import reference,nulls,decision
from run_scaling_matrix import generate,load_responses

BASE=Path('/root/autodl-tmp/token-integrity')
CAL=BASE/'runs/calibration-7b-100-geo-v1'
ROOT=BASE/'runs/fingerprint-7b-mcc-vs-top-v1'

def test_training():
    pairs=[
      ('水在标准大气压下的沸点是多少摄氏度？','100摄氏度。'),('地球唯一的天然卫星叫什么？','月球。'),
      ('太阳系最大的行星是哪颗？','木星。'),('植物进行光合作用主要吸收哪种气体？','二氧化碳。'),
      ('人体负责泵血的器官是什么？','心脏。'),('人类用哪种器官进行呼吸？','肺。'),
      ('铁的化学元素符号是什么？','Fe。'),('黄金的化学元素符号是什么？','Au。'),
      ('汉字“明”由哪两个常见字组成？','日和月。'),('汉字“休”通常表示什么意思？','休息。'),
      ('一年通常分为哪四个季节？','春、夏、秋、冬。'),('彩虹通常说有几种颜色？','七种。'),
      ('指南针的北端通常指向哪个方向？','北方。'),('测量温度通常使用什么仪器？','温度计。'),
      ('测量物体质量通常使用什么仪器？','天平。'),('观察微小物体常用什么光学仪器？','显微镜。'),
      ('请用一句话解释什么是蒸发。','液体表面转化为气体的过程叫蒸发。'),('请用一句话解释什么是凝固。','液体转化为固体的过程叫凝固。'),
      ('请简要说明为什么要节约用水。','淡水资源有限，节约用水有助于保护资源。'),('请简要说明为什么要洗手。','洗手可以减少手上的污物和病原体。'),
      ('请给出一个常见的可再生能源名称。','太阳能。'),('请给出一种常见的金属材料。','铝。'),
      ('请给出一种常见的交通工具。','自行车。'),('请给出一种常用的书写工具。','铅笔。'),
      ('将英文单词water翻译成中文。','水。'),('将英文单词sun翻译成中文。','太阳。'),
      ('将英文单词book翻译成中文。','书。'),('将英文单词school翻译成中文。','学校。'),
      ('请将“我喜欢阅读”翻译成英语。','I like reading.'),('请将“早上好”翻译成英语。','Good morning.'),
      ('请写一句提醒他人按时休息的话。','请合理安排时间，记得按时休息。'),('请写一句鼓励他人坚持学习的话。','每天进步一点，坚持学习就会有所收获。')]
    return [dict(id=f'independent_train_{i:02d}',prompt=q,expected_answer=a,source='prespecified_synthetic_general_knowledge_v1') for i,(q,a) in enumerate(pairs)]

def prepare(root):
    cp=read(CAL/'CALIBRATION_PLAN.json');cs=read(CAL/'CALIBRATION.json');assert cs['valid']
    sources=read(CAL/'OPTIMIZATION_POOL.json');assert len(sources)==300
    assert cs['macro_aggregation']=='category_median_equal_weight_geometric_mean'
    assert cs['micro_weight']==9.619061603348785
    manifest=read(CAL/'MODEL_MANIFEST.json')
    for r in manifest['files']:assert sha(Path(cp['model']['name'])/r['path'])==r['sha256']
    for v in cp['variants']:
        if v['family']=='finetuning':assert file_identity(CAL/'adapters'/v['variant_id'])==cp['adapters_sha256'][v['variant_id']]
    train=test_training();mother=read(CAL/'MOTHER_POOL.json')
    forbidden={r['prompt'].strip() for r in mother}
    forbidden|={json.loads(x)['prompt'].strip() for x in (CAL/'train.jsonl').read_text().splitlines()}
    assert not forbidden&{r['prompt'] for r in train}
    tp=root/'test_train.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train)
    if tp.exists():assert tp.read_text(encoding='utf8')==text
    else:tp.write_text(text,encoding='utf8')
    variants=[]
    for si,strength in enumerate([.001,.0025,.004,.006]):
        for k in range(5):variants.append(dict(family='gaussian_noise',variant_id=f'test_gaussian_{si}_{k}',seed=100510000+si*100+k,strength=strength,
            configuration=dict(method='relative_gaussian',std_ratio=strength,target_scope='attention_ffn',measure_realized=True)))
    for si,steps in enumerate([5,10,20,40]):
        for k in range(5):variants.append(dict(family='finetuning',variant_id=f'test_lora_{si}_{k}',seed=100520000+si*100+k,strength=steps,
            configuration=dict(method='lora',rank=8,alpha=16,dropout=0.,learning_rate=1e-4,steps=steps,lr_scheduler_type='constant',target_scope='attention_ffn')))
    assert not {v['seed'] for v in variants}&{v['seed'] for v in cp['variants']}
    settings=dict(rounds=5,probes=8,candidate_positions=4,candidates_per_position=16,rerank_candidates=8,max_length=128,max_edit_ratio=.5,
        ppl_ratio_limit=1e12,minimum_nondegraded_families=5,family_relative_tolerance=.01,variants_per_family=5,anchor_variants_per_family=0,
        gradient_restarts=1,family_gate_aggregation='mean',require_task_preservation=False,sequential_model_execution=False,
        representative_layer_count=4,block_schedule='balanced',task_validation_mode='strict_r1',macro_proxy='js',macro_top_k=10,
        enforce_surface_compatibility=False,enforce_perplexity=False)
    p=dict(schema='7b-mcc-vs-top-v1',model=cp['model'],model_revision=cp['model_revision'],sources=sources,calibration=cs,
        calibration_plan_sha256=sha(CAL/'CALIBRATION_PLAN.json'),calibration_sha256=sha(CAL/'CALIBRATION.json'),
        blocks=cp['blocks'],layers=cp['layers'],search_variants=cp['variants'],search_settings=settings,
        search_seed=100010000,sampler_seed=100000000,ranking_probe_seed=100100000,
        ranking='micro_weight * mean of all 7 normalized block scores + mean of 5 normalized five-variant mean JS scores',
        candidate_policy='proxy-search accepted, different from source, unique text; no fallback originals; one candidate/source; no extra test-based filters',
        search_caveat='No task-preservation/PPL gate, consistent with preceding search. Accepted means proxy-search acceptance, not proof of semantic preservation.',
        mcc=dict(attention_entropy_fraction=.70,ffn_quantile=.95,residual_threshold=.50,max_ffn_units_per_layer=128,
            repetitions=2,minimum_jaccard=.999,stable_component_frequency=1.,max_length=128,
            weights=dict(attention=1.2,ffn=1.,attention_residual=.8,mlp_residual=.8),tie_break='stable prompt ID only',
            universe='union of shared candidates; descriptive coverage only; no claim of covering all model parameters'),
        sizes=[4,8],queries=[25,50,80],normal_panels=20,simulations=10000,seed=100600000,alpha=.05,e_fraction=.2,
        test_variants=variants,test_training_sha256=sha(tp),test_training=dict(max_length=128,batch_size=1,gradient_accumulation_steps=4),
        decoding=dict(temperature=.7,top_k=50,top_p=.9,max_new_tokens=1,reference_temperature_grid=[.5,.7,.9]),
        pairing='same endpoints; same response prefixes at n25/50/80; shared prompts share samples; 4 is prefix of 8 in both arms',
        normal_caveat='20 independent sampling panels from one normal model; not 20 independent model implementations',
        statistical_scope='12 fixed configurations; per-panel FPR controlled, no claim of familywise control across all 12 comparisons')
    freeze(root/'PLAN.json',p)
    repo=Path(__file__).resolve().parents[1];code={}
    for folder in ['scripts','src/llm_integrity']:
        for path in sorted((repo/folder).rglob('*.py')):
            rel=path.relative_to(repo).as_posix();code[rel]=sha(path);dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==code[rel],('code changed',rel)
            else:shutil.copy2(path,dest)
    freeze(root/'CODE_MANIFEST.json',code)
    if not (root/'IDENTITY_AUDIT.json').exists():
        import importlib.metadata,subprocess
        freeze(root/'IDENTITY_AUDIT.json',dict(time=time.time(),model_files_verified=True,adapters_verified=cp['adapters_sha256'],
            python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','bitsandbytes','numpy']},
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True)))
    return p

def search(root,p,limit=None):
    registry={v['variant_id']:str(CAL/'adapters'/v['variant_id']) for v in p['search_variants'] if v['family']=='finetuning'}
    for i,source in enumerate(p['sources']):
        if limit is not None and i>=limit:break
        out=root/'generation'/f'{i:03d}.json'
        if out.exists():
            r=read(out);assert r['plan_identity']==identity(p) and r['source']==source
            if r.get('failure'):raise RuntimeError(f'Technical failure retained at {out}; repair before resuming')
            continue
        status(root,'generation',source_index=i,completed=i,total=300)
        sampler=StratifiedVariantSampler(p['search_variants'],family_weights={f:.2 for f in p['calibration']['macro_scales']},adapter_registry=registry,seed=p['sampler_seed'])
        bundle=load_model(p['model']);optimizer=None;start=time.time()
        try:
            optimizer=DiscreteJointInnerOptimizer(reference=bundle,sampler=sampler,model_config=p['model'],
                micro_scales=p['calibration']['micro_scales'],macro_scales=p['calibration']['macro_scales'],
                micro_weight=p['calibration']['micro_weight'],macro_weight=1.,seed=p['search_seed']+i*100,
                block_types=tuple(p['blocks']),**p['search_settings'])
            result=optimizer.optimize(source)
            record=dict(**result.payload(),source=source,source_index=i,plan_identity=identity(p),seconds=time.time()-start)
            save(out,record)
            if result.failure:raise RuntimeError(f'Technical search failure: {out}: {result.failure}')
        finally:(optimizer.reference if optimizer else bundle).close();clean()
        status(root,'generation',source_index=i,completed=i+1,total=300,accepted=record['accepted'])

def candidates(root,p):
    rows=[read(root/'generation'/f'{i:03d}.json') for i in range(300)]
    chosen={};mapping=[]
    for r in rows:
        assert not r.get('failure') and r['plan_identity']==identity(p)
        if not r['accepted']:continue
        text=r['optimized_prompt'];assert text!=r['source']['prompt']
        h=hashlib.sha256(text.encode()).hexdigest();pid='s_'+h[:20]
        if h not in chosen:chosen[h]=dict(id=pid,prompt=text,category=r['source']['category'],source_id=r['source']['id'],source_index=r['source_index'],prompt_sha256=h)
        mapping.append(dict(source_id=r['source']['id'],candidate_id=chosen[h]['id']))
    pool=sorted(chosen.values(),key=lambda r:r['id'])
    freeze(root/'CANDIDATES.json',dict(prompts=pool,mapping=mapping,accepted_sources=len(mapping),unique_texts=len(pool),fallbacks_in_pool=0))
    if len(pool)<8:status(root,'INSUFFICIENT_CANDIDATES',count=len(pool));raise RuntimeError('Fewer than 8 unique accepted candidates; do not pad or change frozen search')
    return pool

def encoded(bundle,text):
    ids=bundle.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,return_tensors='pt').to(bundle.device)
    assert ids.shape[1]<=128
    return ids,torch.ones_like(ids)

def score_candidates(root,p,pool):
    b=load_model(p['model']);b.model.eval()
    for x in b.model.parameters():x.requires_grad_(False)
    blocks=discover_micro_blocks(b.model,p['blocks'])
    try:
        for row in pool:
            path=root/'ranking_micro'/f"{row['id']}.json"
            if path.exists():assert read(path)['plan_identity']==identity(p);continue
            ids,mask=encoded(b,row['prompt']);emb=b.model.get_input_embeddings()(ids).detach();scores={}
            for bi,kind in enumerate(p['blocks']):
                block=next(x for x in blocks if x.block_type==kind and x.layer_id==p['layers'][bi])
                # Common probes across candidates make ranking comparable.
                r=score_block_micro_proxy(b,emb,mask,block=block,probes=8,seed=p['ranking_probe_seed']+bi*100)
                assert math.isfinite(r.raw_micro_score);scores[kind]=r.raw_micro_score
            save(path,dict(plan_identity=identity(p),scores=scores,token_count=int(ids.shape[1])))
            status(root,'ranking_micro',completed=len(list((root/'ranking_micro').glob('*.json'))),total=len(pool));clean()
        for v in p['search_variants']:
            folder=root/'ranking_macro'/v['variant_id']
            if all((folder/f"{r['id']}.json").exists() for r in pool):continue
            loaded=load_variant(dict(model=p['model']),v,CAL);loaded.bundle.model.eval()
            try:
                for row in pool:
                    path=folder/f"{row['id']}.json"
                    if path.exists():assert read(path)['plan_identity']==identity(p);continue
                    ids,mask=encoded(b,row['prompt'])
                    with torch.inference_mode():
                        emb=b.model.get_input_embeddings()(ids)
                        a=b.model(inputs_embeds=emb,attention_mask=mask,use_cache=False).logits[0,-1].float()
                        z=loaded.bundle.model(inputs_embeds=emb,attention_mask=mask,use_cache=False).logits[0,-1].float()
                        value=float(differentiable_macro_proxy(a,z,proxy='js'))
                    assert math.isfinite(value)
                    save(path,dict(plan_identity=identity(p),score=value,family=v['family'],variant_id=v['variant_id']))
                status(root,'ranking_macro',variant=v['variant_id'])
            finally:loaded.close();clean()
    finally:b.close();clean()
    scores={}
    for row in pool:
        m=read(root/'ranking_micro'/f"{row['id']}.json")['scores']
        a={f:float(np.mean([read(root/'ranking_macro'/v['variant_id']/f"{row['id']}.json")['score'] for v in p['search_variants'] if v['family']==f])) for f in p['calibration']['macro_scales']}
        micro=np.mean([m[k]/p['calibration']['micro_scales'][k] for k in p['blocks']]);macro=np.mean([a[f]/p['calibration']['macro_scales'][f] for f in a])
        scores[row['id']]=dict(micro_raw=m,macro_raw=a,micro_normalized=float(micro),macro_normalized=float(macro),score=float(p['calibration']['micro_weight']*micro+macro))
    freeze(root/'SCORES.json',scores);return scores

def select(root,p,pool,scores):
    if (root/'FINGERPRINTS.json').exists():
        saved=read(root/'FINGERPRINTS.json');assert saved['plan_identity']==identity(p)
        assert all(pid in scores for panel in saved['panels'].values() for pid in panel['prompt_ids'])
        return saved
    b=load_model(p['model']);cfg=p['mcc'];profiles={}
    profiler=TransformerActivationProfiler(**{k:cfg[k] for k in ['attention_entropy_fraction','ffn_quantile','residual_threshold','max_ffn_units_per_layer']})
    try:
        for row in pool:
            out=root/'coverage'/f"{row['id']}.json"
            if out.exists():r=read(out);assert r['plan_identity']==identity(p)
            else:
                a=profiler.profile(b,row['id'],row['prompt'],max_length=128);z=profiler.profile(b,row['id'],row['prompt'],max_length=128)
                j=len(a.components&z.components)/max(1,len(a.components|z.components));assert j>=cfg['minimum_jaccard']
                r=dict(plan_identity=identity(p),components=sorted(a.components&z.components),repeat_jaccard=j,diagnostics=a.diagnostics);save(out,r)
            profiles[row['id']]=set(r['components']);status(root,'coverage',completed=len(profiles),total=len(pool))
    finally:b.close();clean()
    mcc=greedy_mcc(profiles,8,weights=cfg['weights']);top=top_sensitivity({k:v['score'] for k,v in scores.items()},8)
    universe=set().union(*profiles.values());den=sum(component_weight(c,cfg['weights']) for c in universe)
    selections={}
    for method,ids in [('mcc',mcc.selected_ids),('top_sensitivity',top)]:
        for size in [4,8]:
            panel=ids[:size];covered=set().union(*(profiles[x] for x in panel))
            selections[f'{method}_{size}']=dict(method=method,size=size,prompt_ids=panel,
                weighted_candidate_coverage=sum(component_weight(c,cfg['weights']) for c in covered)/den,
                mean_sensitivity=float(np.mean([scores[x]['score'] for x in panel])))
    result=dict(plan_identity=identity(p),frozen_at=time.time(),panels=selections,mcc_trace=mcc.trace,universe_size=len(universe))
    freeze(root/'FINGERPRINTS.json',result);return result

def train_tests(root,p):
    from llm_integrity.paper_finetuning import train_lora_manifest_variant
    selected=read(root/'FINGERPRINTS.json');assert selected['plan_identity']==identity(p)
    texts={r['prompt'] for r in read(root/'CANDIDATES.json')['prompts']}
    assert not texts&{r['prompt'] for r in test_training()}
    for v in p['test_variants']:
        if v['family']!='finetuning':continue
        path=root/'training'/f"{v['variant_id']}.json"
        if path.exists():assert file_identity(root/'adapters'/v['variant_id'])==read(path)['adapter_files_sha256'];continue
        status(root,'test_training',variant=v['variant_id'])
        result=train_lora_manifest_variant(p['model'],v,data_path=root/'test_train.jsonl',output_root=root/'adapters',**p['test_training'])
        assert result.completed_steps==v['configuration']['steps']
        save(path,dict(**asdict(result),adapter_files_sha256=file_identity(root/'adapters'/v['variant_id'])));clean()

def collect(root,p,prompts):
    endpoints=[dict(label=f'normal_{i:02d}',variant=None) for i in range(20)]+[dict(label=v['variant_id'],variant=v) for v in p['test_variants']]
    binding=identity(p)
    for ei,e in enumerate(endpoints):
        path=root/'responses'/f"{e['label']}.jsonl";path.parent.mkdir(parents=True,exist_ok=True)
        rows=load_responses(path);seen=set();indices={r['id']:i for i,r in enumerate(prompts)}
        for r in rows:
            key=r['prompt_id'],r['response_index'];assert key not in seen and key[0] in indices and 0<=key[1]<80
            assert r['plan_identity']==binding and r['endpoint']==e['label'] and r['seed']==100700000+ei*100000+indices[key[0]]*1000+key[1]
            seen.add(key)
        if len(seen)==len(prompts)*80:continue
        loaded=load_variant(p,e['variant'],root) if e['variant'] else None
        b=loaded.bundle if loaded else load_model(p['model'])
        try:
            if loaded:freeze(root/'test_variant_reports'/f"{e['label']}.json",asdict(loaded.report))
            with path.open('a',encoding='utf8') as f:
                for pi,row in enumerate(prompts):
                    for j in range(80):
                        if (row['id'],j) in seen:continue
                        seed=100700000+ei*100000+pi*1000+j;token=generate(b,row,seed)
                        r=dict(endpoint=e['label'],prompt_id=row['id'],response_index=j,seed=seed,token_id=token,plan_identity=binding)
                        f.write(json.dumps(r)+'\n');f.flush()
                    status(root,'detection_sampling',endpoint=e['label'],completed_prompts=pi+1,total_prompts=len(prompts))
        finally:
            if loaded:loaded.close()
            else:b.close()
            clean()

def evaluate(root,p,fp,prompts,refs):
    from fingerprint_comparison_stats import summarize_decisions
    mc=nulls(root,refs);endpoints=[f'normal_{i:02d}' for i in range(20)]+[v['variant_id'] for v in p['test_variants']]
    reports=[]
    for label in endpoints:
        rows=load_responses(root/'responses'/f'{label}.jsonl');assert len(rows)==80*len(prompts)
        tokens={r['id']:[x['token_id'] for x in sorted(rows,key=lambda x:x['response_index']) if x['prompt_id']==r['id']] for r in prompts}
        for name,panel in fp['panels'].items():
            for n in p['queries']:
                decisions={pid:decision(tokens[pid],refs[pid],mc[pid],panel['size'],n,p) for pid in panel['prompt_ids']}
                reports.append(dict(endpoint=label,panel=name,n=n,query_cap=n*panel['size'],alarm=any(r['alarm'] for r in decisions.values()),prompts=decisions))
    freeze(root/'DECISIONS.json',reports);freeze(root/'RESULTS.json',summarize_decisions(p,reports))
    status(root,'COMPLETE',configurations=12,endpoints=60,unique_test_prompts=len(prompts),responses=60*80*len(prompts))

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--phase',choices=['prepare','search','all'],default='all');a.add_argument('--limit',type=int);args=a.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';torch.set_num_threads(4)
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=prepare(root)
        if args.phase=='prepare':status(root,'PREPARED');return
        search(root,p,args.limit)
        if args.phase=='search' or args.limit is not None:return
        pool=candidates(root,p);scores=score_candidates(root,p,pool);fp=select(root,p,pool,scores)
        needed=set(pid for r in fp['panels'].values() for pid in r['prompt_ids']);prompts=[r for r in pool if r['id'] in needed]
        assert len(needed)<=16
        train_tests(root,p);refs=reference(root,p,prompts);collect(root,p,prompts);evaluate(root,p,fp,prompts,refs)

if __name__=='__main__':main()
