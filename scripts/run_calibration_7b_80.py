"""7B calibration only: 80 nested texts, five configurations per family."""
import argparse,fcntl,hashlib,json,math,os,platform,shutil,subprocess,time,importlib.metadata
from pathlib import Path
from dataclasses import asdict
import _bootstrap
import numpy as np
import torch
from run_resf_small import read,save,freeze,identity,file_identity,clean,progress,load_variant
from run_calibration_scale_comparison import sha,summarize
from calibration_80_pool import build,CATS
from llm_integrity.modeling import load_model
from llm_integrity.inner_micro_proxy import discover_micro_blocks,representative_layers,differentiable_block_micro_proxy
from llm_integrity.macro_proxy import differentiable_macro_proxy
from llm_integrity.stage2_contract import BLOCK_TYPES

BASE=Path('/root/autodl-tmp/token-integrity')

def prepare(root):
    previous=read(BASE/'runs/calibration-scale-v1/CALIBRATION_PLAN.json')
    manifest=read(BASE/'models/qwen7b/DOWNLOAD_MANIFEST.json')
    selected=build(previous['selected'])
    train_path=BASE/'runs/fresh-v3/train.jsonl'
    training=[json.loads(x) for x in train_path.read_text().splitlines() if x.strip()]
    exclude=set(previous['excluded_texts'])|{r['prompt'] for r in training}
    assert not {r['prompt'] for r in selected}&exclude
    from transformers import AutoTokenizer
    model=dict(previous['model'],name=str(BASE/'models/qwen7b'),device_map={'':'cuda:0'})
    tokenizer=AutoTokenizer.from_pretrained(model['name'],local_files_only=True)
    ids={r['id']:tokenizer.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in selected}
    assert max(map(len,ids.values()))<=128
    layer_count=manifest['config']['num_hidden_layers'];layers=[round(i*(layer_count-1)/3) for i in range(4)]
    variants=[]
    settings=[('gaussian_noise','std_ratio',[.001,.0025,.004,.006,.008]),
              ('finetuning','steps',[5,10,20,30,40]),
              ('unstructured_pruning','ratio',[.05,.10,.15,.20,.25]),
              ('structured_pruning','ratio',[.001,.0025,.005,.0075,.01]),
              ('quantization','llm_int8_threshold',[2.,4.,6.,8.,10.])]
    for family,key,values in settings:
        template=next(v for v in previous['variants'] if v['family']==family)
        for i,value in enumerate(values):
            variants.append(dict(template,variant_id=f'cal7b_{family}_{i}',configuration=dict(template['configuration'],**{key:value})))
    for item in manifest['files']:
        assert sha(Path(model['name'])/item['path'])==item['sha256']
    plan=dict(schema='calibration-7b-80-five-variants-v1',model=model,model_revision=manifest['revision'],selected=selected,
        token_ids=ids,excluded_texts=sorted(exclude),nested_sizes=[20,40,80],blocks=list(BLOCK_TYPES),layers=[layers[i%4] for i in range(7)],
        representative_layers=layers,probes=8,probe_seed_formula='926020000 + 10000*b + 100*i + r',
        variants=variants,family_weights={f:.2 for f,_,_ in settings},training_data_sha256=sha(train_path),
        training=dict(batch_size=1,gradient_accumulation_steps=4,max_length=128,optimizer='adamw_torch',weight_decay=0.,warmup_ratio=0.),
        family_score='arithmetic mean of five JS scores per prompt',family_gradient='arithmetic mean of five embedding gradient vectors, then norm',
        scale='median across prompts including valid zeros',macro_proxy='js',softmax_temperature=1.,top_k=None,top_p=None,
        lambda_clip=[.1,10.],bootstrap_seed=926030000,bootstrap_repetitions=1000,minimum_valid_scale=1e-20,
        scope='calibration only; no sensitive prompt search, MCC, reference fitting or detector responses',
        distinctness_policy='retain all prespecified configurations; report identical calibration outputs, never replace based on outcome',
        baseline_1p5b_plan_identity=identity(previous),model_manifest_identity=identity(manifest))
    freeze(root/'CALIBRATION_PLAN.json',plan)
    if (root/'train.jsonl').exists():assert sha(root/'train.jsonl')==sha(train_path)
    else:shutil.copy2(train_path,root/'train.jsonl')
    repo=Path(__file__).resolve().parents[1];code={}
    for folder in ['scripts','src/llm_integrity']:
        for p in sorted((repo/folder).rglob('*.py')):
            rel=str(p.relative_to(repo));code[rel]=sha(p);dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==code[rel],('code changed',rel)
            else:shutil.copy2(p,dest)
    freeze(root/'CODE_MANIFEST.json',code)
    freeze(root/'ENVIRONMENT.json',dict(python=platform.python_version(),packages={x:importlib.metadata.version(x) for x in ['torch','transformers','peft','bitsandbytes','numpy','accelerate']},
        cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
        git_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    return plan


def status(root,phase,**kw):
    x=dict(phase=phase,time=time.time(),**kw);save(root/'STATUS.json',x);progress(phase,**kw)


def train(root,plan):
    from llm_integrity.paper_finetuning import train_lora_manifest_variant
    for v in plan['variants']:
        if v['family']!='finetuning':continue
        out=root/'training'/f"{v['variant_id']}.json"
        if out.exists():
            assert read(out)['adapter_files_sha256']==file_identity(root/'adapters'/v['variant_id']);continue
        status(root,'training',variant=v['variant_id'],steps=v['configuration']['steps'])
        result=train_lora_manifest_variant(plan['model'],v,data_path=root/'train.jsonl',output_root=root/'adapters',
            max_length=128,batch_size=1,gradient_accumulation_steps=4)
        assert result.completed_steps==v['configuration']['steps']
        save(out,dict(**asdict(result),adapter_files_sha256=file_identity(root/'adapters'/v['variant_id'])))
        clean()


def inputs(bundle,plan,row):
    ids=torch.tensor([plan['token_ids'][row['id']]],device=bundle.device)
    return bundle.model.get_input_embeddings()(ids).detach().requires_grad_(True),torch.ones_like(ids)


def micro(root,plan):
    binding=identity(plan)
    bundle=load_model(plan['model']);bundle.model.eval()
    for p in bundle.model.parameters():p.requires_grad_(False)
    blocks=discover_micro_blocks(bundle.model,plan['blocks'])
    try:
        for bi,kind in enumerate(plan['blocks']):
            block=next(b for b in blocks if b.block_type==kind and b.layer_id==plan['layers'][bi])
            for i,row in enumerate(plan['selected']):
                path=root/'micro'/f'{bi}_{i:02d}.json'
                if path.exists():assert read(path)['plan_identity']==binding;continue
                e,mask=inputs(bundle,plan,row)
                score=differentiable_block_micro_proxy(bundle,e,mask,block=block,probes=8,seed=926020000+10000*bi+100*i)
                data=score.metadata();assert data['gradient_finite'] and data['parameter_gradient_finite']
                assert math.isfinite(data['raw_micro_score']) and math.isfinite(data['raw_embedding_gradient_norm'])
                save(path,dict(kind='micro',source_index=i,source_id=row['id'],category=row['category'],plan_identity=binding,completed_at=time.time(),**data))
                del score,e,mask;clean()
                status(root,'micro',block=kind,source_index=i,completed=len(list((root/'micro').glob('*.json'))),total=560)
    finally:bundle.close();clean()


def macro(root,plan):
    binding=identity(plan)
    bundle=load_model(plan['model']);bundle.model.eval()
    for p in bundle.model.parameters():p.requires_grad_(False)
    try:
        for v in plan['variants']:
            folder=root/'macro_variants'/v['variant_id'];folder.mkdir(parents=True,exist_ok=True)
            if len(list(folder.glob('*.json')))==80:continue
            status(root,'loading_macro_variant',variant=v['variant_id'])
            loaded=load_variant(plan,v,root);loaded.bundle.model.eval()
            freeze(root/'variant_reports'/f"{v['variant_id']}.json",asdict(loaded.report))
            for p in loaded.bundle.model.parameters():p.requires_grad_(False)
            try:
                for i,row in enumerate(plan['selected']):
                    path=folder/f'{i:02d}.json';gp=folder/f'{i:02d}.npz'
                    if path.exists():assert read(path)['plan_identity']==binding and sha(gp)==read(path)['gradient_sha256'];continue
                    e,mask=inputs(bundle,plan,row)
                    a=bundle.model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits[0,-1].float()
                    b=loaded.bundle.model(inputs_embeds=e,attention_mask=mask,use_cache=False).logits[0,-1].float()
                    score=differentiable_macro_proxy(a,b,proxy='js',top_k=10)
                    grad=torch.autograd.grad(score,e)[0].detach().float()
                    assert torch.isfinite(grad).all() and torch.isfinite(score)
                    # Keep full vectors so aggregation and its norm can be independently audited.
                    np.savez_compressed(gp,gradient=grad.cpu().numpy())
                    save(path,dict(kind='macro_variant',family=v['family'],variant_id=v['variant_id'],source_index=i,
                        source_id=row['id'],category=row['category'],score=float(score),gradient_norm=float(grad.norm()),
                        logits_sha256=hashlib.sha256(b.detach().cpu().numpy().tobytes()).hexdigest(),
                        gradient_sha256=sha(gp),plan_identity=binding,completed_at=time.time()))
                    del e,mask,a,b,score,grad;clean()
                    status(root,'macro',variant=v['variant_id'],source_index=i,
                        completed=len(list((root/'macro_variants').glob('*/*.json'))),total=2000)
            finally:loaded.close();clean()
    finally:bundle.close();clean()
    for family in plan['family_weights']:
        variants=[v for v in plan['variants'] if v['family']==family]
        assert len(variants)==5
        for i,row in enumerate(plan['selected']):
            records=[read(root/'macro_variants'/v['variant_id']/f'{i:02d}.json') for v in variants]
            grads=[np.load(root/'macro_variants'/v['variant_id']/f'{i:02d}.npz')['gradient'].astype(np.float64) for v in variants]
            mean_grad=np.mean(grads,axis=0)
            save(root/'macro_means'/f'{family}_{i:02d}.json',dict(kind='macro',family=family,source_index=i,source_id=row['id'],category=row['category'],
                score=float(np.mean([r['score'] for r in records])),gradient=float(np.linalg.norm(mean_grad)),
                individual_gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in records])),
                variants=[v['variant_id'] for v in variants],plan_identity=binding))


def aggregate(root,plan):
    mi=[read(p) for p in sorted((root/'micro').glob('*.json'))]
    ma=[read(p) for p in sorted((root/'macro_means').glob('*.json'))]
    assert len(mi)==560 and len(ma)==400
    assert len(list((root/'macro_variants').glob('*/*.json')))==2000
    families=list(plan['family_weights']);scales={};ci={};stats={}
    for n in [20,40,80]:
        m=[r for r in mi if r['source_index']<n];a=[r for r in ma if r['source_index']<n]
        scales[str(n)]=summarize(m,a,plan['blocks'],families);assert scales[str(n)]['valid'],('invalid scale',n)
        rng=np.random.default_rng(926030000);boots=[]
        cat_ids=[[i for i,r in enumerate(plan['selected'][:n]) if r['category']==cat] for cat in CATS]
        u=np.array([[next(r['raw_micro_score'] for r in m if r['source_index']==i and r['block']['block_type']==b) for b in plan['blocks']] for i in range(n)])
        g=np.array([[next(r['raw_embedding_gradient_norm'] for r in m if r['source_index']==i and r['block']['block_type']==b) for b in plan['blocks']] for i in range(n)])
        d=np.array([[next(r['score'] for r in a if r['source_index']==i and r['family']==f) for f in families] for i in range(n)])
        h=np.array([[next(r['gradient'] for r in a if r['source_index']==i and r['family']==f) for f in families] for i in range(n)])
        for _ in range(1000):
            ix=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in cat_ids]).tolist()
            # Use matrices for faster bootstrap, with the same paired source resampling.
            sm=np.median(u[ix],axis=0);sd=np.median(d[ix],axis=0)
            if np.any(sm<=1e-20) or np.any(sd<=1e-20):continue
            gm=np.median(g[ix]/sm);ga=np.median(h[ix]/sd)
            w=ga/max(gm,1e-20);boots.append(np.r_[sm,sd,np.clip(w,.1,10),w])
        ci[str(n)]=dict(valid=len(boots),invalid=1000-len(boots),percentiles={k:np.percentile(np.array(boots)[:,j],[2.5,50,97.5]).tolist() for j,k in enumerate(plan['blocks']+families+['micro_weight','micro_weight_unclipped'])})
    freeze(root/'CALIBRATION.json',scales['80']);save(root/'NESTED_CALIBRATION.json',scales);save(root/'BOOTSTRAP.json',ci)
    for kind,rows,keys,field,grad in [('micro',mi,plan['blocks'],'raw_micro_score','raw_embedding_gradient_norm'),('macro',ma,families,'score','gradient')]:
        for key in keys:
            subset=[r for r in rows if (r['block']['block_type'] if kind=='micro' else r['family'])==key]
            stats[key]={}
            for cat in ['all']+CATS:
                chosen=subset if cat=='all' else [r for r in subset if r['category']==cat]
                stats[key][cat]=dict(count=len(chosen),percentiles=np.percentile([r[field] for r in chosen],[0,25,50,75,100]).tolist(),zero_scores=sum(r[field]==0 for r in chosen),zero_gradients=sum(r[grad]==0 for r in chosen))
    save(root/'DESCRIPTIVE_STATS.json',stats)
    distinct={}
    for f in families:
        vs=[v for v in plan['variants'] if v['family']==f];pairs=[]
        for vi,v in enumerate(vs):
            for w in vs[vi+1:]:
                equal=sum(read(root/'macro_variants'/v['variant_id']/f'{i:02d}.json')['logits_sha256']==read(root/'macro_variants'/w['variant_id']/f'{i:02d}.json')['logits_sha256'] for i in range(80))
                pairs.append(dict(a=v['variant_id'],b=w['variant_id'],identical_logits_prompts=equal))
        distinct[f]=pairs
    save(root/'VARIANT_DISTINCTNESS.json',distinct)
    save(root/'CALIBRATION_AUDIT.json',dict(status='PASS',micro_records=560,macro_variant_records=2000,macro_means=400,nested=[20,40,80],plan_identity=identity(plan)))
    status(root,'COMPLETE')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=BASE/'runs/calibration-7b-80-v1');args=parser.parse_args()
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare(root)
        train(root,plan);micro(root,plan);macro(root,plan);aggregate(root,plan)

if __name__=='__main__':main()
