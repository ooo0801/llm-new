"""Fresh 100-text calibration; 300 held-out optimization texts are never evaluated."""
import argparse,fcntl,hashlib,json,math,platform,shutil,subprocess,time,importlib.metadata
from pathlib import Path
from dataclasses import asdict
import _bootstrap
import numpy as np
import torch
from run_resf_small import read,save,freeze,identity,file_identity,clean,load_variant
from run_calibration_scale_comparison import sha
from run_calibration_7b_80 import inputs,status
from calibration_400_pool import build,split,CATS,SUBTASKS
from calibration_geo_stats import calculate
from llm_integrity.modeling import load_model
from llm_integrity.inner_micro_proxy import discover_micro_blocks,differentiable_block_micro_proxy
from llm_integrity.macro_proxy import differentiable_macro_proxy

BASE=Path('/root/autodl-tmp/token-integrity')

def prepare(root):
    oldroot=BASE/'runs/calibration-7b-80-v1';old=read(oldroot/'CALIBRATION_PLAN.json')
    manifest=read(BASE/'models/qwen7b/DOWNLOAD_MANIFEST.json')
    for item in manifest['files']:assert sha(BASE/'models/qwen7b'/item['path'])==item['sha256']
    rows=build();selected,opt=split(rows)
    from audit_calibration_400_pool import audit
    audit(rows,selected,opt)
    train_path=oldroot/'train.jsonl';assert sha(train_path)==old['training_data_sha256']
    training=[json.loads(x) for x in train_path.read_text().splitlines() if x.strip()]
    assert not {r['prompt'].strip() for r in rows}&{r['prompt'].strip() for r in training}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(old['model']['name'],local_files_only=True)
    ids={r['id']:tokenizer.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in rows}
    assert max(map(len,ids.values()))<=128,[(k,len(v)) for k,v in ids.items() if len(v)>128]
    for name,data in [('MOTHER_POOL',rows),('CALIBRATION_POOL',selected),('OPTIMIZATION_POOL',opt)]:freeze(root/f'{name}.json',data)
    pool_manifest={name:sha(root/name) for name in ['MOTHER_POOL.json','CALIBRATION_POOL.json','OPTIMIZATION_POOL.json']}
    freeze(root/'POOL_MANIFEST.json',pool_manifest)
    adapters={}
    for v in old['variants']:
        if v['family']!='finetuning':continue
        vid=v['variant_id'];report=read(oldroot/'training'/f'{vid}.json')
        assert file_identity(oldroot/'adapters'/vid)==report['adapter_files_sha256']
        adapters[vid]=report['adapter_files_sha256']
        dest=root/'adapters'/vid
        if not dest.exists():shutil.copytree(oldroot/'adapters'/vid,dest)
        assert file_identity(dest)==adapters[vid]
        freeze(root/'training'/f'{vid}.json',report)
    plan=dict(schema='calibration-7b-100-category-geometric-v1',model=old['model'],model_revision=manifest['revision'],
        selected=selected,token_ids={r['id']:ids[r['id']] for r in selected},blocks=old['blocks'],layers=old['layers'],probes=8,
        probe_seed_formula='930020000+10000*block_index+100*source_index+probe_index',variants=old['variants'],family_weights=old['family_weights'],
        training_data_sha256=sha(train_path),adapters_sha256=adapters,model_manifest_identity=identity(manifest),
        pool_manifest=pool_manifest,pool_provenance='synthetic_prespecified_v1; not public dataset samples',
        split_seed_formula='930010000+100*category_index+subtask_index',split_strata='category x subtask, 5 of 20',
        micro_aggregation='pooled median of 100 scores per block',macro_aggregation='category_median_equal_weight_geometric_mean',
        family_score='arithmetic mean of 5 full-vocabulary first-token JS scores',family_gradient='norm of arithmetic mean of 5 full embedding gradient vectors',
        macro_proxy='js',softmax_temperature=1.,top_k=None,top_p=None,lambda_clip=[.1,10.],macro_weight=1.,
        bootstrap_seed=930030000,bootstrap_repetitions=1000,bootstrap_strata='category; 25 replacement samples each, shared indices across metrics',
        minimum_valid_scale=1e-20,invalid_policy='retain diagnostics; no smoothing, sample replacement or default scales',
        scope='calibration only; no generation, MCC or detection; 300 held-out texts not evaluated',previous_plan_identity=identity(old))
    assert plan['layers']==[0,9,18,27,0,9,18]
    freeze(root/'CALIBRATION_PLAN.json',plan);freeze(root/'MODEL_MANIFEST.json',manifest)
    if not (root/'train.jsonl').exists():shutil.copy2(train_path,root/'train.jsonl')
    assert sha(root/'train.jsonl')==plan['training_data_sha256']
    distributions={}
    from collections import Counter
    for name,subset in [('mother',rows),('calibration',selected),('optimization',opt)]:
        distributions[name]=dict(count=len(subset),category=dict(Counter(r['category'] for r in subset)),
            subtask=dict(Counter(r['subtask'] for r in subset)),template=dict(Counter(r['template'] for r in subset)),
            difficulty=dict(Counter(r['difficulty'] for r in subset)),length_percentiles=np.percentile([len(ids[r['id']]) for r in subset],[0,25,50,75,100]).tolist(),
            token_lengths={r['id']:len(ids[r['id']]) for r in subset})
    freeze(root/'POOL_AUDIT.json',dict(status='PASS',distributions=distributions,semantic_audit='audit_calibration_400_pool.py',
        exact_training_overlap=0,note='Templates are intentionally shared across the split; not a held-out-template generalization test. Semantic classes are prespecified author labels.'))
    repo=Path(__file__).resolve().parents[1];code={}
    for folder in ['scripts','src/llm_integrity']:
        for p in sorted((repo/folder).rglob('*.py')):
            rel=p.relative_to(repo).as_posix();code[rel]=sha(p);dest=root/'code_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert sha(dest)==code[rel],('code changed',rel)
            else:shutil.copy2(p,dest)
    freeze(root/'CODE_MANIFEST.json',code)
    if not (root/'ENVIRONMENT.json').exists():
        freeze(root/'ENVIRONMENT.json',dict(python=platform.python_version(),packages={x:importlib.metadata.version(x) for x in ['torch','transformers','peft','bitsandbytes','numpy','accelerate']},
            cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
            git_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()))
    return plan

def micro(root,plan):
    binding=identity(plan);bundle=load_model(plan['model']);bundle.model.eval()
    for p in bundle.model.parameters():p.requires_grad_(False)
    blocks=discover_micro_blocks(bundle.model,plan['blocks'])
    try:
        for bi,kind in enumerate(plan['blocks']):
            block=next(b for b in blocks if b.block_type==kind and b.layer_id==plan['layers'][bi])
            for i,row in enumerate(plan['selected']):
                path=root/'micro'/f'{bi}_{i:02d}.json'
                if path.exists():assert read(path)['plan_identity']==binding;continue
                e,mask=inputs(bundle,plan,row)
                score=differentiable_block_micro_proxy(bundle,e,mask,block=block,probes=8,seed=930020000+10000*bi+100*i)
                data=score.metadata();assert data['gradient_finite'] and data['parameter_gradient_finite']
                assert math.isfinite(data['raw_micro_score']) and math.isfinite(data['raw_embedding_gradient_norm'])
                save(path,dict(kind='micro',source_index=i,source_id=row['id'],category=row['category'],plan_identity=binding,completed_at=time.time(),**data))
                del score,e,mask;clean();status(root,'micro',block=kind,source_index=i,completed=len(list((root/'micro').glob('*.json'))),total=700)
    finally:bundle.close();clean()

def macro(root,plan):
    binding=identity(plan);bundle=load_model(plan['model']);bundle.model.eval()
    for p in bundle.model.parameters():p.requires_grad_(False)
    try:
        for v in plan['variants']:
            folder=root/'macro_variants'/v['variant_id'];folder.mkdir(parents=True,exist_ok=True)
            if len(list(folder.glob('*.json')))==100:
                for i in range(100):
                    r=read(folder/f'{i:02d}.json');assert r['plan_identity']==binding and r['gradient_sha256']==sha(folder/f'{i:02d}.npz')
                continue
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
                    np.savez_compressed(gp,gradient=grad.cpu().numpy())
                    save(path,dict(kind='macro_variant',family=v['family'],variant_id=v['variant_id'],source_index=i,
                        source_id=row['id'],category=row['category'],score=float(score),gradient_norm=float(grad.norm()),
                        logits_sha256=hashlib.sha256(b.detach().cpu().numpy().tobytes()).hexdigest(),gradient_sha256=sha(gp),plan_identity=binding,completed_at=time.time()))
                    del e,mask,a,b,score,grad;clean();status(root,'macro',variant=v['variant_id'],source_index=i,completed=len(list((root/'macro_variants').glob('*/*.json'))),total=2500)
            finally:loaded.close();clean()
    finally:bundle.close();clean()
    for family in plan['family_weights']:
        variants=[v for v in plan['variants'] if v['family']==family];assert len(variants)==5
        for i,row in enumerate(plan['selected']):
            records=[read(root/'macro_variants'/v['variant_id']/f'{i:02d}.json') for v in variants]
            grads=[np.load(root/'macro_variants'/v['variant_id']/f'{i:02d}.npz')['gradient'].astype(np.float64) for v in variants]
            mean_grad=np.mean(grads,axis=0)
            save(root/'macro_means'/f'{family}_{i:02d}.json',dict(kind='macro',family=family,source_index=i,source_id=row['id'],category=row['category'],
                score=float(np.mean([r['score'] for r in records])),gradient=float(np.linalg.norm(mean_grad)),
                individual_gradient_norm_mean=float(np.mean([r['gradient_norm'] for r in records])),variants=[v['variant_id'] for v in variants],plan_identity=binding))

def aggregate(root,plan):
    mi=[read(p) for p in sorted((root/'micro').glob('*.json'))];ma=[read(p) for p in sorted((root/'macro_means').glob('*.json'))]
    assert len(mi)==700 and len(ma)==500 and len(list((root/'macro_variants').glob('*/*.json')))==2500
    assert all(r['plan_identity']==identity(plan) for r in mi+ma)
    results=calculate(plan,mi,ma)
    save(root/'STATISTICS.json',results)
    primary=results['summaries']['category_median_equal_weight_geometric_mean']
    freeze(root/'CALIBRATION.json',primary)
    save(root/'CALIBRATION_AUDIT.json',dict(status='PASS' if primary['valid'] else 'INVALID_SCALE',micro_records=700,macro_variant_records=2500,macro_means=500,plan_identity=identity(plan)))
    status(root,'COMPLETE' if primary['valid'] else 'INVALID_SCALE')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=BASE/'runs/calibration-7b-100-geo-v1');parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);plan=prepare(root)
        if args.prepare_only:status(root,'PREPARED');return
        micro(root,plan);macro(root,plan);aggregate(root,plan)

if __name__=='__main__':main()
