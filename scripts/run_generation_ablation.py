"""Frozen same-source objective ablation; no macro computation in micro-only arm."""
import argparse
import gc
import hashlib
import math
import os
from pathlib import Path
import shutil
import tarfile
import time
import _bootstrap
import torch
from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer, select_position_diverse_candidates
from llm_integrity.inner_micro_proxy import score_block_micro_proxy
from llm_integrity.inner_variant_sampler import FAMILIES, StratifiedVariantSampler
from llm_integrity.modeling import load_model
from llm_integrity.stage2_contract import BLOCK_TYPES
from run_resf_small import read, save, freeze, identity, clean, progress
import run_paired_prompt_budget as paired
from evaluate_prompt_budget import reference, nulls

ARMS = ('micro_only', 'macro_only', 'joint_js')


def arm_parameters(arm, weight):
    assert arm in ARMS
    return dict(micro_weight=0.0 if arm == 'macro_only' else weight,
                macro_weight=0.0 if arm == 'micro_only' else 1.0,
                minimum_nondegraded_families=0 if arm == 'micro_only' else 5)


class AblationOptimizer(DiscreteJointInnerOptimizer):
    def _micro_gradient(self, **kw):
        if self.micro_weight == 0:
            return torch.zeros_like(kw['current_embeddings']), 0.0, {'disabled': True}, None
        return super()._micro_gradient(**kw)

    def _macro_gradient(self, **kw):
        if self.macro_weight == 0:
            assert self.minimum_nondegraded_families == 0
            return torch.zeros_like(kw['current_embeddings']), 0.0, [], [], []
        return super()._macro_gradient(**kw)

    def _rerank_candidates(self, **kw):
        if self.macro_weight != 0:
            return super()._rerank_candidates(**kw)
        assert self.minimum_nondegraded_families == 0 and not kw['samples']
        selected = select_position_diverse_candidates(kw['candidates'], limit=self.rerank_candidates)
        baseline = self.micro_weight * kw['baseline_micro_normalized']
        for item in selected:
            embeddings = self._embed_user(item['token_ids'])
            full, mask, _ = self._compose(kw['prefix_ids'], embeddings, kw['suffix_ids'])
            score = score_block_micro_proxy(self.reference, full.detach(), mask,
                block=kw['block'], probes=self.probes, seed=kw['probe_seed'])
            normalized = score.raw_micro_score / self.micro_scales[kw['block_type']]
            objective = self.micro_weight * normalized
            item.update(micro_score_raw=score.raw_micro_score, micro_score_normalized=normalized,
                proxy_objective=objective, proxy_gain=objective-baseline, macro_scores={},
                macro_variant_scores={}, training_family_checks={}, macro_gate_disabled=True,
                constraints_passed=(math.isfinite(objective) and math.isfinite(float(item['ppl']))
                    and item['ppl_ratio'] <= self.ppl_ratio_limit and item['edit_ratio'] <= self.max_edit_ratio))
            del full, mask, embeddings, score
        selected.sort(key=lambda x: x['proxy_gain'], reverse=True)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return selected, dict(micro_score_normalized=kw['baseline_micro_normalized'],
            macro_scores={}, macro_variant_scores={}, proxy_objective=baseline, macro_gate_disabled=True)


def prepare_search(root, previous):
    old = read(previous/'PLAN.json')
    p = dict(old, schema='generation-ablation-search-v1', sources=old['sources'][:20],
        proxies=list(ARMS), seed=323000000, queries=[25,50,100],
        generation='20 identical sources; three newly generated arms; failed optimization retains original; no extra source recruitment',
        search_seed=319010000, sampler_seed=319000000,
        calibration_policy='frozen preceding JS scales and weight, no recalibration or test-driven tuning',
        arm_parameters={a:arm_parameters(a, read(previous/'CALIBRATION.json')['micro_weights']['js']) for a in ARMS},
        search_settings=dict(rounds=5,probes=8,candidate_positions=4,candidates_per_position=16,
            rerank_candidates=8,max_length=128,max_edit_ratio=.5,ppl_ratio_limit=1e12,
            family_relative_tolerance=.01,variants_per_family=1,anchor_variants_per_family=0,
            gradient_restarts=1,family_gate_aggregation='mean',require_task_preservation=False,
            sequential_model_execution=False,representative_layer_count=4,block_schedule='balanced',
            task_validation_mode='strict_r1',macro_proxy='js',macro_top_k=10,
            enforce_surface_compatibility=False,enforce_perplexity=False))
    freeze(root/'PLAN.json',p)
    freeze(root/'CALIBRATION.json',read(previous/'CALIBRATION.json'))
    return p


def search(root,p):
    cal=read(root/'CALIBRATION.json')
    registry={v['variant_id']:str(Path(p['pilot'])/'adapters'/v['variant_id']) for v in p['search_variants'] if v['family']=='finetuning'}
    for si,source in enumerate(p['sources']):
        for arm in ARMS:
            path=root/'search'/f"{arm}_{source['id']}.json"
            if not path.exists():
                progress('ablation_search_start',arm=arm,source=source['id'])
                save(root/'STATUS.json',dict(phase='search',arm=arm,source=source['id'],completed=len(list((root/'search').glob('*.json'))),time=time.time()))
                sampler=StratifiedVariantSampler(p['search_variants'],family_weights={f:.2 for f in FAMILIES},adapter_registry=registry,seed=p['sampler_seed'])
                bundle=load_model(p['model']);optimizer=None;start=time.time()
                try:
                    optimizer=AblationOptimizer(reference=bundle,sampler=sampler,model_config=p['model'],
                        micro_scales=cal['micro_scales'],macro_scales=cal['macro_scales']['js'],
                        **p['arm_parameters'][arm],**p['search_settings'],
                        seed=p['search_seed']+si*100,block_types=tuple(BLOCK_TYPES))
                    result=optimizer.optimize(source)
                    record=dict(**result.payload(),source=source,proxy=arm,seconds=time.time()-start,
                        arm_parameters=p['arm_parameters'][arm],search_seed=p['search_seed']+si*100)
                    save(path,record)
                finally:
                    (optimizer.reference if optimizer else bundle).close();clean()
            r=read(path)
            assert r['source']==source and r['arm_parameters']==p['arm_parameters'][arm]
            if r.get('failure'): raise RuntimeError(f'Technical failure must be repaired, not converted to fallback: {path}: {r["failure"]}')
            if arm=='micro_only':
                for h in r['history']:
                    assert not h['macro'] and not h['gradient_variant_ids'] and not h['anchor_variant_ids']
                    assert all(c.get('macro_gate_disabled') and not c['macro_scores'] for c in h['reranked_candidates'])
            progress('ablation_search_done',arm=arm,source=source['id'],accepted=r['accepted'])
    save(root/'STATUS.json',dict(phase='COMPLETE',jobs=60,time=time.time()))


def prepare_evaluation(root, search_root):
    # Reuse the proven source mapping and paired subset implementation, not old samples.
    staging=root/'prepared_mapping'
    staging.mkdir(parents=True,exist_ok=True)
    p=paired.prepare(staging,search_root)
    p.update(schema='generation-ablation-evaluation-v1',seed=600000000,
        generation='fresh micro_only / macro_only / joint_js searches; same sources, search seeds and budgets',
        sampling_policy='all reference distributions, MC nulls, development and test responses freshly collected on current server',
        caveat='existing attack models; exploratory mechanism ablation, not independent confirmation',
        search_plan_sha256=identity(read(search_root/'PLAN.json')))
    freeze(root/'PLAN.json',p)
    freeze(root/'SUBSETS.json',read(staging/'SUBSETS.json'))
    return p


def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
    a.add_argument('--previous',type=Path,required=True);x=a.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(4)
    import fcntl, subprocess, platform
    root=x.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            sr=root/'generation';sr.mkdir(exist_ok=True)
            p=read(sr/'PLAN.json') if (sr/'PLAN.json').exists() else prepare_search(sr,x.previous.resolve())
            if not (root/'ENVIRONMENT.json').exists():
                freeze(root/'ENVIRONMENT.json',dict(torch=torch.__version__,python=platform.python_version(),
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True),
                    model_sha256=hashlib.sha256((Path(p['model']['name'])/'model.safetensors').read_bytes()).hexdigest()))
            code=root/'code';code.mkdir(exist_ok=True)
            if not (code/'SHA256.json').exists():
                repo=Path(__file__).resolve().parents[1]
                files=list((repo/'src'/'llm_integrity').glob('*.py'))+list((repo/'scripts').glob('*.py'))
                hashes={}
                for f in files:
                    rel=f.relative_to(repo);dst=code/rel;dst.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(f,dst);hashes[str(rel)]=hashlib.sha256(f.read_bytes()).hexdigest()
                freeze(code/'SHA256.json',hashes)
            search(sr,p)
            er=root/'evaluation';er.mkdir(exist_ok=True)
            ep=read(er/'PLAN.json') if (er/'PLAN.json').exists() else prepare_evaluation(er,sr)
            refs=reference(er,ep,ep['prompts']);paired.collect(er,ep,'development')
            mc=nulls(er,refs);ranks=paired.rank(er,ep,refs,mc)
            paired.collect(er,ep,'test');paired.summarize(er,ep,refs,mc,ranks)
            save(root/'STATUS.json',dict(phase='COMPLETE',unique_prompts=len(ep['prompts']),time=time.time()))
            archive=root.parent/(root.name+'-results.tar.gz')
            with tarfile.open(archive,'w:gz') as t:t.add(root,arcname=root.name)
            save(root/'ARCHIVE.json',dict(path=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size))
        except Exception as exc:
            save(root/'STATUS.json',dict(phase='FAILED',error=repr(exc),time=time.time()));raise


if __name__=='__main__':main()
