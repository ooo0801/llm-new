"""Phase-separated evaluation; never modifies the frozen Stage2 search driver.

Run freeze only after both searches; inspect its actual budget before sampling.
Confirmation selection is frozen before held-out responses are generated.
"""
import argparse
import dataclasses
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

from _bootstrap import project_path
from stage2_pilot import OUT, get_design, read, setup, cleanup, loaded_variant, attacks, finite_payload
from stage2_eval_contract import collect_candidates, schedules, sha_value, utility_qualification, candidate_gate, rank_correlation
from llm_integrity.stage1_r1 import canonical, digest, evaluate_task_r1, permutation_test
from llm_integrity.stage1_r2 import freeze_json, atomic_json, read_chain

SEMANTIC={'name':'BAAI/bge-small-zh-v1.5','revision':'7999e1d3359715c523056ef9478215996d62a620',
          'weight_sha256':'354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026',
          'max_seq_length':512,'normalize_embeddings':True}
EVAL=OUT/'evaluation'


def status(phase,**kw):
    value={'phase':phase,'utc_unix':time.time(),**kw}
    atomic_json(EVAL/'STATUS.json',value); print(canonical(value),flush=True)


def freeze_evaluation():
    from run_stage1_r2 import snapshot_files
    p=get_design()
    files=[OUT/'search'/m/(r['id']+'.json') for m in p['methods'] for r in p['prompts']]
    results=[read(f) for f in files]
    if any(r['design_sha256']!=digest(OUT/'DESIGN.json') for r in results): raise ValueError('Search binding differs')
    candidates=collect_candidates(p,results); banks=schedules(p,candidates)
    sem=snapshot_files(SEMANTIC['name'],SEMANTIC['revision'],SEMANTIC['weight_sha256'])
    source_files=[Path(__file__),project_path('scripts/stage2_eval_contract.py'),project_path('scripts/run_stage1_r2.py'),
                  project_path('scripts/build_discrete_candidate_portfolio.py'),project_path('scripts/stage2_prepare.py')]
    plan={'schema':'six-source-pilot-evaluation-v1','design_sha256':digest(OUT/'DESIGN.json'),
          'search_files':{str(f.relative_to(OUT)):digest(f) for f in files},'candidates':candidates,
          'banks':{k:{'count':len(v),'sha256':sha_value(v)} for k,v in banks.items()},
          'behavior_response_count':sum(len(v) for k,v in banks.items() if not k.startswith('utility_')),
          'utility_response_count':sum(len(v) for k,v in banks.items() if k.startswith('utility_')),
          'semantic':SEMANTIC,'semantic_model':sem,
          'packages':{s:importlib.metadata.version(s) for s in ('torch','transformers','sentence-transformers','numpy','scipy','peft','bitsandbytes')},
          'source_hashes':{str(f.relative_to(project_path('.'))).replace('\\','/'):digest(f) for f in source_files},
          'utility_minimum_intact_correct':6,'utility_scope':'Eight independent task instances in two admitted categories; not broad model utility',
          'outer_micro':{'probes':8,'seed':20269991,'scope':'All unique intact-model parameters; full-parameter autograd, no local-block approximation',
                         'role':'Independent-probe audit, not an additional post-hoc selection gate'},
          'prediction':{'x':'Median of three TRAIN variant values within family, except top1 uses mean flip fraction; source-averaged across aliases',
                        'y':'Mean held-out seed MMD2 or raw distance within each family/strength',
                        'unit':'Source; variants from the same source averaged before correlation',
                        'selection':'All capped candidates before behavior filtering; also report source baselines separately',
                        'inference':'Descriptive ranks only; six instances include repeated templates; no proxy superiority test'},
          'behavior_selection':'Confirmation-only gate before held-out generation; all capped candidates still measured for diagnostic comparison',
          'effect':'Report signed unbiased MMD2, raw bounded distances, permutation p and null_std. No epsilon-denominator standardized effects.',
          'determinism':'Fixed seeds and exact saved responses; CUDA sampling and pruning are warn_only, no bitwise rerun guarantee',
          'known_limits':['Only six source instances/two categories, repeated templates','No pure-behavior search or proxy weight ablation',
                          'Same-candidate diagnostics are not six optimized-search arms','Greedy exact-answer preservation is not a complete semantic-equivalence proof',
                          'Independent LoRA runs at each step count, with linear schedules; not shared-trajectory checkpoints',
                          'K8 reduces estimator noise, does not prove exact Jacobian or accurate behavior prediction']}
    freeze_json(EVAL/'PLAN.json',plan)
    status('FROZEN_NO_BEHAVIOR_SAMPLED',unique_prompts=len(candidates['prompts']),
           behavior_response_count=plan['behavior_response_count'],utility_response_count=plan['utility_response_count'])


def context():
    p=get_design(); q=read(EVAL/'PLAN.json')
    if digest(OUT/'DESIGN.json')!=q['design_sha256']: raise ValueError('Design changed')
    for f,s in q['search_files'].items():
        if digest(OUT/f)!=s: raise ValueError('Search changed')
    for f,s in q['source_hashes'].items():
        if digest(project_path(f))!=s: raise ValueError('Evaluation code changed after freeze')
    for package,version in q['packages'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('Evaluation environment changed')
    banks=schedules(p,q['candidates'])
    if {k:{'count':len(v),'sha256':sha_value(v)} for k,v in banks.items()}!=q['banks']: raise ValueError('Schedule changed')
    return p,q,banks


def bank(name, expected, complete=True):
    return read_chain(EVAL/'responses'/(name+'.jsonl'),expected,digest(EVAL/'PLAN.json'),complete)


def generate_bank(bundle,p,q,name,expected):
    import numpy as np
    from llm_integrity.modeling import render_prompt
    torch=setup(); torch.set_num_threads(4)
    rows=bank(name,expected,False)
    if len(rows)==len(expected): return
    prompts=p['utility'] if name.startswith('utility_') else q['candidates']['prompts']
    lookup={r['id']:r for r in prompts}; encodings={}; gen=p['config']['generation']
    for row in prompts:
        rendered=render_prompt(bundle.tokenizer,row['prompt'],None)
        tokens=bundle.tokenizer(rendered,return_tensors='pt',truncation=False)
        if tokens['input_ids'].shape[1]>gen['max_input_tokens']: raise ValueError('Input truncation prohibited')
        encodings[row['id']]={k:v.to(bundle.device) for k,v in tokens.items()}
    path=EVAL/'responses'/(name+'.jsonl'); path.parent.mkdir(parents=True,exist_ok=True)
    previous=rows[-1]['record_sha256'] if rows else '0'*64; start=time.time(); first=len(rows)
    with path.open('a',encoding='utf-8',newline='\n') as f:
        for index in range(first,len(expected)):
            spec=expected[index]; seed=spec['generation_seed']
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            encoding=encodings[spec['prompt_id']]
            kwargs={'max_new_tokens':gen['max_new_tokens'],'do_sample':not name.startswith('utility_'),
                    'pad_token_id':bundle.tokenizer.pad_token_id,'eos_token_id':bundle.tokenizer.eos_token_id}
            if kwargs['do_sample']: kwargs.update({k:gen[k] for k in ('temperature','top_p','top_k')})
            with torch.inference_mode(): tokens=bundle.model.generate(**encoding,**kwargs)
            ids=tokens[0,encoding['input_ids'].shape[1]:].detach().cpu().tolist()
            text=bundle.tokenizer.decode(ids,skip_special_tokens=True)
            eos=bundle.model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos]); eos.add(bundle.tokenizer.eos_token_id)
            truncated=len(ids)>=gen['max_new_tokens'] and (not ids or ids[-1] not in eos)
            task=evaluate_task_r1(text,lookup[spec['prompt_id']])
            passed=task['passed'] is True and not truncated
            row={**spec,'manifest_sha256':digest(EVAL/'PLAN.json'),'previous_sha256':previous,
                 'response':text,'response_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),
                 'completion_token_ids':ids,'completion_token_count':len(ids),'truncated':truncated,
                 'task_evaluation':task,'task_passed':passed,'utc_unix':time.time()}
            row['record_sha256']=sha_value(row); f.write(canonical(row)+'\n'); f.flush(); os.fsync(f.fileno())
            previous=row['record_sha256']
            if (index+1)%24==0 or index+1==len(expected):
                status('generate',bank=name,completed=index+1,total=len(expected),seconds=time.time()-start,
                       responses_per_second=(index+1-first)/max(time.time()-start,1e-9))
    bank(name,expected)


def logits(bundle,prompts):
    import numpy as np
    from llm_integrity.modeling import tokenize_prompts
    torch=setup(); values=[]
    for p in prompts:
        encoded=tokenize_prompts(bundle,[p['prompt']],512)
        with torch.inference_mode(): z=bundle.model(**encoded,use_cache=False).logits[0,-1].float()
        values.append(z.cpu().numpy())
    a=np.stack(values)
    if not np.isfinite(a).all(): raise ValueError('Nonfinite logits')
    return a


def outer_and_intact(p,q,banks):
    import numpy as np
    from llm_integrity.modeling import load_model
    from llm_integrity.paper_outputs import differentiable_next_token_logits
    from llm_integrity.paper_micro import estimate_next_token_jacobian_frobenius
    torch=setup(); bundle=load_model(p['config']['model'])
    try:
        total=sum(par.numel() for par in bundle.model.parameters())
        for row in q['candidates']['prompts']:
            path=EVAL/'outer_micro'/(row['id']+'.json')
            if path.exists():
                if read(path)['plan_sha256']!=digest(EVAL/'PLAN.json'): raise ValueError('Outer binding mismatch')
                continue
            for par in bundle.model.parameters(): par.requires_grad_(True)
            torch.cuda.reset_peak_memory_stats(); start=time.time()
            z=differentiable_next_token_logits(bundle,[row['prompt']],max_length=512)[0]
            result=estimate_next_token_jacobian_frobenius(z,bundle.model.named_parameters(),
                probes=q['outer_micro']['probes'],seed=q['outer_micro']['seed'])
            if result.parameter_count!=total or not np.isfinite(result.estimate): raise ValueError('Invalid full-parameter K8 audit')
            freeze_json(path,{'prompt_id':row['id'],'plan_sha256':digest(EVAL/'PLAN.json'),
                             **dataclasses.asdict(result),'complete_parameter_coverage':True,
                             'seconds':time.time()-start,'peak_bytes':torch.cuda.max_memory_allocated()})
            del z,result; cleanup(); status('outer_micro',prompt_id=row['id'])
        logpath=EVAL/'intact_logits.npy'
        if not logpath.exists(): np.save(logpath,logits(bundle,q['candidates']['prompts']),allow_pickle=False)
        freeze_json(EVAL/'INTACT_LOGITS.json',{'plan_sha256':digest(EVAL/'PLAN.json'),'sha256':digest(logpath)})
        # Fit must be technically valid before spending confirmation/attack response budget.
        for name in ['intact_fit','utility_intact']:
            generate_bank(bundle,p,q,name,banks[name])
    finally: bundle.close(); cleanup()


def confirmation_references(p,q,banks):
    from llm_integrity.modeling import load_model
    if not (EVAL/'FIT.json').exists(): raise ValueError('Fit must precede confirmation responses')
    bundle=load_model(p['config']['model'])
    try:
        for name in ('intact_confirmation','intact_null'): generate_bank(bundle,p,q,name,banks[name])
    finally: bundle.close(); cleanup()


def evaluate_variants(p,q,banks,split):
    import numpy as np
    from llm_integrity.stage2_contract import internal_metrics
    if split=='heldout' and not (EVAL/'CONFIRMATION_SELECTION.json').exists(): raise ValueError('Selection not frozen')
    attacks(p,split)
    intact_path=EVAL/'intact_logits.npy'
    if read(EVAL/'INTACT_LOGITS.json')['sha256']!=digest(intact_path): raise ValueError('Intact logits changed')
    a=np.load(intact_path,allow_pickle=False)
    for v in p['variants']:
        if v['split']!=split: continue
        endpoint=v['variant_id']; metricfile=EVAL/'proxies'/(endpoint+'.json')
        required=['utility_'+endpoint]+([] if split=='train' else [endpoint])
        done=all(len(bank(name,banks[name],False))==len(banks[name]) for name in required)
        if done and metricfile.exists(): continue
        status('load_endpoint',endpoint=endpoint)
        loaded=loaded_variant(p,v)
        try:
            realization=finite_payload(dataclasses.asdict(loaded.report))
            adapter_hashes={}
            if v['family']=='finetuning':
                folder=OUT/'adapters'/endpoint
                adapter_hashes={f.name:digest(f) for f in sorted(folder.iterdir()) if f.is_file()}
            freeze_json(EVAL/'realizations'/(endpoint+'.json'),{'plan_sha256':digest(EVAL/'PLAN.json'),
                        'manifest':v,'realization':realization,'adapter_hashes':adapter_hashes,
                        'note':'Gaussian changed_count from legacy executor is nominal, not measured bitwise changed fraction'})
            if not metricfile.exists():
                b=logits(loaded.bundle,q['candidates']['prompts'])
                records=[{'prompt_id':r['id'],'endpoint':endpoint,'family':v['family'],'strength':v['strength'],
                          'split':split,'attack_seed':v['seed'],**internal_metrics(a[i],b[i],top_k=10)}
                         for i,r in enumerate(q['candidates']['prompts'])]
                freeze_json(metricfile,{'plan_sha256':digest(EVAL/'PLAN.json'),'records':records})
            for name in required: generate_bank(loaded.bundle,p,q,name,banks[name])
        finally: loaded.close(); cleanup()


def feature_tools(q):
    from run_stage1_r2 import NewSemanticCache
    from llm_integrity.features import FeatureExtractor
    cache=NewSemanticCache({'semantic':q['semantic']},EVAL,q)
    return cache,FeatureExtractor(semantic_cache=cache)


def fit_features(p,q,banks):
    import numpy as np
    from llm_integrity.h8_precalibration import (build_h8_feature_schema,fit_family_balanced_scaler,
        global_continuous_exclusion_mask,h8_global_degenerate_bandwidth,h8_median_positive_pairwise_distance)
    rows=bank('intact_fit',banks['intact_fit']); cache,extractor=feature_tools(q)
    try:
        schema=build_h8_feature_schema(512); matrices={}
        for prompt in q['candidates']['prompts']:
            texts=[r['response'] for r in rows if r['prompt_id']==prompt['id']]
            matrices[prompt['id']]=extractor.transform(texts,[prompt]*len(texts))
        pooled=np.concatenate(list(matrices.values())); exclusion=global_continuous_exclusion_mask(pooled,schema)
        scalers={pid:fit_family_balanced_scaler(pid,x,pooled,schema,exclusion_mask=exclusion) for pid,x in matrices.items()}
        transformed={pid:scalers[pid].transform(x,schema) for pid,x in matrices.items()}
        global_sigma=h8_global_degenerate_bandwidth(transformed); bandwidths={}
        for pid,x in transformed.items():
            try: bandwidths[pid]={'value':h8_median_positive_pairwise_distance(x),'source':'prompt_fit'}
            except ValueError: bandwidths[pid]={'value':global_sigma['sigma'],'source':'global_fit_degenerate_fallback'}
        probe=list(dict.fromkeys(r['response'] for r in rows))
        if not np.array_equal(cache.transform(probe),cache.transform(probe[::-1])[::-1]): raise ValueError('Semantic order dependence')
        freeze_json(EVAL/'FIT.json',{'plan_sha256':digest(EVAL/'PLAN.json'),'fit_bank_sha256':digest(EVAL/'responses/intact_fit.jsonl'),
                    'schema_sha256':schema.sha256,'scalers':{pid:s.as_dict() for pid,s in scalers.items()},
                    'bandwidths':bandwidths,'global_sigma':global_sigma,'semantic_identity_sha256':cache.identity_hash,
                    'attack_responses_used':False})
    finally: cache.close()


def analyze(p,q,banks,split):
    from llm_integrity.h8_precalibration import build_h8_feature_schema,FamilyBalancedScaler
    frozen=read(EVAL/'FIT.json')
    if frozen['fit_bank_sha256']!=digest(EVAL/'responses/intact_fit.jsonl'): raise ValueError('Fit changed')
    cache,extractor=feature_tools(q); schema=build_h8_feature_schema(512)
    if frozen['schema_sha256']!=schema.sha256 or frozen['semantic_identity_sha256']!=cache.identity_hash: raise ValueError('Feature identity changed')
    scalers={pid:FamilyBalancedScaler.from_dict(s) for pid,s in frozen['scalers'].items()}
    reference='intact_'+('confirmation' if split=='null' else split)
    left=bank(reference,banks[reference]); targets=[{'variant_id':'intact_null','family':'intact','strength':'null','seed':0}] if split=='null' else [v for v in p['variants'] if v['split']==split]
    try:
        for v in targets:
            endpoint=v['variant_id']; right=bank(endpoint,banks[endpoint]); results=[]
            path=EVAL/'decisions'/(endpoint+'.json')
            binding={'plan_sha256':digest(EVAL/'PLAN.json'),'fit_sha256':digest(EVAL/'FIT.json'),
                     'reference_sha256':digest(EVAL/'responses'/(reference+'.jsonl')),
                     'attack_sha256':digest(EVAL/'responses'/(endpoint+'.jsonl'))}
            if path.exists():
                if any(read(path)[k]!=s for k,s in binding.items()): raise ValueError('Decision inputs changed')
                continue
            for prompt in q['candidates']['prompts']:
                pid=prompt['id']; a=[r for r in left if r['prompt_id']==pid]; b=[r for r in right if r['prompt_id']==pid]
                if len(a)!=8 or len(b)!=8: raise ValueError('Incomplete response group')
                texts=[r['response'] for r in a+b]; feat=extractor.transform(texts,[prompt]*16)
                transformed=scalers[pid].transform(feat,schema)
                seed=int(sha_value({'split':split,'endpoint':endpoint,'prompt_id':pid})[:15],16)
                if p['statistics']!={'permutations':999,'alpha':.05,'tie_atol':1e-12,'channels_separate':True}: raise ValueError('Unexpected statistical contract')
                tested=permutation_test(texts,feat[:,11:523],transformed,8,frozen['bandwidths'][pid]['value'],seed,
                                        permutations=p['statistics']['permutations'],alpha=p['statistics']['alpha'],
                                        tie_atol=p['statistics']['tie_atol'])
                results.append({'prompt_id':pid,'endpoint':endpoint,'split':split,'family':v['family'],'strength':v['strength'],
                                'attack_seed':v['seed'],'intact_correct':sum(r['task_passed'] for r in a),
                                'attack_correct':sum(r['task_passed'] for r in b),**tested})
            freeze_json(path,{**binding,'records':results})
            status('analyze',split=split,endpoint=endpoint)
    finally: cache.close()


def utility_results(p,banks,split):
    baseline=bank('utility_intact',banks['utility_intact']); values={}
    for v in p['variants']:
        if v['split']!=split: continue
        name='utility_'+v['variant_id']; other=bank(name,banks[name])
        values[v['variant_id']]=utility_qualification([r['task_passed'] for r in baseline],[r['task_passed'] for r in other])
    return values


def selection(p,q,banks):
    records=[r for v in p['variants'] if v['split']=='confirmation' for r in read(EVAL/'decisions'/(v['variant_id']+'.json'))['records']]
    by_prompt={r['id']:[d for d in records if d['prompt_id']==r['id']] for r in q['candidates']['prompts']}
    aliases=q['candidates']['aliases']; source_ids={a['source_id']:a['prompt_id'] for a in aliases if a['method']=='source'}
    utility=utility_results(p,banks,'confirmation'); results=[]
    for a in aliases:
        if a['method']=='source': continue
        candidate=by_prompt[a['prompt_id']]; source=by_prompt[source_ids[a['source_id']]]
        result=candidate_gate(candidate,source,utility,candidate[0]['intact_correct'],source[0]['intact_correct'],p['behavior_gate'])
        results.append({**a,**result})
    paths=list((EVAL/'decisions').glob('s2_confirmation_*.json'))
    freeze_json(EVAL/'CONFIRMATION_SELECTION.json',{'plan_sha256':digest(EVAL/'PLAN.json'),
                 'decision_hashes':{f.name:digest(f) for f in paths},'utility':utility,'candidates':results,
                 'heldout_observed':False,'criteria_tuned':False})
    status('CONFIRMATION_SELECTION_FROZEN',passed=sum(r['passed'] for r in results),candidates=len(results))


def heldout(p,q,banks):
    from llm_integrity.modeling import load_model
    selection_file=EVAL/'CONFIRMATION_SELECTION.json'
    selected=read(selection_file)
    if selected['plan_sha256']!=digest(EVAL/'PLAN.json'): raise ValueError('Selection binding changed')
    for f,s in selected['decision_hashes'].items():
        if digest(EVAL/'decisions'/f)!=s: raise ValueError('Confirmation decision changed')
    freeze_json(EVAL/'HELDOUT_ENTRY.json',{'selection_sha256':digest(selection_file),'plan_sha256':digest(EVAL/'PLAN.json')})
    bundle=load_model(p['config']['model'])
    try: generate_bank(bundle,p,q,'intact_heldout',banks['intact_heldout'])
    finally: bundle.close(); cleanup()
    evaluate_variants(p,q,banks,'heldout'); analyze(p,q,banks,'heldout')


def summarize(p,q,banks):
    import csv
    import numpy as np
    from llm_integrity.stage2_contract import PROXIES
    all_decisions=[]; proxies=[]
    for split in ('confirmation','heldout'):
        for v in p['variants']:
            if v['split']==split:
                all_decisions.extend(read(EVAL/'decisions'/(v['variant_id']+'.json'))['records'])
    for v in p['variants']: proxies.extend(read(EVAL/'proxies'/(v['variant_id']+'.json'))['records'])
    null=read(EVAL/'decisions/intact_null.json')['records']
    utility={split:utility_results(p,banks,split) for split in ('train','confirmation','heldout')}
    aliases=q['candidates']['aliases']; source={a['source_id']:a['prompt_id'] for a in aliases if a['method']=='source'}
    original_selected=read(EVAL/'CONFIRMATION_SELECTION.json')
    frozen_alias={(a['method'],a['source_id'],a['prompt_id']) for a in original_selected['candidates'] if a['passed']}
    grouped={split:{row['id']:[r for r in all_decisions if r['split']==split and r['prompt_id']==row['id']]
                    for row in q['candidates']['prompts']} for split in ('confirmation','heldout')}
    transfer=[]
    for a in aliases:
        if a['method']=='source': continue
        c=grouped['heldout'][a['prompt_id']]; b=grouped['heldout'][source[a['source_id']]]
        g=candidate_gate(c,b,utility['heldout'],c[0]['intact_correct'],b[0]['intact_correct'],p['behavior_gate'])
        transfer.append({**a,'confirmation_selected':(a['method'],a['source_id'],a['prompt_id']) in frozen_alias,
                         'heldout_audit_passed':g['passed'],'heldout_reasons':g['reasons'],
                         'used_to_reselect':False})
    correlations=[]
    for method in ('source','legacy','enhanced'):
        for family in ('gaussian_noise','finetuning','unstructured_pruning','structured_pruning','quantization'):
            strengths=('weak','medium','strong') if family in ('gaussian_noise','finetuning') else ('coverage',)
            for strength in strengths:
                qualified_ids={v['variant_id'] for v in p['variants'] if v['split']=='heldout' and v['family']==family and v['strength']==strength}
                all_utility_pass=all(utility['heldout'][v]['qualified'] for v in qualified_ids)
                source_values=[]
                for s in p['prompts']:
                    ids=sorted({a['prompt_id'] for a in aliases if a['method']==method and a['source_id']==s['id']})
                    if not ids: continue
                    xs={proxy:[] for proxy in PROXIES}; ys={ch:[] for ch in ('raw','h8')}; rates={ch:[] for ch in ('raw','h8')}
                    for pid in ids:
                        train=[r for r in proxies if r['split']=='train' and r['family']==family and r['prompt_id']==pid]
                        if len(train)!=3: raise ValueError('Incomplete TRAIN proxy bank')
                        for proxy in PROXIES:
                            aggregate=np.mean if proxy=='top1_flip' else np.median
                            xs[proxy].append(float(aggregate([r[proxy] for r in train])))
                        test=[r for r in grouped['heldout'][pid] if r['endpoint'] in qualified_ids]
                        if len(test)!=len(qualified_ids): raise ValueError('Incomplete heldout group')
                        for ch in ys:
                            if any(r[ch].get('statistic') is None or r[ch]['detected'] is None for r in test): raise ValueError('Unevaluable response statistics')
                            ys[ch].append(float(np.mean([r[ch]['statistic'] for r in test])))
                            rates[ch].append(float(np.mean([r[ch]['detected'] for r in test])))
                    source_values.append({'source_id':s['id'],'candidate_count':len(ids),
                                          'x':{k:float(np.mean(v)) for k,v in xs.items()},
                                          'y':{k:float(np.mean(v)) for k,v in ys.items()},
                                          'rates':{k:float(np.mean(v)) for k,v in rates.items()}})
                for proxy in PROXIES:
                    x=[r['x'][proxy] for r in source_values]
                    for ch in ('raw','h8'):
                        for target in ('distance','detected_fraction'):
                            y=[r['y' if target=='distance' else 'rates'][ch] for r in source_values]
                            correlations.append({'method':method,'family':family,'strength':strength,'proxy':proxy,
                                                 'target_channel':ch,'target':target,'all_endpoints_utility_qualified':all_utility_pass,
                                                 **rank_correlation(x,y),'source_values':source_values})
    methods=[]
    for method in p['methods']:
        items=[r for r in transfer if r['method']==method]; chosen=[r for r in items if r['confirmation_selected']]
        methods.append({'method':method,'candidate_aliases':len(items),'source_instances_with_candidates':len({r['source_id'] for r in items}),
                        'confirmation_selected':len(chosen),'heldout_passing_frozen_selection':sum(r['heldout_audit_passed'] for r in chosen),
                        'heldout_passing_all_capped_candidates':sum(r['heldout_audit_passed'] for r in items),
                        'search_seconds':sum(r['seconds'] for r in q['candidates']['search_audit'] if r['method']==method)})
    null_summary={ch:{'detections':sum(r[ch]['detected'] is True for r in null),'units':len(null),
                      'unevaluable':sum(r[ch]['detected'] is None for r in null),
                      'formal_fpr_certified':False} for ch in ('raw','h8')}
    file_hashes={str(f.relative_to(EVAL)):digest(f) for folder in ('responses','decisions','proxies','outer_micro','realizations') for f in sorted((EVAL/folder).glob('*')) if f.is_file()}
    summary={'plan_sha256':digest(EVAL/'PLAN.json'),'evidence_hashes':file_hashes,'scope':'LIMITED_SIX_SOURCE_PILOT_ONLY',
             'stage2_complete':False,'six_independent_proxy_searches_run':False,'methods':methods,'utility':utility,
             'null':null_summary,'transfer':transfer,'correlations':correlations,
             'proxy_winner':'NOT_ESTABLISHED: descriptive same-candidate correlations, not six-arm causal comparison',
             'limitations':q['known_limits'],'response_rows':sum(len(bank(k,v)) for k,v in banks.items())}
    freeze_json(EVAL/'SUMMARY.json',summary)
    with (EVAL/'response_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['prompt_id','endpoint','split','family','strength','attack_seed','utility_qualified','intact_correct','attack_correct',
                'raw_distance','raw_p','raw_detected','mmd2','h8_p','h8_detected']
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in all_decisions:
            writer.writerow({**{k:r[k] for k in fields[:6]},'utility_qualified':utility[r['split']][r['endpoint']]['qualified'],
                             'intact_correct':r['intact_correct'],'attack_correct':r['attack_correct'],
                             'raw_distance':r['raw']['statistic'],'raw_p':r['raw']['p_value'],'raw_detected':r['raw']['detected'],
                             'mmd2':r['h8']['statistic'],'h8_p':r['h8']['p_value'],'h8_detected':r['h8']['detected']})
    with (EVAL/'proxy_correlations.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['method','family','strength','proxy','target_channel','target','all_endpoints_utility_qualified','rho','n','status']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(correlations)
    text=['# Stage2 六源提示词限域试跑结果','',
          '本报告不代表完整 Stage2 完成。只覆盖逻辑/指令两类任务且有重复模板；六代理是在同候选上诊断，并非六套独立优化。','',
          '| 参数策略 | 候选版本（含来源别名） | 行为确认通过 | 已确认候选的独立测试通过 | 搜索秒数 |',
          '|---|---:|---:|---:|---:|']
    for r in methods:
        text.append(f"| {r['method']} | {r['candidate_aliases']} | {r['confirmation_selected']} | {r['heldout_passing_frozen_selection']} | {r['search_seconds']:.1f} |")
    text += ['',f"本次累计响应 {summary['response_rows']} 条（包括独立任务效用检查，不含搜索内任务守卫）。",
             '', '## 解释边界','',
             '- 六代理结果见 proxy_correlations.csv：按攻击族/强度分层，以来源提示词聚合，不把种子与候选当独立样本。',
             '- 相关性只用于定位机制；常量代理记为不可估计，不强行排序，不宣称显著优胜。',
             '- 只有效用合格的攻击才能支持“细微改动检测”结论；效用不合格的扰动单独标记。',
             '- Gaussian 与 LoRA 每个强度分别检查三个种子；未用强攻击补偿弱攻击。',
             '- null 检查只有每提示词一次 8 对 8，不证明正式 5% FPR。',
             '- 原参数策略与增强参数策略的计算预算不同；本轮不把任何增益归因于单个参数，也不声称等预算优势。',
             '- 完整 Stage2 仍需纯行为搜索、内部权重消融及足够任务覆盖；不得依据本试跑直接宣称论文创新成立。','']
    (EVAL/'试跑结果.md').write_text('\n'.join(text),encoding='utf-8')
    status('LIMITED_PILOT_REPORT_COMPLETE',methods=methods,response_rows=summary['response_rows'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['freeze','confirmation','heldout','report'])
    args=parser.parse_args(); EVAL.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (OUT/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.phase=='freeze': freeze_evaluation(); return
            p,q,banks=context()
            if args.phase=='confirmation':
                outer_and_intact(p,q,banks); fit_features(p,q,banks); confirmation_references(p,q,banks); analyze(p,q,banks,'null')
                evaluate_variants(p,q,banks,'train'); evaluate_variants(p,q,banks,'confirmation')
                analyze(p,q,banks,'confirmation'); selection(p,q,banks)
            elif args.phase=='heldout': heldout(p,q,banks)
            elif args.phase=='report': summarize(p,q,banks)
            status(args.phase.upper()+'_COMPLETE')
        except Exception as exc:
            status('FAILED',error_type=type(exc).__name__,error=str(exc)); raise


if __name__=='__main__': main()
