"""Pure, pre-response rules for the bounded six-source Stage2 evaluation."""
import hashlib
import math
from collections import defaultdict

from build_discrete_candidate_portfolio import candidate_is_eligible
from llm_integrity.stage1_r1 import canonical


def sha_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def collect_candidates(design, results):
    """Final committed version first, then best eligible alternative; never inspect behavior."""
    expected={(m,p['id']) for m in design['methods'] for p in design['prompts']}
    if len(results)!=len(expected) or {(r['method'],r['source']['id']) for r in results}!=expected:
        raise ValueError('Incomplete or duplicate search results')
    union={}; aliases=[]; audit=[]
    def add(text, source, method, **provenance):
        key='p_'+hashlib.sha256(text.encode()).hexdigest()[:20]
        row={**source,'id':key,'prompt':text}
        if key in union:
            old=union[key]
            if any(old.get(k)!=row.get(k) for k in ('prompt','evaluator','expected_answer','category')):
                raise ValueError('Identical prompt has incompatible task definitions')
        else:
            union[key]=row
        aliases.append({'prompt_id':key,'source_id':source['id'],'method':method,**provenance})
    for source in design['prompts']:
        add(source['prompt'],source,'source')
    for result in results:
        if result.get('failure'):
            raise ValueError('Technical search failure cannot be treated as zero candidates')
        method=result['method']; source=result['source']; eligible={}; inspected=0
        gate=result['effective_parameters']['minimum_nondegraded_families']
        for h in result.get('history',[]):
            if 'round' not in h: continue
            for index,c in enumerate(h.get('reranked_candidates',[])):
                inspected+=1
                if not candidate_is_eligible(c,gate): continue
                text=c['decoded_prompt']  # Exact text, no whitespace canonicalization for response caching.
                if text==source['prompt'] or not 0<c['edit_ratio']<=.25 or c['ppl_ratio']>2: continue
                rank=(-int(c['training_nondegraded_families']),-float(c['proxy_gain']),float(c['ppl_ratio']),h['round'],index)
                if text not in eligible or rank<eligible[text][0]:
                    eligible[text]=(rank,c,h['round'],index)
        chosen=[]
        final=result.get('optimized_prompt')
        if result.get('accepted') and final in eligible: chosen.append(final)
        chosen.extend(t for t in sorted(eligible,key=lambda t:eligible[t][0]) if t not in chosen)
        chosen=chosen[:design['maximum_candidates_per_source_per_method']]
        for t in chosen:
            _,c,r,i=eligible[t]
            add(t,source,method,round=r,candidate_index=i,final_committed=t==final,
                inner_proxy_gain=c['proxy_gain'],inner_family_count=c['training_nondegraded_families'])
        audit.append({'method':method,'source_id':source['id'],'inspected':inspected,
                      'eligible_unique':len(eligible),'retained':len(chosen),
                      'rounds_completed':result.get('rounds_completed'),'seconds':result['seconds']})
    if len(union)>design['maximum_unique_response_prompts']: raise ValueError('Candidate budget exceeded')
    return {'prompts':list(union.values()),'aliases':aliases,'search_audit':audit,
            'correlation_population':'Originals plus capped internal-filter survivors; not the full HotFlip proposal population'}


def schedules(design, candidates):
    banks={}; seed=400000000
    endpoints=[('intact_'+r,r) for r in design['intact_roles']]
    endpoints += [(v['variant_id'],v['split']) for v in design['variants'] if v['split']!='train']
    for endpoint,role in endpoints:
        rows=[]
        for i in range(design['response_repetitions']):
            for p in candidates['prompts']:
                rows.append({'endpoint':endpoint,'role':role,'prompt_id':p['id'],
                             'response_index':i,'generation_seed':seed})
                seed+=1
        banks[endpoint]=rows
    for endpoint in ['intact']+[v['variant_id'] for v in design['variants']]:
        banks['utility_'+endpoint]=[{'endpoint':endpoint,'role':'utility','prompt_id':p['id'],
                                     'response_index':0,'generation_seed':500000000+j}
                                    for j,p in enumerate(design['utility'])]
    n=sum(len(rows) for k,rows in banks.items() if not k.startswith('utility_'))
    u=sum(len(rows) for k,rows in banks.items() if k.startswith('utility_'))
    if n>design['maximum_behavior_response_rows'] or u>design['maximum_utility_generations']:
        raise ValueError('Sampling budget exceeded')
    return banks


def utility_qualification(intact, attacked, minimum_intact=6, max_drop=.15):
    if len(intact)!=8 or len(attacked)!=8: raise ValueError('Incomplete utility set')
    if any(x is None for x in intact+attacked):
        return {'status':'unevaluable','qualified':False,'drop':None}
    correct=[i for i,b in enumerate(intact) if b]
    drop=sum(not attacked[i] for i in correct)/len(correct) if correct else None
    return {'status':'evaluable' if len(correct)>=minimum_intact else 'insufficient_intact_correct',
            'qualified':len(correct)>=minimum_intact and drop<=max_drop,
            'intact_correct':len(correct),'attack_correct':sum(attacked),'drop':drop}


def candidate_gate(candidate_rows, source_rows, utility, intact_correct, source_correct, gate):
    """Confirmation-only rule: every strength must stand on its own, no missing-as-pass."""
    reasons=[]; by_group=defaultdict(list)
    for r in candidate_rows: by_group[(r['family'],r['strength'])].append(r)
    if intact_correct<gate['intact_minimum_correct_of_8']: reasons.append('intact_task')
    if (source_correct-intact_correct)/8>gate['max_intact_drop_from_source']: reasons.append('task_drop')
    evidence={}
    for family in ('gaussian_noise','finetuning'):
        for strength in ('weak','medium','strong'):
            rows=by_group[(family,strength)]
            if len(rows)!=3 or len({r['attack_seed'] for r in rows})!=3: raise ValueError('Missing or duplicate seed cell')
            good=sum(utility[r['endpoint']]['qualified'] and r['raw']['detected'] is True
                     and r['h8'].get('statistic') is not None and r['h8']['statistic']>0 for r in rows)
            evidence[family+'/'+strength]=good
            if good<gate['gaussian_lora_each_strength_minimum_seeds_raw_detected']:
                reasons.append(family+'/'+strength)
    baseline={r['endpoint']:r for r in source_rows}
    for family in ('unstructured_pruning','structured_pruning','quantization'):
        rows=by_group[(family,'coverage')]
        if len(rows)!=1: raise ValueError('Missing coverage endpoint')
        r=rows[0]; b=baseline[r['endpoint']]
        x,y=r['h8'].get('statistic'),b['h8'].get('statistic')
        if not utility[r['endpoint']]['qualified'] or x is None or y is None or x<y-gate['coverage_non_degradation_absolute_mmd_tolerance']:
            reasons.append(family+'/coverage')
    return {'passed':not reasons,'reasons':reasons,'qualifying_seed_counts':evidence}


def rank_correlation(x,y):
    import numpy as np
    from scipy.stats import spearmanr
    a,b=np.asarray(x,float),np.asarray(y,float)
    if len(a)!=len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid correlation input')
    if len(a)<3 or len(set(a))<2 or len(set(b))<2:
        return {'rho':None,'n':len(a),'status':'undefined_small_or_constant','p_value':None}
    return {'rho':float(spearmanr(a,b).statistic),'n':len(a),'status':'descriptive_only',
            'p_value':None,'reason':'Source/template dependence; no iid significance claim'}
