import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage2_eval_contract import collect_candidates,schedules,utility_qualification,candidate_gate,rank_correlation
from stage2_prepare import design


def fixture():
    sources=[{'id':str(i),'prompt':f'original{i}','expected_answer':'A','evaluator':'exact','category':'logic'} for i in range(6)]
    p={'methods':['legacy','enhanced'],'prompts':sources,'maximum_candidates_per_source_per_method':2,
       'maximum_unique_response_prompts':30,'variants':design(),'response_repetitions':8,
       'intact_roles':['fit','confirmation','heldout','null'],'utility':[{'id':f'u{i}'} for i in range(8)],
       'maximum_behavior_response_rows':11040,'maximum_utility_generations':464}
    results=[]
    for m in p['methods']:
        for s in sources:
            cs=[{'decoded_prompt':f"{s['id']}_{m}_{j}",'constraints_passed':True,'task_validation':{'task_passed':True},
                 'proxy_gain':j+1.,'ppl_ratio':1.4,'edit_ratio':.1,'training_nondegraded_families':5} for j in range(3)]
            results.append({'method':m,'source':s,'effective_parameters':{'minimum_nondegraded_families':3 if m=='legacy' else 5},
                            'accepted':True,'optimized_prompt':cs[0]['decoded_prompt'],'history':[{'round':1,'reranked_candidates':cs}],
                            'rounds_completed':1,'seconds':2.,'failure':None})
    return p,results


def test_candidate_budget_final_and_alternative():
    p,r=fixture(); c=collect_candidates(p,r)
    assert len(c['prompts'])==30
    aliases=[a for a in c['aliases'] if a['method']=='legacy' and a['source_id']=='0']
    assert len(aliases)==2 and aliases[0]['final_committed']
    assert aliases[1]['candidate_index']==2
    banks=schedules(p,c)
    assert sum(len(v) for k,v in banks.items() if not k.startswith('utility_'))==11040
    assert sum(len(v) for k,v in banks.items() if k.startswith('utility_'))==464
    seeds=[r['generation_seed'] for k,v in banks.items() if not k.startswith('utility_') for r in v]
    assert len(seeds)==len(set(seeds))


def test_failures_not_zero_candidates():
    p,r=fixture(); r[0]['failure']={'error':'OOM'}
    with pytest.raises(ValueError,match='Technical'): collect_candidates(p,r)
    with pytest.raises(ValueError,match='Incomplete'): collect_candidates(p,r[:-1])


def test_no_behavior_eligibility_leak_and_zero_candidates():
    p,r=fixture()
    for result in r:
        for c in result['history'][0]['reranked_candidates']: c['task_validation']['task_passed']=False
    c=collect_candidates(p,r)
    assert len(c['prompts'])==6 and len(c['aliases'])==6
    assert all(a['retained']==0 for a in c['search_audit'])


def test_same_prompt_dedup_retains_provenance():
    p,r=fixture()
    for result in r:
        result['history'][0]['reranked_candidates'][0]['decoded_prompt']='shared'
        result['optimized_prompt']='shared'
    c=collect_candidates(p,r)
    assert len([v for v in c['prompts'] if v['prompt']=='shared'])==1
    assert len([a for a in c['aliases'] if a.get('final_committed')])==12


def test_invalid_utility_is_not_sensitive_success():
    assert utility_qualification([True]*8,[True]*7+[False])['qualified']
    assert not utility_qualification([True]*8,[True]*6+[False]*2)['qualified']
    assert not utility_qualification([True]*5+[False]*3,[True]*8)['qualified']
    assert not utility_qualification([True]*8,[None]*8)['qualified']


def gate_fixture():
    rows=[{'endpoint':v['variant_id'],'attack_seed':v['seed'],'family':v['family'],'strength':v['strength'],
           'raw':{'detected':True},'h8':{'statistic':.1}} for v in design() if v['split']=='confirmation']
    utility={r['endpoint']:{'qualified':True} for r in rows}
    gate={'intact_minimum_correct_of_8':6,'max_intact_drop_from_source':.125,
          'gaussian_lora_each_strength_minimum_seeds_raw_detected':2,'coverage_non_degradation_absolute_mmd_tolerance':.01}
    return rows,copy.deepcopy(rows),utility,gate


def test_each_weak_family_must_stand_alone():
    c,b,u,g=gate_fixture()
    assert candidate_gate(c,b,u,8,8,g)['passed']
    weak=[r for r in c if r['family']=='gaussian_noise' and r['strength']=='weak']
    weak[0]['raw']['detected']=False; weak[1]['h8']['statistic']=None
    assert not candidate_gate(c,b,u,8,8,g)['passed']
    with pytest.raises(ValueError,match='Missing'): candidate_gate(c[1:],b,u,8,8,g)


def test_seed_utility_and_task_gate():
    c,b,u,g=gate_fixture()
    assert not candidate_gate(c,b,u,6,8,g)['passed']
    for r in c:
        if r['family']=='finetuning': u[r['endpoint']]['qualified']=False
    assert not candidate_gate(c,b,u,8,8,g)['passed']


def test_constant_proxy_unrankable():
    assert rank_correlation([0,0,0],[1,2,3])['rho'] is None
    assert rank_correlation([1,2,3],[3,2,1])['rho']==-1
    assert rank_correlation([1,2,3],[3,2,1])['p_value'] is None


def test_continuation_requires_all_requested_rounds(tmp_path,monkeypatch):
    import json
    import stage2_finish_pilot as runner
    monkeypatch.setattr(runner,'OUT',tmp_path)
    (tmp_path/'DESIGN.json').write_text('{}')
    from llm_integrity.stage1_r1 import digest
    p={'methods':['legacy'],'prompts':[{'id':'a'}]}
    path=tmp_path/'search/legacy/a.json'; path.parent.mkdir(parents=True)
    row={'failure':None,'design_sha256':digest(tmp_path/'DESIGN.json'),'rounds_completed':3,'rounds_requested':3}
    path.write_text(json.dumps(row))
    assert runner.validate_search_results(p)==[path]
    row['rounds_completed']=2; path.write_text(json.dumps(row))
    with pytest.raises(RuntimeError,match='incomplete'): runner.validate_search_results(p)


def test_evaluation_import_has_no_generation_side_effects():
    import stage2_pilot_evaluate
    assert stage2_pilot_evaluate.SEMANTIC['max_seq_length']==512


def test_cpu_full_statistical_pipeline_constant_response(tmp_path,monkeypatch):
    """Synthetic unit test only: real R1 scaler/permutation/report, no model generation."""
    import json
    import numpy as np
    import stage2_pilot_evaluate as e
    from llm_integrity.features import FeatureExtractor
    from llm_integrity.stage1_r2 import freeze_json
    p,results=fixture(); p['prompts']=p['prompts'][:2]
    p['prompts'][1]['expected_answer']='B'
    results=[r for r in results if r['source']['id'] in ('0','1')]
    for r in results: r['history']=[]; r['accepted']=False
    candidates=collect_candidates(p,results); plans=schedules(p,candidates)
    _,_,_,gate=gate_fixture(); p['behavior_gate']=gate
    p['statistics']={'permutations':999,'alpha':.05,'tie_atol':1e-12,'channels_separate':True}
    q={'candidates':candidates,'known_limits':['synthetic fixture only']}
    monkeypatch.setattr(e,'EVAL',tmp_path)
    (tmp_path/'PLAN.json').write_text('{}',encoding='utf-8')
    (tmp_path/'responses').mkdir(); (tmp_path/'outer_micro').mkdir()
    first_pid=candidates['prompts'][0]['id']
    def fake_bank(name,expected,complete=True):
        return [{**r,'response':'A' if r['prompt_id']==first_pid and not (name=='intact_fit' and r['response_index']%2) else 'B',
                 'task_passed':True} for r in expected]
    monkeypatch.setattr(e,'bank',fake_bank)
    for name,rows in plans.items():
        (tmp_path/'responses'/(name+'.jsonl')).write_text(json.dumps(fake_bank(name,rows)),encoding='utf-8')
    class Cache:
        identity_hash='fake_identity'; manifest={'identity':{'name':'TEST_ONLY'}}
        def transform(self,texts):
            result=np.zeros((len(texts),512))
            for i,t in enumerate(texts): result[i,0 if t=='A' else 1]=1
            return result
        def close(self): pass
    def feature_tools(_):
        c=Cache(); return c,FeatureExtractor(semantic_cache=c)
    monkeypatch.setattr(e,'feature_tools',feature_tools)
    e.fit_features(p,q,plans); e.analyze(p,q,plans,'null')
    for split in ('confirmation','heldout'): e.analyze(p,q,plans,split)
    e.selection(p,q,plans)
    assert not e.read(tmp_path/'CONFIRMATION_SELECTION.json')['candidates']
    from llm_integrity.stage2_contract import PROXIES
    for v in p['variants']:
        records=[{'prompt_id':r['id'],'endpoint':v['variant_id'],'family':v['family'],
                  'strength':v['strength'],'split':v['split'],'attack_seed':v['seed'],**{k:0. for k in PROXIES}}
                 for r in candidates['prompts']]
        freeze_json(tmp_path/'proxies'/(v['variant_id']+'.json'),{'records':records})
    e.summarize(p,q,plans)
    summary=e.read(tmp_path/'SUMMARY.json')
    assert summary['stage2_complete'] is False
    assert all(r['rho'] is None for r in summary['correlations'])
    assert summary['null']['h8']['detections']==0 and summary['null']['raw']['detections']==0
    assert all(r['h8']['p_value']==1 for r in e.read(tmp_path/'decisions/intact_null.json')['records'])
