import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_prompt_budget import decision, validate_rows, response_seed
from run_resf_small import identity
from llm_integrity.resf_token import calibration


def fixture():
    p=dict(seed=319000000,alpha=.05,e_fraction=.2)
    ref=dict(token_ids=[1,2],probabilities=[[.5,.5]])
    mc=calibration(ref['probabilities'],looks=(10,25,50,100),simulations=10000,seed=12)
    return p,ref,mc


def test_budget_does_not_look_at_later_tokens():
    p,r,mc=fixture()
    samples=[1,2]*5+[999]*90
    assert not decision(samples,r,mc,20,10,p)['alarm']
    long=decision(samples,r,mc,20,25,p)
    assert long['rule']=='E' and long['query']==11


def test_sample_prefix_has_exact_allocation_and_terminal_look():
    p,r,mc=fixture()
    d=decision([1,2]*50,r,mc,5,25,p)
    assert not d['alarm']
    assert d['alpha_p_per_look']==pytest.approx(.008)
    assert [t['query'] for t in d['traces']]==[25]


def test_positive_control_inside_support():
    p,r,mc=fixture()
    d=decision([1]*100,r,mc,20,100,p)
    assert d['alarm'] and d['rule']=='P'


def test_resume_rejects_duplicate_or_wrong_binding():
    p,_,_=fixture();prompts=[dict(id='a')]
    row=dict(prompt_id='a',response_index=0,endpoint='normal_panel_00',split='test',
             plan_sha256=identity(p),seed=response_seed(p,'test',0,0,0),token_id=1)
    assert len(validate_rows([row],p,prompts,'test','normal_panel_00',0))==1
    with pytest.raises(RuntimeError):validate_rows([row,row],p,prompts,'test','normal_panel_00',0)
    with pytest.raises(RuntimeError):validate_rows([dict(row,seed=row['seed']+1)],p,prompts,'test','normal_panel_00',0)


def test_response_seed_partitions_are_disjoint():
    p,_,_=fixture()
    dev={response_seed(p,'development',e,i,j) for e in range(6) for i in range(99) for j in (0,99)}
    test={response_seed(p,'test',e,i,j) for e in range(60) for i in range(20) for j in (0,99)}
    assert not dev & test
    assert len(dev)==6*99*2 and len(test)==60*20*2
