import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_paired_prompt_budget import prepare,expected_seed,check_rows
from run_resf_small import save,read,identity


def test_mapping_fallback_and_shared_subsets(tmp_path):
    old=tmp_path/'old';root=tmp_path/'new';root.mkdir()
    sources=[dict(id=f'pool_{i:02d}',prompt=f'original {i}',category='logic') for i in range(20)]
    proxies=['js','topk_continuous','raw_logit_l2']
    save(old/'PLAN.json',dict(sources=sources,proxies=proxies,sizes=[1,2,5,10,20]))
    for proxy in proxies:
        for i,s in enumerate(sources):
            save(old/'search'/f"{proxy}_{s['id']}.json",dict(source=s,accepted=i!=0,optimized_prompt=f'{proxy} edit {i}',failure=None))
    p=prepare(root,old);subsets=read(root/'SUBSETS.json')
    assert p['queries']==[25,50,100]
    assert all(len(g)==20 for g in p['groups'].values())
    assert all(p['groups'][proxy]['pool_00']=='pool_00' for proxy in proxies)
    for m,panels in subsets.items():
        assert len(panels)==(20 if m=='1' else 1 if m=='20' else 100)
        assert len({tuple(s) for s in panels})==len(panels)
        for sources in panels:
            assert len(sources)==len(set(sources))==int(m)
            for group in p['groups'].values():assert all(s in group for s in sources)


def test_reuse_preserves_actual_seed_and_token():
    p=dict(seed=320000000,prompts=[dict(id='a')]);key=('a',0)
    old=dict(seed=420000000,token_id=123)
    row=dict(prompt_id='a',response_index=0,endpoint='normal_panel_00',split='test',
             plan_sha256=identity(p),seed=old['seed'],token_id=old['token_id'],origin='reused')
    assert check_rows([row],p,'test','normal_panel_00',0,{key:old})=={key}
    with pytest.raises(AssertionError):check_rows([dict(row,token_id=124)],p,'test','normal_panel_00',0,{key:old})
    with pytest.raises(AssertionError):check_rows([row,row],p,'test','normal_panel_00',0,{key:old})


def test_fresh_seed_range_does_not_overlap_previous_experiment():
    p=dict(seed=320000000)
    seeds={expected_seed(p,'test',e,i,j) for e in range(60) for i in range(56) for j in (0,99)}
    assert min(seeds)>426000000
    assert len(seeds)==60*56*2
