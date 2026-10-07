import importlib
import json
from pathlib import Path
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('run_scaling_matrix')


def test_legacy_and_jsonlines_read_identical(module,tmp_path):
    records=[dict(prompt_id='p',response_index=0,endpoint='n',seed=284000000,token_id=12)]
    path=tmp_path/'response.jsonl'
    path.write_text(json.dumps(records,indent=2),encoding='utf-8')
    assert module.load_responses(path)==records
    path.write_text(json.dumps(records[0])+'\n',encoding='utf-8')
    assert module.load_responses(path)==records
    plan=dict(prompts=[dict(id='p')],responses_per_prompt=100)
    assert module.check_responses(records,plan,'n',0)=={('p',0)}


@pytest.mark.parametrize('change',[{'seed':12},{'endpoint':'attack'},{'response_index':100},{'prompt_id':'other'},{'token_id':-1}])
def test_resume_rejects_misbound_record(module,change):
    row=dict(prompt_id='p',response_index=0,endpoint='n',seed=284000000,token_id=12)
    row.update(change)
    with pytest.raises(ValueError):
        module.check_responses([row],dict(prompts=[dict(id='p')],responses_per_prompt=100),'n',0)


def test_resume_rejects_duplicate(module):
    row=dict(prompt_id='p',response_index=0,endpoint='n',seed=284000000,token_id=12)
    with pytest.raises(ValueError):
        module.check_responses([row,row],dict(prompts=[dict(id='p')],responses_per_prompt=100),'n',0)
