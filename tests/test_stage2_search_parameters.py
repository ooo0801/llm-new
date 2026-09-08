from types import SimpleNamespace
import pytest
torch=pytest.importorskip("torch")
from llm_integrity.inner_micro_proxy import discover_micro_blocks
from llm_integrity.stage2_contract import BLOCK_TYPES
from llm_integrity.discrete_joint_inner_optimizer import DiscreteJointInnerOptimizer


def test_seven_modules_discovered_and_legacy_default_preserved():
    weights={}
    for layer in range(2):
        for kind in BLOCK_TYPES:
            location="self_attn" if kind in {"q_proj","k_proj","v_proj","o_proj"} else "mlp"
            weights[f"model.layers.{layer}.{location}.{kind}.weight"]=torch.nn.Parameter(torch.ones(2,2))
    model=SimpleNamespace(named_parameters=lambda:weights.items())
    assert len(discover_micro_blocks(model))==6
    assert len(discover_micro_blocks(model,BLOCK_TYPES))==14
    with pytest.raises(ValueError):
        discover_micro_blocks(model,[])


def test_strict_guard_rejects_fence_wrong_type_and_truncation(monkeypatch):
    import llm_integrity.discrete_joint_inner_optimizer as module
    opt=DiscreteJointInnerOptimizer.__new__(DiscreteJointInnerOptimizer)
    opt.require_task_preservation=True
    opt.task_validation_mode="strict_r1"
    opt.reference=None
    opt.task_generation={}
    valid='{"name":"test","level":1,"enabled":true}'
    replies=[{"text":valid,"token_count":20,"truncated":False},
             {"text":"```json\n"+valid+"\n```","token_count":24,"truncated":False},
             {"text":valid,"token_count":128,"truncated":True},
             {"text":'{"name":"x","level":true,"enabled":true}',"token_count":20,"truncated":False}]
    monkeypatch.setattr(module,"generate_texts",lambda *args,**kwargs:replies)
    result=opt._task_validation({"evaluator":"json"},["p"]*4)
    assert [r["task_passed"] for r in result]==[True,False,False,False]


def test_strict_guard_does_not_promote_keyword_proxy_to_success(monkeypatch):
    import llm_integrity.discrete_joint_inner_optimizer as module
    opt=DiscreteJointInnerOptimizer.__new__(DiscreteJointInnerOptimizer)
    opt.require_task_preservation=True; opt.task_validation_mode="strict_r1"
    opt.reference=None; opt.task_generation={}
    monkeypatch.setattr(module,"generate_texts",lambda *args,**kwargs:[{"text":"不能这样做","truncated":False,"token_count":4}])
    assert not opt._task_validation({"evaluator":"refusal","expected_contains":["不能"]},["p"])[0]["task_passed"]
