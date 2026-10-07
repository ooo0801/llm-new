import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_generation_ablation import AblationOptimizer, arm_parameters


def test_arm_objectives_and_gates():
    assert arm_parameters('micro_only',3)==dict(micro_weight=3,macro_weight=0,minimum_nondegraded_families=0)
    assert arm_parameters('macro_only',3)==dict(micro_weight=0,macro_weight=1,minimum_nondegraded_families=5)
    assert arm_parameters('joint_js',3)==dict(micro_weight=3,macro_weight=1,minimum_nondegraded_families=5)


def test_disabled_objectives_never_call_parent():
    obj=AblationOptimizer.__new__(AblationOptimizer)
    obj.micro_weight=0;obj.macro_weight=0;obj.minimum_nondegraded_families=0
    x=torch.ones(1,2,3)
    with patch('run_generation_ablation.DiscreteJointInnerOptimizer._micro_gradient',side_effect=AssertionError('micro called')), \
         patch('run_generation_ablation.DiscreteJointInnerOptimizer._macro_gradient',side_effect=AssertionError('macro called')):
        mg,ms,_,_=obj._micro_gradient(current_embeddings=x)
        ag,ass,trace,samples,anchors=obj._macro_gradient(current_embeddings=x)
    assert torch.count_nonzero(mg)==torch.count_nonzero(ag)==0
    assert ms==ass==0 and not trace and not samples and not anchors


def test_micro_rerank_accepts_gain_without_macro_information():
    obj=AblationOptimizer.__new__(AblationOptimizer)
    obj.micro_weight=3.;obj.macro_weight=0.;obj.minimum_nondegraded_families=0
    obj.rerank_candidates=8;obj.probes=8;obj.reference=None;obj.micro_scales={'q_proj':2.}
    obj.ppl_ratio_limit=1e12;obj.max_edit_ratio=.5
    obj._embed_user=lambda ids:torch.ones(1,1,2)
    obj._compose=lambda *args:(torch.ones(1,1,2),torch.ones(1,1),None)
    candidate=dict(token_ids=[1],ppl=1.,ppl_ratio=1.,edit_ratio=.1)
    with patch('run_generation_ablation.select_position_diverse_candidates',return_value=[candidate]), \
         patch('run_generation_ablation.score_block_micro_proxy',return_value=SimpleNamespace(raw_micro_score=4.)), \
         patch('llm_integrity.discrete_joint_inner_optimizer.load_manifest_variant',side_effect=AssertionError('macro model loaded')):
        result,baseline=obj._rerank_candidates(candidates=[candidate],samples=[],prefix_ids=[],suffix_ids=[],
            block=None,block_type='q_proj',probe_seed=1,baseline_micro_normalized=1.)
    assert result[0]['constraints_passed'] and result[0]['proxy_gain']==3.
    assert not result[0]['macro_scores'] and result[0]['macro_gate_disabled']
    assert baseline['proxy_objective']==3.
