import numpy as np
import pytest
from llm_integrity.stage2_contract import BLOCK_TYPES, balanced_block_schedule, internal_metrics


def test_full_cartesian_and_balanced_half():
    layers = [0, 9, 18, 27]
    for prompt in ["a", "b", "source_3"]:
        schedule = balanced_block_schedule(BLOCK_TYPES, layers, 28, prompt)
        assert len(set(schedule)) == 28
        half = schedule[:14]
        assert all(sum(b == t for b,l in half) == 2 for t in BLOCK_TYPES)
        counts = [sum(l == layer for b,l in half) for layer in layers]
        assert max(counts)-min(counts) <= 1


def test_translation_invariance_and_identity():
    x = np.array([3., 2., 1., -1.])
    identical = internal_metrics(x, x, 2)
    assert all(abs(v) < 1e-12 for v in identical.values())
    shift = internal_metrics(x, x+7., 2)
    assert shift["raw_logit_l2"] == pytest.approx(14.)
    assert all(abs(v)<1e-12 for k,v in shift.items() if k != "raw_logit_l2")


def test_metric_ranges_and_invalid_inputs():
    m = internal_metrics([1000., 0., -1000.], [-1000., 0., 1000.], 1)
    assert m["top1_flip"] == m["topk_change"] == 1
    assert m["js"] == pytest.approx(np.log(2))
    assert m["kl_intact_to_attack"] == pytest.approx(2000)
    with pytest.raises(ValueError):
        internal_metrics([np.nan], [0.])


@pytest.mark.parametrize("modules,layers", [(3, 4), (4, 4), (6, 4), (7, 4), (7, 7)])
def test_schedule_no_shared_factor_omissions(modules, layers):
    schedule = balanced_block_schedule(BLOCK_TYPES[:modules], list(range(layers)), modules*layers, "p")
    assert len(set(schedule)) == modules*layers


def test_primary_metrics_are_six_and_kl_is_secondary():
    from llm_integrity.stage2_contract import PROXIES
    m = internal_metrics([1., 2., 3.], [3., 2., 1.], 2)
    assert len(PROXIES) == 6
    assert set(m) == set(PROXIES) | {"kl_intact_to_attack"}
    assert m["topk_change"] == .5


def test_strict_preflight_contracts():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from stage2_preflight import task_check
    instruction = {"category":"instruction", "expected_answer":"1,2,3,4,5,6"}
    assert task_check(instruction, "1, 2, 3, 4, 5, 6")
    assert not task_check(instruction,"1,2,3,4,5")
    assert not task_check(instruction,"1,2,3,4,5,6",truncated=True)
    row = {"category":"structured"}
    valid = '{"name":"x","level":1,"enabled":true}'
    assert task_check(row,valid)
    assert not task_check(row,"```json\n"+valid+"\n```")
    assert not task_check(row,'{"name":"x","level":true,"enabled":true}')
    assert not task_check(row,'{"name":"x","name":"y","level":1,"enabled":true}')
