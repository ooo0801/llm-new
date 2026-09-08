import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from stage2_pilot import finite_payload,pilot_utility
from stage2_prepare import design,training_data


def test_nonfinite_diagnostics_do_not_modify_prompt_text():
    p=finite_payload({"prompt":"NaN Infinity -Infinity","x":float('nan'),"y":-float('inf')})
    assert p=={"prompt":"NaN Infinity -Infinity","x":None,"y":None}


def test_attack_splits_and_budget():
    v=design()
    assert len(v)==57
    assert sum(r["split"]=="train" for r in v)==15
    for split in ["confirmation","heldout"]:
        for fam in ["gaussian_noise","finetuning"]:
            for strength in ["weak","medium","strong"]:
                group=[r for r in v if r["split"]==split and r["family"]==fam and r["strength"]==strength]
                assert len(group)==3 and len({r["seed"] for r in group})==3
    assert 30*(32+42*8)==11040
    assert 8*(57+1)==464


def test_training_and_utility_disjoint():
    assert len(training_data())==32
    assert not {r["prompt"] for r in training_data()}&{r["prompt"] for r in pilot_utility()}
