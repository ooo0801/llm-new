import numpy as np
import pytest
from llm_integrity.resf_token import early_window, fitted_deviance, calibration, detect


def test_rule_e_budget_floor():
    n = early_window(.003, .01, 100)
    assert n == 3
    assert 1 - (1-.003)**n <= .01
    assert 1 - (1-.003)**(n+1) > .01
    assert early_window(0, .01, 100) == 100
    assert early_window(1, .01, 100) == 0


def test_impossible_and_sparse_deviance():
    q = np.array([[1., 0], [.9, .1]])
    value, fitted = fitted_deviance([90, 10], q)
    assert value[0] == pytest.approx(0, abs=1e-10)
    assert fitted[0] == 1
    assert np.isinf(fitted_deviance([0, 100], [[1.,0]])[0][0])
    with pytest.raises(ValueError):
        fitted_deviance([2, 3], [[.7, .4]])


def test_support_and_shape_and_nuisance():
    q = np.array([[.8, .2], [.6, .4]])
    cal = calibration(q, simulations=2000)
    assert detect([99]*100, [0,1], q, cal, panel_size=4)['rule'] == 'E'
    assert detect([1]*100, [0,1], q, cal, panel_size=4)['rule'] == 'P'
    # Balanced interleaving matches the second nuisance distribution at all looks.
    null = [0,0,0,1,1]*20
    assert not detect(null, [0,1], q, cal, panel_size=4)['alarm']
    with pytest.raises(ValueError):
        detect(null[:4], [0,1], q, cal, panel_size=4)


def test_monte_carlo_null_alarm_smoke():
    q = np.array([[.8, .19, .01], [.6, .35, .05]])
    cal = calibration(q, simulations=2000, seed=18)
    rng = np.random.default_rng(19)
    for row in q:
        alarms = sum(detect(rng.choice(3,100,p=row), [0,1,2], q, cal, panel_size=1)['alarm']
                     for _ in range(200))
        assert alarms <= 20  # Gross calibration regression only; not FPR certification.
