import numpy as np

from llm_integrity.statistics import mmd2_unbiased, mmd_permutation_test, paired_sign_flip_test


def test_mmd_is_symmetric():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 4))
    y = rng.normal(loc=0.5, size=(12, 4))
    assert np.isclose(mmd2_unbiased(x, y), mmd2_unbiased(y, x))


def test_permutation_detects_large_shift():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(20, 6))
    y = x + 2.0
    assert mmd_permutation_test(x, y, permutations=199).reject
    assert paired_sign_flip_test(x, y, permutations=199).reject
