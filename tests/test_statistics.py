import numpy as np

from llm_integrity.statistics import (
    mmd2_unbiased,
    mmd_permutation_test,
    paired_sign_flip_test,
    prompt_stratified_block_mmd_test,
    prompt_stratified_mmd_test,
)


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


def test_prompt_stratified_mmd_is_deterministic_and_reports_correction():
    rng = np.random.default_rng(13)
    x = np.concatenate([rng.normal(-20, 0.1, size=(8, 3)), rng.normal(20, 0.1, size=(8, 3))])
    y = x + 2.0
    strata = ["first"] * 8 + ["second"] * 8
    first = prompt_stratified_mmd_test(x, y, strata, permutations=199, seed=5)
    second = prompt_stratified_mmd_test(x, y, strata, permutations=199, seed=5)
    assert first.reject
    assert first.as_dict() == second.as_dict()
    assert first.p_value >= 1 / 200
    assert first.diagnostics["strata"] == 2


def test_prompt_stratified_mmd_requires_repeats_per_prompt():
    x = np.zeros((2, 2))
    y = np.ones((2, 2))
    import pytest

    with pytest.raises(ValueError, match="fewer than two"):
        prompt_stratified_mmd_test(x, y, ["a", "b"], permutations=9)


def test_prompt_stratified_block_mmd_is_deterministic_and_keeps_blocks():
    rng = np.random.default_rng(21)
    blocks = 8
    prompts = 3
    x_cube = rng.normal(size=(blocks, prompts, 4))
    y_cube = x_cube + 1.5
    x = x_cube.reshape(blocks * prompts, 4)
    y = y_cube.reshape(blocks * prompts, 4)
    strata = [f"prompt_{index}" for _ in range(blocks) for index in range(prompts)]
    block_labels = [block for block in range(blocks) for _ in range(prompts)]
    first = prompt_stratified_block_mmd_test(
        x, y, strata, block_labels, permutations=199, seed=17
    )
    second = prompt_stratified_block_mmd_test(
        x, y, strata, block_labels, permutations=199, seed=17
    )
    assert first.reject
    assert first.as_dict() == second.as_dict()
    assert first.diagnostics["blocks_per_group"] == blocks
    assert first.diagnostics["exchangeable_unit"] == "generation_seed_block"


def test_prompt_stratified_block_mmd_requires_rectangular_blocks():
    import pytest

    x = np.zeros((4, 2))
    y = np.ones((4, 2))
    with pytest.raises(ValueError, match="exactly one"):
        prompt_stratified_block_mmd_test(
            x,
            y,
            ["a", "a", "b", "b"],
            [0, 0, 1, 1],
            permutations=9,
        )
