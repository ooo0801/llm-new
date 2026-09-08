import json

import numpy as np
import pytest

from llm_integrity.features import FeatureExtractor
from llm_integrity.h8_precalibration import h8_mmd2_unbiased_unequal
from llm_integrity.stage1_r1 import (
    FrozenSemanticCache, bind_existing_cache, evaluate_task_r1, majority,
    mmd_statistics, permutation_test, raw_statistics,
)


@pytest.fixture
def bank(tmp_path):
    texts = tmp_path / "texts.json"
    vectors = tmp_path / "vectors.npy"
    texts.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    np.save(vectors, np.eye(3))
    identity = {"name": "test", "revision": "frozen", "normalize_embeddings": True, "max_seq_length": 512}
    path = tmp_path / "manifest.json"
    cache = bind_existing_cache(path, texts, vectors, identity)
    return path, cache, identity


def test_cache_order_batch_resume_and_copy_isolation(bank):
    path, cache, identity = bank
    expected = cache.transform(["a", "b", "a", "c"])
    assert np.array_equal(expected, np.concatenate([cache.transform(["a"]), cache.transform(["b", "a", "c"])]))
    assert np.array_equal(expected, FrozenSemanticCache(path, identity).transform(["a", "b", "a", "c"]))
    assert np.array_equal(expected, cache.transform(["c", "a", "b", "a"])[::-1])
    expected[0] = 999
    assert np.array_equal(cache.transform(["a"]), [[1, 0, 0]])
    assert FeatureExtractor(semantic_cache=cache).transform(["a"]).shape == (1, 19)


def test_cache_unknown_identity_and_corruption_fail_closed(bank):
    path, cache, identity = bank
    with pytest.raises(ValueError, match="cache miss"):
        cache.transform(["unknown"])
    with pytest.raises(ValueError, match="identity"):
        FrozenSemanticCache(path, {**identity, "max_seq_length": 256})
    with pytest.raises(ValueError, match="requested encoder"):
        FeatureExtractor(semantic_model_name="other-model", semantic_cache=cache)
    with pytest.raises(ValueError, match="requested encoder"):
        FeatureExtractor(semantic_model_revision="other-revision", semantic_cache=cache)
    (path.parent / "texts.json").write_text('["b","a","c"]', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        FrozenSemanticCache(path, identity)


def test_mmd_vectorized_matches_direct_unbiased_unequal():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(11, 5))
    kernel = np.exp(-np.square(x[:, None] - x[None, :]).sum(axis=2) / 2)
    w = np.zeros((2, 11))
    w[0, :5] = 1
    w[1, [1, 3, 4, 8, 10]] = 1
    values = mmd_statistics(kernel, w, 5)
    for weights, value in zip(w, values):
        assert value == pytest.approx(h8_mmd2_unbiased_unequal(x[weights == 1], x[weights == 0], 1), abs=1e-12)


def test_identical_null_p_one_and_changed_constant_baseline_detected():
    sem = np.tile([1., 0.], (48, 1))
    x = np.zeros((48, 2))
    identical = permutation_test(["1, 2, 3, 4, 5, 6"] * 48, sem, x, 24, 1, 12, 199)
    assert identical["h8"]["p_value"] == identical["raw"]["p_value"] == 1
    assert identical["h8"]["detected"] is False
    x[24:] = [3, 1]
    sem[24:] = [0., 1.]
    changed = permutation_test(["1, 2, 3, 4, 5, 6"] * 24 + ["1, 2, 3, 4, 5"] * 24, sem, x, 24, 1, 12, 199)
    assert changed["raw"]["detected"] is True
    assert changed["h8"]["detected"] is True
    assert changed == permutation_test(["1, 2, 3, 4, 5, 6"] * 24 + ["1, 2, 3, 4, 5"] * 24, sem, x, 24, 1, 12, 199)


def test_empty_text_and_zero_semantic_is_valid_not_nan():
    result = permutation_test([""] * 8, np.zeros((8, 2)), np.zeros((8, 2)), 4, 1, 2, 99)
    assert result["raw"]["p_value"] == 1
    assert result["h8"]["p_value"] == 1


def test_raw_includes_component_that_was_constant_under_intact():
    # First ten tokens match; only bigrams beyond the prefix differ.
    good = " ".join(str(i) for i in range(20))
    bad = " ".join(str(i) for i in range(10)) + " z z z z z z z z z z"
    sem = np.tile([1., 0.], (48, 1))
    result = permutation_test([good] * 24 + [bad] * 24, sem, np.zeros((48, 2)), 24, 1, 55, 199)
    assert result["raw"]["components"]["prefix10_tv"] == 0
    assert result["raw"]["detected"] is True
    assert result["h8"]["detected"] is False  # No OR: channels remain separate.


def test_raw_matches_simple_tv_js_cases():
    w = np.array([[1., 1., 0., 0.]])
    values, components = raw_statistics(["a b", "a b", "c d", "c d"], np.eye(4), w, 2)
    assert components[0, 0] == pytest.approx(.2)
    assert components[0, 1] == pytest.approx(1)
    assert components[0, 2] == pytest.approx(.5)
    assert values[0] == 1


def test_unevaluable_is_not_false():
    result = permutation_test(["a"] * 4, np.eye(4), np.full((4, 2), np.nan), 2, 1, 2, 99)
    assert result["h8"]["status"] == "unevaluable"
    assert result["h8"]["detected"] is None
    assert majority([True, False, None])["detected"] is None
    assert majority([False, False, None])["detected"] is False
    assert majority([True, True, None])["detected"] is True
    assert majority([None, None, None])["evaluable_count"] == 0


@pytest.mark.parametrize("text,ok", [("360", True), ("答案是360。", True), ("360.0公里", True),
                                     ("> 360公里", True),
                                     ("70乘5等于350，再加10。所以，总共行驶的距离是360公里。", True),
                                     ("计算中出现360，但总距离为370公里。", False),
                                     ("1360", False), ("3600", False), ("-360", False),
                                     ("不是360，是370", False), ("360和370", False)])
def test_numeric_contract(text, ok):
    assert evaluate_task_r1(text, {"evaluator": "numeric", "expected_answer": "360"})["passed"] is ok


@pytest.mark.parametrize("text,ok", [
    ('{"name":"x","level":1,"enabled":true}', True),
    ('{"name":"x","level":true,"enabled":true}', False),
    ('{"name":"x","level":"1","enabled":true}', False),
    ('{"name":"x","level":1,"enabled":"true"}', False),
    ('{"name":"x","name":"y","level":1,"enabled":true}', False),
    ('{"description":"name level enabled"}', False),
    ('{"name":"x","level":1,"enabled":true,"other":NaN}', False),
    ('```json\n{"name":"x","level":1,"enabled":true}\n```', False),
])
def test_json_contract(text, ok):
    assert evaluate_task_r1(text, {"evaluator": "json"})["passed"] is ok


def test_instruction_missing_or_added_number_rejected():
    row = {"evaluator": "exact", "expected_answer": "1,2,3,4,5,6"}
    assert evaluate_task_r1("1, 2, 3, 4, 5, 6", row)["passed"] is True
    for text in ["1,2,3,4,5", "1,2,3,4,5,6,7", "答案：1,2,3,4,5,6"]:
        assert evaluate_task_r1(text, row)["passed"] is False


def test_code_not_executed_and_not_claimed_functionally_correct(tmp_path):
    marker = tmp_path / "never_created"
    text = f"import pathlib\npathlib.Path({str(marker)!r}).touch()\ndef square_sum_26(values):\n return 0"
    value = evaluate_task_r1(text, {"evaluator": "python_syntax"})
    assert value["passed"] is None and value["proxy_passed"] is True
    assert not marker.exists()
    assert evaluate_task_r1("def wrong(values): return 1", {"evaluator": "python_syntax"})["proxy_passed"] is False


def test_limited_proxies_and_unknown_types_remain_explicit():
    for evaluator in ["refusal", "length_and_contains", "contains"]:
        result = evaluate_task_r1("不能实验", {"evaluator": evaluator, "expected_answer": "实验"})
        assert result["passed"] is None and result["status"] == "proxy_only"
    assert evaluate_task_r1("anything", {"evaluator": "unknown"})["status"] == "unevaluable"
