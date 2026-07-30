from llm_integrity.mcc import complementarity, coverage_rate, greedy_mcc


def test_greedy_prefers_largest_weighted_gain():
    mapping = {
        "a": {"attention:0:0", "ffn:0:0"},
        "b": {"attention:0:1", "attention:0:2"},
        "c": {"mlp_residual:0"},
    }
    result = greedy_mcc(mapping, 2)
    assert result.selected_ids[0] == "b"
    assert len(result.selected_ids) == 2
    assert coverage_rate(result.selected_ids, mapping) > 0.5
    assert 0.0 <= complementarity(result.selected_ids, mapping) <= 1.0
    assert len(result.trace) == 2


def test_greedy_returns_exact_k_after_coverage_saturates():
    mapping = {
        "a": {"attention:0:0"},
        "b": {"attention:0:0"},
        "c": {"attention:0:0"},
    }
    result = greedy_mcc(mapping, 3)
    assert result.selected_ids == ["a", "b", "c"]
    assert result.marginal_gains == [1.2, 0, 0]


def test_greedy_rejects_k_larger_than_pool():
    import pytest

    with pytest.raises(ValueError, match="candidate count"):
        greedy_mcc({"a": {"attention:0:0"}}, 2)
