import pytest

torch = pytest.importorskip("torch")

from llm_integrity.activations import _causal_attention_entropy


def test_causal_attention_entropy_normalizes_uniform_prefixes() -> None:
    attention = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]]]],
        dtype=torch.float32,
    )
    _, normalized = _causal_attention_entropy(attention)
    assert normalized.shape == (1,)
    assert normalized.item() == pytest.approx(1.0, abs=1e-5)


def test_causal_attention_entropy_marks_focused_heads() -> None:
    attention = torch.tensor(
        [[[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]],
        dtype=torch.float32,
    )
    _, normalized = _causal_attention_entropy(attention)
    assert normalized.item() < 1e-6
