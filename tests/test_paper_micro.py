from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_integrity.paper_micro import (
    estimate_next_token_jacobian_frobenius,
)


def test_hutchinson_matches_linear_layer_exact_result() -> None:
    """
    对线性层y=Wx+b：

        ||∂y/∂W||_F² = output_dim * ||x||²
        ||∂y/∂b||_F² = output_dim

    因此总Jacobian范数平方为：

        output_dim * (||x||² + 1)

    在线性层和Rademacher探针下，该结果可以精确验证。
    """
    torch.manual_seed(7)

    input_dimension = 3
    output_dimension = 4

    layer = torch.nn.Linear(
        input_dimension,
        output_dimension,
        bias=True,
    )

    x = torch.tensor(
        [1.0, -2.0, 0.5],
        dtype=torch.float32,
    )

    output = layer(x)

    result = estimate_next_token_jacobian_frobenius(
        output,
        layer.named_parameters(),
        probes=3,
        seed=42,
    )

    expected = output_dimension * (
        x.square().sum().item() + 1.0
    )

    assert result.estimate == pytest.approx(
        expected,
        rel=1e-6,
        abs=1e-6,
    )

    assert result.parameter_count == (
        output_dimension * input_dimension
        + output_dimension
    )

    assert result.output_dimension == output_dimension
    assert result.probes == 3

    assert set(result.per_parameter_tensor) == {
        "weight",
        "bias",
    }


def test_estimator_rejects_detached_logits() -> None:
    layer = torch.nn.Linear(2, 3)

    output = layer(
        torch.tensor([1.0, 2.0])
    ).detach()

    with pytest.raises(
        ValueError,
        match="does not require gradients",
    ):
        estimate_next_token_jacobian_frobenius(
            output,
            layer.named_parameters(),
            probes=2,
        )


def test_estimator_rejects_batched_output() -> None:
    layer = torch.nn.Linear(2, 3)

    output = layer(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        estimate_next_token_jacobian_frobenius(
            output,
            layer.named_parameters(),
            probes=2,
        )
