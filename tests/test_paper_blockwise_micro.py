from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from llm_integrity.paper_blockwise_micro import (
    estimate_jacobian_frobenius_blockwise,
)
from llm_integrity.paper_micro import (
    estimate_next_token_jacobian_frobenius,
)
from llm_integrity.paper_parameter_groups import (
    build_parameter_groups,
)


class TinyAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(
            3,
            3,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.q_proj(x)


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            3,
            4,
            bias=False,
        )
        self.down_proj = torch.nn.Linear(
            4,
            3,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.down_proj(
            torch.tanh(
                self.gate_proj(x)
            )
        )


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = TinyAttention()
        self.mlp = TinyMLP()
        self.input_layernorm = torch.nn.LayerNorm(3)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        return self.mlp(x)


class TinyInnerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(
            10,
            3,
        )
        self.layers = torch.nn.ModuleList(
            [TinyLayer()]
        )
        self.norm = torch.nn.LayerNorm(3)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyInnerModel()
        self.lm_head = torch.nn.Linear(
            3,
            5,
            bias=False,
        )

    def forward(
        self,
        token_id: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.embed_tokens(
            token_id
        )
        x = self.model.layers[0](x)
        x = self.model.norm(x)
        return self.lm_head(x)


def test_blockwise_matches_all_parameters_at_once() -> None:
    torch.manual_seed(123)

    model = TinyModel()
    token_id = torch.tensor(4)

    direct_output = model(token_id)

    direct = estimate_next_token_jacobian_frobenius(
        direct_output,
        model.named_parameters(),
        probes=8,
        seed=42,
    )

    groups = build_parameter_groups(model)

    blockwise = estimate_jacobian_frobenius_blockwise(
        model,
        lambda: model(token_id),
        groups=groups,
        probes=8,
        seed=42,
    )

    assert blockwise.complete_parameter_coverage

    assert blockwise.parameter_count == sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert blockwise.output_dimension == 5

    assert blockwise.estimate == pytest.approx(
        direct.estimate,
        rel=1e-5,
        abs=1e-6,
    )

    assert blockwise.probe_estimates == pytest.approx(
        direct.probe_estimates,
        rel=1e-5,
        abs=1e-6,
    )

    assert sum(
        blockwise.component_estimates.values()
    ) == pytest.approx(
        blockwise.estimate,
        rel=1e-6,
        abs=1e-6,
    )


def test_original_gradient_flags_are_restored() -> None:
    model = TinyModel()
    token_id = torch.tensor(2)

    first_parameter = next(
        model.parameters()
    )
    first_parameter.requires_grad_(False)

    original_flags = {
        id(parameter): parameter.requires_grad
        for parameter in model.parameters()
    }

    estimate_jacobian_frobenius_blockwise(
        model,
        lambda: model(token_id),
        probes=2,
        seed=42,
    )

    restored_flags = {
        id(parameter): parameter.requires_grad
        for parameter in model.parameters()
    }

    assert restored_flags == original_flags
