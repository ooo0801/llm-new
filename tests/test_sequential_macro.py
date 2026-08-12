from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from llm_integrity.sequential_macro import squared_l2_logit_vjp


def test_sequential_squared_l2_vjp_matches_joint_autograd() -> None:
    torch.manual_seed(2026082010)
    reference = torch.nn.Linear(5, 7, bias=True)
    variant = torch.nn.Linear(5, 7, bias=True)
    embeddings = torch.randn(2, 3, 5, requires_grad=True)

    reference_logits = reference(embeddings)[:, -1, :]
    variant_logits = variant(embeddings)[:, -1, :]
    joint_score = (variant_logits - reference_logits).square().sum()
    joint_gradient = torch.autograd.grad(joint_score, embeddings)[0]

    reference_input = embeddings.detach().clone().requires_grad_(True)
    reference_only = reference(reference_input)[:, -1, :]
    variant_input = embeddings.detach().clone().requires_grad_(True)
    variant_only = variant(variant_input)[:, -1, :]
    delta = (variant_only - reference_only).detach()
    sequential_gradient = squared_l2_logit_vjp(
        variant_only,
        variant_input,
        delta,
        sign=1.0,
    ) + squared_l2_logit_vjp(
        reference_only,
        reference_input,
        delta,
        sign=-1.0,
    )

    assert joint_score.item() == pytest.approx(delta.square().sum().item())
    assert torch.allclose(joint_gradient, sequential_gradient, atol=1e-6, rtol=1e-6)


def test_sequential_squared_l2_vjp_validates_inputs() -> None:
    embeddings = torch.randn(1, 2, 3, requires_grad=True)
    logits = embeddings.sum(dim=-1)
    with pytest.raises(ValueError, match="sign"):
        squared_l2_logit_vjp(logits, embeddings, logits.detach(), sign=0.0)
    with pytest.raises(ValueError, match="shape"):
        squared_l2_logit_vjp(
            logits,
            embeddings,
            torch.zeros(1, 3),
            sign=1.0,
        )
