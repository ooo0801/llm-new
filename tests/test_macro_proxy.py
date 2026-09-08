import pytest
import torch

from llm_integrity.macro_proxy import SEARCH_PROXIES, differentiable_macro_proxy


@pytest.mark.parametrize("proxy", SEARCH_PROXIES)
def test_proxy_is_zero_for_identical_logits_and_has_finite_gradient(proxy):
    a = torch.tensor([3.0, 1.0, -2.0], requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    score = differentiable_macro_proxy(a, b, proxy=proxy, top_k=2)
    assert float(score.item()) == pytest.approx(0.0, abs=1e-7)
    score.backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


@pytest.mark.parametrize("proxy", SEARCH_PROXIES)
def test_proxy_detects_nonconstant_change(proxy):
    a = torch.tensor([4.0, 1.0, 0.0], requires_grad=True)
    b = torch.tensor([0.0, 1.0, 4.0], requires_grad=True)
    score = differentiable_macro_proxy(a, b, proxy=proxy, top_k=1)
    assert float(score.item()) > 0.0
    score.backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(b.grad).all()


def test_centered_and_probability_ignore_global_logit_shift():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = a + 9.0
    assert differentiable_macro_proxy(
        a, b, proxy="raw_logit_l2", top_k=2
    ).item() > 0.0
    for proxy in ("centered_logit_l2", "probability_l2", "js", "topk_continuous"):
        assert differentiable_macro_proxy(
            a, b, proxy=proxy, top_k=2
        ).item() == pytest.approx(0.0, abs=1e-7)
