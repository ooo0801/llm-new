"""Differentiable next-token proxy objectives used by Stage2 searches.

The functions in this module intentionally return scalar *squared* distances
for the three L2 objectives.  The ordering is identical to the corresponding
L2 norm, while the squared form has the stable gradient used by the historical
raw-logit implementation.
"""
from __future__ import annotations

import torch
from torch import Tensor


SEARCH_PROXIES = (
    "raw_logit_l2",
    "centered_logit_l2",
    "probability_l2",
    "js",
    "topk_continuous",
)


def differentiable_macro_proxy(
    reference_logits: Tensor,
    variant_logits: Tensor,
    *,
    proxy: str,
    top_k: int = 10,
) -> Tensor:
    """Return a finite scalar proxy for one paired next-token distribution.

    ``topk_continuous`` is a symmetric, piecewise-differentiable leakage
    measure.  Within the current top-k regions it measures how much probability
    mass each endpoint loses on its own preferred support relative to the other
    endpoint.  The support indices are treated as fixed during one backward
    pass and exact scores are recomputed for every discrete candidate.
    """
    if proxy not in SEARCH_PROXIES:
        raise ValueError(f"Unsupported macro proxy: {proxy}")
    if reference_logits.ndim != 1 or reference_logits.shape != variant_logits.shape:
        raise ValueError("Expected paired one-dimensional vocabulary logits")
    if not 1 <= int(top_k) <= int(reference_logits.numel()):
        raise ValueError("Invalid top_k")

    if proxy == "raw_logit_l2":
        score = (variant_logits - reference_logits).square().sum()
    elif proxy == "centered_logit_l2":
        reference_centered = reference_logits - reference_logits.mean()
        variant_centered = variant_logits - variant_logits.mean()
        score = (variant_centered - reference_centered).square().sum()
    else:
        reference_logp = torch.log_softmax(reference_logits.float(), dim=-1)
        variant_logp = torch.log_softmax(variant_logits.float(), dim=-1)
        reference_p = reference_logp.exp()
        variant_p = variant_logp.exp()
        if proxy == "probability_l2":
            score = (variant_p - reference_p).square().sum()
        elif proxy == "js":
            log_mixture = torch.logaddexp(reference_logp, variant_logp) - torch.log(
                reference_logp.new_tensor(2.0)
            )
            score = 0.5 * (
                (reference_p * (reference_logp - log_mixture)).sum()
                + (variant_p * (variant_logp - log_mixture)).sum()
            )
            score = score.clamp_min(0.0)
        else:
            reference_top = torch.topk(reference_p.detach(), int(top_k)).indices
            variant_top = torch.topk(variant_p.detach(), int(top_k)).indices
            score = 0.5 * (
                reference_p[reference_top].sum()
                - variant_p[reference_top].sum()
                + variant_p[variant_top].sum()
                - reference_p[variant_top].sum()
            )
            score = score.clamp_min(0.0)
    if score.ndim != 0 or not bool(torch.isfinite(score).item()):
        raise RuntimeError(f"Non-finite {proxy} objective")
    return score
