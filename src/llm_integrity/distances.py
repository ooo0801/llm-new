from __future__ import annotations

from typing import Literal

import numpy as np


def js_divergence_logits(reference, candidate, temperature: float = 1.0):
    """Row-wise Jensen-Shannon divergence for PyTorch logits."""
    import torch
    import torch.nn.functional as F

    p_log = F.log_softmax(reference / temperature, dim=-1)
    q_log = F.log_softmax(candidate / temperature, dim=-1)
    p = p_log.exp()
    q = q_log.exp()
    m = 0.5 * (p + q)
    m_log = torch.log(m.clamp_min(torch.finfo(m.dtype).tiny))
    return 0.5 * ((p * (p_log - m_log)).sum(-1) + (q * (q_log - m_log)).sum(-1))


def symmetric_kl_logits(reference, candidate, temperature: float = 1.0):
    import torch.nn.functional as F

    p_log = F.log_softmax(reference / temperature, dim=-1)
    q_log = F.log_softmax(candidate / temperature, dim=-1)
    p, q = p_log.exp(), q_log.exp()
    return 0.5 * ((p * (p_log - q_log)).sum(-1) + (q * (q_log - p_log)).sum(-1))


def logit_distance(
    reference,
    candidate,
    metric: Literal["js", "symmetric_kl", "l2"] = "js",
    temperature: float = 1.0,
):
    if metric == "js":
        return js_divergence_logits(reference, candidate, temperature)
    if metric == "symmetric_kl":
        return symmetric_kl_logits(reference, candidate, temperature)
    if metric == "l2":
        return ((reference - candidate) ** 2).mean(dim=-1).sqrt()
    raise ValueError(f"Unknown logit distance: {metric}")


def robust_standardize(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    return (values - median) / (q3 - q1 + epsilon)
