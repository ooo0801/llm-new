from __future__ import annotations

import torch


def squared_l2_logit_vjp(
    logits: torch.Tensor,
    embeddings: torch.Tensor,
    delta: torch.Tensor,
    *,
    sign: float,
) -> torch.Tensor:
    """Return one side of the squared-logit-distance input gradient.

    For ``delta = variant_logits - reference_logits``, the variant side
    uses ``sign=1`` and the reference side uses ``sign=-1``. Their sum is
    exactly the VJP obtained by differentiating ``delta.square().sum()``
    with both model graphs live at once.
    """
    if float(sign) not in {-1.0, 1.0}:
        raise ValueError("sign must be -1 or 1")
    if logits.shape != delta.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} does not match "
            f"delta shape {tuple(delta.shape)}"
        )
    return torch.autograd.grad(
        logits,
        embeddings,
        grad_outputs=(2.0 * float(sign) * delta).to(logits.device),
    )[0]
