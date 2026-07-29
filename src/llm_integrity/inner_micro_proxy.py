from __future__ import annotations

import gc
import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import Tensor

from .modeling import ModelBundle


_BLOCK_PATTERNS = {
    "q_proj": re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.q_proj\.weight$"),
    "v_proj": re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.v_proj\.weight$"),
    "down_proj": re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.down_proj\.weight$"),
}


@dataclass(frozen=True)
class MicroBlock:
    block_type: str
    layer_id: int
    parameter_name: str
    parameter_count: int


@dataclass
class DifferentiableMicroResult:
    block: MicroBlock
    probe_seed: int
    probe_count: int
    output_dimension: int
    raw_micro_score: float
    raw_embedding_gradient_norm: float
    maximum_token_gradient_norm: float
    gradient_finite: bool
    gradient_nonzero: bool
    parameter_gradient_finite: bool
    parameter_gradient_nonzero: bool
    embedding_gradient: Tensor
    probe_scores: tuple[float, ...]

    def metadata(self) -> dict:
        payload = asdict(self)
        payload.pop("embedding_gradient")
        payload["block"] = asdict(self.block)
        return payload


@dataclass(frozen=True)
class BlockMicroScore:
    block: MicroBlock
    probe_seed: int
    probe_count: int
    raw_micro_score: float
    parameter_gradient_finite: bool
    parameter_gradient_nonzero: bool
    probe_scores: tuple[float, ...]


def discover_micro_blocks(model) -> list[MicroBlock]:
    blocks: list[MicroBlock] = []
    for name, parameter in model.named_parameters():
        for block_type, pattern in _BLOCK_PATTERNS.items():
            match = pattern.search(name)
            if match is None:
                continue
            blocks.append(
                MicroBlock(
                    block_type=block_type,
                    layer_id=int(match.group(1)),
                    parameter_name=name,
                    parameter_count=int(parameter.numel()),
                )
            )
            break
    blocks.sort(key=lambda item: (item.layer_id, item.block_type))
    if not blocks:
        raise RuntimeError(
            "No q_proj, v_proj, or down_proj parameter blocks were found"
        )
    return blocks


def select_micro_block(
    blocks: Iterable[MicroBlock],
    *,
    block_type: str,
    layer_id: int,
) -> MicroBlock:
    selected = [
        block
        for block in blocks
        if block.block_type == block_type and block.layer_id == layer_id
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {block_type} block at layer {layer_id}, "
            f"found {len(selected)}"
        )
    return selected[0]


def representative_layers(blocks: Iterable[MicroBlock], count: int = 4) -> list[int]:
    layers = sorted({block.layer_id for block in blocks})
    if not layers:
        raise ValueError("No layer ids were found")
    count = max(1, min(int(count), len(layers)))
    if count == 1:
        return [layers[0]]
    positions = [
        round(index * (len(layers) - 1) / (count - 1))
        for index in range(count)
    ]
    return [layers[position] for position in dict.fromkeys(positions)]


def _rademacher(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    values = torch.randint(
        0,
        2,
        shape,
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    return values.to(dtype=dtype).mul_(2).sub_(1)


def differentiable_block_micro_proxy(
    bundle: ModelBundle,
    embeddings: Tensor,
    attention_mask: Tensor,
    *,
    block: MicroBlock,
    probes: int = 1,
    seed: int = 42,
) -> DifferentiableMicroResult:
    if probes <= 0:
        raise ValueError("probes must be positive")
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError("embeddings must have shape [1, sequence, hidden]")
    if not embeddings.requires_grad:
        raise ValueError("embeddings must require gradients")

    named_parameters = dict(bundle.model.named_parameters())
    if block.parameter_name not in named_parameters:
        raise KeyError(f"Parameter not found: {block.parameter_name}")
    selected_parameter = named_parameters[block.parameter_name]

    initially_trainable = [
        name
        for name, parameter in bundle.model.named_parameters()
        if parameter.requires_grad
    ]
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    selected_parameter.requires_grad_(True)

    probe_scores: list[Tensor] = []
    parameter_gradient_finite = True
    parameter_gradient_nonzero = True
    try:
        dtype = bundle.model.get_input_embeddings().weight.dtype
        outputs = bundle.model(
            inputs_embeds=embeddings.to(dtype),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        mask = attention_mask.bool()
        positions = torch.arange(
            mask.shape[1],
            device=mask.device,
        ).unsqueeze(0).expand_as(mask)
        last_position = positions.masked_fill(~mask, -1).max(dim=1).values
        logits = outputs.logits[
            torch.arange(outputs.logits.shape[0], device=mask.device),
            last_position,
        ].float().squeeze(0)

        for probe_index in range(probes):
            probe = _rademacher(
                logits.shape,
                device=logits.device,
                dtype=logits.dtype,
                seed=int(seed) + probe_index,
            )
            projected = (logits * probe).sum()
            parameter_gradient = torch.autograd.grad(
                projected,
                selected_parameter,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0]
            finite = bool(torch.isfinite(parameter_gradient).all().item())
            nonzero = bool(
                finite
                and parameter_gradient.detach().float().norm().item() > 0
            )
            parameter_gradient_finite &= finite
            parameter_gradient_nonzero &= nonzero
            probe_scores.append(
                parameter_gradient.float().square().sum()
            )

        micro_score = torch.stack(probe_scores).mean()
        embedding_gradient = torch.autograd.grad(
            micro_score,
            embeddings,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0].detach().float()

        raw_norm = float(embedding_gradient.norm().item())
        token_norms = embedding_gradient.squeeze(0).norm(dim=-1)
        gradient_finite = bool(
            torch.isfinite(embedding_gradient).all().item()
            and math.isfinite(float(micro_score.detach().item()))
        )
        gradient_nonzero = bool(gradient_finite and raw_norm > 0)

        return DifferentiableMicroResult(
            block=block,
            probe_seed=int(seed),
            probe_count=int(probes),
            output_dimension=int(logits.numel()),
            raw_micro_score=float(micro_score.detach().item()),
            raw_embedding_gradient_norm=raw_norm,
            maximum_token_gradient_norm=float(token_norms.max().item()),
            gradient_finite=gradient_finite,
            gradient_nonzero=gradient_nonzero,
            parameter_gradient_finite=parameter_gradient_finite,
            parameter_gradient_nonzero=parameter_gradient_nonzero,
            embedding_gradient=embedding_gradient,
            probe_scores=tuple(
                float(value.detach().item()) for value in probe_scores
            ),
        )
    finally:
        selected_parameter.requires_grad_(False)
        selected_parameter.grad = None
        for name in initially_trainable:
            if name in named_parameters:
                named_parameters[name].requires_grad_(True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def score_block_micro_proxy(
    bundle: ModelBundle,
    embeddings: Tensor,
    attention_mask: Tensor,
    *,
    block: MicroBlock,
    probes: int = 1,
    seed: int = 42,
) -> BlockMicroScore:
    """Score a discrete candidate without building a second-order graph.

    Candidate re-ranking only needs the small-block sensitivity score, not
    its derivative with respect to the candidate embeddings.  Avoiding
    ``create_graph=True`` keeps the exact proxy score while substantially
    reducing peak memory during discrete candidate evaluation.
    """
    if probes <= 0:
        raise ValueError("probes must be positive")
    if embeddings.ndim != 3 or embeddings.shape[0] != 1:
        raise ValueError("embeddings must have shape [1, sequence, hidden]")

    named_parameters = dict(bundle.model.named_parameters())
    if block.parameter_name not in named_parameters:
        raise KeyError(f"Parameter not found: {block.parameter_name}")
    selected_parameter = named_parameters[block.parameter_name]

    initially_trainable = [
        name
        for name, parameter in bundle.model.named_parameters()
        if parameter.requires_grad
    ]
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    selected_parameter.requires_grad_(True)

    probe_scores: list[Tensor] = []
    parameter_gradient_finite = True
    parameter_gradient_nonzero = True
    try:
        dtype = bundle.model.get_input_embeddings().weight.dtype
        outputs = bundle.model(
            inputs_embeds=embeddings.to(dtype),
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        mask = attention_mask.bool()
        positions = torch.arange(
            mask.shape[1],
            device=mask.device,
        ).unsqueeze(0).expand_as(mask)
        last_position = positions.masked_fill(~mask, -1).max(dim=1).values
        logits = outputs.logits[
            torch.arange(outputs.logits.shape[0], device=mask.device),
            last_position,
        ].float().squeeze(0)

        for probe_index in range(probes):
            probe = _rademacher(
                logits.shape,
                device=logits.device,
                dtype=logits.dtype,
                seed=int(seed) + probe_index,
            )
            projected = (logits * probe).sum()
            parameter_gradient = torch.autograd.grad(
                projected,
                selected_parameter,
                create_graph=False,
                retain_graph=probe_index < probes - 1,
                allow_unused=False,
            )[0]
            finite = bool(torch.isfinite(parameter_gradient).all().item())
            nonzero = bool(
                finite
                and parameter_gradient.detach().float().norm().item() > 0
            )
            parameter_gradient_finite &= finite
            parameter_gradient_nonzero &= nonzero
            probe_scores.append(
                parameter_gradient.detach().float().square().sum()
            )

        micro_score = torch.stack(probe_scores).mean()
        return BlockMicroScore(
            block=block,
            probe_seed=int(seed),
            probe_count=int(probes),
            raw_micro_score=float(micro_score.item()),
            parameter_gradient_finite=parameter_gradient_finite,
            parameter_gradient_nonzero=parameter_gradient_nonzero,
            probe_scores=tuple(float(value.item()) for value in probe_scores),
        )
    finally:
        selected_parameter.requires_grad_(False)
        selected_parameter.grad = None
        for name in initially_trainable:
            if name in named_parameters:
                named_parameters[name].requires_grad_(True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
