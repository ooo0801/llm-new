from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .distances import logit_distance, robust_standardize
from .modeling import ModelBundle, next_token_logits, tokenize_prompts


@dataclass
class PromptSensitivity:
    prompt_id: str
    prompt: str
    macro: float | None = None
    micro: float | None = None
    macro_by_attack: dict[str, float] | None = None
    hybrid: float | None = None
    alpha: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "prompt": self.prompt,
            "macro": self.macro,
            "micro": self.micro,
            "macro_by_attack": self.macro_by_attack or {},
            "hybrid": self.hybrid,
            "alpha": self.alpha,
        }


def macro_sensitivity(
    reference: ModelBundle,
    variants: Sequence[tuple[str, ModelBundle]],
    rows: list[dict[str, Any]],
    metric: str = "js",
    batch_size: int = 4,
    max_length: int = 512,
    temperature: float = 1.0,
) -> list[PromptSensitivity]:
    if not variants:
        raise ValueError("At least one modified model is required")
    results = [
        PromptSensitivity(prompt_id=str(row["id"]), prompt=str(row["prompt"]), macro_by_attack={})
        for row in rows
    ]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch]
        reference_logits = next_token_logits(reference, prompts, max_length)
        attack_values: list[np.ndarray] = []
        for attack_name, variant in variants:
            candidate_logits = next_token_logits(variant, prompts, max_length)
            distances = logit_distance(reference_logits, candidate_logits, metric, temperature)
            values = distances.detach().cpu().numpy().astype(float)
            attack_values.append(values)
            for offset, value in enumerate(values):
                results[start + offset].macro_by_attack[attack_name] = float(value)
        averages = np.mean(np.stack(attack_values, axis=0), axis=0)
        for offset, value in enumerate(averages):
            results[start + offset].macro = float(value)
    return results


def _selected_parameters(model, patterns: Iterable[str], max_tensors: int | None):
    matches = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        if patterns and not any(pattern in name for pattern in patterns):
            continue
        matches.append((name, parameter))
    if max_tensors:
        matches = matches[-max_tensors:]
    if not matches:
        raise ValueError("No trainable parameters matched micro-sensitivity patterns")
    return matches


def micro_sensitivity_one(
    bundle: ModelBundle,
    prompt: str,
    parameter_patterns: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "down_proj"),
    max_parameter_tensors: int | None = 16,
    max_length: int = 512,
) -> float:
    """Operationalized Eq. (7): gradient norm of self-targeted next-token NLL."""
    import torch
    import torch.nn.functional as F

    model = bundle.model
    selected = _selected_parameters(model, tuple(parameter_patterns), max_parameter_tensors)
    original_flags = {id(parameter): parameter.requires_grad for _, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    encoded = tokenize_prompts(bundle, [prompt], max_length=max_length)
    outputs = model(**encoded, use_cache=False, return_dict=True)
    position = int(encoded["attention_mask"].sum().item()) - 1
    logits = outputs.logits[0, position].float()
    pseudo_target = logits.detach().argmax().reshape(1)
    loss = F.cross_entropy(logits.reshape(1, -1), pseudo_target)
    loss.backward()
    squared_norm = torch.zeros((), device=logits.device, dtype=torch.float64)
    for _, parameter in selected:
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().double().pow(2).sum()
    value = float(squared_norm.item())
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(original_flags[id(parameter)])
    return value


def add_micro_scores(
    scores: list[PromptSensitivity],
    bundle: ModelBundle,
    parameter_patterns: Iterable[str],
    max_parameter_tensors: int | None,
    max_length: int,
) -> list[PromptSensitivity]:
    for score in scores:
        score.micro = micro_sensitivity_one(
            bundle,
            score.prompt,
            parameter_patterns,
            max_parameter_tensors,
            max_length,
        )
    return scores


def combine_hybrid(
    scores: list[PromptSensitivity],
    beta: float = 1.0,
    fixed_alpha: float | None = None,
) -> list[PromptSensitivity]:
    if any(score.micro is None or score.macro is None for score in scores):
        raise ValueError("Both micro and macro scores are required")
    micro = robust_standardize(np.array([score.micro for score in scores], dtype=float))
    macro = robust_standardize(np.array([score.macro for score in scores], dtype=float))
    shifted_micro = micro - micro.min() + 1e-8
    shifted_macro = macro - macro.min() + 1e-8
    for index, score in enumerate(scores):
        if fixed_alpha is None:
            ratio = shifted_micro[index] / shifted_macro[index]
            alpha = 1.0 / (1.0 + math.exp(-beta * math.log(ratio)))
        else:
            alpha = float(fixed_alpha)
        score.alpha = alpha
        score.hybrid = float(alpha * micro[index] + (1.0 - alpha) * macro[index])
    return scores
