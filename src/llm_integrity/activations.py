from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .modeling import ModelBundle, tokenize_prompts


@dataclass
class ActivationProfile:
    prompt_id: str
    components: set[str]
    diagnostics: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.prompt_id,
            "components": sorted(self.components),
            "diagnostics": self.diagnostics,
        }


class TransformerActivationProfiler:
    def __init__(
        self,
        attention_entropy_fraction: float = 0.70,
        ffn_quantile: float = 0.95,
        residual_threshold: float = 0.50,
        max_ffn_units_per_layer: int = 128,
    ) -> None:
        self.attention_entropy_fraction = attention_entropy_fraction
        self.ffn_quantile = ffn_quantile
        self.residual_threshold = residual_threshold
        self.max_ffn_units_per_layer = max_ffn_units_per_layer

    def profile(self, bundle: ModelBundle, prompt_id: str, prompt: str, max_length: int = 512) -> ActivationProfile:
        import torch

        captures: dict[str, list[torch.Tensor]] = defaultdict(list)
        handles = []

        def output_hook(name):
            def hook(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(value):
                    captures[name].append(value.detach().float().cpu())
            return hook

        def pre_hook(name):
            def hook(module, inputs):
                if inputs and torch.is_tensor(inputs[0]):
                    captures[name].append(inputs[0].detach().float().cpu())
            return hook

        for name, module in bundle.model.named_modules():
            if name.endswith(".self_attn"):
                handles.append(module.register_forward_hook(output_hook(f"attn_out:{name}")))
            elif name.endswith(".mlp"):
                handles.append(module.register_forward_hook(output_hook(f"mlp_out:{name}")))
            elif name.endswith(".mlp.down_proj"):
                handles.append(module.register_forward_pre_hook(pre_hook(f"ffn_intermediate:{name}")))
        encoded = tokenize_prompts(bundle, [prompt], max_length=max_length)
        try:
            with torch.inference_mode():
                outputs = bundle.model(
                    **encoded,
                    output_attentions=True,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        components: set[str] = set()
        diagnostics: dict[str, Any] = {"attention_entropy": {}, "residual_ratio": {}, "ffn_count": {}}
        attentions = getattr(outputs, "attentions", None)
        if attentions:
            for layer, attention in enumerate(attentions):
                if attention is None:
                    continue
                probabilities = attention.detach().float().clamp_min(1e-12)
                entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=(0, 2))
                uniform_entropy = np.log(probabilities.shape[-1])
                threshold = self.attention_entropy_fraction * uniform_entropy
                for head, value in enumerate(entropy.cpu().tolist()):
                    diagnostics["attention_entropy"][f"{layer}:{head}"] = value
                    if value < threshold:
                        components.add(f"attention:{layer}:{head}")
        hidden_states = getattr(outputs, "hidden_states", None)
        base_norms: dict[int, float] = {}
        if hidden_states:
            for layer, state in enumerate(hidden_states[:-1]):
                base_norms[layer] = float(state.detach().float().norm(dim=-1).mean().cpu().item())
        for name, values in captures.items():
            layer = _layer_index(name)
            if layer is None:
                continue
            value = values[-1]
            if name.startswith("ffn_intermediate:"):
                activation = value.abs().mean(dim=(0, 1))
                threshold = torch.quantile(activation, self.ffn_quantile)
                indices = torch.nonzero(activation >= threshold, as_tuple=False).flatten()
                if indices.numel() > self.max_ffn_units_per_layer:
                    top = activation.topk(self.max_ffn_units_per_layer).indices
                    indices = top
                diagnostics["ffn_count"][str(layer)] = int(indices.numel())
                components.update(f"ffn:{layer}:{int(index)}" for index in indices.tolist())
            elif name.startswith(("attn_out:", "mlp_out:")):
                kind = "attention_residual" if name.startswith("attn_out:") else "mlp_residual"
                ratio = float(value.norm(dim=-1).mean().item()) / max(base_norms.get(layer, 1.0), 1e-8)
                diagnostics["residual_ratio"][f"{kind}:{layer}"] = ratio
                if ratio > self.residual_threshold:
                    components.add(f"{kind}:{layer}")
        return ActivationProfile(prompt_id, components, diagnostics)


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "h", "blocks"}:
            try:
                return int(parts[index + 1])
            except (ValueError, IndexError):
                return None
    return None
