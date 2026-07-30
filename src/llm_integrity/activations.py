from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

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

    def profile(
        self,
        bundle: ModelBundle,
        prompt_id: str,
        prompt: str,
        max_length: int = 512,
        system_prompt: str | None = None,
    ) -> ActivationProfile:
        import torch

        base_norms: dict[int, float] = {}
        residual_norms: dict[tuple[str, int], float] = {}
        ffn_activations: dict[int, torch.Tensor] = {}
        handles = []

        def layer_pre_hook(layer: int):
            def hook(module, inputs):
                if inputs and torch.is_tensor(inputs[0]):
                    base_norms[layer] = float(
                        inputs[0].detach().float().norm(dim=-1).mean().item()
                    )

            return hook

        def residual_hook(kind: str, layer: int):
            def hook(module, inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                if torch.is_tensor(value):
                    residual_norms[(kind, layer)] = float(
                        value.detach().float().norm(dim=-1).mean().item()
                    )

            return hook

        def ffn_pre_hook(layer: int):
            def hook(module, inputs):
                if inputs and torch.is_tensor(inputs[0]):
                    # Qwen2.5 feeds SiLU(gate_proj(x)) * up_proj(x) into
                    # down_proj. Aggregate immediately so a profile never
                    # retains the full [batch, tokens, intermediate] tensor.
                    ffn_activations[layer] = (
                        inputs[0].detach().float().abs().mean(dim=(0, 1)).cpu()
                    )

            return hook

        for name, module in bundle.model.named_modules():
            layer = _layer_index(name)
            if layer is None:
                continue
            if _is_transformer_layer(name, module):
                handles.append(module.register_forward_pre_hook(layer_pre_hook(layer)))
            elif name.endswith(".self_attn"):
                handles.append(
                    module.register_forward_hook(residual_hook("attention_residual", layer))
                )
            elif name.endswith(".mlp"):
                handles.append(module.register_forward_hook(residual_hook("mlp_residual", layer)))
            elif name.endswith(".mlp.down_proj"):
                handles.append(module.register_forward_pre_hook(ffn_pre_hook(layer)))
        encoded = tokenize_prompts(
            bundle,
            [prompt],
            max_length=max_length,
            system_prompt=system_prompt,
        )
        outputs = None
        try:
            with torch.inference_mode():
                outputs = bundle.model(
                    **encoded,
                    output_attentions=True,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if outputs is None:
            raise RuntimeError("Activation profiling forward pass produced no output")
        components: set[str] = set()
        diagnostics: dict[str, Any] = {
            "attention_entropy": {},
            "attention_normalized_entropy": {},
            "residual_ratio": {},
            "ffn_count": {},
            "token_count": int(encoded["attention_mask"].sum().item()),
        }
        attentions = getattr(outputs, "attentions", None)
        if attentions:
            for layer, attention in enumerate(attentions):
                if attention is None:
                    continue
                raw_entropy, normalized_entropy = _causal_attention_entropy(attention)
                for head, value in enumerate(raw_entropy.tolist()):
                    key = f"{layer}:{head}"
                    normalized = float(normalized_entropy[head].item())
                    diagnostics["attention_entropy"][key] = float(value)
                    diagnostics["attention_normalized_entropy"][key] = normalized
                    if normalized < self.attention_entropy_fraction:
                        components.add(f"attention:{layer}:{head}")
        for layer, activation in sorted(ffn_activations.items()):
            count = min(
                self.max_ffn_units_per_layer,
                max(1, int(math.ceil(activation.numel() * (1.0 - self.ffn_quantile)))),
            )
            indices = activation.topk(count, largest=True, sorted=True).indices
            diagnostics["ffn_count"][str(layer)] = int(indices.numel())
            components.update(f"ffn:{layer}:{int(index)}" for index in indices.tolist())
        for (kind, layer), norm in sorted(residual_norms.items()):
            ratio = norm / max(base_norms.get(layer, 0.0), 1e-8)
            diagnostics["residual_ratio"][f"{kind}:{layer}"] = ratio
            if ratio > self.residual_threshold:
                components.add(f"{kind}:{layer}")
        if not components:
            raise RuntimeError(f"Activation profile for {prompt_id!r} selected no components")
        diagnostics["component_count"] = len(components)
        return ActivationProfile(prompt_id, components, diagnostics)


def _causal_attention_entropy(attention):
    """Return mean raw and causally normalized entropy for every head."""
    import torch

    probabilities = attention.detach().float().clamp_min(1e-12)
    if probabilities.ndim != 4:
        raise ValueError("Attention tensor must have shape [batch, heads, query, key]")
    query_count = probabilities.shape[-2]
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    if query_count <= 1:
        zeros = torch.zeros(probabilities.shape[1], dtype=torch.float32)
        return entropy.mean(dim=(0, 2)).cpu(), zeros
    # For causal attention, query q has q+1 reachable keys. Query zero has
    # log(1)=0 and carries no concentration information, so exclude it.
    denominators = torch.arange(
        2,
        query_count + 1,
        device=probabilities.device,
        dtype=torch.float32,
    ).log()
    normalized = entropy[:, :, 1:] / denominators.view(1, 1, -1)
    return entropy.mean(dim=(0, 2)).cpu(), normalized.mean(dim=(0, 2)).cpu()


def _is_transformer_layer(name: str, module: Any) -> bool:
    return bool(
        _layer_index(name) is not None
        and hasattr(module, "self_attn")
        and hasattr(module, "mlp")
    )


def _layer_index(name: str) -> int | None:
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in {"layers", "h", "blocks"}:
            try:
                return int(parts[index + 1])
            except (ValueError, IndexError):
                return None
    return None
