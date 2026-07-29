from __future__ import annotations

import random
from typing import Any, Iterable, Mapping

from .attacks import AttackReport


def _configuration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = manifest.get("configuration", manifest)
    if not isinstance(config, Mapping):
        raise TypeError("Attack configuration must be a mapping")
    return dict(config)


def _manifest_seed(manifest: Mapping[str, Any]) -> int:
    config = _configuration(manifest)
    return int(manifest.get("seed", config.get("seed", 42)))


def execution_mode(manifest: Mapping[str, Any]) -> str:
    """Return the execution lifecycle required by an attack variant."""
    family = str(
        manifest.get(
            "family",
            _configuration(manifest).get("type", ""),
        )
    ).lower()

    if family in {
        "unstructured_pruning",
        "structured_pruning",
        "gaussian_noise",
    }:
        return "in_memory"

    if family == "quantization":
        return "reload"

    if family == "finetuning":
        return "training"

    raise ValueError(f"Unknown attack family: {family!r}")


def _scope_matches(name: str, scope: str) -> bool:
    lowered = name.lower()
    scope = scope.lower()

    attention_markers = (
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    ffn_markers = (
        ".mlp.",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    transformer_marker = ".layers."

    if scope in {"full_model", "all"}:
        return True

    if scope == "attention":
        return any(marker in lowered for marker in attention_markers)

    if scope == "ffn":
        return any(marker in lowered for marker in ffn_markers)

    if scope in {
        "attention_ffn",
        "all_transformer_layers",
        "transformer",
    }:
        return (
            transformer_marker in lowered
            and (
                any(marker in lowered for marker in attention_markers)
                or any(marker in lowered for marker in ffn_markers)
            )
        )

    if scope == "partial_layers":
        return transformer_marker in lowered

    raise ValueError(f"Unknown target scope: {scope!r}")


def _selected_parameters(
    model,
    target_scope: str,
) -> list[tuple[str, Any]]:
    selected: list[tuple[str, Any]] = []

    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if not _scope_matches(name, target_scope):
            continue
        selected.append((name, parameter))

    if not selected:
        raise ValueError(
            f"No floating-point parameters matched target_scope={target_scope!r}"
        )

    return selected


def _torch_generator(device, seed: int):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _sample_global_threshold(
    selected: Iterable[tuple[str, Any]],
    ratio: float,
    seed: int,
    samples_per_tensor: int = 8192,
):
    """Estimate a scalable global threshold without concatenating the model."""
    import torch

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    samples: list[Any] = []

    for _, parameter in selected:
        flat = parameter.detach().abs().reshape(-1)
        take = min(samples_per_tensor, flat.numel())

        if take == flat.numel():
            sample = flat.float().cpu()
        else:
            indices = torch.randint(
                low=0,
                high=flat.numel(),
                size=(take,),
                generator=cpu_generator,
                device="cpu",
            )
            sample = flat[indices.to(flat.device)].float().cpu()

        samples.append(sample)

    combined = torch.cat(samples)
    threshold = torch.quantile(
        combined,
        torch.tensor(ratio, dtype=torch.float32),
    )
    return float(threshold.item()), int(combined.numel())


def apply_unstructured_pruning(
    model,
    configuration: Mapping[str, Any],
    seed: int,
) -> AttackReport:
    import torch

    ratio = float(configuration.get("ratio", 0.30))
    method = str(configuration.get("method", "layerwise_magnitude")).lower()
    target_scope = str(configuration.get("target_scope", "attention_ffn"))

    if not 0.0 <= ratio < 1.0:
        raise ValueError("Pruning ratio must be in [0, 1)")

    if method not in {
        "global_magnitude",
        "layerwise_magnitude",
        "random",
    }:
        raise ValueError(f"Unknown unstructured pruning method: {method!r}")

    selected = _selected_parameters(model, target_scope)
    total_parameters = sum(parameter.numel() for _, parameter in selected)
    changed_parameters = 0
    threshold_value: float | None = None
    threshold_samples: int | None = None

    if ratio == 0.0:
        return AttackReport(
            name="unstructured_pruning",
            changed_parameters=0,
            total_parameters=total_parameters,
            details={
                "ratio": ratio,
                "method": method,
                "target_scope": target_scope,
                "seed": seed,
            },
        )

    if method == "global_magnitude":
        threshold_value, threshold_samples = _sample_global_threshold(
            selected,
            ratio,
            seed,
        )

        for _, parameter in selected:
            mask = parameter.detach().abs() <= threshold_value
            changed_parameters += int(mask.sum().item())

            with torch.no_grad():
                parameter.masked_fill_(mask, 0)

    elif method == "layerwise_magnitude":
        for _, parameter in selected:
            flat = parameter.detach().abs().reshape(-1)
            count = max(1, int(flat.numel() * ratio))
            threshold = torch.kthvalue(
                flat.float(),
                min(count, flat.numel()),
            ).values.to(flat.device)

            mask = parameter.detach().abs() <= threshold
            changed_parameters += int(mask.sum().item())

            with torch.no_grad():
                parameter.masked_fill_(mask, 0)

    else:
        generators: dict[str, Any] = {}

        for index, (_, parameter) in enumerate(selected):
            device_key = str(parameter.device)

            if device_key not in generators:
                generators[device_key] = _torch_generator(
                    parameter.device,
                    seed,
                )

            random_values = torch.rand(
                parameter.shape,
                generator=generators[device_key],
                device=parameter.device,
                dtype=torch.float32,
            )
            mask = random_values < ratio
            changed_parameters += int(mask.sum().item())

            with torch.no_grad():
                parameter.masked_fill_(mask, 0)

    return AttackReport(
        name="unstructured_pruning",
        changed_parameters=changed_parameters,
        total_parameters=total_parameters,
        details={
            "ratio": ratio,
            "method": method,
            "target_scope": target_scope,
            "seed": seed,
            "global_threshold": threshold_value,
            "threshold_sample_count": threshold_samples,
            "selected_tensors": len(selected),
        },
    )


def _transformer_layers(model) -> list[tuple[int, Any]]:
    layers = getattr(
        getattr(model, "model", None),
        "layers",
        None,
    )

    if layers is None:
        raise ValueError(
            "Structured pruning currently requires a model.model.layers "
            "transformer layout"
        )

    return list(enumerate(layers))


def _choose_layers(
    model,
    layer_scope: str,
    seed: int,
) -> list[tuple[int, Any]]:
    layers = _transformer_layers(model)
    layer_scope = layer_scope.lower()

    if layer_scope == "all_layers":
        return layers

    if layer_scope == "random_layer_subset":
        rng = random.Random(seed)
        count = max(1, len(layers) // 2)
        selected_indices = set(
            rng.sample(
                [index for index, _ in layers],
                count,
            )
        )
        return [
            (index, layer)
            for index, layer in layers
            if index in selected_indices
        ]

    raise ValueError(f"Unknown layer_scope: {layer_scope!r}")


def _selected_indices(
    scores,
    count: int,
    selection: str,
    seed: int,
):
    import torch

    count = min(max(int(count), 1), int(scores.numel()))
    selection = selection.lower()

    if selection == "magnitude":
        return torch.topk(
            scores.float(),
            k=count,
            largest=False,
        ).indices

    if selection == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randperm(
            scores.numel(),
            generator=generator,
        )[:count]

    raise ValueError(f"Unknown structured selection method: {selection!r}")


def _prune_attention_heads(
    layer,
    ratio: float,
    selection: str,
    seed: int,
) -> tuple[int, int, list[int]]:
    import torch

    attention = getattr(layer, "self_attn", None)
    if attention is None:
        raise ValueError("Transformer layer has no self_attn module")

    q_proj = attention.q_proj
    o_proj = attention.o_proj

    num_heads = int(
        getattr(
            attention,
            "num_heads",
            getattr(attention.config, "num_attention_heads"),
        )
    )
    head_dim = int(
        getattr(
            attention,
            "head_dim",
            q_proj.weight.shape[0] // num_heads,
        )
    )

    prune_count = max(1, int(num_heads * ratio))
    scores: list[Any] = []

    for head_index in range(num_heads):
        start = head_index * head_dim
        stop = start + head_dim

        q_score = (
            q_proj.weight.detach()[start:stop]
            .float()
            .square()
            .sum()
        )
        o_score = (
            o_proj.weight.detach()[:, start:stop]
            .float()
            .square()
            .sum()
        )
        scores.append((q_score + o_score).cpu())

    score_tensor = torch.stack(scores)
    indices = _selected_indices(
        score_tensor,
        prune_count,
        selection,
        seed,
    )
    selected_heads = sorted(int(value) for value in indices.tolist())

    changed = 0
    total = q_proj.weight.numel() + o_proj.weight.numel()

    with torch.no_grad():
        for head_index in selected_heads:
            start = head_index * head_dim
            stop = start + head_dim

            q_proj.weight[start:stop].zero_()
            o_proj.weight[:, start:stop].zero_()

            changed += q_proj.weight[start:stop].numel()
            changed += o_proj.weight[:, start:stop].numel()

            if q_proj.bias is not None:
                q_proj.bias[start:stop].zero_()
                changed += q_proj.bias[start:stop].numel()
                total += q_proj.bias.numel()

    return changed, total, selected_heads


def _prune_ffn_channels(
    layer,
    ratio: float,
    selection: str,
    seed: int,
) -> tuple[int, int, list[int]]:
    import torch

    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        raise ValueError("Transformer layer has no mlp module")

    gate_proj = mlp.gate_proj
    up_proj = mlp.up_proj
    down_proj = mlp.down_proj

    channels = int(gate_proj.weight.shape[0])
    prune_count = max(1, int(channels * ratio))

    scores = (
        gate_proj.weight.detach().float().square().sum(dim=1)
        + up_proj.weight.detach().float().square().sum(dim=1)
        + down_proj.weight.detach().float().square().sum(dim=0)
    ).cpu()

    indices = _selected_indices(
        scores,
        prune_count,
        selection,
        seed,
    )
    selected_channels = sorted(int(value) for value in indices.tolist())

    device_indices = indices.to(gate_proj.weight.device)
    changed = 0
    total = (
        gate_proj.weight.numel()
        + up_proj.weight.numel()
        + down_proj.weight.numel()
    )

    with torch.no_grad():
        # Advanced tensor indexing returns a copy in PyTorch, so
        # ``weight[indices].zero_()`` does not reliably mutate the model.
        # index_fill_ performs an actual in-place channel mask.
        gate_proj.weight.index_fill_(0, device_indices, 0)
        up_proj.weight.index_fill_(0, device_indices, 0)
        down_proj.weight.index_fill_(1, device_indices, 0)

        changed += gate_proj.weight[device_indices].numel()
        changed += up_proj.weight[device_indices].numel()
        changed += down_proj.weight[:, device_indices].numel()

        if gate_proj.bias is not None:
            gate_proj.bias.index_fill_(0, device_indices, 0)
            changed += gate_proj.bias[device_indices].numel()
            total += gate_proj.bias.numel()

        if up_proj.bias is not None:
            up_proj.bias.index_fill_(0, device_indices, 0)
            changed += up_proj.bias[device_indices].numel()
            total += up_proj.bias.numel()

    return changed, total, selected_channels


def apply_structured_pruning(
    model,
    configuration: Mapping[str, Any],
    seed: int,
) -> AttackReport:
    ratio = float(configuration.get("ratio", 0.20))
    structure = str(
        configuration.get("structure", "attention_heads")
    ).lower()
    selection = str(
        configuration.get("selection", "magnitude")
    ).lower()
    layer_scope = str(
        configuration.get("layer_scope", "all_layers")
    ).lower()
    implementation = str(
        configuration.get("implementation", "mask")
    ).lower()

    if not 0.0 < ratio < 1.0:
        raise ValueError("Structured pruning ratio must be in (0, 1)")

    if implementation != "mask":
        raise ValueError(
            "Only mask-based structured pruning is currently supported"
        )

    layers = _choose_layers(model, layer_scope, seed)
    changed_parameters = 0
    total_parameters = 0
    selections: dict[str, list[int]] = {}

    for layer_index, layer in layers:
        layer_seed = seed + layer_index * 1009

        if structure == "attention_heads":
            changed, total, selected = _prune_attention_heads(
                layer,
                ratio,
                selection,
                layer_seed,
            )
        elif structure == "ffn_channels":
            changed, total, selected = _prune_ffn_channels(
                layer,
                ratio,
                selection,
                layer_seed,
            )
        else:
            raise ValueError(
                f"Unknown structured pruning structure: {structure!r}"
            )

        changed_parameters += changed
        total_parameters += total
        selections[str(layer_index)] = selected

    return AttackReport(
        name="structured_pruning",
        changed_parameters=changed_parameters,
        total_parameters=total_parameters,
        details={
            "ratio": ratio,
            "structure": structure,
            "selection": selection,
            "layer_scope": layer_scope,
            "implementation": implementation,
            "seed": seed,
            "selected_layers": [
                index for index, _ in layers
            ],
            "selected_structures": selections,
        },
    )


def apply_paper_gaussian_noise(
    model,
    configuration: Mapping[str, Any],
    seed: int,
) -> AttackReport:
    import torch

    std_ratio = float(configuration.get("std_ratio", 0.001))
    target_scope = str(
        configuration.get("target_scope", "all_transformer_layers")
    )
    scale_rule = str(
        configuration.get("scale_rule", "parameter_tensor_std")
    )

    if std_ratio < 0.0:
        raise ValueError("std_ratio must be non-negative")

    if scale_rule != "parameter_tensor_std":
        raise ValueError(f"Unknown Gaussian scale rule: {scale_rule!r}")

    selected = _selected_parameters(model, target_scope)

    if target_scope == "partial_layers":
        layer_numbers = sorted(
            {
                int(name.split(".layers.", 1)[1].split(".", 1)[0])
                for name, _ in selected
                if ".layers." in name
            }
        )
        rng = random.Random(seed)
        count = max(1, len(layer_numbers) // 2)
        retained_layers = set(rng.sample(layer_numbers, count))

        selected = [
            (name, parameter)
            for name, parameter in selected
            if int(
                name.split(".layers.", 1)[1].split(".", 1)[0]
            ) in retained_layers
        ]
    else:
        retained_layers = set()

    generators: dict[str, Any] = {}
    changed_parameters = 0
    total_parameters = 0

    for _, parameter in selected:
        total_parameters += parameter.numel()
        scale = (
            float(parameter.detach().float().std().item())
            * std_ratio
        )

        if scale == 0.0:
            continue

        device_key = str(parameter.device)
        if device_key not in generators:
            generators[device_key] = _torch_generator(
                parameter.device,
                seed,
            )

        noise = torch.randn(
            parameter.shape,
            generator=generators[device_key],
            device=parameter.device,
            dtype=torch.float32,
        )
        noise = noise.to(parameter.dtype) * scale

        with torch.no_grad():
            parameter.add_(noise)

        changed_parameters += parameter.numel()

    return AttackReport(
        name="gaussian_noise",
        changed_parameters=changed_parameters,
        total_parameters=total_parameters,
        details={
            "std_ratio": std_ratio,
            "target_scope": target_scope,
            "scale_rule": scale_rule,
            "seed": seed,
            "selected_tensors": len(selected),
            "selected_layers": sorted(retained_layers),
        },
    )


def apply_manifest_in_memory(
    model,
    manifest: Mapping[str, Any],
) -> AttackReport:
    """Apply one isolated manifest variant to an already loaded model."""
    mode = execution_mode(manifest)

    if mode != "in_memory":
        raise ValueError(
            f"Variant requires execution mode {mode!r}, "
            "so it cannot be applied to an existing model instance"
        )

    configuration = _configuration(manifest)
    family = str(
        manifest.get(
            "family",
            configuration.get("type", ""),
        )
    ).lower()
    seed = _manifest_seed(manifest)

    if family == "unstructured_pruning":
        return apply_unstructured_pruning(
            model,
            configuration,
            seed,
        )

    if family == "structured_pruning":
        return apply_structured_pruning(
            model,
            configuration,
            seed,
        )

    if family == "gaussian_noise":
        return apply_paper_gaussian_noise(
            model,
            configuration,
            seed,
        )

    raise ValueError(f"Unsupported in-memory attack family: {family!r}")
