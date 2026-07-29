from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass
class AttackReport:
    name: str
    changed_parameters: int
    total_parameters: int
    details: dict[str, Any]

    @property
    def changed_fraction(self) -> float:
        return self.changed_parameters / max(self.total_parameters, 1)


def _selected_parameters(model, patterns: Iterable[str] | None = None):
    normalized = tuple(patterns or ())
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        if normalized and not any(pattern in name for pattern in normalized):
            continue
        yield name, parameter


def apply_magnitude_pruning(
    model,
    ratio: float,
    module_patterns: Iterable[str] | None = None,
) -> AttackReport:
    import torch

    if not 0.0 <= ratio < 1.0:
        raise ValueError("Pruning ratio must be in [0, 1)")
    changed = 0
    total = 0
    selected = list(_selected_parameters(model, module_patterns))
    if not selected:
        raise ValueError("No parameters matched pruning selection")
    for _, parameter in selected:
        total += parameter.numel()
        if ratio == 0.0:
            continue
        flat = parameter.detach().abs().reshape(-1)
        count = max(1, int(flat.numel() * ratio))
        threshold = torch.kthvalue(flat.float(), min(count, flat.numel())).values.to(flat.device)
        mask = parameter.detach().abs() <= threshold
        with torch.no_grad():
            parameter.masked_fill_(mask, 0)
        changed += int(mask.sum().item())
    return AttackReport(
        name="magnitude_pruning",
        changed_parameters=changed,
        total_parameters=total,
        details={"ratio": ratio, "module_patterns": list(module_patterns or [])},
    )


def apply_gaussian_noise(
    model,
    std_ratio: float,
    module_patterns: Iterable[str] | None = None,
    seed: int = 42,
) -> AttackReport:
    import torch

    if std_ratio < 0:
        raise ValueError("std_ratio must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    changed = 0
    total = 0
    for _, parameter in _selected_parameters(model, module_patterns):
        total += parameter.numel()
        scale = float(parameter.detach().float().std().item()) * std_ratio
        if scale == 0.0:
            continue
        noise = torch.randn(parameter.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(device=parameter.device, dtype=parameter.dtype) * scale
        with torch.no_grad():
            parameter.add_(noise)
        changed += parameter.numel()
    return AttackReport(
        name="gaussian_noise",
        changed_parameters=changed,
        total_parameters=total,
        details={"std_ratio": std_ratio, "seed": seed, "module_patterns": list(module_patterns or [])},
    )


def apply_attack(model, attack: Mapping[str, Any]) -> AttackReport:
    kind = str(attack.get("type", "none")).lower()
    if kind == "pruning":
        return apply_magnitude_pruning(
            model,
            float(attack.get("ratio", 0.3)),
            attack.get("module_patterns"),
        )
    if kind in {"noise", "gaussian_noise"}:
        return apply_gaussian_noise(
            model,
            float(attack.get("std_ratio", 0.01)),
            attack.get("module_patterns"),
            int(attack.get("seed", 42)),
        )
    if kind == "none":
        total = sum(p.numel() for p in model.parameters())
        return AttackReport("none", 0, total, {})
    raise ValueError(
        f"Attack {kind!r} is not an in-memory attack. Quantization and replacement are configured during model loading."
    )
