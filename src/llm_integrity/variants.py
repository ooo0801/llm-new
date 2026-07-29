from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .attacks import AttackReport, apply_attack
from .modeling import ModelBundle, load_model


def load_variant(
    base_model_config: Mapping[str, Any],
    attack_config: Mapping[str, Any],
) -> tuple[ModelBundle, AttackReport | None]:
    """Load one attacked model at a time to keep GPU memory bounded."""
    kind = str(attack_config.get("type", "none")).lower()
    config = deepcopy(dict(base_model_config))
    if kind == "replacement":
        config.update(dict(attack_config["model"]))
        return load_model(config), None
    if kind in {"int8", "8bit", "int4", "4bit"}:
        config["quantization"] = kind
        if "compute_dtype" in attack_config:
            config["compute_dtype"] = attack_config["compute_dtype"]
        return load_model(config), None
    bundle = load_model(config)
    if kind == "lora":
        from peft import PeftModel

        bundle.model = PeftModel.from_pretrained(bundle.model, str(attack_config["adapter_path"]))
        bundle.model.eval()
        return bundle, None
    report = apply_attack(bundle.model, attack_config)
    return bundle, report
