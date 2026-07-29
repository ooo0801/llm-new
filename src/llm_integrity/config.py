from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ConfigError("Top-level configuration must be a mapping")
    value["_config_path"] = str(path.resolve())
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        current = current[part]
    return current


def validate_config(config: Mapping[str, Any]) -> None:
    require(config, "model.name")
    require(config, "data.candidate")
    require(config, "output_dir")
    selected = int(config.get("fingerprint", {}).get("selected_size", 0))
    if selected <= 0:
        raise ConfigError("fingerprint.selected_size must be positive")
    alpha = float(config.get("statistics", {}).get("alpha", 0.05))
    if not 0.0 < alpha < 1.0:
        raise ConfigError("statistics.alpha must be between 0 and 1")
