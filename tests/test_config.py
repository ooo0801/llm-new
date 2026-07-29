from pathlib import Path

from llm_integrity.config import load_config, validate_config


def test_debug_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/debug_qwen_0.5b.yaml")
    validate_config(config)
