from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config, validate_config
from llm_integrity.io import read_jsonl
from llm_integrity.modeling import generate_texts, load_model, model_metadata, next_token_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/debug_qwen_0.5b.yaml"))
    parser.add_argument("--rows", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    rows = read_jsonl(project_path(config["data"]["smoke"]))[: args.rows]
    bundle = load_model(config["model"])
    try:
        prompts = [row["prompt"] for row in rows]
        logits = next_token_logits(bundle, prompts, int(config["sensitivity"]["max_length"]))
        responses = generate_texts(bundle, prompts, config["generation"])
        value = {
            "model": model_metadata(bundle),
            "rows": len(rows),
            "logits_shape": list(logits.shape),
            "responses": responses,
        }
        print(json.dumps(value, ensure_ascii=False, indent=2))
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
