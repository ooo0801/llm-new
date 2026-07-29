from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_jsonl
from llm_integrity.modeling import load_model
from llm_integrity.prompt_optimization import SoftPromptOptimizer
from llm_integrity.variants import load_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--attack", required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    config = load_config(args.config)
    attack = next(item for item in config["attacks"] if item["name"] == args.attack)
    reference = load_model(config["model"])
    modified, _ = load_variant(config["model"], attack)
    settings = config["sensitivity"].get("prompt_optimization", {})
    optimizer = SoftPromptOptimizer(
        reference,
        modified,
        int(settings.get("candidate_tokens", 64)),
        float(settings.get("learning_rate", 0.1)),
        int(settings.get("steps", 30)),
        float(settings.get("initial_temperature", 1.0)),
        float(settings.get("anneal", 0.95)),
        float(settings.get("epsilon", 2.0)),
        float(settings.get("semantic_weight", 0.1)),
    )
    rows = read_jsonl(project_path(config["data"]["candidate"]))[: args.count]
    outputs = []
    try:
        for row in rows:
            result = optimizer.optimize(row["prompt"])
            outputs.append(row | result.__dict__)
    finally:
        modified.close()
        reference.close()
    output = project_path(config["output_dir"]) / "optimized_prompts.jsonl"
    write_jsonl(output, outputs)
    print(json.dumps({"output": str(output), "rows": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
