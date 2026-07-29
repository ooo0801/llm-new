from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path

from llm_integrity.activations import TransformerActivationProfiler
from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_jsonl
from llm_integrity.modeling import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--pool-size", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    scores = read_jsonl(output_dir / "prompt_scores.jsonl")
    rank_key = "hybrid" if scores and scores[0].get("hybrid") is not None else "macro"
    scores = sorted(scores, key=lambda row: float(row.get(rank_key) or 0.0), reverse=True)[: args.pool_size]
    settings = config.get("activations", {})
    profiler = TransformerActivationProfiler(
        float(settings.get("attention_entropy_fraction", 0.70)),
        float(settings.get("ffn_quantile", 0.95)),
        float(settings.get("residual_threshold", 0.50)),
        int(settings.get("max_ffn_units_per_layer", 128)),
    )
    bundle = load_model(config["model"])
    profiles = []
    try:
        for index, row in enumerate(scores, 1):
            print(f"[activation] {index}/{len(scores)} {row['id']}", flush=True)
            profiles.append(profiler.profile(bundle, row["id"], row["prompt"], int(config["sensitivity"]["max_length"])).as_dict())
    finally:
        bundle.close()
    output = output_dir / "activation_profiles.jsonl"
    write_jsonl(output, profiles)
    print(json.dumps({"output": str(output), "rows": len(profiles)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
