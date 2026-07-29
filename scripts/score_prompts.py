from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config, validate_config
from llm_integrity.distances import logit_distance
from llm_integrity.io import read_jsonl, set_seed, write_jsonl
from llm_integrity.modeling import load_model, next_token_logits
from llm_integrity.sensitivity import PromptSensitivity, add_micro_scores, combine_hybrid
from llm_integrity.variants import load_variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate prompts against configured model modifications")
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--with-micro", action="store_true")
    parser.add_argument("--attacks", nargs="*", help="Optional attack-name filter")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    set_seed(int(config.get("seed", 42)))
    rows = read_jsonl(project_path(config["data"]["candidate"]))
    limit = int(config["data"].get("max_candidate_rows", len(rows)))
    rows = rows[:limit]
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(config["sensitivity"].get("batch_size", 4))
    max_length = int(config["sensitivity"].get("max_length", 512))
    metric = str(config["sensitivity"].get("metric", "js"))

    reference = load_model(config["model"])
    reference_batches = []
    try:
        for start in range(0, len(rows), batch_size):
            prompts = [row["prompt"] for row in rows[start : start + batch_size]]
            reference_batches.append(next_token_logits(reference, prompts, max_length))
    finally:
        reference.close()

    scores = [PromptSensitivity(str(row["id"]), str(row["prompt"]), macro_by_attack={}) for row in rows]
    attacks = [a for a in config.get("attacks", []) if not args.attacks or a["name"] in args.attacks]
    if not attacks:
        raise ValueError("No attacks selected")
    for attack in attacks:
        print(f"[attack] {attack['name']}", flush=True)
        variant, report = load_variant(config["model"], attack)
        try:
            for batch_index, start in enumerate(range(0, len(rows), batch_size)):
                prompts = [row["prompt"] for row in rows[start : start + batch_size]]
                candidate_logits = next_token_logits(variant, prompts, max_length)
                values = logit_distance(reference_batches[batch_index], candidate_logits, metric).numpy()
                for offset, value in enumerate(values):
                    scores[start + offset].macro_by_attack[attack["name"]] = float(value)
        finally:
            variant.close()
        gc.collect()
    for score in scores:
        score.macro = float(np.mean(list(score.macro_by_attack.values())))

    if args.with_micro:
        reference = load_model(config["model"])
        try:
            add_micro_scores(
                scores,
                reference,
                config["sensitivity"].get("micro_parameter_patterns", []),
                config["sensitivity"].get("max_parameter_tensors"),
                max_length,
            )
            combine_hybrid(scores, float(config["sensitivity"].get("hybrid_beta", 1.0)))
        finally:
            reference.close()
    output = output_dir / "prompt_scores.jsonl"
    write_jsonl(output, [score.as_dict() | {"category": row.get("category")} for score, row in zip(scores, rows)])
    print(json.dumps({"output": str(output), "rows": len(scores), "attacks": [a["name"] for a in attacks]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
