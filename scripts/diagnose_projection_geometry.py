from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from _bootstrap import project_path
from llm_integrity.config import load_config
from llm_integrity.inner_variant_sampler import read_jsonl
from llm_integrity.modeling import load_model


def stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p05": ordered[max(0, math.ceil(0.05 * len(ordered)) - 1)],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    config = load_config(project_path(args.config))
    prompts = read_jsonl(project_path(args.prompts))
    bundle = load_model(config["model"])
    try:
        tokenizer = bundle.tokenizer
        all_ids: list[int] = []
        digit_positions = 0
        for row in prompts:
            ids = list(
                tokenizer(
                    str(row["prompt"]),
                    add_special_tokens=False,
                )["input_ids"]
            )
            if tokenizer.chat_template:
                full_ids = list(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": str(row["prompt"])}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                )
                prefix_and_suffix = len(full_ids) - len(ids)
            else:
                prefix_and_suffix = 0
            allowed = max(1, args.max_length - prefix_and_suffix)
            active_ids = ids[:allowed]
            all_ids.extend(active_ids)
            digit_positions += sum(
                any(
                    character.isdigit()
                    for character in tokenizer.decode(
                        [token_id],
                        skip_special_tokens=False,
                    )
                )
                for token_id in active_ids
            )

        matrix = bundle.model.get_input_embeddings().weight.detach().float()
        normalized_matrix = F.normalize(matrix, dim=-1)
        special_ids = set(tokenizer.all_special_ids)
        unique_ids = sorted(set(all_ids))
        per_token: dict[int, dict[str, float | int | str]] = {}
        for token_id in unique_ids:
            source = matrix[token_id]
            source_normalized = normalized_matrix[token_id]
            best_score = -float("inf")
            best_id = -1
            for start in range(0, matrix.shape[0], 4096):
                stop = min(start + 4096, matrix.shape[0])
                scores = normalized_matrix[start:stop] @ source_normalized
                if start <= token_id < stop:
                    scores[token_id - start] = -float("inf")
                for special_id in special_ids:
                    if start <= special_id < stop:
                        scores[special_id - start] = -float("inf")
                score, position = scores.max(dim=0)
                if float(score.item()) > best_score:
                    best_score = float(score.item())
                    best_id = start + int(position.item())
            norm = float(source.norm().item())
            cosine_gap = 1.0 - best_score
            angular_boundary_raw = norm * math.sqrt(max(0.0, cosine_gap / 2.0))
            per_token[token_id] = {
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False),
                "nearest_other_token_id": best_id,
                "nearest_other_token": tokenizer.decode(
                    [best_id], skip_special_tokens=False
                ),
                "embedding_norm": norm,
                "nearest_other_cosine": best_score,
                "cosine_gap": cosine_gap,
                "approx_raw_displacement_to_angular_bisector": (
                    angular_boundary_raw
                ),
            }

        position_rows = [per_token[token_id] for token_id in all_ids]
        report: dict[str, Any] = {
            "prompt_count": len(prompts),
            "active_token_positions": len(all_ids),
            "unique_active_token_ids": len(unique_ids),
            "digit_protected_positions": digit_positions,
            "embedding_norm": stats(
                [float(row["embedding_norm"]) for row in position_rows]
            ),
            "nearest_other_cosine": stats(
                [float(row["nearest_other_cosine"]) for row in position_rows]
            ),
            "cosine_gap": stats(
                [float(row["cosine_gap"]) for row in position_rows]
            ),
            "approx_raw_displacement_to_angular_bisector": stats(
                [
                    float(row["approx_raw_displacement_to_angular_bisector"])
                    for row in position_rows
                ]
            ),
            "positions_with_boundary_below_0_0145": sum(
                float(row["approx_raw_displacement_to_angular_bisector"])
                < 0.0145
                for row in position_rows
            ),
            "projection": "cosine_nearest_embedding",
            "per_unique_token": list(per_token.values()),
        }
        output = project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "per_unique_token"},
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
