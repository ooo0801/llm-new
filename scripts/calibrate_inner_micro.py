from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.inner_micro_proxy import (
    differentiable_block_micro_proxy,
    discover_micro_blocks,
    representative_layers,
    select_micro_block,
)
from llm_integrity.io import read_jsonl, write_jsonl
from llm_integrity.modeling import load_model, render_prompt


BLOCK_TYPES = ("q_proj", "v_proj", "down_proj")


def memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def prompt_embeddings(bundle, prompt: str, max_length: int, system_prompt: str | None):
    rendered = render_prompt(bundle.tokenizer, prompt, system_prompt)
    encoded = bundle.tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded["attention_mask"].to(bundle.device)
    with torch.no_grad():
        embeddings = bundle.model.get_input_embeddings()(input_ids)
    return embeddings.detach().float().requires_grad_(True), attention_mask


def robust_threshold(values: list[float], multiplier: float = 5.0) -> dict[str, float]:
    if not values:
        raise ValueError("No values supplied")
    epsilon = 1e-30
    logged = [math.log(max(value, epsilon)) for value in values]
    center = statistics.median(logged)
    deviations = [abs(value - center) for value in logged]
    mad = statistics.median(deviations)
    threshold = math.exp(center + multiplier * max(mad, 1e-12))
    sorted_values = sorted(values)
    p95_index = min(
        len(sorted_values) - 1,
        max(0, math.ceil(0.95 * len(sorted_values)) - 1),
    )
    return {
        "log_median": center,
        "log_mad": mad,
        "mad_multiplier": multiplier,
        "threshold": threshold,
        "empirical_p95": float(sorted_values[p95_index]),
        "maximum": float(sorted_values[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/paper_aligned_qwen_7b_joint_inner.yaml"),
    )
    parser.add_argument(
        "--prompts",
        default=(
            "results/paper_aligned_qwen_7b/prompt_optimization/"
            "seeds_stratified_k60.jsonl"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--probes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--representative-layer-count", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_jsonl(project_path(args.prompts))[: args.max_prompts]
    output = project_path(args.output)
    summary_path = project_path(args.summary)
    completed = (
        read_jsonl(output)
        if args.resume and output.exists()
        else []
    )
    completed_keys = {
        (
            str(row["prompt_id"]),
            int(row["layer_id"]),
            str(row["block_type"]),
        )
        for row in completed
    }
    bundle = load_model(config["model"])
    records = list(completed)
    try:
        blocks = discover_micro_blocks(bundle.model)
        layers = representative_layers(
            blocks,
            args.representative_layer_count,
        )
        system_prompt = config.get("generation", {}).get("system_prompt")
        for prompt_index, row in enumerate(rows):
            prompt_id = str(row.get("id", row.get("prompt_id")))
            layer_id = layers[prompt_index % len(layers)]
            for block_index, block_type in enumerate(BLOCK_TYPES):
                key = (prompt_id, layer_id, block_type)
                if key in completed_keys:
                    continue
                started = time.time()
                record: dict[str, Any] = {
                    "prompt_id": prompt_id,
                    "category": row.get("category", "unknown"),
                    "prompt_index": prompt_index,
                    "block_type": block_type,
                    "layer_id": layer_id,
                    "probe_seed": (
                        args.seed
                        + prompt_index * len(BLOCK_TYPES)
                        + block_index
                    ),
                    "probe_count": args.probes,
                    "status": "started",
                    "valid": False,
                }
                try:
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    embeddings, attention_mask = prompt_embeddings(
                        bundle,
                        str(row["prompt"]),
                        args.max_length,
                        system_prompt,
                    )
                    block = select_micro_block(
                        blocks,
                        block_type=block_type,
                        layer_id=layer_id,
                    )
                    result = differentiable_block_micro_proxy(
                        bundle,
                        embeddings,
                        attention_mask,
                        block=block,
                        probes=args.probes,
                        seed=record["probe_seed"],
                    )
                    record.update(result.metadata())
                    record["actual_token_length"] = int(embeddings.shape[1])
                    record["valid"] = bool(
                        result.gradient_finite
                        and result.gradient_nonzero
                        and result.parameter_gradient_finite
                        and result.parameter_gradient_nonzero
                    )
                    record["status"] = (
                        "passed" if record["valid"] else "invalid_gradient"
                    )
                    del result, embeddings, attention_mask
                except torch.OutOfMemoryError as exc:
                    record.update(
                        {
                            "status": "cuda_out_of_memory",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    record["memory"] = memory_snapshot()
                    record["elapsed_seconds"] = time.time() - started
                    records.append(record)
                    completed_keys.add(key)
                    write_jsonl(output, records)
                    print(
                        json.dumps(
                            {
                                "prompt_id": prompt_id,
                                "block_type": block_type,
                                "layer_id": layer_id,
                                "status": record["status"],
                                "output": str(output),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        bundle.close()

    valid = [record for record in records if record.get("valid")]
    scores = [float(record["raw_micro_score"]) for record in valid]
    scale = statistics.median(scores) if scores else None
    normalized_norms = (
        [
            float(record["raw_embedding_gradient_norm"]) / scale
            for record in valid
        ]
        if scale is not None and scale > 0
        else []
    )
    for record in records:
        if record.get("valid") and scale:
            record["micro_scale"] = scale
            record["normalized_embedding_gradient_norm"] = (
                float(record["raw_embedding_gradient_norm"]) / scale
            )
    write_jsonl(output, records)
    summary = {
        "input": str(project_path(args.prompts)),
        "output": str(output),
        "requested_prompts": len(rows),
        "requested_records": len(rows) * len(BLOCK_TYPES),
        "records": len(records),
        "valid_records": len(valid),
        "invalid_records": len(records) - len(valid),
        "block_types": list(BLOCK_TYPES),
        "representative_layers": layers,
        "max_length": args.max_length,
        "probes": args.probes,
        "seed": args.seed,
        "micro_scale_method": "median_raw_micro_score",
        "micro_scale": scale,
        "normalized_gradient_threshold_method": "log_median_plus_5_mad",
        "normalized_gradient_threshold": (
            robust_threshold(normalized_norms)
            if normalized_norms
            else None
        ),
        "passed": len(valid) == len(rows) * len(BLOCK_TYPES),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
