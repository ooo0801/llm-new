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
from llm_integrity.inner_variant_sampler import (
    FAMILIES,
    StratifiedVariantSampler,
    load_registry,
    read_jsonl,
)
from llm_integrity.io import write_jsonl
from llm_integrity.modeling import load_model, render_prompt
from llm_integrity.paper_variant_executor import load_manifest_variant


def freeze(bundle) -> None:
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def prepare_embeddings(reference, prompt: str, max_length: int, system_prompt: str | None):
    rendered = render_prompt(reference.tokenizer, prompt, system_prompt)
    encoded = reference.tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    ids = encoded["input_ids"].to(reference.device)
    mask = encoded["attention_mask"].to(reference.device)
    with torch.no_grad():
        embeddings = reference.model.get_input_embeddings()(ids)
    return embeddings.detach().float().requires_grad_(True), mask


def next_logits(bundle, embeddings, mask):
    dtype = bundle.model.get_input_embeddings().weight.dtype
    output = bundle.model(
        inputs_embeds=embeddings.to(dtype),
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    )
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    last = positions.expand_as(mask).masked_fill(~mask.bool(), -1).max(dim=1).values
    return output.logits[
        torch.arange(output.logits.shape[0], device=mask.device),
        last,
    ].float()


def memory_snapshot():
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/paper_aligned_qwen_7b_joint_inner.yaml"),
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-registry", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = read_jsonl(project_path(args.prompts))[: args.max_prompts]
    manifests = read_jsonl(project_path(args.manifest))
    registry = load_registry(project_path(args.adapter_registry))
    weights = config["joint_inner"]["macro"]["families"]
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights=weights,
        adapter_registry=registry,
        seed=args.seed,
    )
    samples = sampler.representative_samples("macro_calibration")
    output = project_path(args.output)
    records: list[dict[str, Any]] = []
    reference = load_model(config["model"])
    freeze(reference)
    try:
        for sample in samples:
            loaded = None
            try:
                loaded = load_manifest_variant(
                    config["model"],
                    sample.manifest,
                    adapter_path=sample.adapter_path,
                )
                freeze(loaded.bundle)
                for prompt_index, row in enumerate(prompts):
                    started = time.time()
                    record: dict[str, Any] = {
                        **sample.trace(),
                        "prompt_id": str(row.get("id", row.get("prompt_id"))),
                        "category": row.get("category", "unknown"),
                        "prompt_index": prompt_index,
                        "valid": False,
                        "status": "started",
                    }
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        embeddings, mask = prepare_embeddings(
                            reference,
                            str(row["prompt"]),
                            args.max_length,
                            config.get("generation", {}).get("system_prompt"),
                        )
                        reference_logits = next_logits(reference, embeddings, mask)
                        variant_logits = next_logits(loaded.bundle, embeddings, mask)
                        score = (variant_logits - reference_logits).square().sum()
                        gradient = torch.autograd.grad(score, embeddings)[0].detach().float()
                        record.update(
                            {
                                "actual_token_length": int(embeddings.shape[1]),
                                "raw_macro_score": float(score.detach().item()),
                                "raw_embedding_gradient_norm": float(gradient.norm().item()),
                                "maximum_token_gradient_norm": float(
                                    gradient.squeeze(0).norm(dim=-1).max().item()
                                ),
                                "gradient_finite": bool(
                                    torch.isfinite(gradient).all().item()
                                ),
                                "gradient_nonzero": bool(gradient.norm().item() > 0),
                                "valid": bool(
                                    torch.isfinite(gradient).all().item()
                                    and gradient.norm().item() > 0
                                    and math.isfinite(float(score.detach().item()))
                                ),
                            }
                        )
                        record["status"] = "passed" if record["valid"] else "invalid_gradient"
                        del embeddings, mask, reference_logits, variant_logits, score, gradient
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
                        write_jsonl(output, records)
                        print(
                            json.dumps(
                                {
                                    "family": sample.family,
                                    "prompt_id": record["prompt_id"],
                                    "status": record["status"],
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
            finally:
                if loaded is not None:
                    loaded.close()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        reference.close()

    valid = [record for record in records if record.get("valid")]
    scores = [float(record["raw_macro_score"]) for record in valid]
    scale = statistics.median(scores) if scores else None
    for record in records:
        if record.get("valid") and scale:
            record["macro_scale"] = scale
            record["normalized_embedding_gradient_norm"] = (
                float(record["raw_embedding_gradient_norm"]) / scale
            )
    write_jsonl(output, records)
    summary = {
        "output": str(output),
        "requested_prompts": len(prompts),
        "families": list(FAMILIES),
        "representative_variants": [sample.trace() for sample in samples],
        "requested_records": len(prompts) * len(samples),
        "records": len(records),
        "valid_records": len(valid),
        "macro_scale_method": "median_raw_macro_score",
        "macro_scale": scale,
        "passed": len(valid) == len(prompts) * len(samples),
    }
    summary_path = project_path(args.summary)
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
