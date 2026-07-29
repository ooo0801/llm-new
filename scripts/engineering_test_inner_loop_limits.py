from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from llm_integrity.modeling import ModelBundle, load_model
from llm_integrity.paper_variant_executor import load_manifest_variant


FAMILY_VARIANTS = {
    "unstructured_pruning": "validation_unstructured_pruning_c3cd2f6fc89b",
    "structured_pruning": "validation_structured_pruning_9180b6d85a9e",
    "quantization": "validation_quantization_86e3da4994a3",
    "gaussian_noise": "validation_gaussian_noise_6dfcbc3bffb8",
    "finetuning": "validation_finetuning_b9bd14496c78",
}

MIB = 1024**2
GIB = 1024**3


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def gpu_memory() -> dict[str, float | int]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "free_bytes": 0,
            "total_bytes": 0,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def freeze(bundle: ModelBundle) -> None:
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def adapter_path_for(
    family: str,
    variant_id: str,
    registry: dict[str, str],
) -> str | None:
    if family != "finetuning":
        return None
    return registry[variant_id]


def exact_length_embeddings(
    reference: ModelBundle,
    token_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_text = (
        "请分析云端语言模型完整性验证中提示词敏感性、参数修改和输出差异之间的关系。"
    )
    seed_ids = reference.tokenizer(
        seed_text,
        add_special_tokens=False,
    )["input_ids"]
    if not seed_ids:
        raise RuntimeError("Tokenizer returned no seed tokens")
    repeats = math.ceil(token_length / len(seed_ids))
    exact_ids = (seed_ids * repeats)[:token_length]
    input_ids = torch.tensor(
        [exact_ids],
        dtype=torch.long,
        device=reference.device,
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        embeddings = reference.model.get_input_embeddings()(input_ids)
    return embeddings.detach().float(), attention_mask


def next_logits(
    bundle: ModelBundle,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    dtype = bundle.model.get_input_embeddings().weight.dtype
    output = bundle.model(
        inputs_embeds=embeddings.to(dtype),
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return output.logits[:, -1, :].float()


def raw_l2(
    reference: ModelBundle,
    variant: ModelBundle,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    reference_logits = next_logits(reference, embeddings, attention_mask)
    variant_logits = next_logits(variant, embeddings, attention_mask)
    return (variant_logits - reference_logits).square().sum()


def cleanup_tensors() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_boundary(
    reference: ModelBundle,
    variant: ModelBundle,
    lengths: list[int],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    previous_oom = False
    for token_length in lengths:
        if previous_oom:
            records.append(
                {
                    "token_length": token_length,
                    "status": "skipped_after_shorter_length_oom",
                    "passed": False,
                }
            )
            continue
        record: dict[str, Any] = {
            "token_length": token_length,
            "passed": False,
            "status": "started",
        }
        started = time.time()
        try:
            cleanup_tensors()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            embeddings, attention_mask = exact_length_embeddings(
                reference,
                token_length,
            )
            embeddings.requires_grad_(True)
            score = raw_l2(
                reference,
                variant,
                embeddings,
                attention_mask,
            )
            gradient = torch.autograd.grad(
                score,
                embeddings,
                create_graph=False,
                retain_graph=False,
            )[0]
            memory = gpu_memory()
            record.update(
                {
                    "status": "passed",
                    "passed": True,
                    "actual_token_length": int(embeddings.shape[1]),
                    "raw_squared_l2_logits": float(score.detach().item()),
                    "raw_gradient_norm": float(
                        gradient.detach().float().norm().item()
                    ),
                    "gradient_finite": bool(
                        torch.isfinite(gradient).all().item()
                    ),
                    "peak_allocated_bytes": int(
                        memory["peak_allocated_bytes"]
                    ),
                    "peak_allocated_gib": float(
                        memory["peak_allocated_bytes"] / GIB
                    ),
                    "peak_reserved_bytes": int(
                        memory["peak_reserved_bytes"]
                    ),
                    "peak_reserved_gib": float(
                        memory["peak_reserved_bytes"] / GIB
                    ),
                    "free_bytes_at_measurement": int(memory["free_bytes"]),
                    "free_gib_at_measurement": float(
                        memory["free_bytes"] / GIB
                    ),
                }
            )
            del gradient, score, embeddings, attention_mask
        except torch.OutOfMemoryError as exc:
            previous_oom = True
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
            cleanup_tensors()
            record["memory_after_cleanup"] = gpu_memory()
            record["elapsed_seconds"] = time.time() - started
            records.append(record)
    safe_lengths = [
        int(record["token_length"])
        for record in records
        if record.get("passed")
    ]
    return {
        "mode": "memory_boundary",
        "records": records,
        "maximum_tested_safe_token_length": max(safe_lengths, default=None),
        "all_requested_lengths_passed": all(
            record.get("passed", False) for record in records
        ),
    }


def run_update(
    reference: ModelBundle,
    variant: ModelBundle,
    token_length: int,
    global_scale: float,
    gradient_clip_norm: float,
    step_sizes: list[float],
) -> dict[str, Any]:
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("global_scale must be finite and positive")
    embeddings, attention_mask = exact_length_embeddings(
        reference,
        token_length,
    )
    initial = embeddings.detach().clone()
    embeddings.requires_grad_(True)

    score_before = raw_l2(
        reference,
        variant,
        embeddings,
        attention_mask,
    )
    normalized_before = score_before / global_scale
    raw_gradient = torch.autograd.grad(
        normalized_before,
        embeddings,
        create_graph=False,
        retain_graph=False,
    )[0]
    raw_gradient_norm = raw_gradient.float().norm()
    clip_coefficient = min(
        1.0,
        gradient_clip_norm / (float(raw_gradient_norm.item()) + 1e-12),
    )
    clipped_gradient = raw_gradient * clip_coefficient
    clipped_gradient_norm = clipped_gradient.float().norm()

    trials: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    best: dict[str, Any] | None = None
    for step_size in step_sizes:
        with torch.no_grad():
            candidate = initial + step_size * clipped_gradient
            candidate_score_tensor = raw_l2(
                reference,
                variant,
                candidate,
                attention_mask,
            )
            candidate_score = float(candidate_score_tensor.item())
            delta = candidate - initial
            trial = {
                "step_size": float(step_size),
                "raw_squared_l2_logits_after": candidate_score,
                "raw_score_gain": candidate_score
                - float(score_before.detach().item()),
                "embedding_delta_norm": float(delta.float().norm().item()),
                "relative_embedding_delta": float(
                    delta.float().norm().item()
                    / (initial.float().norm().item() + 1e-12)
                ),
                "embedding_finite": bool(
                    torch.isfinite(candidate).all().item()
                ),
            }
        trials.append(trial)
        if best is None or trial["raw_score_gain"] > best["raw_score_gain"]:
            best = trial
        if (
            selected is None
            and trial["embedding_finite"]
            and trial["embedding_delta_norm"] > 0
            and trial["raw_score_gain"] > 0
        ):
            selected = trial

    if selected is None:
        selected = best
    assert selected is not None
    operation_succeeded = bool(
        torch.isfinite(raw_gradient).all().item()
        and float(raw_gradient_norm.item()) > 0
        and float(clipped_gradient_norm.item())
        <= gradient_clip_norm * (1.0 + 1e-5)
        and selected["embedding_finite"]
        and selected["embedding_delta_norm"] > 0
        and selected["raw_score_gain"] > 0
    )
    result = {
        "mode": "one_step_update",
        "token_length": token_length,
        "normalization": {
            "method": "single_fixed_global_scale",
            "global_scale": float(global_scale),
        },
        "gradient_clip_norm": float(gradient_clip_norm),
        "raw_squared_l2_logits_before": float(score_before.detach().item()),
        "normalized_objective_before": float(
            normalized_before.detach().item()
        ),
        "gradient_norm_before_clip": float(raw_gradient_norm.item()),
        "clip_coefficient": float(clip_coefficient),
        "gradient_norm_after_clip": float(clipped_gradient_norm.item()),
        "trials": trials,
        "selected_update": selected,
        "operation_succeeded": operation_succeeded,
        "memory": gpu_memory(),
    }
    del (
        clipped_gradient,
        raw_gradient,
        normalized_before,
        score_before,
        embeddings,
        attention_mask,
        initial,
    )
    cleanup_tensors()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-registry", type=Path, required=True)
    parser.add_argument("--family", choices=sorted(FAMILY_VARIANTS), required=True)
    parser.add_argument("--mode", choices=("boundary", "update"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--token-length", type=int, default=64)
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--step-sizes",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0],
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    manifests = read_jsonl(args.manifest)
    variant_id = FAMILY_VARIANTS[args.family]
    selected_manifests = [
        item for item in manifests if item.get("variant_id") == variant_id
    ]
    if len(selected_manifests) != 1:
        raise ValueError(
            f"Expected one manifest for {variant_id}, "
            f"got {len(selected_manifests)}"
        )
    manifest = selected_manifests[0]
    registry = json.loads(
        args.adapter_registry.read_text(encoding="utf-8")
    )

    reference: ModelBundle | None = None
    loaded_variant = None
    started = time.time()
    result: dict[str, Any] = {
        "family": args.family,
        "variant_id": variant_id,
        "seed": int(manifest.get("seed", 0)),
        "configuration": manifest.get("configuration", {}),
        "passed": False,
        "memory_before_load": gpu_memory(),
    }
    try:
        cleanup_tensors()
        reference = load_model(model_config)
        freeze(reference)
        loaded_variant = load_manifest_variant(
            model_config,
            manifest,
            adapter_path=adapter_path_for(
                args.family,
                variant_id,
                registry,
            ),
        )
        freeze(loaded_variant.bundle)
        result["execution_report"] = {
            "execution_mode": loaded_variant.report.execution_mode,
            "realized_method": loaded_variant.report.realized_method,
            "isolated_base_reload": loaded_variant.report.isolated_base_reload,
            "details": loaded_variant.report.details,
        }
        result["memory_after_models_loaded"] = gpu_memory()
        if args.mode == "boundary":
            mode_result = run_boundary(
                reference,
                loaded_variant.bundle,
                sorted(set(args.lengths)),
            )
            result["passed"] = bool(
                mode_result["maximum_tested_safe_token_length"] is not None
            )
        else:
            mode_result = run_update(
                reference,
                loaded_variant.bundle,
                args.token_length,
                args.global_scale,
                args.gradient_clip_norm,
                args.step_sizes,
            )
            result["passed"] = bool(mode_result["operation_succeeded"])
        result.update(mode_result)
    except Exception as exc:
        result.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if loaded_variant is not None:
            loaded_variant.close()
        if reference is not None:
            reference.close()
        cleanup_tensors()
        result["memory_after_close"] = gpu_memory()
        result["elapsed_seconds"] = time.time() - started
        result["tested_at"] = datetime.now().astimezone().isoformat()
        result["gpu_name"] = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "family": args.family,
                    "mode": args.mode,
                    "passed": result["passed"],
                    "output": str(args.output),
                    "error_type": result.get("error_type"),
                    "error": result.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
