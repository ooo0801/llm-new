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

from llm_integrity.modeling import ModelBundle, load_model, render_prompt
from llm_integrity.paper_variant_executor import load_manifest_variant


FAMILY_VARIANTS = {
    "unstructured_pruning": "validation_unstructured_pruning_c3cd2f6fc89b",
    "structured_pruning": "validation_structured_pruning_9180b6d85a9e",
    "quantization": "validation_quantization_86e3da4994a3",
    "gaussian_noise": "validation_gaussian_noise_6dfcbc3bffb8",
    "finetuning": "validation_finetuning_b9bd14496c78",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
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
    try:
        return registry[variant_id]
    except KeyError as exc:
        raise KeyError(f"No adapter registered for {variant_id}") from exc


def prepare_embeddings(
    reference: ModelBundle,
    prompt: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = render_prompt(reference.tokenizer, prompt)
    encoded = reference.tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(reference.device)
    attention_mask = encoded["attention_mask"].to(reference.device)
    with torch.no_grad():
        embeddings = reference.model.get_input_embeddings()(input_ids)
    return embeddings.detach(), attention_mask


def next_logits(bundle: ModelBundle, embeddings, attention_mask):
    output = bundle.model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return output.logits[:, -1, :].float()


def test_family(
    *,
    family: str,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    adapter_registry: dict[str, str],
    prompt: str,
    steps: int,
    max_length: int,
) -> dict[str, Any]:
    started = time.time()
    reference: ModelBundle | None = None
    loaded_variant = None
    result: dict[str, Any] = {
        "family": family,
        "variant_id": manifest["variant_id"],
        "seed": int(manifest.get("seed", 0)),
        "configuration": manifest.get("configuration", {}),
        "steps_requested": steps,
        "passed": False,
        "supports_inputs_embeds": False,
        "gradient_exists": False,
        "gradient_finite": False,
        "gradient_nonzero": False,
        "model_parameter_gradients_absent": False,
        "memory_leak_suspected": None,
        "steps": [],
        "memory_before_load": memory_snapshot(),
    }

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        reference = load_model(model_config)
        freeze(reference)
        result["memory_after_reference_load"] = memory_snapshot()

        adapter_path = adapter_path_for(
            family,
            str(manifest["variant_id"]),
            adapter_registry,
        )
        loaded_variant = load_manifest_variant(
            model_config,
            manifest,
            adapter_path=adapter_path,
        )
        freeze(loaded_variant.bundle)
        result["execution_report"] = {
            "execution_mode": loaded_variant.report.execution_mode,
            "realized_method": loaded_variant.report.realized_method,
            "isolated_base_reload": loaded_variant.report.isolated_base_reload,
            "details": loaded_variant.report.details,
        }
        result["memory_after_variant_load"] = memory_snapshot()

        initial_embeddings, attention_mask = prepare_embeddings(
            reference,
            prompt,
            max_length,
        )
        if str(reference.device) != str(loaded_variant.bundle.device):
            raise RuntimeError(
                "Reference and variant input devices differ: "
                f"{reference.device} vs {loaded_variant.bundle.device}"
            )

        all_finite = True
        all_nonzero = True
        all_exist = True
        supports_inputs_embeds = True

        for step_index in range(steps):
            embeddings = initial_embeddings.clone().requires_grad_(True)
            attention = attention_mask.to(embeddings.device)

            reference_logits = next_logits(reference, embeddings, attention)
            variant_logits = next_logits(
                loaded_variant.bundle,
                embeddings,
                attention,
            )
            difference = variant_logits - reference_logits
            squared_l2_logits = difference.square().sum()
            mean_squared_logits = difference.square().mean()
            gradient = torch.autograd.grad(
                squared_l2_logits,
                embeddings,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )[0]

            exists = gradient is not None
            finite = bool(
                exists and torch.isfinite(gradient).all().item()
            )
            gradient_norm = (
                float(gradient.float().norm().item())
                if exists
                else 0.0
            )
            nonzero = bool(finite and gradient_norm > 0.0)

            all_exist = all_exist and exists
            all_finite = all_finite and finite
            all_nonzero = all_nonzero and nonzero
            result["steps"].append(
                {
                    "step": step_index + 1,
                    "squared_l2_logits": float(
                        squared_l2_logits.detach().item()
                    ),
                    "mean_squared_logits": float(
                        mean_squared_logits.detach().item()
                    ),
                    "gradient_norm": gradient_norm,
                    "gradient_finite": finite,
                    "gradient_nonzero": nonzero,
                    "memory": memory_snapshot(),
                }
            )

            del (
                gradient,
                squared_l2_logits,
                mean_squared_logits,
                difference,
                variant_logits,
                reference_logits,
                embeddings,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        parameter_gradient_count = sum(
            parameter.grad is not None
            for bundle in (reference, loaded_variant.bundle)
            for parameter in bundle.model.parameters()
        )
        result["supports_inputs_embeds"] = supports_inputs_embeds
        result["gradient_exists"] = all_exist
        result["gradient_finite"] = all_finite
        result["gradient_nonzero"] = all_nonzero
        result["model_parameter_gradient_tensor_count"] = (
            parameter_gradient_count
        )
        result["model_parameter_gradients_absent"] = (
            parameter_gradient_count == 0
        )

        allocated = [
            int(item["memory"]["allocated_bytes"])
            for item in result["steps"]
        ]
        reserved = [
            int(item["memory"]["reserved_bytes"])
            for item in result["steps"]
        ]
        allocated_growth = allocated[-1] - allocated[0]
        reserved_growth = reserved[-1] - reserved[0]
        result["memory_growth"] = {
            "allocated_bytes_first_to_last": allocated_growth,
            "reserved_bytes_first_to_last": reserved_growth,
        }
        result["memory_leak_suspected"] = bool(
            allocated_growth > 256 * 1024**2
            or reserved_growth > 512 * 1024**2
        )
        result["passed"] = bool(
            supports_inputs_embeds
            and all_exist
            and all_finite
            and all_nonzero
            and parameter_gradient_count == 0
            and not result["memory_leak_suspected"]
        )

    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        if isinstance(exc, TypeError) and "inputs_embeds" in str(exc):
            result["supports_inputs_embeds"] = False
    finally:
        if loaded_variant is not None:
            loaded_variant.close()
        if reference is not None:
            reference.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["memory_after_close"] = memory_snapshot()
        result["elapsed_seconds"] = time.time() - started

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-registry", type=Path, required=True)
    parser.add_argument("--family", choices=sorted(FAMILY_VARIANTS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--prompt",
        default="请用三点简要说明机器学习模型量化的主要影响。",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config = dict(config["model"])
    manifests = read_jsonl(args.manifest)
    variant_id = FAMILY_VARIANTS[args.family]
    selected = [
        item for item in manifests if item.get("variant_id") == variant_id
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one manifest for {variant_id}, got {len(selected)}"
        )
    registry = json.loads(
        args.adapter_registry.read_text(encoding="utf-8")
    )

    result = test_family(
        family=args.family,
        manifest=selected[0],
        model_config=model_config,
        adapter_registry=registry,
        prompt=args.prompt,
        steps=max(1, args.steps),
        max_length=max(8, args.max_length),
    )
    result["tested_at"] = datetime.now().astimezone().isoformat()
    result["torch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    result["cuda_version"] = torch.version.cuda
    result["gpu_name"] = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None
    )
    result["gpu_total_memory_bytes"] = (
        int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else 0
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "family": result["family"],
                "variant_id": result["variant_id"],
                "passed": result["passed"],
                "supports_inputs_embeds": result["supports_inputs_embeds"],
                "gradient_exists": result["gradient_exists"],
                "gradient_finite": result["gradient_finite"],
                "gradient_nonzero": result["gradient_nonzero"],
                "memory_leak_suspected": result["memory_leak_suspected"],
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
