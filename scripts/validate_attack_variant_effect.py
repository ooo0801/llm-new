from __future__ import annotations

import argparse
import gc
import json
import math

import torch

from _bootstrap import project_path
from llm_integrity.config import load_config
from llm_integrity.inner_variant_sampler import load_registry, read_jsonl
from llm_integrity.modeling import load_model, render_prompt
from llm_integrity.paper_variant_executor import load_manifest_variant


def freeze(bundle) -> None:
    bundle.model.eval()
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def next_logits(bundle, embeddings, attention_mask):
    dtype = bundle.model.get_input_embeddings().weight.dtype
    output = bundle.model(
        inputs_embeds=embeddings.to(dtype),
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return output.logits[:, -1, :].float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-registry", default=None)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    manifests = read_jsonl(project_path(args.manifest))
    matches = [
        row
        for row in manifests
        if str(row.get("variant_id")) == args.variant_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one manifest for {args.variant_id!r}, "
            f"found {len(matches)}"
        )
    manifest = matches[0]
    registry = load_registry(
        project_path(args.adapter_registry)
        if args.adapter_registry
        else None
    )
    prompts = read_jsonl(project_path(args.prompts))
    if not prompts:
        raise ValueError("Prompt file is empty")
    prompt = str(prompts[0]["prompt"])

    reference = load_model(config["model"])
    loaded = None
    try:
        freeze(reference)
        rendered = render_prompt(
            reference.tokenizer,
            prompt,
            config.get("generation", {}).get("system_prompt"),
        )
        encoded = reference.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        input_ids = encoded["input_ids"].to(reference.device)
        attention_mask = encoded["attention_mask"].to(reference.device)
        with torch.no_grad():
            embeddings = (
                reference.model.get_input_embeddings()(input_ids)
                .detach()
                .float()
            )
        embeddings.requires_grad_(True)

        loaded = load_manifest_variant(
            config["model"],
            manifest,
            adapter_path=registry.get(args.variant_id),
        )
        freeze(loaded.bundle)
        reference_logits = next_logits(
            reference,
            embeddings,
            attention_mask,
        )
        variant_logits = next_logits(
            loaded.bundle,
            embeddings,
            attention_mask,
        )
        score = (variant_logits - reference_logits).square().sum()
        gradient = torch.autograd.grad(score, embeddings)[0]
        score_value = float(score.detach().item())
        gradient_norm = float(gradient.detach().float().norm().item())
        details = loaded.report.details
        changed_parameters = int(details.get("changed_parameters", 0))
        passed = bool(
            changed_parameters > 0
            and math.isfinite(score_value)
            and score_value > 0
            and math.isfinite(gradient_norm)
            and gradient_norm > 0
        )
        output = {
            "variant_id": args.variant_id,
            "family": manifest.get("family"),
            "configuration": manifest.get("configuration"),
            "prompt_id": prompts[0].get(
                "id",
                prompts[0].get("prompt_id"),
            ),
            "changed_parameters_reported": changed_parameters,
            "macro_score_raw": score_value,
            "embedding_gradient_norm_raw": gradient_norm,
            "gradient_finite": bool(torch.isfinite(gradient).all().item()),
            "attack_report": details,
            "passed": passed,
        }
        output_path = project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        if loaded is not None:
            loaded.close()
        reference.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
