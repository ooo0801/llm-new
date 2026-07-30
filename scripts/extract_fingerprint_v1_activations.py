from __future__ import annotations

import argparse
from collections import defaultdict
import json

from _bootstrap import ROOT, project_path

from llm_integrity.activations import TransformerActivationProfiler
from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_json, write_jsonl
from llm_integrity.modeling import load_model, next_token_logits


def hook_count(model) -> int:
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks)
        for module in model.modules()
    )


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract stable activation coverage for V1")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v1_qwen_7b.yaml"))
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    prompts = read_jsonl(project_path(config["data"]["strict16_manifest"]))
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    settings = config["activations"]
    repetitions = int(args.repetitions or settings["repetitions"])
    profiler = TransformerActivationProfiler(
        attention_entropy_fraction=float(settings["attention_entropy_fraction"]),
        ffn_quantile=float(settings["ffn_quantile"]),
        residual_threshold=float(settings["residual_threshold"]),
        max_ffn_units_per_layer=int(settings["max_ffn_units_per_layer"]),
    )
    bundle = load_model(config["model"])
    profiles = []
    invariance = None
    try:
        initial_hooks = hook_count(bundle.model)
        for repeat in range(repetitions):
            for index, row in enumerate(prompts, 1):
                if repeat == 0 and index == 1:
                    before = next_token_logits(bundle, [row["prompt"]], int(settings["max_length"]))
                profile = profiler.profile(
                    bundle,
                    str(row["id"]),
                    str(row["prompt"]),
                    int(settings["max_length"]),
                    config["generation"].get("system_prompt"),
                ).as_dict()
                if hook_count(bundle.model) != initial_hooks:
                    raise RuntimeError("Activation profiler leaked forward hooks")
                if repeat == 0 and index == 1:
                    after = next_token_logits(bundle, [row["prompt"]], int(settings["max_length"]))
                    import torch

                    invariance = {
                        "allclose": bool(torch.allclose(before, after, rtol=0.0, atol=0.0)),
                        "max_abs_difference": float((before - after).abs().max().item()),
                    }
                    if not invariance["allclose"]:
                        raise RuntimeError("Profiling changed reference model logits")
                profile["profile_repeat"] = repeat
                profile["category"] = row.get("category")
                profile["prompt_sha256"] = row.get("prompt_sha256")
                profiles.append(profile)
                print(json.dumps({"stage": "activation", "repeat": repeat, "index": index, "total": len(prompts), "id": row["id"], "components": len(profile["components"])}, ensure_ascii=False), flush=True)
    finally:
        bundle.close()

    grouped: dict[str, list[set[str]]] = defaultdict(list)
    for row in profiles:
        grouped[str(row["id"])].append(set(row["components"]))
    stability = {}
    for prompt_id, values in grouped.items():
        if len(values) != repetitions:
            raise RuntimeError(f"Missing activation repeat for {prompt_id}")
        pairwise = [jaccard(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
        stability[prompt_id] = min(pairwise) if pairwise else 1.0
        if stability[prompt_id] < 0.999:
            raise RuntimeError(f"Activation components are unstable for {prompt_id}: {stability[prompt_id]}")
    output_dir = project_path(config["output_dir"])
    output = output_dir / "activation_profiles.jsonl"
    write_jsonl(output, profiles)
    audit = {
        "prompts": len(prompts),
        "repetitions": repetitions,
        "rows": len(profiles),
        "minimum_jaccard": min(stability.values()),
        "per_prompt_jaccard": stability,
        "logit_invariance": invariance,
        "passed": bool(invariance and invariance["allclose"] and min(stability.values()) >= 0.999),
    }
    write_json(output_dir / "activation_audit.json", audit)
    print(json.dumps(audit | {"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
