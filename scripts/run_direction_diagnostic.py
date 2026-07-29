from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.inner_variant_sampler import (
    StratifiedVariantSampler,
    load_registry,
    read_jsonl,
)
from llm_integrity.io import write_jsonl
from llm_integrity.joint_inner_optimizer_diagnostic import JointInnerOptimizer
from llm_integrity.modeling import load_model


def log_mad_threshold(values: list[float], multiplier: float = 5.0) -> float:
    logged = [math.log(max(value, 1e-30)) for value in values]
    center = statistics.median(logged)
    mad = statistics.median(abs(value - center) for value in logged)
    return math.exp(center + multiplier * max(mad, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/paper_aligned_qwen_7b_joint_inner.yaml"),
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-registry", required=True)
    parser.add_argument("--micro-calibration", required=True)
    parser.add_argument("--macro-calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--micro-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--global-clip-norm", type=float, default=None)
    parser.add_argument(
        "--maximum-relative-update",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--clip-mode",
        choices=["global", "tokenwise"],
        default="global",
    )
    parser.add_argument("--top-token-updates", type=int, default=None)
    parser.add_argument("--projection-interval", type=int, default=15)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    joint = config["joint_inner"]
    prompts = read_jsonl(project_path(args.prompts))
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    manifests = read_jsonl(project_path(args.manifest))
    registry = load_registry(project_path(args.adapter_registry))
    micro_records = [
        row
        for row in read_jsonl(project_path(args.micro_calibration))
        if row.get("valid")
    ]
    macro_records = [
        row
        for row in read_jsonl(project_path(args.macro_calibration))
        if row.get("valid")
    ]
    micro_scale = statistics.median(
        float(row["raw_micro_score"]) for row in micro_records
    )
    macro_scale = statistics.median(
        float(row["raw_macro_score"]) for row in macro_records
    )
    micro_norms = [
        float(row["raw_embedding_gradient_norm"]) / micro_scale
        for row in micro_records
    ]
    macro_norms = [
        float(row["raw_embedding_gradient_norm"]) / macro_scale
        for row in macro_records
    ]
    micro_median = statistics.median(micro_norms)
    macro_median = statistics.median(macro_norms)
    calibrated_micro_weight = (
        macro_median / max(micro_median, 1e-30)
    )
    micro_weight = (
        float(args.micro_weight)
        if args.micro_weight is not None
        else calibrated_micro_weight
    )
    micro_clip = log_mad_threshold(micro_norms)
    macro_clip = log_mad_threshold(macro_norms)
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights=joint["macro"]["families"],
        adapter_registry=registry,
        seed=int(config.get("seed", 42)),
    )

    output = project_path(args.output)
    results = (
        read_jsonl(output)
        if args.resume and output.exists()
        else []
    )
    completed = {str(row["prompt_id"]) for row in results}
    reference = load_model(config["model"])
    try:
        optimizer = JointInnerOptimizer(
            reference=reference,
            sampler=sampler,
            model_config=config["model"],
            micro_scale=micro_scale,
            macro_scale=macro_scale,
            micro_weight=micro_weight,
            macro_weight=float(joint.get("macro_weight", 1.0)),
            micro_component_clip=micro_clip,
            macro_component_clip=macro_clip,
            global_clip_norm=float(
                args.global_clip_norm
                if args.global_clip_norm is not None
                else joint.get("global_gradient_clip_norm", 1.0)
            ),
            learning_rate=float(
                args.learning_rate
                if args.learning_rate is not None
                else joint.get("learning_rate", 0.01)
            ),
            maximum_relative_update=float(
                args.maximum_relative_update
                if args.maximum_relative_update is not None
                else joint.get(
                    "maximum_relative_embedding_update",
                    0.001,
                )
            ),
            clip_mode=args.clip_mode,
            top_token_updates=args.top_token_updates,
            projection_interval=args.projection_interval,
            probes=int(joint.get("probes_inner", 1)),
            seed=int(config.get("seed", 42)),
            max_length=args.max_length,
            max_edit_ratio=float(joint.get("max_edit_ratio", 0.25)),
            ppl_ratio_limit=float(joint.get("ppl_ratio_limit", 2.0)),
        )
        for index, row in enumerate(prompts, 1):
            prompt_id = str(row.get("id", row.get("prompt_id")))
            if prompt_id in completed:
                continue
            result = optimizer.optimize(row)
            payload = {
                **result.payload(),
                "category": row.get("category", "unknown"),
                "source": row.get("source"),
            }
            results.append(payload)
            write_jsonl(output, results)
            completed.add(prompt_id)
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(prompts),
                        "prompt_id": prompt_id,
                        "accepted": result.accepted,
                        "steps_completed": result.steps_completed,
                        "failure": result.failure,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        reference.close()

    summary = {
        "output": str(output),
        "requested_prompts": len(prompts),
        "results": len(results),
        "accepted": sum(bool(row.get("accepted")) for row in results),
        "failed": sum(not bool(row.get("accepted")) for row in results),
        "complete_15_steps": sum(
            int(row.get("steps_completed", 0)) == 15 for row in results
        ),
        "micro_scale": micro_scale,
        "macro_scale": macro_scale,
        "micro_weight": micro_weight,
        "calibrated_micro_weight": calibrated_micro_weight,
        "micro_component_clip": micro_clip,
        "macro_component_clip": macro_clip,
        "learning_rate": (
            float(args.learning_rate)
            if args.learning_rate is not None
            else float(joint.get("learning_rate", 0.01))
        ),
        "global_clip_norm": (
            float(args.global_clip_norm)
            if args.global_clip_norm is not None
            else float(joint.get("global_gradient_clip_norm", 1.0))
        ),
        "maximum_relative_update": (
            float(args.maximum_relative_update)
            if args.maximum_relative_update is not None
            else float(
                joint.get(
                    "maximum_relative_embedding_update",
                    0.001,
                )
            )
        ),
        "clip_mode": args.clip_mode,
        "top_token_updates": args.top_token_updates,
        "projection_interval": args.projection_interval,
        "changed_discrete_prompts": sum(
            row.get("optimized_prompt") != row.get("initial_prompt")
            for row in results
        ),
        "maximum_boundary_ratio": max(
            (
                float(step.get("maximum_boundary_ratio", 0.0))
                for row in results
                for step in row.get("history", [])
            ),
            default=0.0,
        ),
        "projection_change_checks": sum(
            int(step.get("projection_changed_tokens", 0)) > 0
            for row in results
            for step in row.get("history", [])
        ),
        "passed": bool(
            len(results) >= len(prompts)
            and all(int(row.get("steps_completed", 0)) == 15 for row in results)
        ),
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
