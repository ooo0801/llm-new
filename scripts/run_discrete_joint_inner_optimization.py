from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.discrete_joint_inner_optimizer import (
    DiscreteJointInnerOptimizer,
)
from llm_integrity.inner_variant_sampler import (
    StratifiedVariantSampler,
    load_registry,
    read_jsonl,
)
from llm_integrity.io import write_jsonl
from llm_integrity.modeling import load_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs/paper_aligned_qwen_7b_joint_discrete_v2.yaml"
        ),
    )
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--adapter-registry", default=None)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--micro-weight", type=float, default=None)
    parser.add_argument("--required-accepted", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = project_path(args.config)
    config = load_config(config_path)
    settings = config["discrete_joint_inner"]
    prompt_path = project_path(
        args.prompts or settings["prompt_pool"]
    )
    manifest_path = project_path(
        args.manifest or settings["training_manifest"]
    )
    registry_path = project_path(
        args.adapter_registry
        or settings["training_adapter_registry"]
    )
    calibration_path = project_path(args.calibration)
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    if int(calibration.get("version", 0)) != 2:
        raise ValueError("Discrete calibration version 2 is required")

    prompts = read_jsonl(prompt_path)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    manifests = read_jsonl(manifest_path)
    registry = load_registry(registry_path)
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights=settings["macro"]["families"],
        adapter_registry=registry,
        seed=int(config.get("seed", 42)),
    )

    micro_scales = {
        key: float(value["scale"])
        for key, value in calibration["micro"].items()
    }
    macro_scales = {
        key: float(value["scale"])
        for key, value in calibration["macro"].items()
    }
    micro_clips = {
        key: float(
            value["normalized_gradient_norm"][
                "clip_threshold_log_median_plus_5_mad"
            ]
        )
        for key, value in calibration["micro"].items()
    }
    macro_clips = {
        key: float(
            value["normalized_gradient_norm"][
                "clip_threshold_log_median_plus_5_mad"
            ]
        )
        for key, value in calibration["macro"].items()
    }
    micro_weight = (
        float(args.micro_weight)
        if args.micro_weight is not None
        else float(calibration["recommended_micro_weight"])
    )

    output = project_path(args.output)
    results = (
        read_jsonl(output)
        if args.resume and output.exists()
        else []
    )
    completed = {str(row["prompt_id"]) for row in results}
    reference = load_model(config["model"])
    optimizer = None
    try:
        optimizer = DiscreteJointInnerOptimizer(
            reference=reference,
            sampler=sampler,
            model_config=config["model"],
            micro_scales=micro_scales,
            macro_scales=macro_scales,
            micro_component_clips=micro_clips,
            macro_component_clips=macro_clips,
            micro_weight=micro_weight,
            macro_weight=float(settings.get("macro_weight", 1.0)),
            rounds=int(
                args.rounds
                if args.rounds is not None
                else settings.get("rounds", 3)
            ),
            candidate_positions=int(
                settings.get("candidate_positions", 4)
            ),
            candidates_per_position=int(
                settings.get("candidates_per_position", 8)
            ),
            rerank_candidates=int(
                settings.get("rerank_candidates", 4)
            ),
            minimum_proxy_gain=float(
                settings.get("minimum_proxy_gain", 0.0)
            ),
            probes=int(settings.get("probes_inner", 1)),
            seed=int(config.get("seed", 42)),
            max_length=int(
                args.max_length
                if args.max_length is not None
                else settings.get("max_input_tokens", 64)
            ),
            max_edit_ratio=float(
                settings.get("max_edit_ratio", 0.25)
            ),
            ppl_ratio_limit=float(
                settings.get("ppl_ratio_limit", 2.0)
            ),
            minimum_nondegraded_families=int(
                settings.get(
                    "minimum_nondegraded_families",
                    0,
                )
            ),
            family_relative_tolerance=float(
                settings.get(
                    "family_relative_tolerance",
                    0.01,
                )
            ),
            variants_per_family=int(
                settings.get("variants_per_family", 1)
            ),
            anchor_variants_per_family=int(
                settings.get("anchor_variants_per_family", 0)
            ),
            gradient_restarts=int(
                settings.get("gradient_restarts", 1)
            ),
            family_gate_aggregation=str(
                settings.get("family_gate_aggregation", "mean")
            ),
            require_task_preservation=bool(
                settings.get("task_guard", {}).get(
                    "enabled",
                    False,
                )
            ),
            task_max_input_tokens=int(
                settings.get("task_guard", {}).get(
                    "max_input_tokens",
                    512,
                )
            ),
            task_max_new_tokens=int(
                settings.get("task_guard", {}).get(
                    "max_new_tokens",
                    128,
                )
            ),
            sequential_model_execution=bool(
                settings.get("sequential_model_execution", False)
            ),
        )
        for index, row in enumerate(prompts, start=1):
            prompt_id = str(row.get("id", row.get("prompt_id")))
            if prompt_id in completed:
                continue
            result = optimizer.optimize(row)
            payload = {
                **result.payload(),
                "category": row.get("category", "unknown"),
                "source": row.get("source"),
                "evaluator": row.get("evaluator"),
                "expected_answer": row.get("expected_answer"),
                "expected_contains": row.get("expected_contains"),
                "language": row.get("language"),
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
                        "edit_count": result.edit_count,
                        "committed_rounds": result.committed_rounds,
                        "proxy_objective_gain": (
                            result.proxy_objective_gain
                        ),
                        "failure": result.failure,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if optimizer is not None:
            optimizer.reference.close()
        else:
            reference.close()

    accepted = [row for row in results if row.get("accepted")]
    complete = [
        row for row in results
        if row.get("failure") is None
    ]
    variants_per_family = int(settings.get("variants_per_family", 1))
    anchor_variants_per_family = int(
        settings.get("anchor_variants_per_family", 0)
    )
    family_gate_aggregation = str(
        settings.get("family_gate_aggregation", "mean")
    )
    gradient_restarts = int(settings.get("gradient_restarts", 1))
    method = (
        "discrete_joint_hotflip_multi_direction_inner_dev_rerank"
        if anchor_variants_per_family > 0 and gradient_restarts > 1
        else "discrete_joint_hotflip_disjoint_inner_dev_rerank"
        if anchor_variants_per_family > 0
        else "discrete_joint_hotflip_multi_variant_mean_rerank"
        if variants_per_family > 1
        else "discrete_joint_hotflip_exact_proxy_rerank"
    )
    summary = {
        "version": int(settings.get("version", 2)),
        "method": method,
        "output": str(output),
        "requested_prompts": len(prompts),
        "results": len(results),
        "technically_complete": len(complete),
        "accepted": len(accepted),
        "rejected": len(results) - len(accepted),
        "changed_discrete_prompts": sum(
            row.get("optimized_prompt") != row.get("initial_prompt")
            for row in results
        ),
        "total_committed_rounds": sum(
            int(row.get("committed_rounds", 0)) for row in results
        ),
        "positive_proxy_gain_prompts": sum(
            float(row.get("proxy_objective_gain", 0.0)) > 0
            for row in results
        ),
        "micro_weight": micro_weight,
        "variants_per_family": variants_per_family,
        "anchor_variants_per_family": anchor_variants_per_family,
        "gradient_restarts": gradient_restarts,
        "family_gate_aggregation": family_gate_aggregation,
        "required_accepted": int(args.required_accepted),
        "technical_passed": bool(
            len(results) == len(prompts)
            and len(complete) == len(prompts)
        ),
        "scientific_gate_passed": bool(
            len(accepted) >= int(args.required_accepted)
        ),
        "code_sha256": {
            "config": sha256(config_path),
            "calibration": sha256(calibration_path),
            "optimizer": sha256(
                ROOT
                / "src/llm_integrity/"
                "discrete_joint_inner_optimizer.py"
            ),
            "runner": sha256(Path(__file__).resolve()),
        },
    }
    summary["passed"] = bool(
        summary["technical_passed"]
        and summary["scientific_gate_passed"]
    )
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
