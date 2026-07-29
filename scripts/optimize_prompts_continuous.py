from __future__ import annotations

import argparse
import json
import traceback

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.continuous_prompt_optimization import (
    ContinuousPromptOptimizer,
)
from llm_integrity.io import read_jsonl, set_seed, write_jsonl
from llm_integrity.modeling import load_model
from llm_integrity.variants import load_variant


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=str(
            ROOT / "configs/paper_aligned_qwen_7b.yaml"
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=60,
        help="本次优化的提示词数量",
    )

    parser.add_argument(
        "--seed-file",
        default=None,
        help="Override sensitivity.prompt_optimization.seed_file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path; defaults to optimized_prompts_k<count>.jsonl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ids already saved in the output file",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later prompts after logging a failed prompt",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    settings = config["sensitivity"]["prompt_optimization"]
    seed = int(config.get("seed", 42))
    set_seed(seed)

    attack_name = str(settings["training_attack"])

    attack_config = next(
        attack
        for attack in config["attacks"]
        if attack["name"] == attack_name
    )

    seed_file = project_path(str(args.seed_file or settings["seed_file"]))
    rows = read_jsonl(seed_file)[: args.count]

    if not rows:
        raise ValueError(f"No prompts found in {seed_file}")

    output_path = (
        project_path(args.output)
        if args.output is not None
        else (
            project_path(config["output_dir"])
            / "prompt_optimization"
            / f"optimized_prompts_k{args.count}.jsonl"
        )
    )
    outputs = (
        read_jsonl(output_path)
        if args.resume and output_path.exists()
        else []
    )
    completed_ids = {
        str(row.get("id", row.get("prompt_id")))
        for row in outputs
    }
    selected_ids = {
        str(row.get("id", row.get("prompt_id")))
        for row in rows
    }
    unexpected_ids = completed_ids - selected_ids
    if unexpected_ids:
        raise ValueError(
            "Resume output contains ids outside the selected seed pool: "
            + ", ".join(sorted(unexpected_ids))
        )

    print(
        json.dumps(
            {
                "stage": "loading_models",
                "reference": config["model"]["name"],
                "training_attack": attack_name,
                "prompt_rows": len(rows),
                "already_completed": len(completed_ids),
                "output": str(output_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    reference = load_model(config["model"])
    modified, attack_report = load_variant(
        config["model"],
        attack_config,
    )

    optimizer = ContinuousPromptOptimizer(
        reference=reference,
        modified=modified,
        candidate_tokens=int(
            settings.get("candidate_tokens", 16)
        ),
        learning_rate=float(
            settings.get("learning_rate", 0.05)
        ),
        steps=int(settings.get("steps", 10)),
        initial_temperature=float(
            settings.get("initial_temperature", 1.0)
        ),
        anneal=float(settings.get("anneal", 0.95)),
        epsilon=float(settings.get("epsilon", 1.0)),
        semantic_weight=float(
            settings.get("semantic_weight", 0.1)
        ),
        ppl_ratio_limit=float(
            settings.get("ppl_ratio_limit", 2.0)
        ),
        max_edit_ratio=float(
            settings.get("max_edit_ratio", 0.25)
        ),
        convergence_tolerance=float(
            settings.get("convergence_tolerance", 1e-4)
        ),
    )

    try:
        for index, row in enumerate(rows, start=1):
            prompt_id = str(row.get("id", row.get("prompt_id")))
            if prompt_id in completed_ids:
                print(
                    f"[optimization] skip completed {prompt_id}",
                    flush=True,
                )
                continue
            prompt = str(row["prompt"])

            print(
                f"[optimization] {index}/{len(rows)} "
                f"{row.get('category', 'unknown')} "
                f"{row.get('id', row.get('prompt_id', 'unknown'))}",
                flush=True,
            )

            try:
                result = optimizer.optimize(prompt)
            except Exception:
                print(
                    json.dumps(
                        {
                            "stage": "optimization_error",
                            "prompt_id": prompt_id,
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if args.continue_on_error:
                    continue
                raise

            output_row = dict(row)
            optimization_payload = dict(result.__dict__)
            optimization_payload["proxy_accepted"] = result.accepted
            # Final acceptance is deliberately unknown until the complete
            # micro + five-family macro hard-validation stage has run.
            optimization_payload["accepted"] = None
            output_row["optimization"] = optimization_payload
            output_row["training_attack"] = attack_name
            outputs.append(output_row)
            completed_ids.add(prompt_id)
            write_jsonl(output_path, outputs)

            print(
                json.dumps(
                    {
                        "proxy_accepted": result.proxy_accepted,
                        "acceptance_stage": result.acceptance_stage,
                        "requires_hard_validation": (
                            result.requires_hard_validation
                        ),
                        "initial_score": result.initial_score,
                        "final_score": result.final_score,
                        "score_gain": result.score_gain,
                        "ppl_ratio": result.ppl_ratio,
                        "edit_ratio": result.edit_ratio,
                        "steps_run": result.steps_run,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    finally:
        modified.close()
        reference.close()

    write_jsonl(output_path, outputs)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(outputs),
                "attack_report": (
                    attack_report.__dict__
                    if attack_report is not None
                    else None
                ),
                "next_stage": (
                    "prepare_hybrid_validation_prompts.py, then compute "
                    "full micro and five-family macro scores, then run "
                    "finalize_hybrid_prompt_optimization.py"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
