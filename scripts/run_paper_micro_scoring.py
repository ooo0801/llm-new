from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_jsonl
from llm_integrity.modeling import load_model
from llm_integrity.paper_blockwise_micro import (
    estimate_prompt_jacobian_blockwise,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score prompts with the full blockwise Hutchinson micro estimator"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/paper_aligned_qwen_7b.yaml"),
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument(
        "--output",
        default=(
            "results/paper_aligned_qwen_7b/prompt_optimization/"
            "hybrid_micro_scores.jsonl"
        ),
    )
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--probes", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_jsonl(project_path(args.prompts))
    if args.max_prompts is not None:
        rows = rows[: args.max_prompts]
    if not rows:
        raise ValueError("No prompts selected")

    output = project_path(args.output)
    completed = (
        read_jsonl(output)
        if args.resume and output.exists()
        else []
    )
    completed_ids = {str(row["prompt_id"]) for row in completed}

    micro_config = config["sensitivity"]["micro"]
    probes = int(
        args.probes
        if args.probes is not None
        else micro_config.get("probes_formal", 4)
    )
    max_length = int(config["sensitivity"].get("max_length", 512))
    system_prompt = config.get("generation", {}).get("system_prompt")
    seed = int(micro_config.get("seed", config.get("seed", 42)))

    bundle = load_model(config["model"])
    try:
        for index, row in enumerate(rows, 1):
            prompt_id = str(row.get("prompt_id", row.get("id")))
            if prompt_id in completed_ids:
                print(f"[micro] skip completed {prompt_id}", flush=True)
                continue
            print(
                f"[micro] {index}/{len(rows)} {prompt_id}",
                flush=True,
            )
            result = estimate_prompt_jacobian_blockwise(
                bundle,
                str(row["prompt"]),
                max_length=max_length,
                system_prompt=system_prompt,
                probes=probes,
                seed=seed,
                show_progress=True,
            )
            completed.append(
                {
                    "prompt_id": prompt_id,
                    "source_prompt_id": row.get("source_prompt_id"),
                    "candidate_role": row.get("candidate_role"),
                    "category": row.get("category", "unknown"),
                    "micro_raw": result.estimate,
                    "micro_per_parameter": result.estimate_per_parameter,
                    "standard_error": result.standard_error,
                    "parameter_count": result.parameter_count,
                    "output_dimension": result.output_dimension,
                    "probes": result.probes,
                    "seed": result.seed,
                    "component_estimates": result.component_estimates,
                    "complete_parameter_coverage": (
                        result.complete_parameter_coverage
                    ),
                }
            )
            write_jsonl(output, completed)
            completed_ids.add(prompt_id)
    finally:
        bundle.close()

    print(
        json.dumps(
            {"output": str(output), "rows": len(completed)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
