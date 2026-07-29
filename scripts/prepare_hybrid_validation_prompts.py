from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path
from llm_integrity.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build initial/optimized prompt pairs for hard scoring"
    )
    parser.add_argument(
        "--input",
        default=(
            "results/paper_aligned_qwen_7b/prompt_optimization/"
            "optimized_prompts.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/paper_aligned_qwen_7b/prompt_optimization/"
            "hybrid_validation_prompts.jsonl"
        ),
    )
    args = parser.parse_args()

    rows = read_jsonl(project_path(args.input))
    prompts: list[dict] = []
    for row in rows:
        source_id = str(row.get("id", row.get("prompt_id")))
        optimization = row.get("optimization", {})
        for role, field in (
            ("initial", "initial_prompt"),
            ("optimized", "optimized_prompt"),
        ):
            prompt = str(optimization.get(field, "")).strip()
            if not prompt:
                raise ValueError(
                    f"{source_id!r} has no optimization.{field}"
                )
            prompts.append(
                {
                    "id": f"{source_id}::{role}",
                    "prompt_id": f"{source_id}::{role}",
                    "source_prompt_id": source_id,
                    "candidate_role": role,
                    "category": row.get("category", "unknown"),
                    "prompt": prompt,
                }
            )

    output = project_path(args.output)
    write_jsonl(output, prompts)
    print(
        json.dumps(
            {"output": str(output), "rows": len(prompts)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
