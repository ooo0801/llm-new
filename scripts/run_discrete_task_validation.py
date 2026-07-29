from __future__ import annotations

import argparse
import json

from _bootstrap import project_path
from llm_integrity.config import load_config
from llm_integrity.inner_variant_sampler import read_jsonl
from llm_integrity.io import write_jsonl
from llm_integrity.modeling import generate_texts, load_model
from llm_integrity.task_validation import evaluate_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(project_path(args.config))
    rows = read_jsonl(project_path(args.prompts))
    output = project_path(args.output)
    results = (
        read_jsonl(output)
        if args.resume and output.exists()
        else []
    )
    completed = {str(row["prompt_id"]) for row in results}
    bundle = load_model(config["model"])
    generation = {
        "max_input_tokens": int(args.max_input_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": False,
        "system_prompt": None,
    }
    try:
        for index, row in enumerate(rows, start=1):
            prompt_id = str(row.get("id", row.get("prompt_id")))
            if prompt_id in completed:
                continue
            generated = generate_texts(
                bundle,
                [str(row["prompt"])],
                generation,
            )[0]
            passed, rule = evaluate_task(row, generated)
            result = {
                "prompt_id": prompt_id,
                "source_prompt_id": row.get("source_prompt_id"),
                "candidate_role": row.get("candidate_role"),
                "category": row.get("category"),
                "evaluator": row.get("evaluator"),
                "expected_answer": row.get("expected_answer"),
                "expected_contains": row.get("expected_contains"),
                "generated_text": generated,
                "task_passed": bool(passed),
                "evaluation_rule": rule,
                "deterministic_generation": True,
            }
            results.append(result)
            write_jsonl(output, results)
            completed.add(prompt_id)
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(rows),
                        "prompt_id": prompt_id,
                        "task_passed": passed,
                        "evaluation_rule": rule,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        bundle.close()

    report = {
        "records": len(results),
        "passed": sum(bool(row["task_passed"]) for row in results),
        "failed": sum(not bool(row["task_passed"]) for row in results),
        "deterministic_generation": True,
    }
    report_path = output.with_suffix(".summary.json")
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
