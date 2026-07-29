from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.io import read_jsonl, write_jsonl
from llm_integrity.paper_hybrid_validation import validate_proxy_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard-validate proxy-optimized prompts with full hybrid scores"
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/paper_aligned_qwen_7b.yaml"),
    )
    parser.add_argument("--proxy-results", required=True)
    parser.add_argument("--micro-scores", required=True)
    parser.add_argument("--macro-records", required=True)
    parser.add_argument(
        "--output",
        default=(
            "results/paper_aligned_qwen_7b/prompt_optimization/"
            "hybrid_validated_prompts.jsonl"
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config["sensitivity"]["prompt_optimization"]
    hybrid = settings.get("hard_hybrid_validation", {})

    outputs, report = validate_proxy_candidates(
        read_jsonl(project_path(args.proxy_results)),
        read_jsonl(project_path(args.micro_scores)),
        read_jsonl(project_path(args.macro_records)),
        metric=str(hybrid.get("macro_metric", "macro_l2_raw")),
        weighting_mode=str(hybrid.get("weighting_mode", "adaptive")),
        hybrid_beta=float(hybrid.get("hybrid_beta", 1.0)),
        micro_weight=float(hybrid.get("micro_prior_weight", 0.5)),
        macro_weight=float(hybrid.get("macro_prior_weight", 0.5)),
        minimum_gain=float(hybrid.get("minimum_gain", 0.0)),
    )

    output = project_path(args.output)
    write_jsonl(output, outputs)
    report_path = output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "report": str(report_path), **report},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
