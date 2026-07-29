from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import project_path
from llm_integrity.paper_hybrid_sensitivity import fit_robust_calibration
from llm_integrity.paper_hybrid_validation import (
    aggregate_prompt_macro,
    index_micro_scores,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro-scores", required=True)
    parser.add_argument("--macro-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="macro_l2_raw")
    parser.add_argument("--minimum-samples", type=int, default=10)
    args = parser.parse_args()

    micro_path = project_path(args.micro_scores)
    macro_path = project_path(args.macro_records)
    micro = index_micro_scores(read_jsonl(micro_path))
    macro = aggregate_prompt_macro(
        read_jsonl(macro_path),
        metric=args.metric,
    )
    initial_ids = sorted(
        prompt_id
        for prompt_id in set(micro) & set(macro)
        if prompt_id.endswith("::initial")
    )
    if len(initial_ids) < int(args.minimum_samples):
        raise ValueError(
            f"Need at least {args.minimum_samples} initial prompts, "
            f"found {len(initial_ids)}"
        )

    micro_calibration = fit_robust_calibration(
        micro[prompt_id] for prompt_id in initial_ids
    )
    macro_calibration = fit_robust_calibration(
        macro[prompt_id] for prompt_id in initial_ids
    )
    payload = {
        "version": 1,
        "method": "fixed_initial_development_pool_log1p_median_mad",
        "metric": args.metric,
        "sample_count": len(initial_ids),
        "initial_prompt_ids": initial_ids,
        "micro": micro_calibration.to_dict(),
        "macro": macro_calibration.to_dict(),
        "source_sha256": {
            "micro_scores": sha256(micro_path),
            "macro_records": sha256(macro_path),
        },
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
