from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from _bootstrap import project_path
from llm_integrity.inner_variant_sampler import FAMILIES
from llm_integrity.io import read_jsonl


BLOCK_TYPES = ("q_proj", "v_proj", "down_proj")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def robust_summary(values: list[float]) -> dict[str, float]:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Calibration values must be non-empty and positive")
    logged = [math.log(value) for value in values]
    center = statistics.median(logged)
    mad = statistics.median(abs(value - center) for value in logged)
    threshold = math.exp(center + 5.0 * max(mad, 1e-12))
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "log_mad": float(mad),
        "clip_threshold_log_median_plus_5_mad": float(threshold),
    }


def grouped_calibration(
    records: list[dict[str, Any]],
    *,
    group_key: str,
    groups: tuple[str, ...],
    score_key: str,
) -> dict[str, Any]:
    valid = [
        row
        for row in records
        if row.get("valid")
        and math.isfinite(float(row[score_key]))
        and float(row[score_key]) > 0
        and math.isfinite(float(row["raw_embedding_gradient_norm"]))
        and float(row["raw_embedding_gradient_norm"]) > 0
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        grouped[str(row[group_key])].append(row)
    missing = [group for group in groups if not grouped[group]]
    if missing:
        raise ValueError(f"Missing valid calibration groups: {missing}")

    output: dict[str, Any] = {}
    for group in groups:
        rows = grouped[group]
        scale = statistics.median(float(row[score_key]) for row in rows)
        normalized_norms = [
            float(row["raw_embedding_gradient_norm"]) / scale
            for row in rows
        ]
        output[group] = {
            "scale": float(scale),
            "score": robust_summary(
                [float(row[score_key]) for row in rows]
            ),
            "normalized_gradient_norm": robust_summary(normalized_norms),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build block-specific micro and family-specific macro "
            "calibration for the discrete joint optimizer"
        )
    )
    parser.add_argument("--micro-records", required=True)
    parser.add_argument("--macro-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-types", default=",".join(BLOCK_TYPES))
    parser.add_argument(
        "--family-weights",
        default="0.2,0.2,0.2,0.2,0.2",
    )
    args = parser.parse_args()
    block_types = tuple(args.block_types.split(","))
    if not all(block_types) or len(set(block_types)) != len(block_types):
        raise ValueError("Invalid block type list")

    micro_path = project_path(args.micro_records)
    macro_path = project_path(args.macro_records)
    micro_records = read_jsonl(micro_path)
    macro_records = read_jsonl(macro_path)
    micro = grouped_calibration(
        micro_records,
        group_key="block_type",
        groups=block_types,
        score_key="raw_micro_score",
    )
    macro = grouped_calibration(
        macro_records,
        group_key="family",
        groups=FAMILIES,
        score_key="raw_macro_score",
    )

    weights = [
        float(value) for value in args.family_weights.split(",")
    ]
    if len(weights) != len(FAMILIES) or any(value < 0 for value in weights):
        raise ValueError("Five non-negative family weights are required")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Family weights must sum to a positive value")
    family_weights = {
        family: weight / total_weight
        for family, weight in zip(FAMILIES, weights, strict=True)
    }

    micro_target = statistics.median(
        micro[block]["normalized_gradient_norm"]["median"]
        for block in block_types
    )
    macro_target = sum(
        family_weights[family]
        * macro[family]["normalized_gradient_norm"]["median"]
        for family in FAMILIES
    )
    micro_weight = macro_target / max(micro_target, 1e-30)

    output = {
        "version": 2,
        "method": (
            "block_specific_micro_family_specific_macro_gradient_balance"
        ),
        "micro": micro,
        "macro": macro,
        "family_weights": family_weights,
        "recommended_micro_weight": float(micro_weight),
        "recommended_macro_weight": 1.0,
        "source": {
            "micro_records": str(micro_path),
            "micro_sha256": sha256(micro_path),
            "macro_records": str(macro_path),
            "macro_sha256": sha256(macro_path),
        },
    }
    output_path = project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
