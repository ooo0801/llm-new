from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.inner_variant_sampler import FAMILIES
from llm_integrity.paper_hybrid_validation import (
    validate_proxy_candidates,
)
from llm_integrity.paper_hybrid_sensitivity import RobustCalibration


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def family_means(
    records: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = {}
    for row in records:
        prompt_id = str(row["prompt_id"])
        family = str(row["family"])
        values.setdefault(prompt_id, {}).setdefault(family, []).append(
            float(row[metric])
        )
    means: dict[str, dict[str, float]] = {}
    for prompt_id, by_family in values.items():
        missing = set(FAMILIES) - set(by_family)
        if missing:
            raise ValueError(
                f"{prompt_id}: missing families {sorted(missing)}"
            )
        means[prompt_id] = {
            family: sum(by_family[family]) / len(by_family[family])
            for family in FAMILIES
        }
    return means


def load_fixed_calibrations(
    path: Path,
) -> tuple[RobustCalibration, RobustCalibration]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != 1:
        raise ValueError("Hybrid calibration version 1 is required")

    def build(name: str) -> RobustCalibration:
        item = payload[name]
        result = RobustCalibration(
            center=float(item["center"]),
            scale=float(item["scale"]),
            sample_count=int(item["sample_count"]),
            transform=str(item.get("transform", "log1p")),
            scale_method=str(item.get("scale_method", "unknown")),
        )
        if not math.isfinite(result.center):
            raise ValueError(f"{name} calibration center must be finite")
        if not math.isfinite(result.scale) or result.scale <= 0:
            raise ValueError(f"{name} calibration scale must be positive")
        if result.sample_count < 2:
            raise ValueError(
                f"{name} fixed calibration requires at least 2 samples"
            )
        if result.transform != "log1p":
            raise ValueError(f"Unsupported {name} calibration transform")
        return result

    return build("micro"), build("macro")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "configs/paper_aligned_qwen_7b_joint_outer_validation.yaml"
        ),
    )
    parser.add_argument("--proxy-results", required=True)
    parser.add_argument("--micro-scores", required=True)
    parser.add_argument("--macro-records", required=True)
    parser.add_argument("--task-records", required=True)
    parser.add_argument("--hybrid-calibration", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metric", default="macro_l2_raw")
    parser.add_argument(
        "--minimum-nondegraded-families",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--family-relative-tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument("--required-accepted", type=int, default=6)
    args = parser.parse_args()

    if not 1 <= args.minimum_nondegraded_families <= len(FAMILIES):
        raise ValueError("minimum nondegraded families is invalid")
    if args.family_relative_tolerance < 0:
        raise ValueError("family relative tolerance must be nonnegative")

    config = load_config(project_path(args.config))
    validation = config.get("hybrid_validation")
    if validation is None:
        validation = (
            config.get("sensitivity", {})
            .get("prompt_optimization", {})
            .get("hard_hybrid_validation", {})
        )
    proxy_rows = read_jsonl(project_path(args.proxy_results))
    micro_rows = read_jsonl(project_path(args.micro_scores))
    macro_rows = read_jsonl(project_path(args.macro_records))
    task_rows = read_jsonl(project_path(args.task_records))
    task_by_prompt = {
        str(row["prompt_id"]): row for row in task_rows
    }
    if len(task_by_prompt) != len(task_rows):
        raise ValueError("Duplicate task-validation prompt ids")
    fixed_micro = None
    fixed_macro = None
    if args.hybrid_calibration is not None:
        fixed_micro, fixed_macro = load_fixed_calibrations(
            project_path(args.hybrid_calibration)
        )
    outputs, hybrid_report = validate_proxy_candidates(
        proxy_rows,
        micro_rows,
        macro_rows,
        metric=args.metric,
        weighting_mode=str(
            validation.get("weighting_mode", "adaptive")
        ),
        hybrid_beta=float(validation.get("hybrid_beta", 1.0)),
        micro_weight=float(
            validation.get(
                "micro_weight",
                validation.get("micro_prior_weight", 0.5),
            )
        ),
        macro_weight=float(
            validation.get(
                "macro_weight",
                validation.get("macro_prior_weight", 0.5),
            )
        ),
        minimum_gain=float(validation.get("minimum_gain", 0.0)),
        micro_calibration=fixed_micro,
        macro_calibration=fixed_macro,
    )
    macro_by_prompt = family_means(
        macro_rows,
        metric=args.metric,
    )

    strict_accepted = 0
    for row in outputs:
        source_id = str(row.get("id", row.get("prompt_id")))
        initial_id = f"{source_id}::initial"
        optimized_id = f"{source_id}::optimized"
        initial = macro_by_prompt[initial_id]
        optimized = macro_by_prompt[optimized_id]
        if (
            initial_id not in task_by_prompt
            or optimized_id not in task_by_prompt
        ):
            raise ValueError(f"{source_id}: missing task validation")
        initial_task_passed = bool(
            task_by_prompt[initial_id]["task_passed"]
        )
        optimized_task_passed = bool(
            task_by_prompt[optimized_id]["task_passed"]
        )
        task_preserved = bool(
            initial_task_passed and optimized_task_passed
        )
        family_details: dict[str, dict[str, Any]] = {}
        nondegraded = 0
        for family in FAMILIES:
            baseline = float(initial[family])
            candidate = float(optimized[family])
            tolerance = (
                args.family_relative_tolerance
                * max(abs(baseline), 1e-12)
            )
            gain = candidate - baseline
            passed = bool(gain >= -tolerance)
            nondegraded += int(passed)
            family_details[family] = {
                "initial": baseline,
                "optimized": candidate,
                "gain": gain,
                "relative_gain": gain / max(abs(baseline), 1e-12),
                "nondegraded": passed,
                "relative_tolerance": (
                    args.family_relative_tolerance
                ),
            }

        optimization = row["optimization"]
        hybrid = optimization["hybrid_validation"]
        initial_prompt = str(optimization["initial_prompt"])
        optimized_prompt = str(optimization["optimized_prompt"])
        changed = initial_prompt != optimized_prompt
        finite = all(
            math.isfinite(float(value))
            for value in (
                hybrid["initial_objective"],
                hybrid["optimized_objective"],
                hybrid["objective_gain"],
            )
        )
        constraints_passed = bool(
            int(optimization.get("edit_count", 0)) >= 1
            and float(optimization.get("edit_ratio", math.inf)) <= 0.25
            and float(optimization.get("ppl_ratio", math.inf)) <= 2.0
            and optimization.get("discrete_change_verified") is True
        )
        base_accepted = bool(optimization["accepted"])
        accepted = bool(
            base_accepted
            and finite
            and changed
            and constraints_passed
            and task_preserved
            and nondegraded
            >= int(args.minimum_nondegraded_families)
        )
        strict_accepted += int(accepted)
        optimization["hybrid_base_accepted"] = base_accepted
        optimization["accepted"] = accepted
        optimization["acceptance_stage"] = (
            "heldout_full_micro_five_family_strict_validation"
        )
        optimization["strict_validation"] = {
            "finite_objective": finite,
            "discrete_prompt_changed": changed,
            "proxy_constraints_passed": constraints_passed,
            "initial_task_passed": initial_task_passed,
            "optimized_task_passed": optimized_task_passed,
            "task_preserved": task_preserved,
            "nondegraded_families": nondegraded,
            "minimum_nondegraded_families": int(
                args.minimum_nondegraded_families
            ),
            "family_relative_tolerance": float(
                args.family_relative_tolerance
            ),
            "families": family_details,
        }

    technical_passed = bool(
        len(outputs) == len(proxy_rows)
        and all(
            row["optimization"]["strict_validation"]["finite_objective"]
            for row in outputs
        )
    )
    scientific_gate_passed = bool(
        strict_accepted >= int(args.required_accepted)
    )
    report = {
        "version": 2,
        "method": (
            "heldout_full_micro_plus_five_family_macro_strict_validation"
        ),
        "rows": len(outputs),
        "proxy_accepted": sum(
            bool(row["optimization"]["proxy_accepted"])
            for row in outputs
        ),
        "hybrid_base_accepted": sum(
            bool(row["optimization"]["hybrid_base_accepted"])
            for row in outputs
        ),
        "initial_task_passed": sum(
            bool(
                row["optimization"]["strict_validation"][
                    "initial_task_passed"
                ]
            )
            for row in outputs
        ),
        "optimized_task_passed": sum(
            bool(
                row["optimization"]["strict_validation"][
                    "optimized_task_passed"
                ]
            )
            for row in outputs
        ),
        "strict_accepted": strict_accepted,
        "required_accepted": int(args.required_accepted),
        "minimum_nondegraded_families": int(
            args.minimum_nondegraded_families
        ),
        "family_relative_tolerance": float(
            args.family_relative_tolerance
        ),
        "technical_passed": technical_passed,
        "scientific_gate_passed": scientific_gate_passed,
        "passed": technical_passed and scientific_gate_passed,
        "hybrid_report": hybrid_report,
    }
    write_jsonl(project_path(args.output), outputs)
    report_path = project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
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
    if not technical_passed:
        raise SystemExit(2)
    if not scientific_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
