from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    experiment = (
        root
        / "results/paper_aligned_qwen_7b/joint_inner_20260727"
    )
    backup = root / "backups/joint_inner_20260727_before_implementation"
    stage2 = load(experiment / "02_micro_proxy_smoke/summary.json")
    micro = load(experiment / "03_calibration/micro_summary_12x3.json")
    macro = load(experiment / "03_calibration/macro_summary_12x5.json")
    stage4 = load(experiment / "04_stratified_sampler/schedule_audit_60.json")
    stage5 = load(experiment / "05_joint_optimizer/integration_acceptance.json")
    stage6 = load(experiment / "06_joint_smoke_1x15_attempt2/summary.json")
    tail = load(
        experiment
        / "06_long_prompt_tail_regression_1x15_attempt2/summary.json"
    )
    macro12 = load(
        experiment
        / "07_small_comparison/macro_only_a0_summary_12_tail_safe.json"
    )
    joint12 = load(
        experiment
        / "07_small_comparison/joint_summary_12_tail_safe.json"
    )
    comparison = load(
        experiment
        / "07_small_comparison/comparison_summary_12_tail_safe.json"
    )
    joint60 = load(experiment / "08_formal_60/joint_summary_60.json")
    hard_report = load(
        experiment
        / "08_formal_60/outer_validation/"
        "hybrid_validated_prompts_60.report.json"
    )

    directories = {
        "1_backup_and_fixed_config": {
            "result_directory": str(
                experiment / "01_backup_and_fixed_config"
            ),
            "backup_directory": str(backup),
            "passed": (
                backup.is_dir()
                and (
                    experiment
                    / "01_backup_and_fixed_config/"
                    "paper_aligned_qwen_7b_joint_inner.yaml"
                ).is_file()
            ),
        },
        "2_small_block_micro_proxy": {
            "result_directory": str(experiment / "02_micro_proxy_smoke"),
            "passed": bool(stage2.get("passed")),
            "valid_records": stage2.get("valid_records"),
        },
        "3_micro_macro_calibration": {
            "result_directory": str(experiment / "03_calibration"),
            "passed": bool(micro.get("passed") and macro.get("passed")),
            "micro_scale": micro.get("micro_scale"),
            "macro_scale": macro.get("macro_scale"),
        },
        "4_five_family_stratified_sampler": {
            "result_directory": str(
                experiment / "04_stratified_sampler"
            ),
            "passed": bool(stage4.get("passed")),
            "valid_schedules": stage4.get("valid_schedules"),
        },
        "5_joint_inner_optimizer": {
            "result_directory": str(experiment / "05_joint_optimizer"),
            "passed": bool(stage5.get("passed")),
        },
        "6_one_prompt_15_step_smoke": {
            "result_directory": str(
                experiment / "06_joint_smoke_1x15_attempt2"
            ),
            "tail_regression_directory": str(
                experiment / "06_long_prompt_tail_regression_1x15_attempt2"
            ),
            "passed": bool(stage6.get("passed") and tail.get("passed")),
        },
        "7_twelve_prompt_comparison": {
            "result_directory": str(
                experiment / "07_small_comparison"
            ),
            "passed": bool(
                macro12.get("passed")
                and joint12.get("passed")
                and comparison.get("same_prompt_ids_across_arms")
            ),
            "macro_only_accepted": macro12.get("accepted"),
            "joint_accepted": joint12.get("accepted"),
        },
        "8_sixty_prompt_formal_and_outer_validation": {
            "result_directory": str(experiment / "08_formal_60"),
            "outer_validation_directory": str(
                experiment / "08_formal_60/outer_validation"
            ),
            "passed": bool(
                joint60.get("passed")
                and int(hard_report.get("rows", 0)) == 60
            ),
            "inner_accepted": joint60.get("accepted"),
            "hard_validated_rows": hard_report.get("rows"),
            "hard_accepted": hard_report.get("accepted"),
        },
    }
    code_files = [
        root / "configs/paper_aligned_qwen_7b_joint_inner.yaml",
        root / "configs/paper_aligned_qwen_7b_joint_outer_validation.yaml",
        root / "src/llm_integrity/inner_micro_proxy.py",
        root / "src/llm_integrity/inner_variant_sampler.py",
        root / "src/llm_integrity/joint_inner_optimizer.py",
        root / "scripts/run_joint_inner_optimization.py",
        root / "scripts/run_stage8_outer_validation.sh",
    ]
    report = {
        "passed": all(bool(item["passed"]) for item in directories.values()),
        "stages": directories,
        "code_sha256": {
            str(path.relative_to(root)): sha256(path) for path in code_files
        },
        "notes": {
            "aborted_evidence": str(
                experiment / "08_formal_60/aborted_max64"
            ),
            "active_inner_prefix_tokens": 64,
            "outer_validation_max_tokens": 128,
            "outer_micro_probes": 4,
            "outer_macro_validation_variants": 10,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
