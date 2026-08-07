from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


STATIC_FILES = [
    "experiments/prompt-robust-14b-f2/protocol.md",
    "configs/experiment_f2_qwen14b_inner.yaml",
    "configs/experiment_f2_qwen14b_development.yaml",
    "configs/experiment_f2_qwen14b_confirmation.yaml",
    "experiments/prompt-robust-14b-f2/inputs/design_manifest.json",
    "experiments/prompt-robust-14b-f2/inputs/smoke_sources2.jsonl",
    "experiments/prompt-robust-14b-f2/inputs/expansion_sources32.jsonl",
    "experiments/prompt-robust-14b-f2/inputs/attack_manifest_development_3each.jsonl",
    "experiments/prompt-robust-14b-f2/inputs/attack_manifest_confirmation_2each.jsonl",
    "experiments/prompt-robust-14b-f1/inputs/candidate_pairs64.jsonl",
]

CONSTRUCTION_FILES = [
    "model_smoke.json",
    "gate_status.txt",
    "smoke_results.jsonl",
    "smoke_summary.json",
    "joint_results_32.jsonl",
    "joint_summary_32.json",
    "new_portfolio_candidates.jsonl",
    "new_portfolio_mapping.jsonl",
    "new_portfolio_report.json",
    "candidate_pairs_f2.jsonl",
    "pool_assembly_report.json",
]

DEVELOPMENT_FILES = [
    "gate_status.txt",
    "task_validation.jsonl",
    "task_validation.summary.json",
    "micro_scores_unique.jsonl",
    "micro_scores_expanded.jsonl",
    "macro/macro_records_validation.jsonl",
    "macro/macro_summary_validation.json",
    "macro_records_expanded.jsonl",
    "analysis/report.json",
    "analysis/per_prompt_decisions.jsonl",
    "analysis/frozen30_pairs.jsonl",
]

CONFIRMATION_FILES = [
    "gate_status.txt",
    "task_validation.jsonl",
    "task_validation.summary.json",
    "micro_scores_unique.jsonl",
    "micro_scores_expanded.jsonl",
    "macro/macro_records_test.jsonl",
    "macro/macro_summary_test.json",
    "macro_records_expanded.jsonl",
    "analysis/report.json",
    "analysis/per_prompt_decisions.jsonl",
    "analysis/confirmed_fingerprint.jsonl",
    "analysis/strict_5of5_subset.jsonl",
]

ADAPTER_FILES = [
    "registry_development.json",
    "registry_confirmation.json",
    "development/training_reports_validation.json",
    "confirmation/training_reports_test.json",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required archive input is missing: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_terminal_state(results_root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    construction_gate = (results_root / "00_construction/gate_status.txt").read_text(
        encoding="utf-8"
    ).strip()
    development_gate = (results_root / "01_development/gate_status.txt").read_text(
        encoding="utf-8"
    ).strip()
    confirmation_gate = (results_root / "02_confirmation/gate_status.txt").read_text(
        encoding="utf-8"
    ).strip()
    if construction_gate != "CONSTRUCTION_GO":
        raise RuntimeError(f"unexpected F2 construction gate: {construction_gate}")
    if development_gate != "DEVELOPMENT_GO":
        raise RuntimeError(f"unexpected F2 development gate: {development_gate}")

    development = load_json(results_root / "01_development/analysis/report.json")
    confirmation = load_json(results_root / "02_confirmation/analysis/report.json")
    if development.get("technical_passed") is not True or development.get("gate_passed") is not True:
        raise RuntimeError("F2 archive requires a passed development parent gate")
    if confirmation.get("technical_passed") is not True:
        terminal_state = "confirmation_technical_failure"
    elif confirmation.get("gate_passed") is True and confirmation.get("hypothesis_supported") is True:
        terminal_state = "confirmation_go"
    else:
        terminal_state = "confirmation_no_go"
    expected_gate = "CONFIRMATION_GO" if terminal_state == "confirmation_go" else "CONFIRMATION_NO_GO"
    if confirmation_gate != expected_gate:
        raise RuntimeError(
            f"confirmation report/gate mismatch: report={terminal_state}, gate={confirmation_gate}"
        )
    return development, confirmation, terminal_state


def archive(root: Path, results_root: Path, release_dir: Path) -> dict[str, Any]:
    development, confirmation, terminal_state = validate_terminal_state(results_root)
    if release_dir.exists() and any(release_dir.iterdir()):
        raise FileExistsError(f"release directory is not empty: {release_dir}")
    release_dir.mkdir(parents=True, exist_ok=True)

    for relative in STATIC_FILES:
        copy_required(root / relative, release_dir / "frozen" / relative)
    for relative in CONSTRUCTION_FILES:
        copy_required(results_root / "00_construction" / relative, release_dir / "00_construction" / relative)
    for relative in DEVELOPMENT_FILES:
        copy_required(results_root / "01_development" / relative, release_dir / "01_development" / relative)
    for relative in CONFIRMATION_FILES:
        copy_required(results_root / "02_confirmation" / relative, release_dir / "02_confirmation" / relative)
    for relative in ADAPTER_FILES:
        copy_required(results_root / "adapters" / relative, release_dir / "adapters" / relative)
    copy_required(results_root / "pipeline.log", release_dir / "pipeline.log")

    final_report = {
        "schema_version": "experiment_f2_archive_1.0",
        "experiment": "Qwen2.5-14B targeted logic/summary expansion and independent confirmation",
        "hypothesis": "H-F2",
        "terminal_state": terminal_state,
        "hypothesis_supported": confirmation.get("hypothesis_supported") is True,
        "development": development,
        "confirmation": confirmation,
        "git_revision": git_revision(root),
        "claim_boundary": (
            "F2 confirmation uses the frozen 30-row development selection and independent endpoints. "
            "A No-Go is archived without candidate repair, threshold changes, or downstream G2/H2 execution."
        ),
        "excluded_large_artifacts": ["LoRA adapter weight tensors"],
    }
    (release_dir / "FINAL_REPORT.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_rows = []
    for path in sorted(item for item in release_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        relative = path.relative_to(release_dir).as_posix()
        checksum_rows.append(f"{sha256(path)}  {relative}")
    (release_dir / "SHA256SUMS").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")
    return final_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and archive terminal F2 evidence")
    parser.add_argument("--results-root", default="results/experiment_f2_qwen14b_targeted_20260808")
    parser.add_argument("--release-dir", default="reproducibility/experiment_f2_qwen14b_targeted_20260808")
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    release_dir = Path(args.release_dir)
    if not release_dir.is_absolute():
        release_dir = ROOT / release_dir
    report = archive(ROOT, results_root, release_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
