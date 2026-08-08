from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from llm_integrity.config import load_config


FROZEN_FILES = [
    "configs/fingerprint_h4_qwen14b_paired.yaml",
    "experiments/paired-mmd-14b-h4/protocol.md",
    "reproducibility/fingerprint_h4_qwen14b_paired_20260808/protocol.json",
    "reproducibility/fingerprint_h4_qwen14b_paired_20260808/attack_manifest.jsonl",
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


def detect_h4_terminal(output_dir: Path) -> dict[str, Any]:
    report = load_json(output_dir / "final_mmd_report.json")
    if report.get("technical_passed") is not True:
        terminal_state = "h4_technical_failure"
    elif report.get("hypothesis_supported") is True:
        terminal_state = "h4_go"
    else:
        terminal_state = "h4_no_go"
    return {
        "terminal_state": terminal_state,
        "hypothesis_supported": report.get("hypothesis_supported") is True,
        "specificity_gate_passed": report.get("intact_correct") is True,
        "sensitivity_count_gate_passed": int(report.get("modified_detected", -1))
        >= int(report.get("required_modified_detected", 9)),
        "family_coverage_gate_passed": report.get("family_coverage_passed") is True,
        "mmd_report": report,
    }


def write_checksums(release_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in release_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        rows.append(f"{sha256(path)}  {path.relative_to(release_dir).as_posix()}")
    (release_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def archive(root: Path, output_dir: Path, release_dir: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    terminal = detect_h4_terminal(output_dir)
    report = terminal["mmd_report"]
    expected_states = int(config["statistics"]["expected_intact_states"]) + int(
        config["statistics"]["expected_modified_states"]
    )
    if report.get("technical_passed") is not True or int(report.get("states", -1)) != expected_states:
        raise RuntimeError("H4 archive requires a technically valid complete 14-state report")

    for relative in FROZEN_FILES:
        copy_required(root / relative, release_dir / "frozen" / relative)

    artifact = str(config["fingerprint"]["artifact_name"])
    prefix = str(config["fingerprint"]["verification_prefix"])
    copy_required(output_dir / "fingerprints" / artifact, release_dir / "fingerprints" / artifact)
    copy_required(output_dir / "reference_report.json", release_dir / "reference_report.json")
    copy_required(output_dir / "final_mmd_report.json", release_dir / "final_mmd_report.json")

    verification_files = sorted((output_dir / "verification").glob(f"{prefix}__*.json"))
    if len(verification_files) != expected_states:
        raise RuntimeError(
            f"H4 archive requires {expected_states} verification files, found {len(verification_files)}"
        )
    for path in verification_files:
        copy_required(path, release_dir / "verification" / path.name)

    copy_required(
        output_dir / "finetuning_adapters/adapter_registry.json",
        release_dir / "adapter_registry.json",
    )
    training_reports = sorted((output_dir / "finetuning_adapters").glob("training_reports*.json"))
    if not training_reports:
        raise FileNotFoundError("H4 LoRA training report is missing")
    for path in training_reports:
        copy_required(path, release_dir / "training" / path.name)

    combined = {
        "schema_version": "fingerprint_h4_archive_1.0",
        "git_revision": git_revision(root),
        "h4": terminal,
        "parent_fingerprint_sha256": str(config["data"]["source_fingerprint_sha256"]),
        "claim_boundary": (
            "H4 is a fixed-instance paired common-random-number experiment. "
            "It is not a population detection-rate estimate and does not repair H3."
        ),
        "excluded_large_artifacts": ["LoRA adapter weight tensors", "runtime model caches"],
    }
    (release_dir / "FINAL_REPORT.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(release_dir)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive terminal H4 paired false-positive remediation evidence")
    parser.add_argument("--config", default="configs/fingerprint_h4_qwen14b_paired.yaml")
    parser.add_argument("--output-dir", default="results/fingerprint_h4_qwen14b_paired_20260808")
    parser.add_argument("--release-dir", default="reproducibility/fingerprint_h4_qwen14b_paired_20260808")
    args = parser.parse_args()
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    release_dir = Path(args.release_dir)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if not release_dir.is_absolute():
        release_dir = ROOT / release_dir
    report = archive(ROOT, output_dir, release_dir, config_path)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
