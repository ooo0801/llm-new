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
    "configs/fingerprint_g3_qwen14b_mcc.yaml",
    "experiments/stable-components-14b-g3/protocol.md",
    "experiments/stratified-mmd-14b-h3/protocol.md",
    "results/experiment_f2_qwen14b_targeted_20260808/02_confirmation/analysis/confirmed_fingerprint.jsonl",
]

G3_FILES = [
    "activation_audit.json",
    "global_component_universe.json",
    "mcc_selection.json",
    "g3.log",
]

G3_EXTERNAL_PROFILE_FILES = [
    "activations/build_profiles.jsonl",
    "activations/audit_profiles.jsonl",
    "activations/candidate_profiles.jsonl",
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


def detect_g3_terminal(output_dir: Path) -> dict[str, Any]:
    audit = load_json(output_dir / "activation_audit.json")
    universe = load_json(output_dir / "global_component_universe.json")
    selection = load_json(output_dir / "mcc_selection.json")
    supported = bool(
        audit.get("passed") is True
        and universe.get("gates", {}).get("passed") is True
        and selection.get("passed") is True
    )
    return {
        "terminal_state": "g3_go" if supported else "g3_no_go",
        "hypothesis_supported": supported,
        "activation_audit": audit,
        "global_universe": universe,
        "mcc_selection": selection,
    }


def detect_h3_terminal(output_dir: Path, g3: dict[str, Any]) -> dict[str, Any]:
    if g3.get("hypothesis_supported") is not True:
        raise RuntimeError("H3 cannot be archived because G3 did not pass")
    report = load_json(output_dir / "final_mmd_report.json")
    if report.get("technical_passed") is not True:
        terminal_state = "h3_technical_failure"
    elif report.get("hypothesis_supported") is True:
        terminal_state = "h3_go"
    else:
        terminal_state = "h3_no_go"
    return {
        "terminal_state": terminal_state,
        "hypothesis_supported": report.get("hypothesis_supported") is True,
        "mmd_report": report,
    }


def write_checksums(release_dir: Path) -> None:
    rows = []
    for path in sorted(item for item in release_dir.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
        rows.append(f"{sha256(path)}  {path.relative_to(release_dir).as_posix()}")
    (release_dir / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def archive(root: Path, output_dir: Path, release_dir: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    for relative in FROZEN_FILES:
        copy_required(root / relative, release_dir / "frozen" / relative)

    g3 = detect_g3_terminal(output_dir)
    legacy_profiles = (release_dir / "g3/activations").resolve()
    resolved_release = release_dir.resolve()
    if legacy_profiles.is_dir():
        if resolved_release not in legacy_profiles.parents:
            raise RuntimeError("refusing to clean activation profiles outside the release directory")
        shutil.rmtree(legacy_profiles)
    for relative in G3_FILES:
        copy_required(output_dir / relative, release_dir / "g3" / relative)
    profile_manifest = []
    for relative in G3_EXTERNAL_PROFILE_FILES:
        path = output_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"required external activation profile is missing: {path}")
        with path.open("rb") as handle:
            rows = sum(1 for line in handle if line.strip())
        profile_manifest.append(
            {
                "path": relative,
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "storage": "server_and_local_external_raw_results",
            }
        )
    (release_dir / "g3/activation_profiles_manifest.json").write_text(
        json.dumps(profile_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / "G3_FINAL_REPORT.json").write_text(
        json.dumps(g3, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    h3 = detect_h3_terminal(output_dir, g3)
    artifact = str(config["fingerprint"]["artifact_name"])
    prefix = str(config["fingerprint"]["verification_prefix"])
    copy_required(output_dir / "fingerprints" / artifact, release_dir / "h3/fingerprints" / artifact)
    verification_files = sorted((output_dir / "verification").glob(f"{prefix}__*.json"))
    if len(verification_files) != 12:
        raise RuntimeError(f"H3 archive requires 12 verification files, found {len(verification_files)}")
    for path in verification_files:
        copy_required(path, release_dir / "h3/verification" / path.name)
    copy_required(output_dir / "final_mmd_report.json", release_dir / "h3/final_mmd_report.json")
    copy_required(output_dir / "h3.log", release_dir / "h3/h3.log")
    copy_required(
        output_dir / "finetuning_adapters/adapter_registry.json",
        release_dir / "h3/adapter_registry.json",
    )
    copy_required(
        output_dir / "finetuning_adapters/training_reports_fingerprint_14b_h3_endpoint.json",
        release_dir / "h3/training_reports.json",
    )
    (release_dir / "H3_FINAL_REPORT.json").write_text(
        json.dumps(h3, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    combined = {
        "schema_version": "fingerprint_g3_h3_archive_1.0",
        "git_revision": git_revision(root),
        "g3": g3,
        "h3": h3,
        "claim_boundary": {
            "g3": "Recovery MCC12 selected from the exact byte-frozen F2 legacy-retained 19-prompt pool.",
            "h3": "Fixed registered model-state instances; H3 is not a population detection-rate estimate.",
        },
        "excluded_large_artifacts": [
            "raw activation profiles (hash/row/size manifest retained)",
            "activation checkpoints",
            "LoRA adapter weight tensors",
        ],
    }
    (release_dir / "FINAL_REPORT.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(release_dir)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive terminal G3/H3 recovery fingerprint evidence")
    parser.add_argument("--config", default="configs/fingerprint_g3_qwen14b_mcc.yaml")
    parser.add_argument("--output-dir", default="results/fingerprint_g3_qwen14b_mcc_20260808")
    parser.add_argument("--release-dir", default="reproducibility/fingerprint_g3_qwen14b_mcc_20260808")
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
