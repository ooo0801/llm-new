from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


FROZEN_FILES = [
    "configs/fingerprint_g_qwen14b_mcc.yaml",
    "experiments/stable-components-14b-g1/protocol.md",
    "experiments/stratified-mmd-14b-h1/protocol.md",
]

G1_PROFILE_FILES = [
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


def detect_g1_terminal(output_dir: Path) -> dict[str, Any]:
    audit_path = output_dir / "activation_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("G1 has no terminal activation audit")
    audit = load_json(audit_path)
    if audit.get("passed") is not True:
        return {
            "terminal_state": "activation_no_go",
            "hypothesis_supported": False,
            "activation_audit": audit,
            "global_universe": None,
            "mcc_selection": None,
        }

    universe_path = output_dir / "global_component_universe.json"
    if not universe_path.is_file():
        raise RuntimeError("activation gate passed but G1 has no terminal universe report")
    universe = load_json(universe_path)
    if universe.get("gates", {}).get("passed") is not True:
        return {
            "terminal_state": "universe_no_go",
            "hypothesis_supported": False,
            "activation_audit": audit,
            "global_universe": universe,
            "mcc_selection": None,
        }

    selection_path = output_dir / "mcc_selection.json"
    if not selection_path.is_file():
        raise RuntimeError("universe gate passed but G1 has no terminal MCC report")
    selection = load_json(selection_path)
    supported = selection.get("passed") is True
    return {
        "terminal_state": "g1_go" if supported else "selection_no_go",
        "hypothesis_supported": supported,
        "activation_audit": audit,
        "global_universe": universe,
        "mcc_selection": selection,
    }


def detect_h1_terminal(output_dir: Path, g1_report: dict[str, Any]) -> dict[str, Any]:
    if g1_report.get("hypothesis_supported") is not True:
        raise RuntimeError("H1 cannot be archived because the conditional G1 gate did not pass")
    report_path = output_dir / "final_mmd_report.json"
    if not report_path.is_file():
        raise RuntimeError("H1 has no terminal MMD report")
    report = load_json(report_path)
    if report.get("technical_passed") is not True:
        terminal_state = "h1_technical_failure"
    elif report.get("hypothesis_supported") is True:
        terminal_state = "h1_go"
    else:
        terminal_state = "h1_no_go"
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


def archive(root: Path, output_dir: Path, release_dir: Path, stage: str) -> dict[str, Any]:
    release_dir.mkdir(parents=True, exist_ok=True)
    for relative in FROZEN_FILES:
        copy_required(root / relative, release_dir / "frozen" / relative)

    g1 = detect_g1_terminal(output_dir)
    for relative in G1_PROFILE_FILES:
        copy_required(output_dir / relative, release_dir / "g1" / relative)
    copy_required(output_dir / "activation_audit.json", release_dir / "g1/activation_audit.json")
    if (output_dir / "global_component_universe.json").is_file():
        copy_required(output_dir / "global_component_universe.json", release_dir / "g1/global_component_universe.json")
    if (output_dir / "mcc_selection.json").is_file():
        copy_required(output_dir / "mcc_selection.json", release_dir / "g1/mcc_selection.json")
    g1_log = output_dir / "g1.log"
    if g1_log.is_file():
        copy_required(g1_log, release_dir / "g1/g1.log")
    (release_dir / "G1_FINAL_REPORT.json").write_text(
        json.dumps(g1, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    h1 = None
    if stage == "h1":
        h1 = detect_h1_terminal(output_dir, g1)
        fingerprint = output_dir / "fingerprints/global_mcc_14b.json"
        copy_required(fingerprint, release_dir / "h1/fingerprints/global_mcc_14b.json")
        verification_files = sorted((output_dir / "verification").glob("global_mcc_14b__*.json"))
        if len(verification_files) != 12:
            raise RuntimeError(f"H1 archive requires 12 verification files, found {len(verification_files)}")
        for path in verification_files:
            copy_required(path, release_dir / "h1/verification" / path.name)
        copy_required(output_dir / "final_mmd_report.json", release_dir / "h1/final_mmd_report.json")
        registry = output_dir / "finetuning_adapters/adapter_registry.json"
        if registry.is_file():
            copy_required(registry, release_dir / "h1/adapter_registry.json")
        h1_log = output_dir / "h1.log"
        if h1_log.is_file():
            copy_required(h1_log, release_dir / "h1/h1.log")
        (release_dir / "H1_FINAL_REPORT.json").write_text(
            json.dumps(h1, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    combined = {
        "schema_version": "fingerprint_14b_followup_archive_1.0",
        "git_revision": git_revision(root),
        "g1": g1,
        "h1": h1,
        "claim_boundary": {
            "g1": "Direct transfer of frozen 7B component thresholds to the fixed 14B revision.",
            "h1": "Fixed registered model-state instances; not a population detection-rate estimate.",
        },
        "excluded_large_artifacts": ["activation checkpoints", "LoRA adapter weight tensors"],
    }
    (release_dir / "FINAL_REPORT.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(release_dir)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive terminal Qwen2.5-14B G1/H1 evidence")
    parser.add_argument("--stage", choices=["g1", "h1"], required=True)
    parser.add_argument("--output-dir", default="results/fingerprint_g_qwen14b_mcc_20260807")
    parser.add_argument("--release-dir", default="reproducibility/fingerprint_g_qwen14b_mcc_20260807")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    release_dir = Path(args.release_dir)
    if not release_dir.is_absolute():
        release_dir = ROOT / release_dir
    report = archive(ROOT, output_dir, release_dir, args.stage)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
