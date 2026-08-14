from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from _bootstrap import ROOT


CORE_CATEGORIES = {
    "code",
    "instruction",
    "knowledge",
    "logic",
    "reasoning",
    "safety",
    "structured",
    "summary",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required H6 archive file is missing: {path}")
    return path


def validate_release(release_dir: Path) -> dict[str, Any]:
    paths = {
        "protocol": require_file(release_dir / "protocol.json"),
        "candidate_manifest": require_file(release_dir / "candidate_manifest.jsonl"),
        "calibration_build_manifest": require_file(
            release_dir / "calibration_build_manifest.jsonl"
        ),
        "calibration_audit_manifest": require_file(
            release_dir / "calibration_audit_manifest.jsonl"
        ),
        "final_test_report": require_file(
            release_dir / "confirmation/final_test_report.json"
        ),
        "prompt_gate_report": require_file(
            release_dir / "confirmation/prompt_gate_report.json"
        ),
        "confirmed_prompt_pool": require_file(
            release_dir / "confirmation/confirmed_prompt_pool.jsonl"
        ),
        "activation_audit": require_file(
            release_dir / "components/activation_audit.json"
        ),
        "global_component_universe": require_file(
            release_dir / "components/global_component_universe.json"
        ),
        "mcc_selection": require_file(release_dir / "components/mcc_selection.json"),
        "fingerprint": require_file(
            release_dir / "components/fingerprints/global_mcc_32b_h6.json"
        ),
        "frozen_config": require_file(
            release_dir / "frozen/configs/fingerprint_h6_qwen32b_mcc.yaml"
        ),
        "frozen_protocol": require_file(
            release_dir
            / "frozen/experiments/prompt-reconstruction-32b-h6/protocol.md"
        ),
    }

    protocol = load_json(paths["protocol"])
    final_test = load_json(paths["final_test_report"])
    gate = load_json(paths["prompt_gate_report"])
    pool = load_jsonl(paths["confirmed_prompt_pool"])
    audit = load_json(paths["activation_audit"])
    universe = load_json(paths["global_component_universe"])
    selection = load_json(paths["mcc_selection"])
    fingerprint = load_json(paths["fingerprint"])

    candidate_manifest = load_jsonl(paths["candidate_manifest"])
    calibration_build = load_jsonl(paths["calibration_build_manifest"])
    calibration_audit = load_jsonl(paths["calibration_audit_manifest"])

    assert protocol["schema_version"] == "fingerprint_h6_qwen32b_mcc_protocol_1.0"
    assert len(pool) == int(gate["confirmed_rows"]) == 39
    assert len({row["prompt_id"] for row in pool}) == len(pool)
    assert sha256(paths["confirmed_prompt_pool"]) == gate["candidate_source_sha256"]
    assert gate["candidate_source_sha256"] == protocol["candidate_source_sha256"]
    assert gate["hypothesis"] == "H6-P"
    assert gate["hypothesis_supported"] is True
    assert int(gate["confirmed_rows"]) >= int(gate["minimum_confirmed"])
    assert gate["technical_errors"] == []
    assert gate["missing_core_categories"] == []
    assert CORE_CATEGORIES <= set(gate["confirmed_by_category"])

    assert final_test["passed"] is True
    assert final_test["technical_passed"] is True
    assert final_test["scientific_gate_passed"] is True
    assert int(final_test["strict_accepted"]) == len(pool)
    assert int(final_test["rows"]) == int(gate["input_rows"]) == 43

    assert len(candidate_manifest) == int(protocol["candidate_rows"]) == len(pool)
    assert len(calibration_build) == int(protocol["calibration_build_prompts"]) == 280
    assert len(calibration_audit) == int(protocol["calibration_audit_prompts"]) == 56
    assert canonical_sha256(candidate_manifest) == protocol["candidate_manifest_sha256"]
    assert canonical_sha256(calibration_build) == protocol["calibration_build_sha256"]
    assert canonical_sha256(calibration_audit) == protocol["calibration_audit_sha256"]
    assert sha256(paths["frozen_config"]) == protocol["config_sha256"]

    assert audit["passed"] is True
    assert universe["gates"]["passed"] is True
    assert selection["passed"] is True
    assert selection["selection_deterministic"] is True
    assert selection["factorization_passed"] is True
    assert selection["missing_required_categories"] == []
    assert len(selection["selected_ids"]) == 12
    assert len(set(selection["selected_ids"])) == 12
    assert CORE_CATEGORIES <= set(selection["selected_categories"])
    assert selection["global_universe_sha256"] == universe["components_sha256"]
    assert fingerprint["metadata"]["global_universe_sha256"] == universe["components_sha256"]
    assert [entry["prompt_id"] for entry in fingerprint["entries"]] == selection[
        "selected_ids"
    ]
    assert len(fingerprint["entries"]) == 12

    artifacts = {
        name: {
            "path": path.relative_to(release_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
    }
    return {
        "schema_version": "fingerprint_h6_qwen32b_archive_1.0",
        "terminal_state": "h6_go_mcc12_complete",
        "hypotheses": {
            "H6-P": {
                "supported": True,
                "confirmed_rows": len(pool),
                "input_rows": int(gate["input_rows"]),
                "minimum_confirmed": int(gate["minimum_confirmed"]),
                "confirmed_by_category": gate["confirmed_by_category"],
                "missing_core_categories": [],
            },
            "H6-MCC12": {
                "supported": True,
                "selected_prompts": len(selection["selected_ids"]),
                "selection_deterministic": True,
                "factorization_passed": True,
                "activation_audit_passed": True,
                "global_component_gates_passed": True,
                "coverage_metrics": selection["coverage_metrics"],
            },
        },
        "model": {
            "name": universe["model_name"],
            "revision": universe["model_revision"],
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "H6 reconstructs a model-specific Qwen2.5-32B prompt pool and MCC12. "
            "It does not reuse 14B endpoints or component identities, and downstream "
            "modified-state detection verification remains a separate preregistered experiment."
        ),
        "excluded_large_artifacts": [
            "raw activation profiles (audited summaries and component universe retained)",
            "activation checkpoints",
            "LoRA adapter weight tensors",
            "model weight shards",
        ],
    }


def write_checksums(release_dir: Path) -> None:
    rows = []
    for path in sorted(
        item
        for item in release_dir.rglob("*")
        if item.is_file() and item.name != "SHA256SUMS"
    ):
        rows.append(f"{sha256(path)}  {path.relative_to(release_dir).as_posix()}")
    (release_dir / "SHA256SUMS").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def finalize(release_dir: Path, source_server_revision: str | None) -> dict[str, Any]:
    report = validate_release(release_dir)
    report["archive_git_revision"] = git_revision(ROOT)
    report["source_server_revision"] = source_server_revision
    (release_dir / "FINAL_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_checksums(release_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and finalize terminal H6 Qwen2.5-32B MCC12 evidence"
    )
    parser.add_argument(
        "--release-dir",
        default="reproducibility/fingerprint_h6_qwen32b_mcc_20260812",
    )
    parser.add_argument("--source-server-revision", default=None)
    args = parser.parse_args()
    release_dir = Path(args.release_dir)
    if not release_dir.is_absolute():
        release_dir = ROOT / release_dir
    report = finalize(release_dir, args.source_server_revision)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
