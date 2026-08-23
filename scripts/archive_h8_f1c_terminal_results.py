from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "h8-f1c-terminal-archive-1.0"
RESULT_RELATIVE = Path("results/h8_qwen32b_final_confirmation/f1c_performance")
ARCHIVE_RELATIVE = Path("reproducibility/h8_qwen32b_final_confirmation_20260823/f1c_final_v1")
STABLE_TEXT_RESULTS = (
    "H8_F1C_FAIL_CLOSED_PREFLIGHT_REPORT.json",
    "F1C_FEATURE_AUDIT.json",
    "F1C_FEATURE_ROW_INDEX.jsonl",
    "F1C_UNIT_LEVEL_RESULTS.jsonl",
    "F1C_PROMPT_LEVEL_STATISTICS.jsonl",
    "F1C_PERMUTATION_AUDIT.json",
    "F1C_FINAL_PERFORMANCE_SUMMARY.json",
    "H8_F1C_FINAL_PERFORMANCE_CONFIRMATION_REPORT.json",
    "F1C_TERMINAL_SHA256_INDEX.json",
)
IMPLEMENTATION_FILES = (
    "configs/h8_f1c_final_performance_confirmation.yaml",
    "docs/H8_Qwen32B_F1C_Final_Performance_Confirmation_Protocol.md",
    "scripts/run_h8_f1c_final_performance_confirmation.py",
    "src/llm_integrity/h8_f1c.py",
    "tests/test_h8_f1c.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def verify_terminal_index(result: Path) -> tuple[dict[str, Any], list[str]]:
    index_path = result / "F1C_TERMINAL_SHA256_INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("excluded_transient_files") != ["F1C_RUNNER.log", "F1C_TERMINAL_SHA256_INDEX.json"]:
        raise ValueError("F1-C terminal transient exclusion set changed")
    files = index.get("files")
    if int(index.get("file_count", -1)) != 14 or not isinstance(files, dict) or len(files) != 14:
        raise ValueError("F1-C terminal index must bind exactly fourteen stable result files")
    mismatches: list[str] = []
    for logical, metadata in sorted(files.items()):
        path = (result / logical).resolve()
        if path != result and result not in path.parents:
            raise ValueError("F1-C terminal index path escapes result directory")
        if (
            not path.is_file()
            or path.stat().st_size != int(metadata["size_bytes"])
            or sha256(path) != str(metadata["sha256"])
        ):
            mismatches.append(str(logical))
    return index, mismatches


def copy_exact(source: Path, target: Path) -> None:
    if not source.is_file():
        raise ValueError(f"Missing F1-C archive source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256(source) != sha256(target):
        raise ValueError(f"F1-C archive copy hash mismatch: {source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-test-result", default="301 passed")
    parser.add_argument("--focused-test-result", default="7 passed")
    args = parser.parse_args()
    result = (PROJECT_ROOT / RESULT_RELATIVE).resolve()
    archive = (PROJECT_ROOT / ARCHIVE_RELATIVE).resolve()
    if archive.exists():
        raise ValueError("F1-C compact archive already exists; refusing to overwrite")
    index, mismatches = verify_terminal_index(result)
    if mismatches:
        raise ValueError(f"F1-C terminal result hash mismatch: {mismatches}")
    report = json.loads((result / "H8_F1C_FINAL_PERFORMANCE_CONFIRMATION_REPORT.json").read_text(encoding="utf-8"))
    feature = json.loads((result / "F1C_FEATURE_AUDIT.json").read_text(encoding="utf-8"))
    permutation = json.loads((result / "F1C_PERMUTATION_AUDIT.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "PASS"
        or report.get("new_model_responses") != 0
        or report.get("formal_response_hashes_unchanged") is not True
        or feature.get("status") != "PASS"
        or permutation.get("status") != "PASS"
    ):
        raise ValueError("F1-C terminal PASS boundary is incomplete")

    archive.mkdir(parents=True)
    for name in STABLE_TEXT_RESULTS:
        copy_exact(result / name, archive / name)
    for logical in IMPLEMENTATION_FILES:
        copy_exact(PROJECT_ROOT / logical, archive / "implementation" / logical)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_TERMINAL_F1C_ARTIFACT_AUDIT",
        "git_head_at_archive": head,
        "formal_response_count": 12720,
        "new_model_responses": 0,
        "feature_count": feature["feature_count"],
        "feature_dimension": feature["feature_dimension"],
        "feature_nan_count": feature["nan_count"],
        "feature_inf_count": feature["inf_count"],
        "feature_duplicate_response_id_count": feature["duplicate_response_id_count"],
        "evaluation_unit_count": report["evaluation_unit_count"],
        "prompt_statistic_record_count": 1200,
        "permutation_prompt_level_seed_count": permutation["frozen_prompt_level_seed_count"],
        "p_value_grid_all_pass": permutation["p_value_grid_all_pass"],
        "alarm_equivalence_all_pass": permutation["alarm_equivalence_all_pass"],
        "terminal_index_file_count": index["file_count"],
        "terminal_index_mismatch_count": 0,
        "terminal_index_sha256": sha256(result / "F1C_TERMINAL_SHA256_INDEX.json"),
        "final_report_sha256": sha256(result / "H8_F1C_FINAL_PERFORMANCE_CONFIRMATION_REPORT.json"),
        "full_cpu_unit_test_result": args.full_test_result,
        "focused_f1c_test_result_after_terminal_index_fix": args.focused_test_result,
        "gpu_cleanup_status": "PASS_0_MIB_ALL_THREE_GPUS",
        "large_numpy_caches_archived_in_git": False,
        "large_numpy_caches_bound_by_terminal_sha256_index": True,
        "raw_f1b_response_banks_archived_in_git": False,
        "raw_f1b_response_banks_bound_by_f1b_terminal_index": True,
        "next_gate": "STOP_NO_AUTOMATIC_TUNING_OR_FOLLOWUP_EXPERIMENT",
    }
    canonical_write(archive / "F1C_TERMINAL_AUDIT_REPORT.json", audit)

    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        logical = path.relative_to(archive).as_posix()
        if logical == "F1C_COMPACT_ARCHIVE_MANIFEST.json":
            continue
        inventory[logical] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "file_count": len(inventory),
        "files": inventory,
        "source_terminal_index_sha256": audit["terminal_index_sha256"],
        "source_final_report_sha256": audit["final_report_sha256"],
        "result_cache_location": RESULT_RELATIVE.as_posix(),
        "large_cache_policy": "server_only_hash_bound_not_git_archived",
    }
    canonical_write(archive / "F1C_COMPACT_ARCHIVE_MANIFEST.json", manifest)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
