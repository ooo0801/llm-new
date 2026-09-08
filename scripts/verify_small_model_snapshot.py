"""Verify the published source/evidence snapshot without models or GPU calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "reproducibility/small-model-stage2-20260908"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = read(ARCHIVE / "SOURCE_PROVENANCE.json")
    failures = []
    for row in manifest["files"]:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append({"path": row["path"], "reason": "missing/unsafe path"})
        elif digest(path) != row["sha256"]:
            failures.append({"path": row["path"], "reason": "hash mismatch"})
    evidence = ARCHIVE / "evidence/stage2_three_proxy_ceiling_v3_20260906"
    plan = read(evidence / "PLAN.json")
    for name, expected in plan["source_hashes"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != expected:
            failures.append({"path": name, "reason": "frozen search source mismatch"})
    promotion = read(evidence / "PROMOTION.json")
    matrix = evidence / "pilot_behavior/evaluation/PILOT_MATRIX.json"
    if digest(matrix) != promotion["pilot_matrix_sha256"]:
        failures.append({"path": str(matrix.relative_to(ROOT)), "reason": "promotion matrix mismatch"})
    print(json.dumps({"files_checked": len(manifest["files"]),
                      "frozen_search_sources_checked": len(plan["source_hashes"]),
                      "decision": promotion["decision"], "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
