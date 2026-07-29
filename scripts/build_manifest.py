from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from _bootstrap import ROOT


def tracked_paths() -> list[Path]:
    """Return Git-tracked files so local results and model artifacts stay out."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    paths = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / raw_path.decode("utf-8")
        if path.name != "PROJECT_MANIFEST.json" and path.is_file():
            paths.append(path)
    return sorted(paths)


def main() -> None:
    entries = []
    for path in tracked_paths():
        content = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    payload = {
        "schema_version": "2.0",
        "scope": "git_tracked_files_except_manifest",
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }
    (ROOT / "PROJECT_MANIFEST.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ["file_count", "total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
