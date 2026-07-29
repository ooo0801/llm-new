from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    invalid = [
        row
        for row in rows
        if (
            row.get("optimized_prompt") != row.get("initial_prompt")
            and float(row.get("edit_ratio", 0.0)) == 0.0
        )
    ]
    valid = [row for row in rows if row not in invalid]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in valid),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "retained_rows": len(valid),
                "removed_inconsistent_rows": len(invalid),
                "removed_prompt_ids": [row["prompt_id"] for row in invalid],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
