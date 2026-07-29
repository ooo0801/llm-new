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
    parser.add_argument("--id", action="append", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    by_id = {str(row["id"]): row for row in rows}
    missing = [prompt_id for prompt_id in args.id if prompt_id not in by_id]
    if missing:
        raise ValueError(f"Missing prompt ids: {missing}")
    selected = [by_id[prompt_id] for prompt_id in args.id]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected": len(selected),
                "prompt_ids": [str(row["id"]) for row in selected],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
