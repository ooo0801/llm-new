from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
EXPECTED_SHARDS = {
    "model-00001-of-00008.safetensors": 3_885_154_816,
    "model-00002-of-00008.safetensors": 3_995_327_992,
    "model-00003-of-00008.safetensors": 3_995_328_080,
    "model-00004-of-00008.safetensors": 3_995_338_432,
    "model-00005-of-00008.safetensors": 3_979_624_824,
    "model-00006-of-00008.safetensors": 3_995_328_080,
    "model-00007-of-00008.safetensors": 3_995_328_080,
    "model-00008-of-00008.safetensors": 1_698_703_696,
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_snapshot(snapshot: Path) -> dict[str, object]:
    if snapshot.name != REVISION:
        raise RuntimeError(f"unexpected snapshot revision path: {snapshot}")

    shards: list[dict[str, object]] = []
    for name, expected_size in EXPECTED_SHARDS.items():
        path = snapshot / name
        if not path.exists():
            raise FileNotFoundError(f"missing model shard: {path}")
        size = path.stat().st_size
        if size != expected_size:
            raise RuntimeError(
                f"wrong shard size for {name}: {size} != {expected_size}"
            )
        shards.append(
            {
                "name": name,
                "bytes": size,
                "is_symlink": path.is_symlink(),
                "blob": str(path.resolve()),
            }
        )

    required_small = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]
    missing_small = [name for name in required_small if not (snapshot / name).exists()]
    if missing_small:
        raise FileNotFoundError(f"missing metadata/tokenizer files: {missing_small}")

    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "weight_shards": shards,
        "weight_bytes": sum(int(item["bytes"]) for item in shards),
        "required_small_files": required_small,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", default="/root/autodl-tmp/huggingface"
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/prompt-transfer-14b-a1/results/"
            "model_download/download_result.json"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    started = now()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    cache = Path(args.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    base: dict[str, object] = {
        "status": "running",
        "started_at": started,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "cache_dir": str(cache),
        "max_workers": args.max_workers,
        "disk_before": shutil.disk_usage(cache)._asdict(),
        "pid": os.getpid(),
    }
    atomic_json(output, base)
    print(json.dumps(base, ensure_ascii=False), flush=True)

    try:
        snapshot_value = snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            cache_dir=str(cache),
            max_workers=args.max_workers,
        )
        validation = validate_snapshot(Path(snapshot_value).resolve())
        result = {
            **base,
            **validation,
            "status": "complete",
            "finished_at": now(),
            "disk_after": shutil.disk_usage(cache)._asdict(),
        }
        atomic_json(output, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result = {
            **base,
            "status": "failed",
            "finished_at": now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "disk_after": shutil.disk_usage(cache)._asdict(),
        }
        atomic_json(output, result)
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
