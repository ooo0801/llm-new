from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"
REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
SHARDS = [f"model-{index:05d}-of-00017.safetensors" for index in range(1, 18)]
REQUIRED = [
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    temporary.replace(path)


def validate(snapshot: Path, *, hash_weights: bool) -> dict[str, object]:
    if snapshot.name != REVISION:
        raise RuntimeError(f"unexpected snapshot revision: {snapshot}")
    missing = [name for name in SHARDS + REQUIRED if not (snapshot / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing Qwen32B files: {missing}")
    shards = []
    for name in SHARDS:
        path = snapshot / name
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"empty shard: {name}")
        shards.append({"name": name, "bytes": size, "sha256": sha256(path) if hash_weights else None, "blob": str(path.resolve())})
    small = {name: {"bytes": (snapshot / name).stat().st_size, "sha256": sha256(snapshot / name)} for name in REQUIRED}
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if int(config.get("num_hidden_layers", -1)) != 64:
        raise RuntimeError(f"unexpected 32B layer count: {config.get('num_hidden_layers')}")
    tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template = str(tokenizer_config.get("chat_template", ""))
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "weight_shards": shards,
        "weight_bytes": sum(int(row["bytes"]) for row in shards),
        "metadata_files": small,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "vocab_size": config.get("vocab_size"),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/huggingface")
    parser.add_argument("--output", default="results/experiment_h6_qwen32b_reconstruction_20260812/00_environment/model_download.json")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--hash-weights", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    cache = Path(args.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    base = {"status": "running", "started_at": now(), "model_id": MODEL_ID, "revision": REVISION, "cache_dir": str(cache), "max_workers": args.max_workers, "hash_weights": args.hash_weights, "disk_before": shutil.disk_usage(cache)._asdict(), "pid": os.getpid()}
    atomic_json(output, base)
    print(json.dumps(base, ensure_ascii=False), flush=True)
    try:
        snapshot = Path(snapshot_download(repo_id=MODEL_ID, revision=REVISION, cache_dir=str(cache), max_workers=args.max_workers)).resolve()
        result = {**base, **validate(snapshot, hash_weights=args.hash_weights), "status": "complete", "finished_at": now(), "disk_after": shutil.disk_usage(cache)._asdict()}
        atomic_json(output, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result = {**base, "status": "failed", "finished_at": now(), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "disk_after": shutil.disk_usage(cache)._asdict()}
        atomic_json(output, result)
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
