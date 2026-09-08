"""Download only public required 1.5B inference files, resolve revision first."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/huggingface")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


def main():
    from huggingface_hub import HfApi, snapshot_download
    out = Path("results/stage2_qwen15b_20260905")
    out.mkdir(parents=True, exist_ok=True)
    plan = out / "ASSET_DOWNLOAD_PLAN.json"
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    if plan.exists():
        revision = json.loads(plan.read_text())["revision"]
    else:
        revision = HfApi().model_info(name).sha
        plan.write_text(json.dumps({"model": name, "revision": revision,
            "download_endpoint": os.environ["HF_ENDPOINT"],
            "created_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    print("FROZEN_MODEL", name, revision, flush=True)
    path = Path(snapshot_download(name, revision=revision, max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.json", "merges.txt"]))
    files = {}
    for p in sorted(path.iterdir()):
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(8*1024*1024), b""):
                    h.update(chunk)
            files[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    report = {"status": "COMPLETE", "model": name, "revision": revision,
              "path": str(path), "files": files}
    (out / "ASSETS.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
