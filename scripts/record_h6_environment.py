from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from _bootstrap import ROOT
from llm_integrity.config import load_config


def command(*args: str) -> str:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_h6_qwen32b_development.yaml")
    parser.add_argument("--cache-dir", default="/root/autodl-tmp/huggingface")
    parser.add_argument("--output", default="results/experiment_h6_qwen32b_reconstruction_20260812/00_environment/environment.json")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    model = config["model"]
    snapshot = Path(args.cache_dir) / "models--Qwen--Qwen2.5-32B-Instruct" / "snapshots" / str(model["revision"])
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    files = ["config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    file_hashes = {name: sha256(snapshot / name) for name in files}
    tokenizer_config = json.loads((snapshot / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template = str(tokenizer_config.get("chat_template", ""))
    try:
        import accelerate
        accelerate_version = accelerate.__version__
    except ImportError:
        accelerate_version = None
    try:
        import bitsandbytes
        bitsandbytes_version = bitsandbytes.__version__
    except ImportError:
        bitsandbytes_version = None
    report = {
        "schema_version": "experiment_h6_environment_1.0",
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_revision": command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        "git_branch": command("git", "-C", str(ROOT), "branch", "--show-current"),
        "model": {"name": model["name"], "revision": model["revision"], "dtype": model["dtype"], "device_map": model["device_map"], "max_memory": model["max_memory"], "eager_attention": model["eager_attention"], "snapshot": str(snapshot), "files_sha256": file_hashes, "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest()},
        "software": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "accelerate": accelerate_version, "bitsandbytes": bitsandbytes_version},
        "gpu": {"count": torch.cuda.device_count(), "devices": [{"index": index, "name": torch.cuda.get_device_name(index), "total_memory": torch.cuda.get_device_properties(index).total_memory} for index in range(torch.cuda.device_count())], "topology": command("nvidia-smi", "topo", "-m")},
        "no_cpu_or_disk_offload_required": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
