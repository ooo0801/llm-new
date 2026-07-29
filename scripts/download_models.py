from __future__ import annotations

import argparse
import json
import os

from _bootstrap import ROOT

from llm_integrity.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--include-replacements", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    names = {config["model"]["name"]}
    if args.include_replacements:
        for attack in config.get("attacks", []):
            if attack.get("type") == "replacement":
                names.add(attack["model"]["name"])
    from huggingface_hub import snapshot_download

    results = {}
    for name in sorted(names):
        results[name] = snapshot_download(
            repo_id=name,
            revision="main",
            cache_dir=os.environ.get("HF_HOME"),
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
