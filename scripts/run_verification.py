from __future__ import annotations

import argparse
import json

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import write_json
from llm_integrity.variants import load_variant
from llm_integrity.verification import verify_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    fingerprint = ModelFingerprint.load(project_path(args.fingerprint))
    attack = next((item for item in config.get("attacks", []) if item["name"] == args.attack), None)
    if attack is None:
        if args.attack == "intact":
            attack = {"name": "intact", "type": "none"}
        else:
            raise ValueError(f"Unknown attack: {args.attack}")
    bundle, report = load_variant(config["model"], attack)
    try:
        result = verify_model(
            fingerprint,
            bundle,
            config["statistics"],
            config["features"],
            repetitions=args.repetitions,
        )
    finally:
        bundle.close()
    payload = {
        "attack": args.attack,
        "selection_method": fingerprint.selection_method,
        "modified": result.modified,
        "test": result.test.as_dict(),
        "prompts": result.prompts,
        "repetitions": result.repetitions,
        "attack_report": report.__dict__ if report else None,
    }
    output_dir = project_path(config["output_dir"]) / "verification"
    output = output_dir / f"{fingerprint.selection_method}__{args.attack}.json"
    write_json(output, payload)
    print(json.dumps(payload | {"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
