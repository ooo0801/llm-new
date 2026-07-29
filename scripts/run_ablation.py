from __future__ import annotations

import argparse
import json
import subprocess
import sys

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or execute the verification experiment matrix")
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repetitions", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    fingerprint_dir = project_path(config["output_dir"]) / "fingerprints"
    fingerprint_paths = sorted(fingerprint_dir.glob("*.json"))
    if not fingerprint_paths:
        methods = config["fingerprint"].get("methods", [])
        fingerprint_paths = [fingerprint_dir / f"{method}.json" for method in methods]
    attacks = ["intact"] + [item["name"] for item in config.get("attacks", [])]
    jobs = []
    for fingerprint in fingerprint_paths:
        method = fingerprint.stem
        for attack in attacks:
            command = [
                sys.executable,
                str(ROOT / "scripts/run_verification.py"),
                "--config",
                args.config,
                "--fingerprint",
                str(fingerprint),
                "--attack",
                attack,
                "--repetitions",
                str(args.repetitions),
            ]
            jobs.append({"method": method, "attack": attack, "command": command})
    manifest = project_path(config["output_dir"]) / "ablation_manifest.json"
    write_json(manifest, jobs)
    if args.execute:
        for index, job in enumerate(jobs, 1):
            print(f"[job] {index}/{len(jobs)} {job['method']} {job['attack']}", flush=True)
            subprocess.run(job["command"], cwd=ROOT, check=True)
    print(json.dumps({"manifest": str(manifest), "jobs": len(jobs), "executed": args.execute}, ensure_ascii=False))


if __name__ == "__main__":
    main()
