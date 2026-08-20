#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _bootstrap import ROOT
from llm_integrity.h8_d1c import make_split_manifest
from llm_integrity.h8_score_calibration import load_frozen_mmd_measurement


DEFAULT_CONFIG = ROOT / "configs" / "h8_d1c_score_parameter_fit_audit.yaml"


def resolve(logical: str) -> Path:
    path = (ROOT / logical).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError("Configured path escapes repository")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    frozen = config["frozen_inputs"]
    binding = load_frozen_mmd_measurement(
        resolve(frozen["mmd_archive"]),
        expected_manifest_sha256=frozen["mmd_manifest_sha256"],
        expected_mmd_implementation_commit=frozen["mmd_implementation_commit"],
    )
    split = config["cpu_split"]
    manifest = make_split_manifest(
        binding.prompt_ids,
        fit_root_seed=int(split["fit_root_seed_uint64"]),
        audit_root_seed=int(split["audit_root_seed_uint64"]),
        trials_per_stream=int(split["trials_per_structure_prompt_role"]),
    )
    output = resolve(split["manifest"])
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise ValueError("Existing D1-C split manifest differs; refusing overwrite")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8", newline="\n")
    import hashlib

    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), **manifest["payload"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

