from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import write_json
from llm_integrity.modeling import generate_texts, load_model, model_metadata


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def comparison(seed: int, observed: list[str], expected: list[str]) -> dict[str, Any]:
    matches = [left == right for left, right in zip(observed, expected)]
    return {
        "seed": seed,
        "responses": len(observed),
        "exact_matches": sum(matches),
        "all_exact": all(matches),
        "observed_sha256": digest(observed),
        "expected_sha256": digest(expected),
        "mismatch_indices": [index for index, match in enumerate(matches) if not match],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--intact-verification", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    fingerprint = ModelFingerprint.load(project_path(args.fingerprint))
    verification = json.loads(project_path(args.intact_verification).read_text(encoding="utf-8"))
    prompts = [entry.prompt for entry in fingerprint.entries]
    reference_seeds = [int(value) for value in fingerprint.metadata["reference_seeds"]]
    target_seeds = [int(value) for value in verification["target_seeds"]]
    target_by_key = {
        (int(row["repetition"]), str(row["prompt_id"])): str(row["response"])
        for row in verification["response_records"]
    }
    report: dict[str, Any] = {
        "schema_version": "h3_intact_replay_1.0",
        "classification": "engineering_reproducibility_diagnostic",
        "reference": [],
        "target": [],
    }
    bundle = load_model(config["model"])
    try:
        report["model"] = model_metadata(bundle)
        for repetition, seed in enumerate(reference_seeds):
            observed = generate_texts(bundle, prompts, fingerprint.generation_config, seed=seed)
            expected = [entry.reference_responses[repetition] for entry in fingerprint.entries]
            report["reference"].append(comparison(seed, observed, expected))
        for repetition, seed in enumerate(target_seeds):
            observed = generate_texts(bundle, prompts, fingerprint.generation_config, seed=seed)
            expected = [target_by_key[(repetition, entry.prompt_id)] for entry in fingerprint.entries]
            report["target"].append(comparison(seed, observed, expected))
    finally:
        bundle.close()
    report["reference_all_exact"] = all(row["all_exact"] for row in report["reference"])
    report["target_all_exact"] = all(row["all_exact"] for row in report["target"])
    report["all_exact"] = report["reference_all_exact"] and report["target_all_exact"]
    write_json(ROOT / args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
