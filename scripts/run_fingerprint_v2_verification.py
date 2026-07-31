from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import time

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import read_jsonl, write_json
from llm_integrity.modeling import load_model
from llm_integrity.paper_variant_executor import load_manifest_variant
from llm_integrity.verification import verify_model


def load_registry(path):
    if path is None or not path.exists():
        return {}
    return {
        str(key): str(value)
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one isolated V2 global-fingerprint model state")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--adapter-registry")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    fingerprint = ModelFingerprint.load(output_dir / "fingerprints" / "global_mcc_v2.json")
    manifests = read_jsonl(project_path(config["data"]["attack_manifest"]))
    matches = [row for row in manifests if str(row["variant_id"]) == args.variant_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest row for {args.variant_id!r}, got {len(matches)}")
    manifest = matches[0]
    family = str(manifest["family"])
    registry_path = project_path(args.adapter_registry) if args.adapter_registry else None
    registry = load_registry(registry_path)
    loaded = None
    bundle = None
    start = time.perf_counter()
    try:
        if family == "intact":
            bundle = load_model(config["model"])
            realization = {
                "variant_id": args.variant_id,
                "family": "intact",
                "execution_mode": "fresh_base_reload",
                "realized_method": "none",
                "isolated_base_reload": True,
            }
        else:
            loaded = load_manifest_variant(
                config["model"],
                manifest,
                adapter_path=registry.get(args.variant_id),
            )
            bundle = loaded.bundle
            realization = asdict(loaded.report)
        repetitions = int(config["statistics"]["repetitions"])
        seed_base = int(config["statistics"]["target_seed_base"])
        target_seeds = [seed_base + index for index in range(repetitions)]
        result = verify_model(
            fingerprint,
            bundle,
            config["statistics"],
            config["features"],
            repetitions=repetitions,
            target_seeds=target_seeds,
        )
    finally:
        if loaded is not None:
            loaded.close()
        elif bundle is not None:
            bundle.close()
    elapsed = time.perf_counter() - start
    response_text = "\n".join(row["response"] for row in result.response_records)
    payload = {
        "schema_version": "fingerprint_v2_global_verification_1.0",
        "variant_id": args.variant_id,
        "family": family,
        "ground_truth_modified": family != "intact",
        "predicted_modified": result.modified,
        "correct": result.modified == (family != "intact"),
        "primary_test": result.test.as_dict(),
        "secondary_tests": {
            key: value.as_dict()
            for key, value in result.secondary_tests.items()
        },
        "prompts": result.prompts,
        "repetitions": result.repetitions,
        "queries": result.prompts * result.repetitions,
        "target_seeds": target_seeds,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        "response_records": result.response_records,
        "variant_realization": realization,
        "elapsed_seconds": elapsed,
    }
    target = output_dir / "verification" / f"global_mcc_v2__{args.variant_id}.json"
    write_json(target, payload)
    print(
        json.dumps(
            {
                "variant_id": args.variant_id,
                "family": family,
                "predicted_modified": result.modified,
                "correct": payload["correct"],
                "primary_p_value": result.test.p_value,
                "primary_statistic": result.test.statistic,
                "pooled_p_value": (
                    result.secondary_tests["pooled_mmd"].p_value
                    if "pooled_mmd" in result.secondary_tests
                    else None
                ),
                "queries": payload["queries"],
                "elapsed_seconds": elapsed,
                "output": str(target),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
