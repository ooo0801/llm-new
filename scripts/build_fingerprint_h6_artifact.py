from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.fingerprint import FingerprintEntry, ModelFingerprint
from llm_integrity.global_coverage import stable_prompt_components
from llm_integrity.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_h6_qwen32b_mcc.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    candidates = read_jsonl(project_path(config["data"]["candidate_manifest"]))
    by_id = {str(row["id"]): row for row in candidates}
    profiles = read_jsonl(output_dir / "activations" / "candidate_profiles.jsonl")
    stable = stable_prompt_components(profiles, repetitions=int(config["activations"]["repetitions"]), stable_frequency=float(config["activations"]["stable_component_frequency"]))
    selection = json.loads((output_dir / "mcc_selection.json").read_text(encoding="utf-8"))
    universe = json.loads((output_dir / "global_component_universe.json").read_text(encoding="utf-8"))
    if not selection.get("passed") or not universe.get("gates", {}).get("passed"):
        raise RuntimeError("H6 component universe and MCC gates must pass")
    selected_ids = [str(value) for value in selection["selected_ids"]]
    if len(selected_ids) != 12 or len(set(selected_ids)) != 12:
        raise ValueError("H6 selected IDs are not a unique MCC12")
    entries = []
    for prompt_id in selected_ids:
        row = by_id[prompt_id]
        entries.append(
            FingerprintEntry(
                prompt_id=prompt_id,
                prompt=str(row["prompt"]),
                category=row.get("category"),
                sensitivity=None,
                components=sorted(stable[prompt_id]),
                metadata={
                    "evaluator": row.get("evaluator"),
                    "expected_answer": row.get("expected_answer"),
                    "expected_contains": row.get("expected_contains"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "parent_source_prompt_id": row.get("parent_source_prompt_id"),
                },
            )
        )
    fingerprint = ModelFingerprint(
        model_name=str(config["model"]["name"]),
        model_revision=str(config["model"]["revision"]),
        selection_method="global_unweighted_mcc",
        entries=entries,
        generation_config=dict(config["generation"]),
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_version": config["experiment_name"],
            "reference_responses_built": False,
            "verification_deferred_to": "separately_preregistered_32b_integrity_stage",
            "selection_trace": selection["trace"],
            "coverage_metrics": selection["coverage_metrics"],
            "per_type_coverage": selection["per_type_coverage"],
            "global_universe_sha256": universe["components_sha256"],
        },
        schema_version="h6-32b-construction-1.0",
    )
    target = output_dir / "fingerprints" / config["fingerprint"]["artifact_name"]
    fingerprint.save(target)
    print(json.dumps({"fingerprint": str(target.relative_to(ROOT).as_posix()), "selected_ids": selected_ids, "passed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
