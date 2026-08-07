from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.fingerprint import FingerprintEntry, ModelFingerprint
from llm_integrity.global_coverage import stable_prompt_components
from llm_integrity.io import read_jsonl, write_json
from llm_integrity.modeling import generate_texts, load_model, model_metadata
from llm_integrity.task_validation import evaluate_task


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the global-calibrated V2 reference fingerprint")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    candidate_key = "candidate_manifest" if "candidate_manifest" in config["data"] else "strict16_manifest"
    prompts = read_jsonl(project_path(config["data"][candidate_key]))
    by_id = {str(row["id"]): row for row in prompts}
    profiles = read_jsonl(output_dir / "activations" / "candidate_profiles.jsonl")
    stable = stable_prompt_components(
        profiles,
        repetitions=int(config["activations"]["repetitions"]),
        stable_frequency=float(config["activations"]["stable_component_frequency"]),
    )
    selection_path = output_dir / "mcc_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    universe = json.loads((output_dir / "global_component_universe.json").read_text(encoding="utf-8"))
    if not selection.get("passed") or not universe.get("gates", {}).get("passed"):
        raise RuntimeError("V2 coverage and selection gates must pass before fingerprint construction")
    selected_ids = [str(value) for value in selection["selected_ids"]]
    expected_size = int(config["fingerprint"]["selected_size"])
    if len(selected_ids) != expected_size or len(set(selected_ids)) != expected_size:
        raise ValueError("V2 selected prompt IDs are not a unique MCC12")
    selected_rows = [by_id[prompt_id] for prompt_id in selected_ids]
    selected_prompts = [str(row["prompt"]) for row in selected_rows]

    reference_repetitions = int(config["fingerprint"]["reference_repetitions"])
    seed_base = int(config["fingerprint"]["reference_seed_base"])
    reference_seeds = [seed_base + index for index in range(reference_repetitions)]
    reference_responses = [[] for _ in selected_rows]
    task_checks = [[] for _ in selected_rows]
    bundle = load_model(config["model"])
    try:
        metadata = model_metadata(bundle)
        for repetition, seed in enumerate(reference_seeds):
            generated = generate_texts(bundle, selected_prompts, config["generation"], seed=seed)
            for index, response in enumerate(generated):
                reference_responses[index].append(response)
                passed, reason = evaluate_task(selected_rows[index], response)
                task_checks[index].append({"passed": passed, "reason": reason})
            print(
                json.dumps(
                    {
                        "stage": "v2_reference_generation",
                        "repetition": repetition,
                        "total": reference_repetitions,
                        "seed": seed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        bundle.close()

    entries = []
    for index, row in enumerate(selected_rows):
        prompt_id = str(row["id"])
        entries.append(
            FingerprintEntry(
                prompt_id=prompt_id,
                prompt=str(row["prompt"]),
                category=row.get("category"),
                sensitivity=float(row.get("v6_hybrid_gain", 0.0)),
                components=sorted(stable[prompt_id]),
                reference_responses=reference_responses[index],
                reference_response_seeds=reference_seeds,
                metadata={
                    "evaluator": row.get("evaluator"),
                    "expected_answer": row.get("expected_answer"),
                    "expected_contains": row.get("expected_contains"),
                    "initial_prompt": row.get("initial_prompt"),
                    "prompt_sha256": row.get("prompt_sha256"),
                    "v6_nondegraded_families": row.get("v6_nondegraded_families"),
                    "reference_task_checks": task_checks[index],
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
            "experiment_version": str(config.get("experiment_name", "fingerprint_v2_global_20260731")),
            "v1_parent_tag": config["fingerprint"].get("parent_tag", "v1-fingerprint-closed-loop"),
            "model": metadata,
            "reference_seeds": reference_seeds,
            "selection_trace": selection["trace"],
            "coverage_metrics": selection["coverage_metrics"],
            "per_type_coverage": selection["per_type_coverage"],
            "global_universe_sha256": universe["components_sha256"],
            "feature_config": dict(config["features"]),
            "statistics_config": dict(config["statistics"]),
        },
        schema_version="2.0",
    )
    artifact_name = str(config["fingerprint"].get("artifact_name", "global_mcc_v2.json"))
    fingerprint_path = output_dir / "fingerprints" / artifact_name
    fingerprint.save(fingerprint_path)
    response_count = sum(len(entry.reference_responses) for entry in entries)
    selection["reference_responses"] = response_count
    selection["reference_task_pass_rate"] = (
        sum(check["passed"] for checks in task_checks for check in checks) / response_count
    )
    selection["reference_response_sha256"] = sha256_text(
        "\n".join(response for values in reference_responses for response in values)
    )
    selection["reference_seeds"] = reference_seeds
    selection["fingerprint"] = str(fingerprint_path.relative_to(ROOT).as_posix())
    selection["passed"] = bool(selection["passed"] and response_count == expected_size * reference_repetitions)
    write_json(selection_path, selection)
    print(
        json.dumps(
            {
                "selected_ids": selected_ids,
                "reference_responses": response_count,
                "reference_task_pass_rate": selection["reference_task_pass_rate"],
                "fingerprint": str(fingerprint_path),
                "passed": selection["passed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
