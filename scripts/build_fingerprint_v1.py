from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.fingerprint import FingerprintEntry, ModelFingerprint
from llm_integrity.io import read_jsonl, write_json
from llm_integrity.mcc import coverage_rate, greedy_mcc
from llm_integrity.modeling import generate_texts, load_model, model_metadata
from llm_integrity.task_validation import evaluate_task


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select MCC12 and build the V1 reference fingerprint")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v1_qwen_7b.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    prompts = read_jsonl(project_path(config["data"]["strict16_manifest"]))
    profiles = read_jsonl(output_dir / "activation_profiles.jsonl")
    by_id = {str(row["id"]): row for row in prompts}
    if len(by_id) != 16:
        raise ValueError(f"Expected 16 frozen prompts, got {len(by_id)}")

    grouped: dict[str, list[set[str]]] = defaultdict(list)
    diagnostics: dict[str, list[dict]] = defaultdict(list)
    for row in profiles:
        prompt_id = str(row["id"])
        if prompt_id in by_id:
            grouped[prompt_id].append(set(row["components"]))
            diagnostics[prompt_id].append(dict(row["diagnostics"]))
    expected_repetitions = int(config["activations"]["repetitions"])
    frequency = float(config["activations"]["stable_component_frequency"])
    stable_components: dict[str, set[str]] = {}
    selection_metadata = {}
    for prompt_id in sorted(by_id):
        values = grouped.get(prompt_id, [])
        if len(values) != expected_repetitions:
            raise ValueError(f"Expected {expected_repetitions} activation profiles for {prompt_id}, got {len(values)}")
        counts = Counter(component for components in values for component in components)
        required = max(1, int(expected_repetitions * frequency + 0.999999))
        stable = {component for component, count in counts.items() if count >= required}
        if not stable:
            raise ValueError(f"No stable activation components for {prompt_id}")
        stable_components[prompt_id] = stable
        pair_union = values[0] | values[-1]
        stability = len(values[0] & values[-1]) / len(pair_union) if pair_union else 1.0
        selection_metadata[prompt_id] = {
            "category": by_id[prompt_id].get("category"),
            "stability": stability,
            "token_count": int(diagnostics[prompt_id][0]["token_count"]),
        }

    k = int(config["fingerprint"]["selected_size"])
    weights = config["fingerprint"]["weights"]
    selection = greedy_mcc(stable_components, k, weights, selection_metadata)
    if len(selection.selected_ids) != k:
        raise RuntimeError(f"MCC selected {len(selection.selected_ids)} prompts instead of {k}")
    curves = {}
    for curve_k in (4, 8, 12, 16):
        result = greedy_mcc(stable_components, curve_k, weights, selection_metadata)
        curves[str(curve_k)] = {
            "selected_ids": result.selected_ids,
            "coverage_rate": coverage_rate(result.selected_ids, stable_components),
            "weighted_marginal_gain": sum(result.marginal_gains),
            "covered_components": len(result.covered_components),
        }

    reference_repetitions = int(config["fingerprint"]["reference_repetitions"])
    seed_base = int(config["fingerprint"]["reference_seed_base"])
    reference_seeds = [seed_base + index for index in range(reference_repetitions)]
    selected_rows = [by_id[prompt_id] for prompt_id in selection.selected_ids]
    selected_prompts = [str(row["prompt"]) for row in selected_rows]
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
            print(json.dumps({"stage": "reference_generation", "repetition": repetition, "total": reference_repetitions, "seed": seed}, ensure_ascii=False), flush=True)
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
                components=sorted(stable_components[prompt_id]),
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
        selection_method="mcc",
        entries=entries,
        generation_config=dict(config["generation"]),
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "experiment_version": "fingerprint_v1_20260730",
            "model": metadata,
            "reference_seeds": reference_seeds,
            "selection_trace": selection.trace,
            "coverage_curves": curves,
            "feature_config": dict(config["features"]),
            "statistics_config": dict(config["statistics"]),
        },
    )
    fingerprint_path = output_dir / "fingerprints" / "mcc_v1.json"
    fingerprint.save(fingerprint_path)
    response_count = sum(len(entry.reference_responses) for entry in entries)
    selection_report = {
        "candidate_prompts": len(prompts),
        "selected_prompts": len(entries),
        "selected_ids": selection.selected_ids,
        "selected_categories": [entry.category for entry in entries],
        "trace": selection.trace,
        "coverage_curves": curves,
        "stable_components_all": len(set().union(*stable_components.values())),
        "stable_components_selected": len(selection.covered_components),
        "reference_responses": response_count,
        "reference_task_pass_rate": sum(check["passed"] for checks in task_checks for check in checks) / response_count,
        "reference_response_sha256": sha256_text("\n".join(response for values in reference_responses for response in values)),
        "fingerprint": str(fingerprint_path),
        "passed": len(entries) == 12 and response_count == 120,
    }
    write_json(output_dir / "mcc_selection.json", selection_report)
    print(json.dumps(selection_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
