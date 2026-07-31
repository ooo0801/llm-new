from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from statistics import mean

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.global_coverage import (
    COMPONENT_KINDS,
    component_counts,
    coverage_metrics,
    cumulative_coverage_curve,
    per_type_coverage,
    restrict_to_universe,
    stable_prompt_components,
    type_balanced_weights,
    union_components,
)
from llm_integrity.io import read_jsonl, write_json
from llm_integrity.mcc import coverage_rate, greedy_mcc


def canonical_sha256(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_stable(config: dict, output_dir, split: str) -> dict[str, set[str]]:
    rows = read_jsonl(output_dir / "activations" / f"{split}_profiles.jsonl")
    return stable_prompt_components(
        rows,
        repetitions=int(config["activations"]["repetitions"]),
        stable_frequency=float(config["activations"]["stable_component_frequency"]),
    )


def selection_metadata(candidate_profiles: list[dict], prompts: dict[str, dict]) -> dict[str, dict]:
    diagnostics: dict[str, list[dict]] = defaultdict(list)
    values: dict[str, list[set[str]]] = defaultdict(list)
    for row in candidate_profiles:
        prompt_id = str(row["id"])
        diagnostics[prompt_id].append(dict(row["diagnostics"]))
        values[prompt_id].append(set(row["components"]))
    result = {}
    for prompt_id, source in prompts.items():
        profiles = values[prompt_id]
        intersection = set.intersection(*profiles)
        union = set.union(*profiles)
        result[prompt_id] = {
            "category": source.get("category"),
            "stability": len(intersection) / len(union) if union else 1.0,
            "token_count": int(diagnostics[prompt_id][0]["token_count"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V2 global universe and select global MCC12")
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_v2_global_qwen_7b.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    build_manifest = read_jsonl(project_path(config["data"]["calibration_build_manifest"]))
    audit_manifest = read_jsonl(project_path(config["data"]["calibration_audit_manifest"]))
    strict16 = read_jsonl(project_path(config["data"]["strict16_manifest"]))
    prompts = {str(row["id"]): row for row in strict16}
    if len(prompts) != 16:
        raise ValueError("V2 requires exactly 16 candidate prompts")

    activation_audit = json.loads((output_dir / "activation_audit.json").read_text(encoding="utf-8"))
    if not activation_audit.get("passed"):
        raise RuntimeError("V2 activation gate did not pass")
    build_components = load_stable(config, output_dir, "build")
    audit_components = load_stable(config, output_dir, "audit")
    candidate_components = load_stable(config, output_dir, "candidate")
    if set(build_components) != {str(row["id"]) for row in build_manifest}:
        raise ValueError("Build activation IDs differ from the frozen manifest")
    if set(audit_components) != {str(row["id"]) for row in audit_manifest}:
        raise ValueError("Audit activation IDs differ from the frozen manifest")
    if set(candidate_components) != set(prompts):
        raise ValueError("Candidate activation IDs differ from strict16")

    build_universe = union_components(build_components)
    audit_universe = union_components(audit_components)
    global_universe = build_universe | audit_universe
    audit_only = audit_universe - build_universe
    audit_novelty_rate = len(audit_only) / len(audit_universe) if audit_universe else 1.0
    build_audit_union = build_universe | audit_universe
    build_audit_jaccard = (
        len(build_universe & audit_universe) / len(build_audit_union)
        if build_audit_union
        else 1.0
    )
    ordered_ids = [str(row["id"]) for row in build_manifest + audit_manifest]
    all_calibration = dict(build_components) | dict(audit_components)
    curve = cumulative_coverage_curve(
        all_calibration,
        ordered_ids,
        batch_size=int(config["calibration"]["batch_size"]),
    )
    last_count = int(config["calibration"]["last_batches_for_saturation"])
    last_rows = curve[-last_count:]
    gain_threshold = float(config["calibration"]["max_last_batch_relative_gain"])
    novelty_threshold = float(config["calibration"]["max_audit_novelty_rate"])
    saturation_passed = all(float(row["relative_gain"]) <= gain_threshold for row in last_rows)
    audit_novelty_passed = audit_novelty_rate <= novelty_threshold
    universe_gate_passed = saturation_passed and audit_novelty_passed
    universe_report = {
        "schema_version": "global_component_universe_v2_1.0",
        "definition": "Union of stable V1-schema activation-observable components over the frozen independent V2 build and audit calibration prompts.",
        "claim_boundary": "Empirical global observable component estimate; not the mathematical set of every critical model component.",
        "model_name": str(config["model"]["name"]),
        "model_revision": str(config["model"]["revision"]),
        "component_id_schema": "activation_observable_v1",
        "calibration_build_prompts": len(build_manifest),
        "calibration_audit_prompts": len(audit_manifest),
        "calibration_prompts": len(ordered_ids),
        "repetitions": int(config["activations"]["repetitions"]),
        "stable_frequency": float(config["activations"]["stable_component_frequency"]),
        "component_count": len(global_universe),
        "component_counts_by_type": component_counts(global_universe),
        "components": sorted(global_universe),
        "components_sha256": canonical_sha256(sorted(global_universe)),
        "build_component_count": len(build_universe),
        "audit_component_count": len(audit_universe),
        "audit_only_component_count": len(audit_only),
        "audit_novelty_rate": audit_novelty_rate,
        "build_audit_jaccard": build_audit_jaccard,
        "cumulative_coverage_curve": curve,
        "gates": {
            "last_batches": last_count,
            "maximum_last_batch_relative_gain": gain_threshold,
            "observed_last_batch_relative_gains": [float(row["relative_gain"]) for row in last_rows],
            "saturation_passed": saturation_passed,
            "maximum_audit_novelty_rate": novelty_threshold,
            "audit_novelty_passed": audit_novelty_passed,
            "passed": universe_gate_passed,
        },
    }
    write_json(output_dir / "global_component_universe.json", universe_report)
    if not universe_gate_passed:
        raise RuntimeError(
            "Frozen V2 global-universe gate failed; expand the calibration protocol before endpoint execution: "
            + json.dumps(universe_report["gates"], ensure_ascii=False)
        )

    restricted = restrict_to_universe(candidate_components, global_universe)
    metadata = selection_metadata(
        read_jsonl(output_dir / "activations" / "candidate_profiles.jsonl"),
        prompts,
    )
    k = int(config["fingerprint"]["selected_size"])
    ablation_weights = dict(config["fingerprint"]["ablation_weights"])
    selections = {
        "candidate_weighted_v1": greedy_mcc(candidate_components, k, ablation_weights, metadata),
        "global_weighted_v1": greedy_mcc(restricted, k, ablation_weights, metadata),
        "global_unweighted_primary": greedy_mcc(restricted, k, {}, metadata),
        "global_type_balanced": greedy_mcc(
            restricted,
            k,
            type_balanced_weights(global_universe),
            metadata,
        ),
    }
    repeat_primary = greedy_mcc(restricted, k, {}, metadata)
    primary = selections["global_unweighted_primary"]
    deterministic = (
        primary.selected_ids == repeat_primary.selected_ids
        and primary.trace == repeat_primary.trace
    )
    curves = {}
    for curve_k in (4, 8, 12, 16):
        current = greedy_mcc(restricted, curve_k, {}, metadata)
        metrics = coverage_metrics(candidate_components, current.selected_ids, global_universe)
        curves[str(curve_k)] = {
            "selected_ids": current.selected_ids,
            "global_components_covered": metrics.selected_global_components,
            "final_global_coverage": metrics.final_global_coverage,
            "selection_efficiency": metrics.selection_efficiency,
            "unweighted_marginal_gain": sum(current.marginal_gains),
        }
    metrics = coverage_metrics(candidate_components, primary.selected_ids, global_universe)
    typed = per_type_coverage(candidate_components, primary.selected_ids, global_universe)
    nonempty_type_coverage = [
        float(row["final_global_coverage"])
        for row in typed.values()
        if int(row["global_components"]) > 0
    ]
    factorization_passed = metrics.factorization_error < 1e-12
    selection_gate_passed = bool(
        deterministic
        and len(primary.selected_ids) == k
        and len(set(primary.selected_ids)) == k
        and metrics.selected_global_components > 0
        and factorization_passed
        and all(
            0.0 <= float(curves[str(value)]["final_global_coverage"]) <= 1.0
            for value in (4, 8, 12, 16)
        )
        and all(
            float(curves[str(left)]["final_global_coverage"])
            <= float(curves[str(right)]["final_global_coverage"])
            for left, right in ((4, 8), (8, 12), (12, 16))
        )
    )
    report = {
        "schema_version": "fingerprint_v2_global_mcc_selection_1.0",
        "primary_objective": "global_unweighted_mcc",
        "candidate_prompts": len(candidate_components),
        "selected_prompts": len(primary.selected_ids),
        "selected_ids": primary.selected_ids,
        "selected_categories": [prompts[prompt_id].get("category") for prompt_id in primary.selected_ids],
        "trace": primary.trace,
        "coverage_metrics": metrics.as_dict(),
        "per_type_coverage": typed,
        "type_macro_final_global_coverage": mean(nonempty_type_coverage),
        "candidate_relative_coverage_v1_definition": coverage_rate(primary.selected_ids, candidate_components),
        "global_coverage_curves": curves,
        "objective_ablations": {
            name: {
                "selected_ids": result.selected_ids,
                "jaccard_with_primary": (
                    len(set(result.selected_ids) & set(primary.selected_ids))
                    / len(set(result.selected_ids) | set(primary.selected_ids))
                ),
                "coverage_metrics": coverage_metrics(
                    candidate_components,
                    result.selected_ids,
                    global_universe,
                ).as_dict(),
            }
            for name, result in selections.items()
        },
        "global_universe_sha256": universe_report["components_sha256"],
        "candidate_components_sha256": canonical_sha256(
            {key: sorted(value) for key, value in sorted(candidate_components.items())}
        ),
        "selection_deterministic": deterministic,
        "factorization_passed": factorization_passed,
        "passed": selection_gate_passed,
    }
    write_json(output_dir / "mcc_selection.json", report)
    if not selection_gate_passed:
        raise RuntimeError("V2 global MCC selection gate failed")
    print(
        json.dumps(
            {
                "global_universe": {
                    key: value
                    for key, value in universe_report.items()
                    if key not in {"components", "cumulative_coverage_curve"}
                },
                "selection": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
