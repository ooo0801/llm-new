from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from _bootstrap import ROOT, project_path

from llm_integrity.config import load_config
from llm_integrity.fingerprint import FingerprintEntry, ModelFingerprint
from llm_integrity.io import read_jsonl
from llm_integrity.mcc import complementarity, coverage_rate, greedy_mcc, random_selection, top_sensitivity
from llm_integrity.modeling import generate_texts, load_model, model_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/main_qwen_1.5b.yaml"))
    parser.add_argument("--methods", nargs="*", choices=["random", "top_sensitivity", "mcc"])
    parser.add_argument("--k", type=int, help="Override fingerprint size for k-curve experiments")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = project_path(config["output_dir"])
    rows = read_jsonl(project_path(config["data"]["candidate"]))
    by_id = {str(row["id"]): row for row in rows}
    score_rows = read_jsonl(output_dir / "prompt_scores.jsonl")
    scores = {str(row["id"]): float(row.get("hybrid") if row.get("hybrid") is not None else row["macro"]) for row in score_rows}
    profiles_path = output_dir / "activation_profiles.jsonl"
    components = {}
    if profiles_path.exists():
        components = {str(row["id"]): set(row["components"]) for row in read_jsonl(profiles_path)}
    k = int(args.k or config["fingerprint"]["selected_size"])
    methods = args.methods or list(config["fingerprint"].get("methods", ["random", "top_sensitivity", "mcc"]))
    selections = {}
    if "random" in methods:
        selections["random"] = random_selection(scores, k, int(config.get("seed", 42)))
    if "top_sensitivity" in methods:
        selections["top_sensitivity"] = top_sensitivity(scores, k)
    if "mcc" in methods:
        if not components:
            raise FileNotFoundError("MCC requires activation_profiles.jsonl; run extract_activations.py")
        pool = {key: value for key, value in components.items() if key in scores}
        selections["mcc"] = greedy_mcc(pool, k, config["fingerprint"].get("weights")).selected_ids

    bundle = load_model(config["model"])
    try:
        metadata = model_metadata(bundle)
        repetitions = int(config["fingerprint"].get("reference_repetitions", 5))
        for method, ids in selections.items():
            prompts = [by_id[item]["prompt"] for item in ids]
            reference_responses = [[] for _ in ids]
            for repetition in range(repetitions):
                print(f"[reference] {method} repetition {repetition + 1}/{repetitions}", flush=True)
                generated = generate_texts(bundle, prompts, config["generation"])
                for index, response in enumerate(generated):
                    reference_responses[index].append(response)
            entries = []
            for index, item in enumerate(ids):
                row = by_id[item]
                entries.append(
                    FingerprintEntry(
                        prompt_id=item,
                        prompt=row["prompt"],
                        category=row.get("category"),
                        sensitivity=scores[item],
                        components=sorted(components.get(item, set())),
                        reference_responses=reference_responses[index],
                        metadata={key: value for key, value in row.items() if key not in {"id", "prompt", "category"}},
                    )
                )
            diagnostics = {
                "coverage_rate": coverage_rate(ids, components) if components else None,
                "complementarity": complementarity(ids, components) if components else None,
                "mean_sensitivity": sum(scores[item] for item in ids) / len(ids),
                "k": k,
            }
            fingerprint = ModelFingerprint(
                model_name=config["model"]["name"],
                model_revision=str(config["model"].get("revision", "main")),
                selection_method=method,
                entries=entries,
                generation_config=dict(config["generation"]),
                metadata={"created_at": datetime.now(timezone.utc).isoformat(), "model": metadata, "diagnostics": diagnostics},
            )
            suffix = f"_k{k}" if args.k else ""
            target = output_dir / "fingerprints" / f"{method}{suffix}.json"
            fingerprint.save(target)
            print(json.dumps({"method": method, "output": str(target), "prompts": len(entries)}, ensure_ascii=False))
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
