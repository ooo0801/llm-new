from __future__ import annotations

import argparse
import copy
import hashlib

from _bootstrap import ROOT, project_path
from llm_integrity.config import load_config
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import write_json
from llm_integrity.modeling import generate_texts, load_model, model_metadata
from llm_integrity.task_validation import evaluate_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/fingerprint_h5_qwen14b_mismatch.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    source = project_path(config["data"]["source_fingerprint"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["data"]["source_fingerprint_sha256"]:
        raise ValueError("H5 source fingerprint hash mismatch")
    fingerprint = copy.deepcopy(ModelFingerprint.load(source))
    prompts = [entry.prompt for entry in fingerprint.entries]
    repetitions = int(config["fingerprint"]["reference_repetitions"])
    seed_base = int(config["fingerprint"]["reference_seed_base"])
    seeds = [seed_base + index for index in range(repetitions)]
    responses = [[] for _ in fingerprint.entries]
    checks = [[] for _ in fingerprint.entries]
    bundle = load_model(config["model"])
    try:
        metadata = model_metadata(bundle)
        for repetition, seed in enumerate(seeds):
            generated = generate_texts(bundle, prompts, config["generation"], seed=seed)
            for index, response in enumerate(generated):
                responses[index].append(response)
                passed, reason = evaluate_task(
                    fingerprint.entries[index].metadata | {"category": fingerprint.entries[index].category},
                    response,
                )
                checks[index].append({"passed": passed, "reason": reason})
            print({"stage": "h5_reference", "repetition": repetition, "seed": seed}, flush=True)
    finally:
        bundle.close()
    for index, entry in enumerate(fingerprint.entries):
        entry.reference_responses = responses[index]
        entry.reference_response_seeds = seeds
        entry.metadata["h5_reference_task_checks"] = checks[index]
    fingerprint.generation_config = dict(config["generation"])
    fingerprint.metadata.update(
        {
            "experiment_version": config["experiment_name"],
            "h5_parent_fingerprint_sha256": config["data"]["source_fingerprint_sha256"],
            "reference_seeds": seeds,
            "model": metadata,
            "statistics_config": dict(config["statistics"]),
            "feature_config": dict(config["features"]),
        }
    )
    output = project_path(config["output_dir"]) / "fingerprints" / config["fingerprint"]["artifact_name"]
    fingerprint.save(output)
    summary = {
        "schema_version": "fingerprint_14b_h5_reference_1.0",
        "fingerprint": str(output.relative_to(ROOT).as_posix()),
        "responses": sum(map(len, responses)),
        "task_pass_rate": sum(row["passed"] for values in checks for row in values) / sum(map(len, checks)),
        "reference_seeds": seeds,
        "source_fingerprint_sha256": config["data"]["source_fingerprint_sha256"],
    }
    write_json(project_path(config["output_dir"]) / "reference_report.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
