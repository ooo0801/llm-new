from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ROOT
from llm_integrity.features import FeatureExtractor, Standardizer
from llm_integrity.fingerprint import ModelFingerprint
from llm_integrity.io import write_json
from llm_integrity.statistics import prompt_stratified_mmd_test


def response_payload(
    fingerprint: ModelFingerprint,
    verification: dict[str, Any],
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    by_key = {
        (str(row["prompt_id"]), int(row["repetition"])): str(row["response"])
        for row in verification["response_records"]
    }
    repetitions = int(verification["repetitions"])
    reference: list[str] = []
    target: list[str] = []
    rows: list[dict[str, Any]] = []
    strata: list[str] = []
    for repetition in range(repetitions):
        for entry in fingerprint.entries:
            reference.append(entry.reference_responses[repetition])
            target.append(by_key[(entry.prompt_id, repetition)])
            rows.append(entry.metadata | {"category": entry.category})
            strata.append(entry.prompt_id)
    return reference, target, rows, strata


def standardized(
    x: np.ndarray,
    y: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "reference_only":
        scaler = Standardizer().fit(x)
    elif mode == "pooled_symmetric":
        scaler = Standardizer().fit(np.concatenate([x, y], axis=0))
    elif mode == "none":
        return x, y
    else:
        raise ValueError(mode)
    return scaler.transform(x), scaler.transform(y)


def run_test(
    x: np.ndarray,
    y: np.ndarray,
    strata: list[str],
    mode: str,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    left, right = standardized(x, y, mode)
    return prompt_stratified_mmd_test(
        left,
        right,
        strata=strata,
        permutations=permutations,
        alpha=0.05,
        seed=seed,
    ).as_dict()


def block_features(
    extractor: FeatureExtractor,
    reference: list[str],
    target: list[str],
    rows: list[dict[str, Any]],
    block: str,
) -> tuple[np.ndarray, np.ndarray]:
    flags = {
        "surface": (True, False, False),
        "semantic": (False, True, False),
        "task": (False, False, True),
        "all": (True, True, True),
    }[block]
    return (
        extractor.transform(reference, rows, *flags),
        extractor.transform(target, rows, *flags),
    )


def null_resplits(
    x: np.ndarray,
    y: np.ndarray,
    strata: list[str],
    mode: str,
    splits: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    labels = list(dict.fromkeys(strata))
    strata_array = np.asarray(strata, dtype=object)
    rng = np.random.default_rng(seed)
    p_values: list[float] = []
    for split in range(splits):
        left = np.empty_like(x)
        right = np.empty_like(y)
        for label in labels:
            indices = np.flatnonzero(strata_array == label)
            combined = np.concatenate([x[indices], y[indices]], axis=0)
            order = rng.permutation(len(combined))
            left[indices] = combined[order[: len(indices)]]
            right[indices] = combined[order[len(indices) :]]
        result = run_test(
            left,
            right,
            strata,
            mode,
            permutations,
            seed + 1000 + split,
        )
        p_values.append(float(result["p_value"]))
    return {
        "splits": splits,
        "permutations_per_split": permutations,
        "rejections_at_0_05": sum(value < 0.05 for value in p_values),
        "rejection_rate": sum(value < 0.05 for value in p_values) / splits,
        "p_values": p_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--verification-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--semantic-revision", default="7999e1d3359715c523056ef9478215996d62a620")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--null-splits", type=int, default=20)
    parser.add_argument("--null-permutations", type=int, default=199)
    parser.add_argument("--seed", type=int, default=2026081501)
    args = parser.parse_args()

    fingerprint = ModelFingerprint.load(ROOT / args.fingerprint)
    verification_dir = ROOT / args.verification_dir
    intact_path = next(verification_dir.glob("*__intact_14b_h3.json"))
    intact = json.loads(intact_path.read_text(encoding="utf-8"))
    reference, target, rows, strata = response_payload(fingerprint, intact)
    extractor = FeatureExtractor(
        semantic_model_name=args.semantic_model,
        semantic_model_revision=args.semantic_revision,
        semantic_device="cpu",
        semantic_local_files_only=True,
        hashed_dimension=128,
    )

    x_all, y_all = block_features(extractor, reference, target, rows, "all")
    report: dict[str, Any] = {
        "schema_version": "h3_false_positive_diagnostic_1.0",
        "classification": "exploratory_reanalysis_of_frozen_h3_responses",
        "intact": {
            mode: run_test(
                x_all,
                y_all,
                strata,
                mode,
                args.permutations,
                args.seed,
            )
            for mode in ["reference_only", "pooled_symmetric", "none"]
        },
        "pooled_symmetric_feature_blocks": {},
        "existing_state_reanalysis": {},
        "null_resplits": {},
    }
    for block in ["surface", "semantic", "task"]:
        x_block, y_block = block_features(extractor, reference, target, rows, block)
        report["pooled_symmetric_feature_blocks"][block] = run_test(
            x_block,
            y_block,
            strata,
            "pooled_symmetric",
            args.permutations,
            args.seed,
        )

    for path in sorted(verification_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        ref, tgt, metadata, state_strata = response_payload(fingerprint, row)
        x_state, y_state = block_features(extractor, ref, tgt, metadata, "all")
        report["existing_state_reanalysis"][str(row["variant_id"])] = {
            "family": row["family"],
            "original_reference_only": row["primary_test"],
            "pooled_symmetric": run_test(
                x_state,
                y_state,
                state_strata,
                "pooled_symmetric",
                args.permutations,
                args.seed,
            ),
        }

    for mode in ["reference_only", "pooled_symmetric"]:
        report["null_resplits"][mode] = null_resplits(
            x_all,
            y_all,
            strata,
            mode,
            args.null_splits,
            args.null_permutations,
            args.seed + (0 if mode == "reference_only" else 100),
        )

    write_json(ROOT / args.output, report)
    summary = {
        "intact": {
            mode: {"statistic": value["statistic"], "p_value": value["p_value"], "reject": value["reject"]}
            for mode, value in report["intact"].items()
        },
        "feature_blocks": {
            block: {"p_value": value["p_value"], "reject": value["reject"]}
            for block, value in report["pooled_symmetric_feature_blocks"].items()
        },
        "state_rejections_pooled_symmetric": {
            variant: value["pooled_symmetric"]["reject"]
            for variant, value in report["existing_state_reanalysis"].items()
        },
        "null_resplits": report["null_resplits"],
        "output": str(Path(args.output).as_posix()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
