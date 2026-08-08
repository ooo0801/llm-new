from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .features import FeatureExtractor, Standardizer
from .fingerprint import ModelFingerprint
from .modeling import ModelBundle, generate_texts
from .statistics import (
    TestResult,
    mmd_permutation_test,
    paired_block_sign_flip_test,
    paired_sign_flip_test,
    prompt_stratified_mmd_test,
)


@dataclass
class VerificationResult:
    modified: bool
    test: TestResult
    prompts: int
    repetitions: int
    target_responses: list[str]
    response_records: list[dict[str, Any]]
    secondary_tests: dict[str, TestResult]


def verify_model(
    fingerprint: ModelFingerprint,
    target: ModelBundle,
    statistics_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    repetitions: int = 1,
    target_seeds: list[int] | None = None,
) -> VerificationResult:
    if repetitions < 2 and str(statistics_config.get("method")) == "prompt_stratified_mmd":
        raise ValueError("Prompt-stratified MMD requires at least two repetitions per prompt")
    prompts = [entry.prompt for entry in fingerprint.entries]
    rows = [entry.metadata | {"category": entry.category} for entry in fingerprint.entries]

    reference_texts: list[str] = []
    target_texts: list[str] = []
    expanded_rows: list[dict[str, Any]] = []
    strata: list[str] = []
    blocks: list[int] = []
    response_records: list[dict[str, Any]] = []
    if target_seeds is None:
        seed_base = int(statistics_config.get("target_seed_base", 52000))
        target_seeds = [seed_base + index for index in range(repetitions)]
    if len(target_seeds) != repetitions:
        raise ValueError("target_seeds must contain exactly one seed per repetition")

    for repetition in range(repetitions):
        generated = generate_texts(
            target,
            prompts,
            fingerprint.generation_config,
            seed=int(target_seeds[repetition]),
        )
        target_texts.extend(generated)
        expanded_rows.extend(rows)
        strata.extend(
            str(entry.prompt_id)
            for entry in fingerprint.entries
        )
        blocks.extend([repetition] * len(fingerprint.entries))


        for entry in fingerprint.entries:
            if not entry.reference_responses:
                raise ValueError("Fingerprint has no reference responses")
            if repetition >= len(entry.reference_responses):
                raise ValueError(
                    f"Fingerprint entry {entry.prompt_id!r} has fewer than "
                    f"{repetitions} reference responses"
                )
            reference_texts.append(entry.reference_responses[repetition])
        response_records.extend(
            {
                "prompt_id": entry.prompt_id,
                "repetition": repetition,
                "seed": int(target_seeds[repetition]),
                "response": response,
            }
            for entry, response in zip(fingerprint.entries, generated)
        )
    extractor = FeatureExtractor(
        semantic_model_name=feature_config.get("semantic_model"),
        semantic_model_revision=feature_config.get("semantic_model_revision"),
        semantic_device=str(feature_config.get("semantic_device", "cpu")),
        semantic_local_files_only=bool(feature_config.get("semantic_local_files_only", False)),
        hashed_dimension=int(feature_config.get("hashed_dimension", 128)),
    )
    reference_features = extractor.transform(
        reference_texts,
        expanded_rows,
        bool(feature_config.get("surface", True)),
        bool(feature_config.get("semantic", True)),
        bool(feature_config.get("task", True)),
    )
    target_features = extractor.transform(
        target_texts,
        expanded_rows,
        bool(feature_config.get("surface", True)),
        bool(feature_config.get("semantic", True)),
        bool(feature_config.get("task", True)),
    )
    standardization = str(statistics_config.get("standardization", "reference_only"))
    if standardization == "reference_only":
        standardizer = Standardizer().fit(reference_features)
    elif standardization == "pooled_symmetric":
        standardizer = Standardizer().fit(
            np.concatenate([reference_features, target_features], axis=0)
        )
    else:
        raise ValueError(f"Unknown standardization mode: {standardization}")
    x = standardizer.transform(reference_features)
    y = standardizer.transform(target_features)

    method = str(
        statistics_config.get(
            "method",
            "paired_sign_flip",
        )
    )

    common = {
        "permutations": int(
            statistics_config.get(
                "permutations",
                1000,
            )
        ),
        "alpha": float(
            statistics_config.get(
                "alpha",
                0.05,
            )
        ),
        "seed": int(
            statistics_config.get(
                "seed",
                42,
            )
        ),
    }

    if method == "paper_mmd":
        test = mmd_permutation_test(
            x,
            y,
            **common,
        )

    elif method == "prompt_stratified_mmd":
        test = prompt_stratified_mmd_test(
            x,
            y,
            strata=strata,
            **common,
        )

    elif method == "paired_sign_flip":
        test = paired_sign_flip_test(
            x,
            y,
            **common,
        )

    elif method == "paired_block_sign_flip":
        test = paired_block_sign_flip_test(
            x,
            y,
            blocks=blocks,
            exact=bool(statistics_config.get("exact_sign_flips", True)),
            **common,
        )

    else:
        raise ValueError(
            f"Unknown verification method: {method}"
        )

    secondary_tests: dict[str, TestResult] = {}
    if bool(statistics_config.get("report_stratified_ablation", False)) and method != "prompt_stratified_mmd":
        secondary_tests["prompt_stratified_mmd"] = prompt_stratified_mmd_test(
            x, y, strata=strata, **common
        )
    if bool(statistics_config.get("report_pooled_baseline", True)) and method != "paper_mmd":
        secondary_tests["pooled_mmd"] = mmd_permutation_test(x, y, **common)

    return VerificationResult(
        modified=test.reject,
        test=test,
        prompts=len(prompts),
        repetitions=repetitions,
        target_responses=target_texts,
        response_records=response_records,
        secondary_tests=secondary_tests,
    )
