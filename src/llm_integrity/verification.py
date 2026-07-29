from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .features import FeatureExtractor, Standardizer
from .fingerprint import ModelFingerprint
from .modeling import ModelBundle, generate_texts
from .statistics import (
    TestResult,
    mmd_permutation_test,
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


def verify_model(
    fingerprint: ModelFingerprint,
    target: ModelBundle,
    statistics_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    repetitions: int = 1,
) -> VerificationResult:
    prompts = [entry.prompt for entry in fingerprint.entries]
    rows = [entry.metadata | {"category": entry.category} for entry in fingerprint.entries]

    reference_texts: list[str] = []
    target_texts: list[str] = []
    expanded_rows: list[dict[str, Any]] = []
    strata: list[str] = []

    for repetition in range(repetitions):
        generated = generate_texts(target, prompts, fingerprint.generation_config)
        target_texts.extend(generated)
        expanded_rows.extend(rows)
        strata.extend(
            str(entry.prompt_id)
            for entry in fingerprint.entries
        )


        for entry in fingerprint.entries:
            if not entry.reference_responses:
                raise ValueError("Fingerprint has no reference responses")
            reference_texts.append(entry.reference_responses[repetition % len(entry.reference_responses)])
    extractor = FeatureExtractor(
        semantic_model_name=feature_config.get("semantic_model"),
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
    standardizer = Standardizer().fit(reference_features)
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

    else:
        raise ValueError(
            f"Unknown verification method: {method}"
        )

    return VerificationResult(
        modified=test.reject,
        test=test,
        prompts=len(prompts),
        repetitions=repetitions,
        target_responses=target_texts,
    )
