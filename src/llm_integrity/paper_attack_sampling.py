from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class AttackVariant:
    variant_id: str
    split: str
    family: str
    weight: float
    seed: int
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _configuration_signature(
    family: str,
    configuration: Mapping[str, Any],
) -> str:
    payload = {
        "family": family,
        "configuration": dict(configuration),
    }
    return hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _variant_id(
    split: str,
    family: str,
    seed: int,
    configuration: Mapping[str, Any],
) -> str:
    payload = {
        "split": split,
        "family": family,
        "seed": int(seed),
        "configuration": dict(configuration),
    }
    digest = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()[:12]

    return f"{split}_{family}_{digest}"


def _validate_weights(
    family_weights: Mapping[str, float],
) -> dict[str, float]:
    if not family_weights:
        raise ValueError("family_weights cannot be empty")

    weights = {
        str(family): float(weight)
        for family, weight in family_weights.items()
    }

    if any(weight <= 0.0 for weight in weights.values()):
        raise ValueError("Every family weight must be positive")

    total = sum(weights.values())

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "Family weights must sum to 1.0; "
            f"received {total}"
        )

    return weights


def _partition_configurations(
    configurations: Sequence[Mapping[str, Any]],
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    items = [dict(item) for item in configurations]
    count = len(items)

    if count < 3:
        raise ValueError(
            "Each attack family requires at least three "
            "different configurations so train, validation "
            "and test remain isolated"
        )

    train_count = max(1, int(count * train_fraction))
    validation_count = max(
        1,
        int(count * validation_fraction),
    )

    while train_count + validation_count >= count:
        if train_count > validation_count:
            train_count -= 1
        else:
            validation_count -= 1

    return {
        "train": items[:train_count],
        "validation": items[
            train_count:
            train_count + validation_count
        ],
        "test": items[
            train_count + validation_count:
        ],
    }


def build_attack_manifests(
    family_spaces: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    family_weights: Mapping[str, float],
    split_seeds: Mapping[str, Sequence[int]],
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[AttackVariant]]:
    if train_fraction <= 0.0:
        raise ValueError("train_fraction must be positive")

    if validation_fraction <= 0.0:
        raise ValueError(
            "validation_fraction must be positive"
        )

    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            "train_fraction + validation_fraction "
            "must be smaller than 1.0"
        )

    weights = _validate_weights(family_weights)

    if set(family_spaces) != set(weights):
        raise ValueError(
            "family_spaces and family_weights must contain "
            "exactly the same attack families"
        )

    if set(split_seeds) != set(VALID_SPLITS):
        raise ValueError(
            "split_seeds must contain train, validation "
            "and test"
        )

    normalized_seeds = {
        split: tuple(int(value) for value in values)
        for split, values in split_seeds.items()
    }

    for split, values in normalized_seeds.items():
        if not values:
            raise ValueError(
                f"No random seeds configured for {split}"
            )

        if len(values) != len(set(values)):
            raise ValueError(
                f"Duplicate random seeds inside {split}"
            )

    seed_sets = {
        split: set(values)
        for split, values in normalized_seeds.items()
    }

    for index, first_split in enumerate(VALID_SPLITS):
        for second_split in VALID_SPLITS[index + 1:]:
            overlap = (
                seed_sets[first_split]
                & seed_sets[second_split]
            )

            if overlap:
                raise ValueError(
                    "Random seeds must be isolated between "
                    f"{first_split} and {second_split}: "
                    f"{sorted(overlap)}"
                )

    manifests: dict[str, list[AttackVariant]] = {
        split: []
        for split in VALID_SPLITS
    }

    generator = random.Random(seed)

    for family in sorted(weights):
        configurations = [
            dict(item)
            for item in family_spaces[family]
        ]

        signatures = [
            _configuration_signature(
                family,
                configuration,
            )
            for configuration in configurations
        ]

        if len(signatures) != len(set(signatures)):
            raise ValueError(
                f"Duplicate configurations in family "
                f"{family!r}"
            )

        generator.shuffle(configurations)

        partitioned = _partition_configurations(
            configurations,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

        for split in VALID_SPLITS:
            for configuration in partitioned[split]:
                for variant_seed in normalized_seeds[split]:
                    manifests[split].append(
                        AttackVariant(
                            variant_id=_variant_id(
                                split,
                                family,
                                variant_seed,
                                configuration,
                            ),
                            split=split,
                            family=family,
                            weight=weights[family],
                            seed=variant_seed,
                            configuration=dict(
                                configuration
                            ),
                        )
                    )

    validate_attack_manifests(
        manifests,
        expected_families=set(weights),
    )
    return manifests


def validate_attack_manifests(
    manifests: Mapping[
        str,
        Sequence[AttackVariant],
    ],
    expected_families: set[str] | None = None,
) -> None:
    if set(manifests) != set(VALID_SPLITS):
        raise ValueError(
            "Manifests must contain train, validation "
            "and test"
        )

    ids_by_split: dict[str, set[str]] = {}
    signatures_by_split: dict[str, set[str]] = {}
    seeds_by_split: dict[str, set[int]] = {}

    for split in VALID_SPLITS:
        variants = list(manifests[split])

        if not variants:
            raise ValueError(f"{split} manifest is empty")

        variant_ids: set[str] = set()
        signatures: set[str] = set()
        seeds: set[int] = set()
        present_families: set[str] = set()

        for variant in variants:
            if variant.split != split:
                raise ValueError(
                    f"Variant {variant.variant_id} has "
                    f"incorrect split {variant.split!r}"
                )

            if variant.variant_id in variant_ids:
                raise ValueError(
                    f"Duplicate variant ID: "
                    f"{variant.variant_id}"
                )

            variant_ids.add(variant.variant_id)
            signatures.add(
                _configuration_signature(
                    variant.family,
                    variant.configuration,
                )
            )
            seeds.add(int(variant.seed))
            present_families.add(variant.family)

        if (
            expected_families is not None
            and present_families != expected_families
        ):
            raise ValueError(
                f"{split} does not contain every family; "
                f"received {sorted(present_families)}"
            )

        ids_by_split[split] = variant_ids
        signatures_by_split[split] = signatures
        seeds_by_split[split] = seeds

    for index, first_split in enumerate(VALID_SPLITS):
        for second_split in VALID_SPLITS[index + 1:]:
            id_overlap = (
                ids_by_split[first_split]
                & ids_by_split[second_split]
            )
            configuration_overlap = (
                signatures_by_split[first_split]
                & signatures_by_split[second_split]
            )
            seed_overlap = (
                seeds_by_split[first_split]
                & seeds_by_split[second_split]
            )

            if id_overlap:
                raise ValueError(
                    "Variant IDs overlap between "
                    f"{first_split} and {second_split}"
                )

            if configuration_overlap:
                raise ValueError(
                    "Attack configurations overlap between "
                    f"{first_split} and {second_split}"
                )

            if seed_overlap:
                raise ValueError(
                    "Random seeds overlap between "
                    f"{first_split} and {second_split}"
                )


def write_attack_manifests(
    manifests: Mapping[
        str,
        Sequence[AttackVariant],
    ],
    output_directory: str | Path,
) -> dict[str, str]:
    validate_attack_manifests(manifests)

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}

    for split in VALID_SPLITS:
        output_path = (
            directory
            / f"attack_manifest_{split}.jsonl"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for variant in manifests[split]:
                file.write(
                    json.dumps(
                        variant.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        outputs[split] = str(output_path)

    return outputs
