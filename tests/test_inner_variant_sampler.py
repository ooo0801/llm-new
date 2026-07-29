from __future__ import annotations

from llm_integrity.inner_variant_sampler import (
    FAMILIES,
    StratifiedVariantSampler,
)


def build_manifests() -> tuple[list[dict], dict[str, str]]:
    manifests: list[dict] = []
    adapters: dict[str, str] = {}
    for family in FAMILIES:
        for index in range(4):
            variant_id = f"{family}_{index}"
            configuration = {}
            if family == "quantization":
                configuration = {
                    "target_scope": "full_model",
                    "compute_dtype": "bfloat16",
                }
            manifests.append(
                {
                    "family": family,
                    "variant_id": variant_id,
                    "seed": index,
                    "split": "train",
                    "configuration": configuration,
                }
            )
            if family == "finetuning":
                adapters[variant_id] = f"/tmp/{variant_id}"
    return manifests, adapters


def test_family_minibatch_is_deterministic_and_without_replacement() -> None:
    manifests, adapters = build_manifests()
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights={family: 0.2 for family in FAMILIES},
        adapter_registry=adapters,
        seed=42,
    )

    first = sampler.sample_family_variants(
        "prompt-a",
        "quantization",
        count=2,
        cycle=0,
    )
    repeated = sampler.sample_family_variants(
        "prompt-a",
        "quantization",
        count=2,
        cycle=0,
    )
    second_cycle = sampler.sample_family_variants(
        "prompt-a",
        "quantization",
        count=2,
        cycle=1,
    )

    first_ids = [sample.variant_id for sample in first]
    assert first_ids == [sample.variant_id for sample in repeated]
    assert len(set(first_ids)) == 2
    assert set(first_ids).isdisjoint(
        sample.variant_id for sample in second_cycle
    )


def test_family_minibatch_rejects_oversized_request() -> None:
    manifests, adapters = build_manifests()
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights={family: 0.2 for family in FAMILIES},
        adapter_registry=adapters,
        seed=42,
    )

    try:
        sampler.sample_family_variants(
            "prompt-a",
            "finetuning",
            count=5,
        )
    except ValueError as error:
        assert "only 4 are executable" in str(error)
    else:
        raise AssertionError("Expected oversized family batch to fail")


def test_disjoint_search_and_anchor_variants_are_stable_and_separate() -> None:
    manifests, adapters = build_manifests()
    sampler = StratifiedVariantSampler(
        manifests,
        family_weights={family: 0.2 for family in FAMILIES},
        adapter_registry=adapters,
        seed=42,
    )

    search0, anchors0 = sampler.sample_disjoint_family_variants(
        "prompt-a",
        "unstructured_pruning",
        search_count=1,
        anchor_count=2,
        cycle=0,
    )
    search1, anchors1 = sampler.sample_disjoint_family_variants(
        "prompt-a",
        "unstructured_pruning",
        search_count=1,
        anchor_count=2,
        cycle=1,
    )

    anchor_ids0 = {sample.variant_id for sample in anchors0}
    assert anchor_ids0 == {sample.variant_id for sample in anchors1}
    assert search0[0].variant_id not in anchor_ids0
    assert search1[0].variant_id not in anchor_ids0
    assert search0[0].variant_id != search1[0].variant_id
