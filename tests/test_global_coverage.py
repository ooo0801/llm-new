import pytest

from llm_integrity.global_coverage import (
    coverage_metrics,
    cumulative_coverage_curve,
    per_type_coverage,
    restrict_to_universe,
    stable_prompt_components,
    type_balanced_weights,
    validate_component_id,
)


def test_stable_components_and_global_coverage_factorization() -> None:
    profiles = [
        {"id": "a", "profile_repeat": 0, "components": ["ffn:0:1", "attention:0:0"]},
        {"id": "a", "profile_repeat": 1, "components": ["ffn:0:1", "attention:0:0"]},
        {"id": "b", "profile_repeat": 0, "components": ["ffn:0:2"]},
        {"id": "b", "profile_repeat": 1, "components": ["ffn:0:2"]},
    ]
    prompt_components = stable_prompt_components(
        profiles,
        repetitions=2,
        stable_frequency=1.0,
    )
    universe = {"ffn:0:1", "ffn:0:2", "ffn:0:3", "attention:0:0"}
    metrics = coverage_metrics(prompt_components, ["a"], universe)
    assert metrics.candidate_reachable_coverage == pytest.approx(0.75)
    assert metrics.selection_efficiency == pytest.approx(2 / 3)
    assert metrics.final_global_coverage == pytest.approx(0.5)
    assert metrics.factorization_error < 1e-12


def test_outside_components_are_not_counted_as_global_coverage() -> None:
    mapping = {
        "a": {"ffn:0:1", "ffn:0:999"},
        "b": {"attention:0:0"},
    }
    universe = {"ffn:0:1", "attention:0:0"}
    restricted = restrict_to_universe(mapping, universe)
    assert restricted["a"] == {"ffn:0:1"}
    metrics = coverage_metrics(mapping, ["a"], universe)
    assert metrics.selected_global_components == 1
    assert metrics.selected_outside_components == 1
    assert metrics.final_global_coverage == 0.5


def test_per_type_coverage_and_balanced_weights() -> None:
    universe = {"ffn:0:1", "ffn:0:2", "ffn:0:3", "attention:0:0"}
    mapping = {"a": {"ffn:0:1", "attention:0:0"}, "b": {"ffn:0:2"}}
    values = per_type_coverage(mapping, ["a"], universe)
    assert values["attention"]["final_global_coverage"] == 1.0
    assert values["ffn"]["final_global_coverage"] == pytest.approx(1 / 3)
    weights = type_balanced_weights(universe)
    assert weights["attention"] > weights["ffn"]
    assert weights["attention"] == pytest.approx(2.0)
    assert weights["ffn"] == pytest.approx(2 / 3)


def test_cumulative_curve_is_monotonic() -> None:
    mapping = {
        "a": {"ffn:0:1"},
        "b": {"ffn:0:1", "ffn:0:2"},
        "c": {"attention:0:0"},
    }
    curve = cumulative_coverage_curve(mapping, ["a", "b", "c"], batch_size=1)
    counts = [row["component_count"] for row in curve]
    assert counts == [1, 2, 3]
    assert all(0.0 <= row["relative_gain"] <= 1.0 for row in curve)


@pytest.mark.parametrize(
    "value",
    ["unknown:0", "attention:bad:0", "ffn:0", "mlp_residual:-1"],
)
def test_invalid_component_ids_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        validate_component_id(value)
