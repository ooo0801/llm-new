from __future__ import annotations

import json

import numpy as np
import pytest

from llm_integrity.h8_precalibration import (
    H8_BANDWIDTH_CONVENTION,
    FamilyBalancedScaler,
    _stability_gate,
    build_h8_feature_schema,
    canonical_sha256,
    fit_family_balanced_scaler,
    h8_bandwidth_stability,
    h8_median_positive_pairwise_distance,
    h8_mmd2_biased,
    h8_mmd2_unbiased_unequal,
    h8_rbf_kernel,
    load_frozen_scaler_and_bandwidth,
    load_h8_artifact,
    make_bandwidth_payload,
    save_h8_artifact,
)


def random_features(rows: int = 20, semantic_dimension: int = 2) -> tuple[np.ndarray, object]:
    schema = build_h8_feature_schema(semantic_dimension)
    rng = np.random.default_rng(8)
    values = rng.normal(size=(rows, schema.dimension))
    # Bounded/binary columns remain in their declared ranges.
    for feature in schema.features:
        if feature.feature_type == "bounded":
            values[:, feature.index] = rng.uniform(0, 1, rows)
        elif feature.feature_type == "binary":
            values[:, feature.index] = rng.integers(0, 2, rows)
    return values.astype(np.float64), schema


def test_schema_is_frozen_float64_with_family_slices() -> None:
    schema = build_h8_feature_schema(512)
    assert schema.dimension == 528
    assert schema.family_slices == {"surface": (0, 11), "semantic": (11, 523), "task": (523, 528)}
    assert len(schema.sha256) == 64
    assert schema.as_dict()["dtype"] == "float64"


def test_family_balanced_scaler_uses_ddof0_and_weights() -> None:
    values, schema = random_features()
    scaler = fit_family_balanced_scaler("p", values, values, schema)
    expected = values.std(axis=0, ddof=0)
    assert np.allclose(scaler.scale, expected)
    transformed = scaler.transform(values, schema)
    for family, (start, stop) in schema.family_slices.items():
        assert scaler.family_weights[family] == pytest.approx(1 / np.sqrt(stop - start))
        assert np.allclose(transformed[:, start:stop].mean(axis=0), 0.0, atol=1e-12)


def test_constant_binary_and_bounded_use_unit_scale() -> None:
    values, schema = random_features()
    for feature in schema.features:
        if feature.feature_type in {"binary", "bounded"}:
            values[:, feature.index] = 0.0
    scaler = fit_family_balanced_scaler("p", values, values, schema)
    for feature in schema.features:
        if feature.feature_type in {"binary", "bounded"}:
            assert scaler.scale[feature.index] == 1.0
            assert scaler.scale_source[feature.index] == "unit_range_constant"


def test_constant_continuous_uses_pooled_fit_only_scale() -> None:
    values, schema = random_features()
    values[:, 0] = 4.0
    pooled = np.concatenate([values, values.copy()])
    pooled[len(values) :, 0] = np.arange(len(values))
    scaler = fit_family_balanced_scaler("p", values, pooled, schema)
    assert scaler.scale_source[0] == "pooled_fit_only_std_ddof0"
    assert scaler.scale[0] == pytest.approx(pooled[:, 0].std(ddof=0))


def test_globally_constant_continuous_feature_fails() -> None:
    values, schema = random_features()
    values[:, 0] = 4.0
    with pytest.raises(ValueError, match="constant both"):
        fit_family_balanced_scaler("p", values, values, schema)


def test_h8_bandwidth_is_median_positive_euclidean_not_h4() -> None:
    values = np.asarray([[0.0], [2.0], [5.0]], dtype=np.float64)
    # Positive distances are 2, 5, 3; median is 3.
    assert h8_median_positive_pairwise_distance(values) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="No positive"):
        h8_median_positive_pairwise_distance(np.zeros((3, 2)))


def test_rbf_kernel_contract() -> None:
    values = np.asarray([[0.0], [2.0]])
    kernel = h8_rbf_kernel(values, values, bandwidth=2.0)
    assert kernel.shape == (2, 2)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.allclose(kernel, kernel.T)
    assert np.isfinite(kernel).all()


def test_unequal_unbiased_mmd_matches_bruteforce_and_is_symmetric() -> None:
    x = np.asarray([[0.0], [1.0], [2.0]])
    y = np.asarray([[0.5], [3.0], [4.0], [5.0], [6.0]])
    bandwidth = 1.7
    kxx = h8_rbf_kernel(x, x, bandwidth)
    kyy = h8_rbf_kernel(y, y, bandwidth)
    kxy = h8_rbf_kernel(x, y, bandwidth)
    brute = (
        sum(kxx[i, j] for i in range(len(x)) for j in range(len(x)) if i != j)
        / (len(x) * (len(x) - 1))
        + sum(kyy[i, j] for i in range(len(y)) for j in range(len(y)) if i != j)
        / (len(y) * (len(y) - 1))
        - 2 * kxy.sum() / (len(x) * len(y))
    )
    actual = h8_mmd2_unbiased_unequal(x, y, bandwidth)
    assert actual == pytest.approx(brute)
    assert actual == pytest.approx(h8_mmd2_unbiased_unequal(y, x, bandwidth))


def test_unbiased_mmd_can_be_negative_and_biased_is_nonnegative() -> None:
    x = np.asarray([[0.0], [2.0]])
    y = np.asarray([[1.0], [3.0]])
    assert h8_mmd2_unbiased_unequal(x, y, 1.0) < 0.0
    assert h8_mmd2_biased(x, y, 1.0) >= 0.0


def test_mmd_is_invariant_to_row_order() -> None:
    x = np.asarray([[0.0], [1.0], [3.0]])
    y = np.asarray([[2.0], [4.0], [7.0], [9.0]])
    expected = h8_mmd2_unbiased_unequal(x, y, 2.0)
    assert h8_mmd2_unbiased_unequal(x[::-1], y[[2, 0, 3, 1]], 2.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("ratios", "status"),
    [
        ([0.9] * 100, "PASS"),
        ([0.75] * 100, "WARN"),
        ([0.6] * 100, "FAIL"),
    ],
)
def test_stability_gate(ratios: list[float], status: str) -> None:
    assert _stability_gate(ratios)["status"] == status


def test_two_bandwidth_stability_tests_are_deterministic() -> None:
    values_a, schema = random_features(rows=30)
    values_b, _ = random_features(rows=30)
    values_b = values_b + np.linspace(0, 0.2, len(values_b))[:, None]
    pooled = np.concatenate([values_a, values_b])
    scaler = fit_family_balanced_scaler("a", values_a, pooled, schema)
    first = h8_bandwidth_stability(
        "a", {"a": values_a, "b": values_b}, scaler, schema, repetitions=8, seed=22
    )
    second = h8_bandwidth_stability(
        "a", {"a": values_a, "b": values_b}, scaler, schema, repetitions=8, seed=22
    )
    assert first == second
    assert set(first) >= {"fixed_full_scaler", "refit_subsample_scaler", "overall_status"}


def test_serialization_hash_and_fail_closed_loading(tmp_path) -> None:
    values, schema = random_features()
    scaler = fit_family_balanced_scaler("p", values, values, schema)
    scaler_path = tmp_path / "scaler.json"
    scaler_sha = save_h8_artifact(scaler_path, "h8_family_balanced_scaler", scaler.as_dict())
    bandwidth = h8_median_positive_pairwise_distance(scaler.transform(values, schema))
    bandwidth_payload = make_bandwidth_payload(
        "p", bandwidth, scaler_sha, "c" * 64, schema.sha256
    )
    bandwidth_path = tmp_path / "bandwidth.json"
    bandwidth_sha = save_h8_artifact(
        bandwidth_path, "h8_prompt_bandwidth", bandwidth_payload
    )
    loaded_scaler, loaded_bandwidth = load_frozen_scaler_and_bandwidth(
        scaler_path,
        bandwidth_path,
        schema,
        expected_scaler_payload_sha256=scaler_sha,
        expected_bandwidth_payload_sha256=bandwidth_sha,
    )
    assert isinstance(loaded_scaler, FamilyBalancedScaler)
    assert loaded_bandwidth == pytest.approx(bandwidth)
    envelope = json.loads(scaler_path.read_text(encoding="utf-8"))
    envelope["payload"]["mean"][0] += 1
    scaler_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_h8_artifact(scaler_path, "h8_family_balanced_scaler")


def test_legacy_h4_bandwidth_is_rejected(tmp_path) -> None:
    values, schema = random_features()
    scaler = fit_family_balanced_scaler("p", values, values, schema)
    scaler_path = tmp_path / "scaler.json"
    scaler_sha = save_h8_artifact(scaler_path, "h8_family_balanced_scaler", scaler.as_dict())
    payload = make_bandwidth_payload("p", 1.0, scaler_sha, "c" * 64, schema.sha256)
    payload["method"] = "sqrt_half_median_squared_distance"
    bandwidth_path = tmp_path / "bandwidth.json"
    save_h8_artifact(bandwidth_path, "h8_prompt_bandwidth", payload)
    with pytest.raises(ValueError, match="legacy H4"):
        load_frozen_scaler_and_bandwidth(scaler_path, bandwidth_path, schema)


def test_payload_hash_is_canonical() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
    assert H8_BANDWIDTH_CONVENTION == "median_positive_pairwise_euclidean_distance"

