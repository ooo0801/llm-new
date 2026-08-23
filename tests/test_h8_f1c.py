from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from llm_integrity.h8_f1c import (
    evaluate_final_unit,
    expand_frozen_prompt_seeds,
    permutation_mmd2_from_explicit_seeds,
    score_array,
)
from llm_integrity.h8_m0ij import mmd2_unbiased_from_kernel
from llm_integrity.h8_precalibration import h8_rbf_kernel


PROMPTS = tuple(f"prompt_{index:02d}" for index in range(12))


def _kernel(offset: float = 0.0) -> np.ndarray:
    x = np.linspace(-2.0, 2.0, 70, dtype=np.float64)[:, None]
    values = np.concatenate([x, np.sin(x + offset)], axis=1)
    return h8_rbf_kernel(values, values, 1.1)


def _parameters() -> dict[str, dict[str, float | str]]:
    return {
        prompt_id: {
            "mapping": "nondegenerate_centered",
            "effective_scale": 0.1,
            "null_median_m": 0.0,
        }
        for prompt_id in PROMPTS
    }


def test_frozen_seed_expansion_is_deterministic_and_digest_is_iteration_ordered() -> None:
    first, first_digest = expand_frozen_prompt_seeds(123456789, PROMPTS)
    second, second_digest = expand_frozen_prompt_seeds(123456789, tuple(reversed(PROMPTS)))
    assert first_digest == second_digest
    assert all(np.array_equal(first[prompt_id], second[prompt_id]) for prompt_id in PROMPTS)
    manual = hashlib.sha256()
    for permutation_id in range(999):
        for prompt_id in sorted(PROMPTS):
            manual.update(int(first[prompt_id][permutation_id]).to_bytes(8, "little"))
    assert first_digest == manual.hexdigest()


def test_vectorized_explicit_seed_mmd_matches_scalar_and_preserves_negative_values() -> None:
    kernel = _kernel()
    seeds, _ = expand_frozen_prompt_seeds(987654321, PROMPTS)
    values = permutation_mmd2_from_explicit_seeds(kernel, seeds[PROMPTS[0]])
    assert values.shape == (999,)
    assert np.any(values < 0.0)
    for index in (0, 1, 17, 998):
        order = np.random.Generator(np.random.PCG64(int(seeds[PROMPTS[0]][index]))).permutation(70)
        expected = mmd2_unbiased_from_kernel(kernel, order[:60], order[60:])
        assert values[index] == pytest.approx(expected, abs=1e-14)


def test_score_layer_only_clamps_and_uses_both_frozen_mappings() -> None:
    raw = np.linspace(-0.2, 0.2, 999, dtype=np.float64)
    centered = score_array(
        raw,
        {"mapping": "nondegenerate_centered", "effective_scale": 0.1, "null_median_m": 0.05},
    )
    degenerate = score_array(
        raw,
        {"mapping": "degenerate_global_scale", "effective_scale": 0.2, "null_median_m": 99.0},
    )
    assert centered[0] == 0.0
    assert degenerate[0] == 0.0
    assert centered[-1] == pytest.approx(1.5)
    assert degenerate[-1] == pytest.approx(1.0)


def test_final_unit_dynamic_top2_explicit_stream_and_p_grid() -> None:
    kernels = {prompt_id: _kernel(index / 20.0) for index, prompt_id in enumerate(PROMPTS)}
    seeds, digest = expand_frozen_prompt_seeds(11223344, PROMPTS)
    assert len(seeds) == 12
    result = evaluate_final_unit(
        kernels,
        _parameters(),
        stream_seed=11223344,
        expected_stream_digest_sha256=digest,
    )
    ranked = sorted(result.observed_scores, key=lambda key: (-result.observed_scores[key], key))
    assert result.top_prompt_ids == tuple(ranked[:2])
    assert result.observed_top2 == pytest.approx(sum(result.observed_scores[key] for key in ranked[:2]))
    assert result.permutation_top2.shape == (999,)
    assert 0.0 < result.p_global <= 1.0
    assert result.p_global * 1000 == pytest.approx(round(result.p_global * 1000), abs=1e-12)
    assert result.alarm == (result.exceedance_count <= 49)


def test_final_unit_rejects_tampered_frozen_seed_digest() -> None:
    kernels = {prompt_id: _kernel() for prompt_id in PROMPTS}
    with pytest.raises(ValueError, match="seed stream SHA256 mismatch"):
        evaluate_final_unit(
            kernels,
            _parameters(),
            stream_seed=7,
            expected_stream_digest_sha256="0" * 64,
        )


def test_formal_runner_is_offline_analysis_only() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_h8_f1c_final_performance_confirmation.py").read_text(
        encoding="utf-8"
    )
    forbidden_calls = (
        ".generate(",
        "AutoModelForCausalLM",
        "materialize_attack",
        "train_lora",
        "run_h8_f1b_formal_sampling",
    )
    assert all(marker not in source for marker in forbidden_calls)
    assert "audit_formal_records" in source
    assert "verify_f1b_terminal_index(config)" in source
    assert "evaluate_final_unit" in source
