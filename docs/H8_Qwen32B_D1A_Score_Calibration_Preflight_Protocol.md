# H8 D1-A Score Calibration Layer Preflight Protocol

## 1. Scope and immutable dependency

This phase is code preflight only. It loads the H8 MMD measurement layer from
`MMD_FROZEN_MANIFEST.json` and fails closed unless every archived file digest,
feature-schema binding, twelve scaler payload digests, twelve bandwidth payload
digests, global-bandwidth payload digest, and MMD implementation commit match the
frozen manifest and final M0-I/J report.

No model sampling and no formal Reference, held-out, Target, or attack access are
authorized. Frozen MMD artifacts must not be modified. Existing M0 fit-only
pseudo-MMD values may be used only as `development_dry_run`; they cannot freeze
score parameters or select sample size or Top-r.

## 2. Preserved sample structures

The score layer preserves four independent structures:

- `r40_q10`
- `r40_q20`
- `r60_q10`
- `r60_q20`

No cross-structure pooling and no sample-size selection are permitted. Final
calibration parameters must be fitted independently for every structure.

## 3. Primary one-sided robust score

For a nondegenerate fingerprint `j`, using only future
`score_calibration_fit_only` null statistics of the same sample structure:

`m_j = median(D_null,j)`

`a_j = 1.4826 * median(|D_null,j - m_j|)`

`S_j(D) = max(0, (D - m_j) / a_j)`

Raw generalized unbiased `MMD^2` values, including negative values, remain
unchanged in the measurement layer. The one-sided truncation exists only in the
score mapping.

If a nondegenerate fingerprint has `a_j <= numerical_zero_threshold`, calibration
fails closed. It must not silently borrow the degenerate fallback.

## 4. Frozen structural-degeneracy fallback

The fallback in this section is frozen before any new score-calibration data are
generated. For each sample structure, compute

`a_global = median({a_k : k is nondegenerate and a_k > numerical_zero_threshold})`.

For a structurally feature-degenerate fingerprint whose own robust scale is at or
below the numerical threshold, define

`S_j(D) = max(0, D / a_global)`.

No epsilon replacement or artificial scale amplification is allowed. A missing,
nonfinite, or nonpositive `a_global` is a hard failure. Structural degeneracy is
read exclusively from the frozen prompt-bandwidth artifacts; it cannot be inferred
or changed from new Target or attack observations.

## 5. Final calibration data boundary

The proposed future calibration manifest must declare every record as
`data_role=score_calibration_fit_only`, with all formal-reference, held-out, Target,
and attack eligibility flags false. It binds response/split IDs, prompt ID,
sample-structure ID, measurement-manifest SHA256, feature-schema SHA256, scaler and
bandwidth payload SHA256, raw unbiased MMD statistic, and deterministic split seed.

Only these null records may fit `m_j`, `a_j`, and `a_global`. Target and attack data
are not accepted by the fitting API.

## 6. Serialization and phase boundary

Score parameters use canonical UTF-8 JSON, float64 semantics, a frozen score-schema
SHA256, and a canonical payload SHA256. Loading validates both the score artifact
and the current frozen MMD archive. A byte change in either dependency is fatal.

D1-A produces a preflight report only. It does not freeze detector parameters and
must stop before any new 32B response, final D-to-S fitting, Top-r/Max/Energy,
global permutation, or downstream detector calibration.
