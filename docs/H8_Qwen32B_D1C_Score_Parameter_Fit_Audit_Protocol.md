# H8 D1-C Score Parameter Fit + Independent Stability Audit

## Scope

This phase performs no model inference. Its only response input is the frozen D1-B1 file containing 1,200 `score_calibration_fit_only` and 1,200 `score_calibration_stability_audit_only` responses. Formal Reference, held-out, Target, and attack data are forbidden.

The frozen H8 MMD measurement layer is immutable. Feature schema, exclusion mask, twelve family-balanced scalers, twelve bandwidths, global fallback bandwidth, structural-degeneracy labels, and the unequal-size unbiased MMD implementation are loaded fail-closed from `MMD_FROZEN_MANIFEST.json`.

## Frozen CPU resampling

Four structures are retained without selection: `r40_q10`, `r40_q20`, `r60_q10`, and `r60_q20`. Fit and Audit have distinct 64-bit root seeds. For every role × structure × prompt × trial, a trial seed is derived by domain-separated SHA-256 and supplied to NumPy PCG64. Exactly `N_R+N_Q` indices are sampled without replacement from the relevant 100-response bank; the first `N_R` form pseudo-reference and the remainder form pseudo-target. The seed derivation and every split-schedule digest are frozen before any D-to-S parameter is calculated.

There are 96 independent streams and 96,000 pseudo splits: 2 roles × 4 structures × 12 prompts × 1,000 trials. Fit and Audit never share response records or resampling streams.

## Fit

For each structure and nondegenerate prompt, raw generalized unequal-size unbiased MMD² values remain signed. From the 1,000 Fit-only null values:

`m_j = median(D_null)`

`a_j = 1.4826 × median(|D_null - m_j|)`

Any nondegenerate `a_j <= 1e-12` is an immediate FAIL. It may not use the degenerate fallback. For prompts already labelled structurally degenerate by the frozen MMD layer, `a_global` is the median of the positive nondegenerate `a_j` values within the same sample structure. No value is pooled across structures.

The score map is applied only after raw MMD calculation:

- nondegenerate: `S_j = max(0, (D_j - m_j) / a_j)`;
- structurally degenerate: `S_j = max(0, D_j / a_global)`.

## Independent Audit gate

Fit parameters are immutable before Audit is evaluated. Audit-only responses produce report-only `m_audit` and `a_audit`; these values never replace or update Fit parameters.

For every nondegenerate prompt and structure, both conditions must hold:

- `0.5 <= a_audit / a_fit <= 2.0`;
- `abs(m_audit - m_fit) / a_fit <= 1.0`.

For every structurally degenerate prompt, all Audit raw MMD values must satisfy `abs(D) <= 1e-12`, and the frozen Fit map must return exactly zero. Every raw D and score must be finite. Failure of any gate stops the phase without parameter tuning or a frozen score manifest.

## PASS artifact state

Only an all-gates PASS may create `SCORE_CALIBRATION_FROZEN_MANIFEST.json`. The final state is:

- `measurement_layer = frozen`;
- `score_layer = frozen`;
- `sample_size = not_selected`;
- `aggregation = not_selected`;
- `detector = not_frozen`.

