# Claims

## C01: Multi-seed robust selection can freeze a balanced 30-prompt set
- **Statement**: The frozen F1 ranking and quota procedure can select 30 unique-source candidates with at least three from each of eight categories.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The selector returns fewer than 30 rows or any category has fewer than three selected unique sources.
- **Proof**: [`reproducibility/experiment_f_qwen14b_robust_20260807/01_development/analysis/report.json`, N04, N05]
- **Dependencies**: []
- **Tags**: Qwen2.5-14B, prompt selection, category coverage

## C02: F1 development endpoints are technically complete
- **Statement**: All frozen F1 development task, complete-block micro, and fifteen-variant macro endpoints execute with the expected counts and no technical error.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: Missing endpoint records, incomplete parameter coverage, nonfinite values, unregistered variants, or a technical error in the terminal report.
- **Proof**: [`reproducibility/experiment_f_qwen14b_robust_20260807/FINAL_REPORT.json`, `reproducibility/experiment_f_qwen14b_robust_20260807/SHA256SUMS`, N04]
- **Dependencies**: []
- **Tags**: technical gate, reproducibility

## C03: Multi-seed F1 supports independent confirmation
- **Statement**: F1 can freeze an eligible 30-row development set and then retain at least 23 rows with all eight categories under isolated confirmation variants.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The development selection gate fails, or confirmation retains fewer than 23 rows, or confirmation lacks a category.
- **Proof**: [`reproducibility/experiment_f_qwen14b_robust_20260807/FINAL_REPORT.json`, N05]
- **Dependencies**: [C01, C02]
- **Tags**: H-F1, independent confirmation

## C04: Frozen 7B stable-component thresholds transfer to 14B
- **Statement**: Unchanged 7B activation thresholds yield stable repeated 14B profiles, a universe passing saturation and audit-novelty gates, and a deterministic MCC12.
- **Status**: untested
- **Provenance**: ai-suggested
- **Falsification criteria**: Any activation, universe, or selection gate fails under the frozen G1 protocol.
- **Proof**: [pending]
- **Dependencies**: [C03]
- **Tags**: H-G1, stable components, MCC

## C05: Prompt-stratified MMD detects registered 14B modifications
- **Statement**: The intact state is not rejected, at least 9 of 11 modified states are rejected, and every modified family has at least one detection.
- **Status**: untested
- **Provenance**: ai-suggested
- **Falsification criteria**: Intact rejection, fewer than 9 modified detections, or zero detections in a registered family.
- **Proof**: [pending]
- **Dependencies**: [C04]
- **Tags**: H-H1, prompt-stratified MMD

## C06: H8 D-to-S parameters are stable across independent intact response banks
- **Statement**: For all four retained sample structures, every nondegenerate Qwen2.5-32B fingerprint satisfies `0.5 <= a_audit/a_fit <= 2.0` and `abs(m_audit-m_fit)/a_fit <= 1.0`, while every structurally degenerate fingerprint preserves zero raw MMD and zero score on the independent Audit bank.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any nondegenerate structure-prompt pair violates either registered gate, any degenerate Audit MMD exceeds `1e-12`, any mapped degenerate score is nonzero, or Audit responses enter the Fit interface.
- **Proof**: [`reproducibility/h8_qwen32b_score_calibration_20260820/score_calibration_frozen_v1/SCORE_CALIBRATION_STABILITY_AUDIT_REPORT.json`, `reproducibility/h8_qwen32b_score_calibration_20260820/score_calibration_frozen_v1/d1c_raw_unbiased_mmd2_float64.npy`, N08]
- **Dependencies**: []
- **Tags**: H8, Qwen2.5-32B, score calibration, independent audit, MMD
