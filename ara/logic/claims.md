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

## C07: H8 D2-A detector-development inputs are reproducibly frozen without evaluation leakage
- **Statement**: Given the frozen H8 MMD and score manifests, D2-A deterministically constructs exactly twelve Top-r development configurations, nested sample memberships, role-restricted development interfaces, global-permutation streams, and proposed response schedules without reading final held-out/final-attack data or generating model responses.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any frozen dependency or payload hash can be altered without rejection; the candidate set differs from four structures times Top-2/3/4; R40 is not a subset of R60 or Q10 is not a subset of Q20; final-heldout/final-attack roles enter development selection; a proposed generation seed is duplicated or overlaps an available frozen repository seed; or rerunning with the same manifests changes global-permutation output.
- **Proof**: [`reproducibility/h8_qwen32b_detector_development_20260820/d2a/H8_D2A_DETECTOR_DEVELOPMENT_PREFLIGHT_REPORT.json`, `reproducibility/h8_qwen32b_detector_development_20260820/d2a/D2A_ARTIFACT_SHA256_INDEX.json`, `tests/test_h8_d2a.py`, N10]
- **Dependencies**: [C06]
- **Scope note**: This is a reproducibility and leakage-control claim, not evidence that any detector configuration has adequate power or specificity.
- **Tags**: H8, Qwen2.5-32B, detector development, global permutation, preregistration

## C08: H8 D2-B0 materializes the eight registered fresh-development attack endpoints without detector evaluation
- **Statement**: The D2-B0 authorization layer can materialize and smoke-test exactly two Gaussian, two pruning, two LoRA, and two quantization endpoints while preserving unique smoke-only seeds, excluding all smoke records from formal development banks, verifying the realized attack state, and leaving detector statistics and configuration selection untouched.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any endpoint fails to materialize; any smoke seed or response ID duplicates or overlaps the formal development schedule; a quantized endpoint silently loads as the intact model; a LoRA adapter overlaps an available H6 adapter hash or uses MCC12 data; GPU workers remain after completion; or any smoke response enters detector evaluation or configuration selection.
- **Proof**: [`reproducibility/h8_qwen32b_detector_development_20260820/d2b0/H8_D2B0_DEVELOPMENT_SAMPLING_AUTHORIZATION_PREFLIGHT_REPORT.json`, `reproducibility/h8_qwen32b_detector_development_20260820/d2b0/D2B0_FRESHNESS_MATERIALIZATION_AUDIT.json`, `reproducibility/h8_qwen32b_detector_development_20260820/d2b0/D2B0_FINAL_SHA256_INDEX.json`, N12]
- **Dependencies**: [C07]
- **Scope note**: H6 did not archive artifact hashes for its in-memory Gaussian, pruning, or quantized states. D2 freshness for those families is therefore supported by configuration, seed, and realized-state evidence, not by a claim of verified artifact-level zero overlap. Eight smoke responses provide no evidence of detector power or specificity.
- **Tags**: H8, Qwen2.5-32B, fresh attacks, materialization, smoke preflight, provenance
