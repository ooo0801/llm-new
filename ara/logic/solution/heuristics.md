# Heuristics

## H01: Archive scientific No-Go states as first-class results
- **Rationale**: A failed frozen gate is evidence about the method and must remain distinguishable from technical failure or incomplete execution.
- **Provenance**: ai-executed
- **Sensitivity**: high
- **Code ref**: [`scripts/archive_experiment_f_robust.py`, `scripts/archive_fingerprint_14b_followup.py`]

## H02: Score unique texts once, then expand through a frozen pair mapping
- **Rationale**: Shared initial/source prompts need only one complete-block or macro evaluation; deterministic expansion preserves all 64 pair endpoints while reducing redundant model work.
- **Provenance**: ai-executed
- **Sensitivity**: medium
- **Code ref**: [`scripts/prepare_discrete_hard_validation.py`, `scripts/expand_hard_validation_scores.py`]

## H03: Pin line endings for hash-bound release files
- **Rationale**: Git CRLF conversion can invalidate Linux-generated SHA-256 manifests after a Windows checkout even when semantic text is unchanged.
- **Provenance**: ai-executed
- **Sensitivity**: high
- **Code ref**: [`.gitattributes`, `reproducibility/experiment_f_qwen14b_robust_20260807/SHA256SUMS`]

## H04: Freeze independent resampling streams before score fitting
- **Rationale**: Domain-separated Fit and Audit roots, per-trial PCG64 seeds, and schedule digests prevent response-role leakage and result-dependent resampling while keeping 96,000 pseudo splits compactly reproducible.
- **Provenance**: ai-executed
- **Sensitivity**: high
- **Code ref**: [`src/llm_integrity/h8_d1c.py`, `reproducibility/h8_qwen32b_score_calibration_20260820/d1c/D1C_CPU_SPLIT_MANIFEST.json`]

## H05: Freeze one maximal response bank and compare sample sizes through nested memberships
- **Rationale**: Precommitting R40 within R60 and Q10 within Q20 lets all four sample structures share the same future 60-reference and 20-target banks, isolates the effect of sample size from independent generation noise, and prevents output-dependent subset selection.
- **Provenance**: ai-executed
- **Sensitivity**: high
- **Code ref**: [`src/llm_integrity/h8_d2a.py`, `reproducibility/h8_qwen32b_detector_development_20260820/d2a/D2A_NESTED_SUBSET_MANIFEST.json`]
