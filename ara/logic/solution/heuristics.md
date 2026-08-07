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
