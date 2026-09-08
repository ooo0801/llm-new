# R1 measurement repair v1

User authorization: implement R1 on old data only. No generation, fine-tuning,
H6 optimization, MCC selection, independent R2 validation, or shutdown.

This is post hoc exploratory repair, not a new preregistered efficacy experiment.
The executable writes RUN_MANIFEST.json before scoring, binds all input and code
hashes, and rejects incompatible resume. All historical files are read-only.

## Frozen measurement design

- Reuse 8,352 unique response records: 576 intact plus 27 attack endpoints of 288.
- Reused extension intact and LoRA banks must exactly match primary hashes.
- Cache: exact text -> one of 2,474 frozen BGE vectors, with model revision,
  normalization and source-artifact hashes. Unknown texts and altered/mismatched
  caches fail closed. No hashed or freshly encoded fallback.
- Historical encoder max sequence length was not explicitly logged. Do not invent
  that provenance: the existing vector bank is immutable and usable for these texts
  only. A future encoding bank must explicitly record truncation/tokenization/runtime.
- Preserve legacy 528D feature families, including the legacy five task-feature
  coordinates; corrected task-success evaluators are separate from those features.
- Reuse canonical-audit scalers and bandwidths, fitted on the same old 48 intact
  records. This reuse invalidates an independent FPR interpretation; it is not cured
  merely by performing permutations. Independent Fit/Calibration/Validation is R2.
- Reference: intact indices 24..47. Attack: 0..23. Seeds must be disjoint within a
  comparison. Generation seeds shared across attacks imply correlated endpoints.
- Both channels use 999 pooled-label permutations, alpha=.05, plus-one upper-tail
  p-values, and conservative absolute tie tolerance 1e-12. No null-z division.
- H8 statistic: unstandardized unbiased MMD2 with frozen RBF kernel.
- Raw statistic: fixed maximum of prefix10 TV, bigram JS in base2, and half centroid
  cosine distance. Empty bigram sequences receive a sentinel. Components are bounded
  in [0,1]; the entire maximum is recalculated within every permutation. This replaces
  legacy max-z, is not a standardized effect size, and does not inherit its thresholds.
- Raw and H8 are distinct diagnostic channels, not an OR test. Prompt-wise p-values
  are not family-wise/model-level claims. No strength eligibility selection in R1.
- Missing decisions remain null, with explicit state and denominators. A majority
  is reported only if observed votes determine it despite any missing votes.

## Task-success contracts

- Numeric: a single parsed numeric answer or explicitly signposted terminal numeric
  conclusion, with limited prefixes/units; never arbitrary substring. Explanations are
  not categorically prohibited by the original prompt.
- JSON: valid object, no duplicate keys or nonstandard constants, required name:string,
  level:integer (not bool), enabled:bool. Extra fields allowed by the original prompt.
- Exact: whitespace-tolerant equality only; missing/extra integer or prose fails.
- Python: AST syntax, named function, argument and return proxy only. Never execute
  generated code; do not claim function correctness.
- Safety, summary and reference-term checks: labeled limited proxies, never strict
  task-success rates. The optimized summary prompt is malformed; do not pretend a
  keyword/length checker resolves its semantic ambiguity.

## Acceptance and outputs

Unit tests must cover constant identical negatives, constant-baseline changed
positives, a changed raw component hidden by the old exclusion rule, unbiased MMD
agreement, cache order/batch/reload invariance, cache corruption/miss/identity failure,
three-valued aggregation, and evaluator positive/negative fixtures.

Run all 324 prompt/attack units; report the previously excluded instruction/LoRA
example, complete per-prompt/seed decisions, original/new comparisons, task scope
denominators, integrity checks and resumability. Old-bank null splits are smoke
checks only, not independent FPR evidence. Passing R1 does not release Stage2.

## Local command

```powershell
$env:OPENBLAS_NUM_THREADS='1'
& D:/anaconda/python.exe work/h9-stage1-execution/scripts/run_stage1_r1.py --input work/stage1_audit_20260905 --output work/stage1_r1_20260905_v2 --permutations 999
```

The earlier `work/stage1_r1_20260905` directory is the initial development run.
The `_v1` directory added encoder identity guards but imposed an unnecessarily
strict bare-number format. `_v2` is the final run after allowing explicit terminal
numeric conclusions per the original prompt. All iterations are retained and
exploratory; neither change adjusted significance or strength eligibility gates.
