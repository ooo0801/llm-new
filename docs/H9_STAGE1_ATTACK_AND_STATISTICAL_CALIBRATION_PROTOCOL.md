# H9 Stage 1 attack-strength and statistical-calibration protocol

## Purpose

This stage calibrates Gaussian and LoRA attack magnitudes on the frozen
Qwen2.5-0.5B-Instruct target before changing H6 prompt optimization, MCC selection, or
the H8 representation. It is a mechanism experiment, not a final fingerprint result.

## Frozen design

- Use the 12 prompts and isolated LoRA files frozen by commit `a3e2f592`.
- Generate 48 intact responses per prompt. Use 24 responses per prompt for every
  attack endpoint. Attack endpoints share response seeds with one another to support
  controlled cross-attack comparisons, but the tested intact half-bank uses disjoint
  response seeds because H8 MMD is an unpaired two-sample statistic.
- Gaussian: three nominal standard-deviation ratios and three attack seeds.
- LoRA: one controlled rank/alpha/learning-rate/target-module setting, three optimizer
  step budgets, and three training seeds. Report actual adapter update magnitude and
  held-out token NLL; optimizer steps alone are not treated as attack strength.
- Do not reuse any 32B scaler, null distribution, MAD, or RBF bandwidth.

## Statistical contract

The intact bank alone fits prompt-specific 528D H8 scalers and bandwidths. For each
prompt, disjoint 24-vs-24 intact pseudo-splits form 199 null-fit and 200 null-evaluation
statistics. Raw response distribution components are prefix-10 total variation,
bigram Jensen-Shannon divergence, and BGE semantic-centroid distance.

Each raw component is standardized against its intact null. A component with null
standard deviation below `1e-12` is explicitly excluded; it is never divided by a
  numeric floor. The composite statistic applies `max` to standardized components inside
  every null split, and the observed max is compared with that composite null. Therefore
  the multiple-component search is part of the calibration rather than added afterward.
  All decisions use the finite-sample empirical upper-tail p-value with the `+1`
  correction. This avoids anti-conservative threshold ties when a null is discrete.

H8 MMD uses the frozen 528D definition, population standardization, feature-family
balancing, prompt bandwidths with a global fallback for degenerate prompts, and the
generalized unequal-size unbiased estimator. Its null is calibrated separately.

## Aggregation and decision

First aggregate the three attack seeds at each fixed strength for each prompt. Only
then aggregate prompts and compare strengths. Report q25/median effects, empirical
detection fractions, task-pass loss, actual Gaussian relative Frobenius change or LoRA
relative adapter delta, and monotonicity. The recommended operational strength is the
weakest level meeting all predeclared response-sensitivity and task-integrity gates. If
none does, the family is reported as not calibrated; thresholds are not relaxed after
seeing results.

This stage preserves the paper's innovation boundary: it supplies trustworthy attacks
and null statistics but does not claim the later behavior-aware H6 objective or
sensitivity-constrained MCC contribution.
