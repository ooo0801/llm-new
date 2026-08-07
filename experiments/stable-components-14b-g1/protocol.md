# Experiment G1 Protocol: Direct Threshold Migration for 14B Stable Components and MCC

Status: **CONDITIONAL ON H-F1 SUPPORT; MUST BE COMMITTED BEFORE ANY G1 MODEL ENDPOINT**

## Research question and hypothesis

G1 tests whether the stable-component extraction definitions and numerical thresholds frozen for the 7B V2 study can be transferred unchanged to the fixed Qwen2.5-14B model. It is a confirmatory threshold-migration test, not a new threshold calibration.

**H-G1:** the unchanged thresholds produce technically stable repeated activation profiles, an empirical global component universe that passes the frozen saturation and audit-novelty gates, and a deterministic 12-prompt global-unweighted MCC selection from the independently confirmed F1 fingerprint.

If F1 does not support H-F1, G1 does not run. If the G1 universe or selection gate fails, the result is a No-Go and any 14B-specific threshold calibration must be preregistered separately as G2.

## Frozen input and model

- Candidate source: the F1 independently confirmed legacy-retained fingerprint, requiring at least 23 prompts and all eight categories.
- Candidate text: the frozen optimized prompt from each F1 pair.
- Model and revision: `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- BF16, balanced three-GPU layer sharding, 30 GiB maximum per GPU, no CPU/disk offload.
- No candidate, threshold, or calibration prompt is edited after a G1 activation endpoint is produced.

## Frozen component extraction

- Two repeated profiles per prompt; a component is stable only when present in both repeats.
- Attention component: normalized causal attention entropy below `0.70`.
- FFN component: activation in the top 5% within the layer, capped at 128 units per layer.
- Attention/MLP residual component: residual-to-layer-input norm ratio above `0.50`.
- Repeated component-set Jaccard must be at least `0.999` for every split.

These are the exact 7B V2 values. They are not tuned on 14B.

## Independent empirical global universe

- Fourteen frozen task families.
- Build split: 20 prompts per family, 280 total.
- Audit split: 4 prompts per family, 56 total.
- Calibration prompts must be unique and remain below trigram Jaccard `0.72` against every candidate.
- The global empirical universe is the union of stable build and audit components.
- The last three 14-prompt build batches must each add at most 2% relative components.
- Audit-only novelty must be at most 8% of the audit universe.

The universe is an empirical observable-component estimate, not the mathematical set of all important model parameters.

## MCC selection and reported metrics

- Primary selection: deterministic greedy global-unweighted MCC.
- Selected size: 12 prompts.
- Report candidate reachability, selection efficiency, final global coverage, factorization error, per-component-type coverage, and k=4/8/12 coverage curves.
- Weighted and type-balanced objectives are ablations and do not replace the primary selection.

H-G1 requires successful technical profiling, universe gates, deterministic unique MCC12 selection, and exact coverage factorization. No minimum final global coverage is invented after results.
