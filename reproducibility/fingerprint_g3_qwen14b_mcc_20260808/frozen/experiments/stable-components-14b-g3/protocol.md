# Experiment G3 Protocol: Frozen-19 Recovery MCC12

Status: **PREREGISTERED BEFORE ANY G3 MODEL ENDPOINT**

**H-G3:** the exact 19-prompt legacy-retained subset produced by the completed F2 independent confirmation supports a technically stable, saturated and auditable empirical Qwen2.5-14B component universe and a deterministic 12-prompt global-unweighted MCC selection under the unchanged component thresholds.

F2 remains a confirmation No-Go at 19/30 versus 23 required. G3 is a new recovery experiment, not a reinterpretation of H-F2. Its candidate source is frozen byte-for-byte at SHA-256 `6ecad275352bd532dd944154b569c6935ea849d5538ca7b0da2daf7ed7aa31e1`, contains exactly 19 unique prompt IDs and texts, and covers all eight categories: 1 code, 4 instruction, 2 knowledge, 4 logic, 3 reasoning, 2 safety, 1 structured and 2 summary.

Two repeated activation profiles are collected per prompt. Frozen thresholds remain attention-entropy fraction `0.70`, layerwise FFN quantile `0.95` capped at 128 units, residual ratio `0.50`, stable-component frequency `1.0`, and repeat Jaccard at least `0.999`.

The independent empirical universe uses 14 task families, 280 build prompts and 56 audit prompts. Calibration/candidate trigram Jaccard must remain below `0.72`; each of the final three 14-prompt cumulative batches may add at most 2% relative components; audit-only novelty may be at most 8%. The final universe is the build/audit union.

If and only if activation, saturation, audit and selection gates pass, deterministic global-unweighted MCC selects exactly 12 unique prompts from the frozen 19. The implementation must reproduce the same IDs and trace on immediate recomputation, preserve monotone coverage for k=4/8/12/16, cover at least one global component, and pass the numerical factorization audit. No threshold, source row or component definition may be changed after endpoints.
