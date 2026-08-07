# Experiment G2 Protocol: Final 14B Stable Components and MCC12

Status: **CONDITIONAL ON H-F2 SUPPORT; FROZEN BEFORE ANY F2/G2 MODEL ENDPOINT**

**H-G2:** the independently confirmed F2 set supports a technically stable, saturated and auditable empirical 14B component universe and a deterministic 12-prompt global-unweighted MCC selection under the unchanged 7B thresholds.

The candidate source must contain at least 23 independently confirmed F2 prompts and all eight categories. Two repeated activation profiles are collected per prompt. Frozen thresholds are attention-entropy fraction `0.70`, layerwise FFN quantile `0.95` capped at 128 units, residual ratio `0.50`, and repeat Jaccard at least `0.999`.

The independent empirical universe uses 14 task families, 280 build prompts and 56 audit prompts. Calibration/candidate trigram Jaccard must remain below `0.72`; each of the final three build batches may add at most 2% relative components; audit-only novelty may be at most 8%. The final selection is deterministic global-unweighted MCC with 12 prompts. A failed gate is archived without threshold adjustment.
