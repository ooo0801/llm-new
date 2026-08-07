# Experiment H1 Protocol: Prompt-Stratified MMD Detection on Qwen2.5-14B

Status: **CONDITIONAL ON H-F1 AND H-G1 SUPPORT; MUST BE COMMITTED BEFORE ANY H1 REFERENCE OR TARGET RESPONSE**

## Research question and hypothesis

H1 tests whether the G1 MCC12 fingerprint distinguishes the intact fixed Qwen2.5-14B revision from independently materialized pruning, quantization, Gaussian-noise, and LoRA states using the already implemented prompt-stratified MMD detector.

**H-H1:** the intact state is not rejected, at least 9 of 11 registered modified states are rejected, and at least one instance from every registered modified family is detected.

The gate is frozen before reference generation. Per-variant outcomes, pooled-MMD ablations, Gaussian-noise strength behavior, and confidence intervals are reported regardless of the primary result.

## Frozen fingerprint and response protocol

- Candidate set: the deterministic G1 global-unweighted MCC12 selection.
- Ten reference responses per prompt with seeds `2026080800` through `2026080809`.
- Ten target responses per prompt and model state with seeds `2026080900` through `2026080909`.
- Generation: sampling temperature 0.7, top-p 0.9, top-k 50, no system prompt.
- Feature representation: frozen surface, semantic, and task features; semantic encoder `BAAI/bge-small-zh-v1.5@7999e1d...` on CPU.

## Frozen primary statistic

- RBF kernel with prompt-specific median bandwidth.
- Unbiased MMD within every prompt stratum.
- Primary statistic: mean prompt-wise MMD.
- Exactly 999 permutations.
- Labels are permuted only inside the same prompt stratum.
- Significance level `alpha=0.05`.
- Ordinary pooled MMD is an ablation only and cannot replace the primary result.

## Registered model states

One intact state and eleven modified states are frozen:

- two unstructured-pruning instances;
- two structured-pruning instances covering FFN channels and attention heads;
- three Gaussian-noise strengths (`0.00025`, `0.0005`, `0.001`);
- INT8 and NF4 quantization;
- two independently trained rank-16, 50-step LoRA adapters.

All modified states are independently materialized from the fixed base revision. The two LoRA adapters are trained from the isolated frozen attack dataset only after this protocol is committed.

## Gates and claim boundary

The technical gate requires all 12 states, 120 target responses per state, finite feature arrays, a valid 999-permutation primary test, and recorded realization metadata. H-H1 additionally requires:

1. intact predicted unmodified;
2. at least 9/11 modified variants predicted modified;
3. nonzero detection in unstructured pruning, structured pruning, Gaussian noise, quantization, and finetuning.

The experiment reports fixed-instance performance with exact binomial intervals. It does not claim a population detection rate for arbitrary future models, attacks, or perturbation strengths.
