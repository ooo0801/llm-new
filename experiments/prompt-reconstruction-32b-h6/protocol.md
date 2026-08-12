# H6 Protocol: Qwen2.5-32B Prompt Reconstruction and MCC12

Status: **PREREGISTERED BEFORE ANY QWEN2.5-32B MODEL ENDPOINT**

## Scope

H6 constructs a new model-specific sensitive-prompt pool and a new MCC12 for
`Qwen/Qwen2.5-32B-Instruct@5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.
H5 and every earlier 14B endpoint remain immutable. H6 reuses only clean task
texts and general algorithms: it does not reuse 14B token IDs, gradients,
calibration values, responses, activations, component IDs, LoRA weights, or
attack realizations.

This stage ends after prompt reconstruction, independent confirmation,
stable-component extraction, and deterministic MCC12 construction. Integrity
verification of intact and modified 32B states is a later, separately
preregistered experiment.

## Hypotheses

- **H6-P:** Model-aware construction on the fixed 32B model freezes at least
  30 development prompts and independently confirms at least 19 sensitive
  prompts, with all eight core categories represented.
- **H6-G:** The independently confirmed pool supports a stable, saturated and
  auditable 32B empirical component universe and deterministic global
  unweighted MCC selection of exactly 12 prompts.

The core categories are `code`, `instruction`, `knowledge`, `logic`,
`reasoning`, `safety`, `structured`, and `summary`. The shared clean source set
also contains `math`, `translation`, and `long_context`; these are diagnostic
and may enter the global fill after the core-category development quotas.

## Frozen model and runtime requirements

- Model revision: `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.
- BF16, eager attention, one process sharded over GPUs 0/1/2.
- `device_map: balanced`, maximum 30 GiB per GPU.
- CPU and disk offload are forbidden.
- The exact tokenizer files and chat template are hashed after download and
  before the first model endpoint.
- The environment record includes GPU topology, CUDA, Python, PyTorch,
  Transformers, Accelerate and BitsAndBytes versions.

The engineering gate requires full-model load, nonempty deterministic
generation, same-seed replay, one stable-component hook pass, and one complete
blockwise Hutchinson probe over every parameter group. OOM or incomplete
coverage is an engineering No-Go for the current 3x32 GiB runtime. It may be
resumed on a larger GPU host without changing scientific inputs; CPU/disk
offload or a quantized reference cannot repair the gate.

## Prompt construction and isolation

The 60 clean task texts are byte-frozen from Experiment C source SHA-256
`f6c55d4897e37c65e6779e24e1cbfc80e63a7d790ada15ac7a0a2b96282e7a4c`.
H6 generates new prompt IDs and new train/development/confirmation attack IDs
and seeds before endpoints. The first 12 H6 source rows fit 32B-specific inner
micro/macro scales. All LoRA adapters are trained from scratch on 32B in their
registered split.

Construction uses the existing three-round discrete joint optimizer with
`q_proj`, `v_proj`, and `down_proj`, four editable positions, 24 HotFlip
directions, 12 exact reranks, PPL ratio at most 2, edit ratio at most 0.25,
digit protection, task preservation, positive proxy gain, equal five-family
weighting, and at least 3/5 non-degraded families within 1%.

Development freezes at most one candidate per source. It must freeze at least
30 candidates and include at least one prompt from every core category before
confirmation begins. Confirmation uses disjoint attack seeds and may not edit,
rerank, replace, or re-optimize the frozen rows.

H6-P passes only when at least 19 independently confirmed rows remain and all
eight core categories are represented. A lower count or missing category is a
scientific No-Go; thresholds and rows are not repaired in H6.

## Stable components and MCC12

Only after H6-P passes, H6-G retokenizes the confirmed optimized texts on 32B
and extracts two activation profiles per prompt. The frozen direct-transfer
threshold test uses attention entropy fraction 0.70, layerwise FFN quantile
0.95 capped at 128 units, residual ratio 0.50, stable frequency 1.0, and
minimum repeat Jaccard 0.999.

The independent empirical universe uses 14 task families, 280 build prompts
and 56 audit prompts. Calibration/candidate trigram Jaccard must be below 0.72;
each of the last three 14-prompt batches may add at most 2% relative
components; audit-only novelty may be at most 8%. Failure is a direct-threshold
transfer No-Go and cannot be repaired within H6.

Global-unweighted MCC must select exactly 12 unique prompts, reproduce its IDs
and trace on immediate recomputation, preserve monotone k=4/8/12/16 coverage,
and pass the numerical factorization audit. Candidate-reachable selection
efficiency and empirical-global coverage are reported separately.

## Seeds and endpoints

- Input/manifest construction seed namespace: `20260820xx`.
- Inner calibration and optimizer seed: `2026082010`.
- Development micro/bootstrap seeds: `2026082020` / `2026082024`.
- Confirmation micro/bootstrap seeds: `2026082030` / `2026082034`.
- Component preparation seed: `2026082040`.

Every manifest records exact per-variant seeds. Development, confirmation and
future verification endpoints are disjoint.

## Evidence and negative-result policy

Raw results and adapters stay under `results/` and are not committed. Compact
protocols, manifests, reports, confirmed prompts, component audits, MCC trace,
final fingerprint, environment record and SHA-256 manifest are archived under
`reproducibility/`. Text is written as UTF-8 with LF endings and hashes are
computed over repository bytes. Scientific No-Go exit codes are valid terminal
results and must not trigger seed, threshold, candidate or attack replacement.
