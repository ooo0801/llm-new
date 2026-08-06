# Experiment C Protocol: Model-Aware 14B Prompt Construction

Status: **FROZEN BEFORE ANY EXPERIMENT C MODEL ENDPOINT**

## Research question and prediction

Experiment A tested direct transfer of 7B prompt instances. Experiment C tests a different claim: whether the already implemented V6 discrete construction procedure can be rerun on the fixed 14B model to produce a model-specific complement that survives an independent outer attack split.

H-C1: at least four prompts frozen after 14B construction/development calibration satisfy the unchanged strict validation gate on the held-out test attack split.

The threshold is fixed before construction. A nonempty result below four is descriptive and does not support H-C1.

## Frozen data flow

1. Verify the original synthetic candidate pool contains exactly 336 rows.
2. Reuse the deterministic V6 stratified sample of 60 source prompts; verify every source ID belongs to the 336-row pool and lock its SHA256.
3. Use the first 12 frozen source prompts only to fit model-specific inner micro/macro scales.
4. Run the unchanged three-round V6 HotFlip construction on all 60 prompts.
5. Build at most two portfolio candidates per source and score them on the development attack split.
6. Fit the fixed outer hybrid calibration from development **initial prompts only**, then freeze at most one selected candidate per source.
7. Only if at least four development candidates pass, evaluate the frozen set on the held-out test attack split.

No Experiment A or B sensitivity outcome is used to select, edit, rank, or repair an Experiment C prompt. Development and held-out test scores are stored separately.

## Frozen model and construction method

- Model: `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`, BF16.
- Single-process layer sharding across GPUs 0/1/2, 30 GiB per GPU; no CPU/disk offload.
- Raw prompt text is retokenized with the 14B tokenizer; no 7B token IDs are reused.
- Three discrete rounds, one probe per inner step, block cycle `q_proj`, `v_proj`, `down_proj`.
- Four editable positions, 24 HotFlip directions per position, 12 position-diverse exact reranks.
- PPL ratio <=2, edit ratio <=0.25, digit preservation, task preservation, positive proxy gain.
- One search and two disjoint anchor variants per family; at least 3/5 non-degraded families with 1% relative tolerance.
- Five families remain equally weighted.

Before the full 60 prompts, a one-prompt engineering smoke run must complete without OOM, invalid gradient, missing adapter, or device mismatch. Its scientific outcome cannot change the frozen configuration.

## Attack split isolation

Manifests are selected deterministically from the original V6 train, validation, and test manifests without outcome metrics:

- inner construction pool: all executable original train variants; for LoRA, only the exact variant IDs registered by the original V6 train registry are retrained for 14B, and unavailable LoRA rows are filtered by the existing sampler;
- development outer validation: 2 variants per family;
- held-out test: 2 variants per family.

Variant IDs must be disjoint across splits. All 14B LoRA adapters are trained from scratch in split-specific registries. The test adapters are not trained or loaded until the development set is frozen. Deterministic families are fresh materializations of their split-specific configurations; stochastic families use the registered split seeds.

## Outer validation gate

For every candidate pair, use the same deterministic task check, complete 147-block four-probe micro score, equal-weight five-family macro score, positive micro/macro direction, and 1% family non-degeneration rule used in Experiment A. The final accepted flag is produced by the existing frozen hybrid-validation implementation plus the explicit task/family safeguards. No threshold is changed after results.

## Outputs

- `experiments/prompt-construction-14b-c1/`
- `results/experiment_c_qwen14b_20260806/`
- `reproducibility/experiment_c_qwen14b_20260806/`
- `to_human/experiment_c_qwen14b_20260806/`

Existing V1/V2/V6 and Experiments A/B remain immutable. Raw logs and large adapters are untracked; compact manifests, selected prompts, metrics, checksums, and reports are committed.
