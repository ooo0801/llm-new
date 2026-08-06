# Experiment B Protocol: Independent-Seed Replication of 14B Survivors

Status: **FROZEN BEFORE ANY EXPERIMENT B MODEL ENDPOINT**

## Question and confirmatory predictions

Experiment A produced eight legacy-retained prompts, four of which passed the strict 5/5 family gate. Experiment B asks whether those frozen survivors repeat under independently derived stochastic perturbations rather than merely surviving one attack draw.

- H-B1: at least 3 of the 4 strict survivors again satisfy the complete strict gate.
- H-B2: at least 6 of the 8 legacy survivors again satisfy the legacy gate.

Both predictions are confirmatory. Prompt text, task evaluators, model revision, perturbation strengths, five-family weights, micro/macro definitions, 1% family tolerance, and retention gates are unchanged from Experiment A.

## Frozen cohorts

Source: `results/experiment_a_qwen14b_20260805/analysis/per_prompt_transfer_results.jsonl` from result commit `d09e012`.

- Legacy cohort: exactly the eight rows where `legacy_retained == true`.
- Strict cohort: exactly the four rows where `strict_5of5_retained == true`; this is a nested subset of the legacy cohort.
- Paired texts come unchanged from `experiments/prompt-transfer-14b-a1/inputs/accepted16_pairs.jsonl`.

No prompt is added, edited, repaired, or removed using Experiment B outcomes.

## Independent randomization

Seeds are deterministically derived as the first three big-endian uint32 words of `SHA256("experiment-b-survivor-replication-v1")`:

- macro/attack seeds: `2435199792`, `4072939906`;
- blockwise Hutchinson seed: `1685839144` with four probes.

Structured pruning, Gaussian noise, and LoRA therefore use new stochastic instances. LoRA adapters are retrained from scratch for both seeds with the frozen 50-step configuration. Global-magnitude unstructured pruning and NF4 quantization are deterministic configurations; for those two families Experiment B is a fresh process/materialization repeat, not a new random perturbation. This limitation must be reported explicitly.

## Model and device policy

- `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- BF16 base model; NF4 only for the registered quantization family.
- One process shards layers across GPUs 0/1/2 with 30 GiB maximum per GPU.
- No CPU or disk offload; one endpoint model resident at a time.
- Stop on OOM, NaN/Inf, missing adapter, attack no-op, device mismatch, corrupted output, or insufficient disk.

## Frozen endpoints and gates

For each of eight prompt pairs:

1. deterministic task evaluation of initial and optimized text;
2. complete 147-block micro score for both texts, four probes with the new seed;
3. five-family macro score using two registered variants per family;
4. legacy retained iff technical pass, both task endpoints pass, micro gain is positive, equal-family macro gain is positive, and at least 3/5 families are non-degraded;
5. strict retained iff the legacy gate passes and all 5/5 families are non-degraded.

Family non-degeneration remains `optimized >= initial - 0.01 * abs(initial)`. Experiment A values are never pooled into Experiment B gate decisions.

## Outputs

- `experiments/prompt-transfer-14b-b1/`
- `results/experiment_b_qwen14b_20260806/`
- `reproducibility/experiment_b_qwen14b_20260806/`
- `to_human/experiment_b_qwen14b_20260806/`

Existing V1/V2/V6 and Experiment A outputs are immutable. Large adapters and raw runtime logs remain untracked; compact manifests, metrics, checksums, and reports are committed.
