# Experiment E Protocol: Joint Confirmation of the Frozen 30-Prompt Qwen2.5-14B Fingerprint

## Status and temporal lock

This document, the frozen pair manifest, attack manifest, seeds, configuration, and analysis code must be committed before any Experiment E model endpoint is produced. Experiments V1/V2/V6 and A/B/C/D are immutable evidence sources. Experiment E writes only to `experiments/prompt-union-14b-e1`, `results/experiment_e_qwen14b_final_20260807`, `reproducibility/experiment_e_qwen14b_final_20260807`, and `to_human/experiment_e_qwen14b_final_20260807`.

## Research question and hypothesis

The descriptive Experiment D union contains four independently repeated cross-model strict-core prompts and 26 non-overlapping Qwen2.5-14B-specific prompts. Experiment E asks whether this exact 30-prompt union remains a usable single fingerprint when every pair is evaluated together under one new, preregistered set of micro probes and attack instances.

**H-E1:** at least 23 of the 30 frozen prompt pairs (75%, rounded up) will satisfy the unchanged legacy retention rule, and the retained set will contain at least one prompt from each of the eight frozen categories: code, instruction, knowledge, logic, reasoning, safety, structured, and summary.

The strict 5/5 retained count is a secondary descriptive endpoint with no success threshold. H-E1 is supported only if the technical gate, the 23-prompt gate, and eight-category coverage all pass.

## Frozen model and candidate data

- Model: `Qwen/Qwen2.5-14B-Instruct`.
- Revision: `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- Dtype: BF16.
- Placement: balanced layer sharding across GPUs 0/1/2, maximum 30 GiB per GPU, with no CPU or disk offload.
- Candidate source: `reproducibility/experiment_d_combined14b_20260806/combined_fingerprint.jsonl`.
- Frozen pairs: `inputs/final30_pairs.jsonl`, reconstructing the original initial prompt from Experiment B or C provenance while requiring the optimized text and hash to match Experiment D exactly.
- Composition: four `cross_model_strict_core` and 26 `qwen14b_model_specific_complement` rows.

No candidate may be added, removed, edited, reranked, or replaced after an Experiment E endpoint is observed. A scientific failure does not authorize tuning.

## Independent randomness and perturbations

- Complete Hutchinson micro seed: `2026080701`, four Rademacher probes over all 147 parameter blocks.
- Two attack seeds per family: `2026080702` and `2026080703`.
- Macro bootstrap seed: `2026080704`.
- The seeds are distinct from recorded Experiment A/B/C seeds `42`, `777`, `3407`, `1685839144`, `2435199792`, and `4072939906`.
- Five family definitions remain fixed: global-magnitude 30% attention unstructured pruning; random 5% FFN-channel structured pruning on a random layer subset; full-model double-quantized NF4 BF16; FFN Gaussian noise at 0.0005 parameter-tensor standard deviation; and 50-step rank-16 LoRA at learning rate 1e-5.
- The LoRA attack data remain the isolated frozen `data/attack_train_lora.jsonl`; two new adapters are trained from the new seeds.

## Staged execution

1. Validate all 30 pair hashes, provenance, counts, and seed separation without loading the model.
2. Load tokenizer/config offline and require all initial/optimized strings to fit both the 512-token task limit and 128-token sensitivity limit.
3. Load the complete BF16 model on all three GPUs and generate four tokens without CPU/disk offload.
4. Train the two new registered LoRA attack adapters.
5. Run one frozen pair through two task endpoints, two complete micro endpoints, and 20 macro observations. This is an engineering smoke gate; scientific direction does not stop or tune the confirmatory run.
6. Run all 30 pairs: 60 deterministic task endpoints, 60 complete micro endpoints, and 600 macro observations.
7. Apply the gates below once, write the final retained set, and archive evidence.

Task and micro runners use resume semantics. A technical failure may be repaired only if the fix does not change candidates, seeds, attack definitions, evaluators, scoring, or thresholds. All failed logs remain preserved.

## Frozen per-prompt gates

For every source prompt, compare the initial and optimized text within the target 14B model.

1. **Task preservation:** both deterministic greedy task endpoints pass the frozen evaluator.
2. **Micro direction:** optimized `micro_per_parameter` is strictly greater than initial.
3. **Macro direction:** the equal-weight mean of the five family-specific mean `macro_l2_raw` values is strictly greater for optimized than initial.
4. **Family non-degeneration:** a family passes when optimized is no worse than 1% below initial. The legacy rule requires at least 3/5 families; the strict rule requires 5/5.
5. **Legacy retained:** task preservation, positive micro, positive macro, and at least 3/5 non-degraded families.
6. **Strict retained:** task preservation, positive micro, positive macro, and 5/5 non-degraded families.

The final operational `final_14b_fingerprint.jsonl` contains every legacy-retained row. The strict subset is preserved separately.

## Technical and scientific gates

The technical gate requires exactly 60 unique task records, 60 unique complete-coverage micro records, and 600 finite macro records matching the two frozen seeds in every family, with no missing model, adapter, prompt role, or family.

H-E1 requires all of the following:

- technical gate passes;
- at least 23/30 legacy-retained prompts;
- all eight categories appear in the legacy-retained set.

Family pass counts, failure waterfall, component-specific retention, and the strict 5/5 count are reported regardless of H-E1 outcome. They do not replace the preregistered primary gate.

## Claim boundary

If H-E1 is supported, the retained subset is a jointly confirmed Qwen2.5-14B fingerprint under the new frozen seed set. This does not establish transfer to other architectures, revisions, quantization methods beyond NF4, task evaluators, or future perturbation distributions. If H-E1 is refuted, the exact 30-row union remains descriptive prior evidence and the smaller observed retained subset is reported without retrospective repair.
