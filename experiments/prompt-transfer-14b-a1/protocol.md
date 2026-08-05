# Experiment A Protocol: Frozen 7B Prompt Transfer to Qwen2.5-14B

Status: **FROZEN BEFORE 14B MODEL DOWNLOAD OR ENDPOINT SCORING**

Protocol clarification A1-P1 (2026-08-05): after the fixed-revision download
started but before any 14B endpoint was loaded or scored, a direct code audit
corrected two descriptive statements below. The V6 task entry point uses
deterministic greedy generation without a system prompt, and the formal
blockwise Hutchinson entry point covers all parameter groups. No prompt,
threshold, family configuration, seed, or endpoint was changed.

## 1. Research question and prediction

Do the 16 prompts that strictly passed the Qwen2.5-7B V6 frozen test remain task-valid and sensitive when their raw text is transferred without editing or re-optimization to Qwen2.5-14B-Instruct?

Confirmatory prediction H-A1: at least 12 of 16 prompts satisfy the legacy transfer-retention gate. The retained count is the primary endpoint. A strict 5/5 retained count is secondary and descriptive.

## 2. Immutable provenance

- Git parent: `v2-global-calibrated-coverage` (`ae27ec18f5409f337439425b8f0f647a05f9e6a6`).
- Source evidence: `reproducibility/v6_20260729/final_test_validated_prompts.jsonl`.
- Source model: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`.
- Target model: `Qwen/Qwen2.5-14B-Instruct@cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8`.
- Target storage: eight BF16 safetensors shards totaling 29,540,134,000 bytes in `/root/autodl-tmp/huggingface`.
- Target dtype: BF16. No base-model quantization is allowed except when materializing the frozen NF4 attack family.
- Existing `results/fingerprint_v1_20260730`, `results/fingerprint_v2_global_20260731`, and all V6/V1/V2 reproducibility evidence are immutable.

The prompt source SHA256, accepted-set SHA256, model file manifest, tokenizer file hashes, chat-template SHA256, environment versions, and actual device map must be recorded before scoring.

## 3. Frozen prompt-set construction

1. Read all 22 rows from the source evidence.
2. Select exactly rows where `optimization.accepted == true`; expected count is exactly 16.
3. For every selected row:
   - paired baseline text = `optimization.initial_prompt`;
   - transferred sensitive text = `optimization.optimized_prompt`;
   - prompt ID, category, evaluator, expected answer, and expected substrings are copied unchanged.
4. Sort by source-file order. Do not select, edit, normalize, repair, or re-rank text using 14B outcomes.
5. Transfer raw Unicode text, never 7B token IDs. Render each text through the target tokenizer and target chat template.

Tokenizer differences are part of the transfer condition. Record token counts and truncation flags. A pair is technically invalid only if tokenizer rendering fails, either text is truncated at the frozen 512-token task limit or 128-token sensitivity limit, or decoded text/hash provenance cannot be reproduced. Token-count differences alone are not failure.

## 4. Target loading and three-GPU policy

- Use GPUs 0, 1, and 2 only.
- Use deterministic Accelerate/Transformers layer sharding with per-GPU `max_memory` of 30 GiB, leaving about 2 GiB safety margin on each GPU.
- Do not use CPU or disk offload for confirmatory scoring.
- Record the complete `hf_device_map`, per-GPU allocated/reserved peaks, wall time, and any cross-device error.
- Only one base or attack endpoint model may be resident at a time. Release it and empty CUDA caches before the next endpoint.

## 5. Frozen task evaluation

Use each row's original evaluator (`exact`, `contains`, `refusal`, `semantic`, `length_and_contains`, or `python_syntax`) with the existing V6 task-validation entry point. Its frozen generation contract is deterministic greedy generation, no system prompt, maximum 512 input tokens, and maximum 128 generated tokens.

Task preservation for a pair requires both the initial and optimized text to pass. All generations and evaluator diagnostics are retained.

## 6. Frozen micro sensitivity

- Existing formal blockwise Hutchinson Jacobian proxy.
- Full parameter coverage through the existing deterministic `build_parameter_groups` implementation. Every group is processed sequentially with only the current group requiring gradients.
- Four Rademacher probes, seed 42.
- Maximum sensitivity length: 128 tokens.
- Output position: last non-padding token; representation: next-token logits cast to float32.
- Record every group name, component, layer index, parameter count, group estimate, total estimate, standard error, and the `complete_parameter_coverage` audit.

For each pair, `micro_gain = optimized_micro - initial_micro` and relative gain uses `max(abs(initial_micro), 1e-12)`. Micro improvement requires finite values and `micro_gain > 0`.

## 7. Frozen macro sensitivity and five-family manifest

Use next-token-logit squared L2 (`macro_l2_raw`) and the exact V6 test configurations below. Each family mean is computed over seeds 3407 and 777.

1. Unstructured pruning: global magnitude, ratio 0.30, attention scope.
2. Structured pruning: random FFN channels, ratio 0.05, random layer subset, mask implementation.
3. Quantization: full-model NF4, BF16 compute, double quantization.
4. Gaussian noise: FFN scope, parameter-tensor-std scaling, std ratio 0.0005.
5. Finetuning: LoRA rank 16, alpha 32, dropout 0.05, learning rate 1e-5, 50 steps, attention+FFN scope, isolated attack training data.

LoRA training also freezes maximum length 256, batch size 1, gradient accumulation 8, and the two manifest seeds. The stage gate first trains one step for seed 3407 only to validate memory and gradient flow; if that passes, the confirmatory adapters use the frozen 50 steps for both seeds. The one-step adapter is engineering evidence only and is never used in endpoint scoring.

Every endpoint must pass a no-op audit: realized method matches the manifest, at least one intended parameter or adapter value changes, logits differ finitely from the base, and device placement is valid. An infrastructure no-op stops the stage; it is not counted as a scientific family failure.

For each pair and family, compare the optimized family mean with the initial family mean. The family is non-degraded when:

`optimized >= initial - 0.01 * max(abs(initial), 1e-12)`.

Macro aggregate is the equal-weight mean of the five family means. Macro improvement requires finite values and optimized aggregate strictly greater than initial aggregate.

## 8. Frozen retention gates

Legacy retained requires all of:

1. tokenizer and truncation technical checks pass;
2. both initial and optimized task evaluations pass;
3. finite micro values with positive micro gain;
4. finite equal-weight macro values with positive macro gain;
5. at least 3 of 5 families pass the 1% non-degeneration rule.

Strict retained requires all legacy conditions and 5 of 5 family passes.

Experiment-level feasibility supports H-A1 when at least 12 of 16 are legacy retained. Report task-pass count, positive micro count, positive macro count, every family pass count, legacy retained count, strict retained count, and exact prompt IDs for all failures. Do not change gates after results.

The existing V6 fixed hybrid calibration is calculated and reported as a continuity diagnostic, but it is not allowed to replace the explicit positive-micro and positive-macro conditions above.

## 9. Staged Go/No-Go sequence

1. Verify deletion target, then delete only the local 7B Hugging Face cache authorized by the user. Verify all V6/V1/V2 evidence remains.
2. Download the exact 14B revision; validate all eight shard sizes and snapshot completeness.
3. Offline tokenizer/config load and accepted-set/tokenization audit.
4. Complete BF16 three-GPU load and four-token generation.
5. One-prompt task validation using the first accepted source row.
6. The same prompt through micro backward and base macro forward.
7. The same prompt through one variant per family, including one-step LoRA, followed by the complete two-seed five-family gate if engineering checks pass.
8. Audit memory and device placement. Run all 16 only if stages 1 through 7 are technically valid.
9. Generate machine-readable outputs, checksums, analysis, and a human report.

Stop a stage on OOM, NaN/Inf, incomplete model files, attack no-op, device mismatch, missing adapter, corrupted output, or insufficient disk. A valid task or sensitivity gate failure is a scientific result and must be recorded without tuning.

## 10. Outputs

- `experiments/prompt-transfer-14b-a1/results/`
- `results/experiment_a_qwen14b_20260805/`
- `reproducibility/experiment_a_qwen14b_20260805/`
- `to_human/experiment_a_qwen14b_20260805/`

Large weights and raw runtime logs remain untracked. Git records protocol, configs, code, compact manifests, checksums, metrics, and reports. Existing V1/V2/V6 paths must never be overwritten.
