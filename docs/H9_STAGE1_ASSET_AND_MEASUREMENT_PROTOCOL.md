# H9 Stage 1 migrated-asset and measurement protocol

## Frozen provenance

- GitHub baseline: `experiment/h8-qwen32b-mmd-precalibration` at commit
  `4ea49f907ce4b536575826c45d8d10f377197c75`.
- Stage 1 work proceeds only on `experiment/h9-stage1-sensitivity-propagation`.
- H8 final files are evidence and implementation sources; they are not overwritten.

## Prompt boundary

Stage 1 uses the frozen 12-row prompt list: eight H6 optimized prompts plus four
task-matched original natural controls. It does not regenerate prompts, run HotFlip, or
rerun MCC selection. Five optimized prompts were nondegenerate and three were degenerate
under the historical 32B H8 intact bank; this is only a balanced-selection stratum, not a
claim about Qwen2.5-0.5B-Instruct. All response variability and feature degeneracy must be
remeasured on the 0.5B model.

## LoRA-data boundary

Stage 1 LoRA training uses only
`data/h8_d2_fresh_isolated_attack_training_data_v1.jsonl` (32 rows). The file passed the
repository isolation audit and has no exact construction/confirmation prompt overlap or
MCC prompt/response substring overlap. `data/attack_train.jsonl` and
`data/attack_train_lora.jsonl` are forbidden because they overlap H6 data. The F1 isolated
file remains held out and is not silently pooled into Stage 1 training.

The preprocessing implementation is `src/llm_integrity/paper_finetuning.py`: prompts are
rendered through the target tokenizer/template, prompt-token labels are masked to `-100`,
and answer plus EOS tokens are supervised. The tokenizer revision and the exact resolved
chat-template text must be hashed in the run manifest.

## Measurement definition reused from H8

Reuse the code-level 528-dimensional response measurement:

- 11 surface features;
- 512 BGE semantic features;
- 5 task features;
- `float64`, population standardization (`ddof=0`);
- feature-family scaling by the square root of each family dimension;
- generalized unequal-size unbiased MMD squared with the registered RBF construction.

Do not copy H8's fitted 32B scaler, MAD/null maps, or RBF bandwidth. Fit all numerical
calibration from a newly generated Qwen2.5-0.5B-Instruct intact reference bank before any
attack comparison. Persist the new scaler, active/excluded dimension mask, per-prompt
bandwidths, global degenerate fallback, source-bank hash, and fit code commit.

## Reproducibility requirements before execution

Every formal run must record the source commit, prompt-list hash, LoRA-data hash, model and
tokenizer revisions, semantic-encoder revision, resolved chat template, generation
parameters, package lock, CUDA/driver/GPU inventory, all random seeds, and hashes of every
fitted 0.5B calibration artifact. The migrated asset-freeze configuration deliberately
leaves attack strengths, reference-bank size, and formal seeds pending; they must be fixed
by the final Stage 1 execution protocol rather than inferred from 32B H8.

## Known provenance limitations at migration time

The old server could not be audited because its SSH host key no longer matched the trusted
fingerprint. No host-key check was bypassed. The recovery source was therefore the clean,
full-history local worktree at the specified commit. Separately, the committed H6 archive's
`SHA256SUMS` entry for `FINAL_REPORT.json` does not match that committed file; this is an
inherited archive inconsistency, not a transfer error, and must remain documented rather
than silently rewritten.
