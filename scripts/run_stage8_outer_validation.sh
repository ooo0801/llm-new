#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
STAGE="$ROOT/results/paper_aligned_qwen_7b/joint_inner_20260727/08_formal_60"
FORMAL_SUMMARY="$STAGE/joint_summary_60.json"
FORMAL_PID_FILE="$STAGE/run.pid"

cd "$ROOT"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

while [[ ! -f "$FORMAL_SUMMARY" ]]; do
  if [[ -f "$FORMAL_PID_FILE" ]]; then
    formal_pid="$(<"$FORMAL_PID_FILE")"
    if ! kill -0 "$formal_pid" 2>/dev/null; then
      echo "Formal optimizer exited without producing $FORMAL_SUMMARY" >&2
      exit 1
    fi
  fi
  sleep 30
done

# Recompute only the two stage-7 records affected by the pre-fix tail
# truncation bug.  The other ten records are retained and the runner resumes
# by prompt id, so this is an exact targeted repair rather than a new arm.
"$PYTHON" scripts/prepare_stage7_tail_safe_resume.py \
  --input results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/macro_only_a0_results_12.jsonl \
  --output results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/macro_only_a0_results_12_tail_safe.jsonl

"$PYTHON" scripts/run_joint_inner_optimization.py \
  --config configs/paper_aligned_qwen_7b_joint_inner.yaml \
  --prompts results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl \
  --manifest results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_train.jsonl \
  --adapter-registry results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_train.json \
  --micro-calibration results/paper_aligned_qwen_7b/joint_inner_20260727/03_calibration/micro_records_12x3.jsonl \
  --macro-calibration results/paper_aligned_qwen_7b/joint_inner_20260727/03_calibration/macro_records_12x5.jsonl \
  --output results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/macro_only_a0_results_12_tail_safe.jsonl \
  --summary results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/macro_only_a0_summary_12_tail_safe.json \
  --max-prompts 12 \
  --max-length 64 \
  --micro-weight 0 \
  --resume

"$PYTHON" scripts/prepare_stage7_tail_safe_resume.py \
  --input results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/joint_results_12.jsonl \
  --output results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/joint_results_12_tail_safe.jsonl

"$PYTHON" scripts/run_joint_inner_optimization.py \
  --config configs/paper_aligned_qwen_7b_joint_inner.yaml \
  --prompts results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl \
  --manifest results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_train.jsonl \
  --adapter-registry results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_train.json \
  --micro-calibration results/paper_aligned_qwen_7b/joint_inner_20260727/03_calibration/micro_records_12x3.jsonl \
  --macro-calibration results/paper_aligned_qwen_7b/joint_inner_20260727/03_calibration/macro_records_12x5.jsonl \
  --output results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/joint_results_12_tail_safe.jsonl \
  --summary results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/joint_summary_12_tail_safe.json \
  --max-prompts 12 \
  --max-length 64 \
  --resume

"$PYTHON" scripts/summarize_small_comparison.py \
  --seeds results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl \
  --old-nf4 results/paper_aligned_qwen_7b/prompt_optimization/optimized_prompts_k60.jsonl \
  --macro-only results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/macro_only_a0_results_12_tail_safe.jsonl \
  --joint results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/joint_results_12_tail_safe.jsonl \
  --output results/paper_aligned_qwen_7b/joint_inner_20260727/07_small_comparison/comparison_summary_12_tail_safe.json

"$PYTHON" scripts/prepare_joint_hard_validation.py \
  --input "$STAGE/joint_results_60.jsonl" \
  --output-dir "$STAGE/outer_validation"

"$PYTHON" scripts/run_paper_micro_scoring.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$STAGE/outer_validation/validation_unique_prompts.jsonl" \
  --output "$STAGE/outer_validation/micro_scores_unique.jsonl" \
  --probes 4 \
  --resume

"$PYTHON" scripts/run_paper_macro_sampling.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$STAGE/outer_validation/validation_unique_prompts.jsonl" \
  --manifest results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_validation_balanced_2each.jsonl \
  --split validation \
  --output-dir "$STAGE/outer_validation/macro_validation_2each" \
  --adapter-registry results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_validation.json \
  --max-length 128 \
  --batch-size 1

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$STAGE/outer_validation/validation_expansion_map.jsonl" \
  --input "$STAGE/outer_validation/micro_scores_unique.jsonl" \
  --kind micro \
  --output "$STAGE/outer_validation/micro_scores_120.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$STAGE/outer_validation/validation_expansion_map.jsonl" \
  --input "$STAGE/outer_validation/macro_validation_2each/macro_records_validation.jsonl" \
  --kind macro \
  --output "$STAGE/outer_validation/macro_records_120x10.jsonl"

"$PYTHON" scripts/finalize_hybrid_prompt_optimization.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$STAGE/outer_validation/proxy_results_60.jsonl" \
  --micro-scores "$STAGE/outer_validation/micro_scores_120.jsonl" \
  --macro-records "$STAGE/outer_validation/macro_records_120x10.jsonl" \
  --output "$STAGE/outer_validation/hybrid_validated_prompts_60.jsonl"

"$PYTHON" scripts/finalize_eight_stage_report.py \
  --root "$ROOT" \
  --output "$ROOT/results/paper_aligned_qwen_7b/joint_inner_20260727/EIGHT_STAGE_COMPLETION_REPORT.json"

echo "STAGE8_OUTER_VALIDATION_COMPLETE"
