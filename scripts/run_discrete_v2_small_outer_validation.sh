#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
STAGE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728/06_small_comparison"
OUTER="$STAGE/outer_validation"

cd "$ROOT"
mkdir -p "$OUTER"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f "$STAGE/joint_summary_12.json" ]]; then
  echo "Missing completed inner result: $STAGE/joint_summary_12.json" >&2
  exit 2
fi

"$PYTHON" scripts/summarize_small_comparison.py \
  --seeds results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl \
  --old-nf4 results/paper_aligned_qwen_7b/prompt_optimization/optimized_prompts_k60.jsonl \
  --macro-only "$STAGE/macro_only_a0_results_12.jsonl" \
  --joint "$STAGE/joint_results_12.jsonl" \
  --output "$STAGE/four_arm_comparison_summary_12.json" \
  --count 12

"$PYTHON" scripts/prepare_discrete_hard_validation.py \
  --input "$STAGE/joint_results_12.jsonl" \
  --output-dir "$OUTER"

"$PYTHON" scripts/run_discrete_task_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_candidates.jsonl" \
  --output "$OUTER/task_validation_24.jsonl" \
  --max-input-tokens 512 \
  --max-new-tokens 128 \
  --resume

"$PYTHON" scripts/run_paper_micro_scoring.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_unique_prompts.jsonl" \
  --output "$OUTER/micro_scores_unique.jsonl" \
  --probes 4 \
  --resume

"$PYTHON" scripts/run_paper_macro_sampling.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_unique_prompts.jsonl" \
  --manifest results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_validation_balanced_2each.jsonl \
  --split validation \
  --output-dir "$OUTER/macro_validation_2each" \
  --adapter-registry results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_validation.json \
  --max-length 128 \
  --batch-size 1

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/micro_scores_unique.jsonl" \
  --kind micro \
  --output "$OUTER/micro_scores_24.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/macro_validation_2each/macro_records_validation.jsonl" \
  --kind macro \
  --output "$OUTER/macro_records_24x10.jsonl"

if "$PYTHON" scripts/finalize_discrete_hard_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$OUTER/proxy_results.jsonl" \
  --micro-scores "$OUTER/micro_scores_24.jsonl" \
  --macro-records "$OUTER/macro_records_24x10.jsonl" \
  --task-records "$OUTER/task_validation_24.jsonl" \
  --output "$OUTER/strict_validated_prompts_12.jsonl" \
  --report "$OUTER/strict_validation_report_12.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 6; then
  echo "SMALL_OUTER_SCIENTIFIC_GATE_PASSED" \
    > "$OUTER/gate_status.txt"
else
  status=$?
  if [[ "$status" -eq 1 ]]; then
    echo "SMALL_OUTER_SCIENTIFIC_GATE_NOT_MET" \
      > "$OUTER/gate_status.txt"
  else
    exit "$status"
  fi
fi

sha256sum \
  configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  scripts/prepare_discrete_hard_validation.py \
  scripts/run_discrete_task_validation.py \
  scripts/finalize_discrete_hard_validation.py \
  scripts/run_discrete_v2_small_outer_validation.sh \
  > "$OUTER/code_and_config_sha256.txt"

echo "DISCRETE_V2_SMALL_OUTER_VALIDATION_COMPLETE"
