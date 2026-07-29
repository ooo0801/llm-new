#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/11_multi_direction_search/01_knowledge_smoke"
OUTER="$STAGE/outer_validation"

cd "$ROOT"
mkdir -p "$OUTER"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" scripts/prepare_discrete_hard_validation.py \
  --input "$STAGE/joint_results_1.jsonl" \
  --output-dir "$OUTER"

"$PYTHON" scripts/run_discrete_task_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_candidates.jsonl" \
  --output "$OUTER/task_validation_2.jsonl" \
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
  --output "$OUTER/micro_scores_2.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/macro_validation_2each/macro_records_validation.jsonl" \
  --kind macro \
  --output "$OUTER/macro_records_2x10.jsonl"

if "$PYTHON" scripts/finalize_discrete_hard_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$OUTER/proxy_results.jsonl" \
  --micro-scores "$OUTER/micro_scores_2.jsonl" \
  --macro-records "$OUTER/macro_records_2x10.jsonl" \
  --task-records "$OUTER/task_validation_2.jsonl" \
  --output "$OUTER/strict_validated_prompt_1.jsonl" \
  --report "$OUTER/strict_validation_report_1.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 1; then
  echo "DISCRETE_V5_KNOWLEDGE_OUTER_GATE_PASSED" \
    > "$OUTER/gate_status.txt"
else
  status=$?
  if [[ "$status" -eq 1 ]]; then
    echo "DISCRETE_V5_KNOWLEDGE_OUTER_GATE_NOT_MET" \
      > "$OUTER/gate_status.txt"
  else
    exit "$status"
  fi
fi

"$PYTHON" scripts/summarize_strict_validation.py \
  "$OUTER/strict_validated_prompt_1.jsonl" \
  --output "$OUTER/strict_validation_detailed_summary.json"

echo "DISCRETE_V5_KNOWLEDGE_OUTER_COMPLETE"
