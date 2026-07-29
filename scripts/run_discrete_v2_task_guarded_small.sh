#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/08_task_guarded_repair/03_small_comparison"
PROMPTS="$BASE/08_task_guarded_repair/01_selected_seeds/seeds_task_valid_k12.jsonl"
CALIBRATION="$BASE/02_calibration/discrete_calibration_v2.json"
CONFIG="$ROOT/configs/paper_aligned_qwen_7b_joint_discrete_v2.yaml"

cd "$ROOT"
mkdir -p "$STAGE"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" scripts/run_discrete_joint_inner_optimization.py \
  --config "$CONFIG" \
  --calibration "$CALIBRATION" \
  --prompts "$PROMPTS" \
  --output "$STAGE/macro_only_a0_results_12.jsonl" \
  --summary "$STAGE/macro_only_a0_summary_12.json" \
  --max-prompts 12 \
  --max-length 64 \
  --rounds 3 \
  --micro-weight 0 \
  --required-accepted 0 \
  --resume

"$PYTHON" scripts/run_discrete_joint_inner_optimization.py \
  --config "$CONFIG" \
  --calibration "$CALIBRATION" \
  --prompts "$PROMPTS" \
  --output "$STAGE/joint_results_12.jsonl" \
  --summary "$STAGE/joint_summary_12.json" \
  --max-prompts 12 \
  --max-length 64 \
  --rounds 3 \
  --required-accepted 0 \
  --resume

"$PYTHON" scripts/summarize_discrete_results.py \
  "$STAGE/macro_only_a0_results_12.jsonl" \
  --output "$STAGE/macro_only_a0_detailed_summary_12.json"

"$PYTHON" scripts/summarize_discrete_results.py \
  "$STAGE/joint_results_12.jsonl" \
  --output "$STAGE/joint_detailed_summary_12.json"

sha256sum \
  "$CONFIG" \
  "$CALIBRATION" \
  "$PROMPTS" \
  src/llm_integrity/discrete_joint_inner_optimizer.py \
  scripts/run_discrete_joint_inner_optimization.py \
  scripts/run_discrete_v2_task_guarded_small.sh \
  > "$STAGE/code_and_config_sha256.txt"

echo "TASK_GUARDED_SMALL_COMPARISON_COMPLETE"
