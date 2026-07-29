#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/10_disjoint_inner_dev/01_smoke"
PROMPTS="$BASE/08_task_guarded_repair/01_selected_seeds/seeds_task_valid_k12.jsonl"
CALIBRATION="$BASE/02_calibration/discrete_calibration_v2.json"
CONFIG="$ROOT/configs/paper_aligned_qwen_7b_joint_discrete_v4_inner_dev.yaml"

cd "$ROOT"
mkdir -p "$STAGE"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" scripts/run_discrete_joint_inner_optimization.py \
  --config "$CONFIG" \
  --calibration "$CALIBRATION" \
  --prompts "$PROMPTS" \
  --output "$STAGE/joint_results_1.jsonl" \
  --summary "$STAGE/joint_summary_1.json" \
  --max-prompts 1 \
  --max-length 64 \
  --rounds 1 \
  --required-accepted 0

echo "DISCRETE_V4_SMOKE_COMPLETE"
