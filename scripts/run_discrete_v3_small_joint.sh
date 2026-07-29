#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/09_multi_variant_repair/02_small_joint"
PROMPTS="$BASE/08_task_guarded_repair/01_selected_seeds/seeds_task_valid_k12.jsonl"
CALIBRATION="$BASE/02_calibration/discrete_calibration_v2.json"
CONFIG="$ROOT/configs/paper_aligned_qwen_7b_joint_discrete_v3_ensemble.yaml"

cd "$ROOT"
mkdir -p "$STAGE"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
  "$STAGE/joint_results_12.jsonl" \
  --output "$STAGE/joint_detailed_summary_12.json"

sha256sum \
  "$CONFIG" \
  "$CALIBRATION" \
  "$PROMPTS" \
  src/llm_integrity/inner_variant_sampler.py \
  src/llm_integrity/discrete_joint_inner_optimizer.py \
  scripts/run_discrete_joint_inner_optimization.py \
  scripts/run_discrete_v3_small_joint.sh \
  results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_train.json \
  > "$STAGE/code_config_registry_sha256.txt"

echo "DISCRETE_V3_SMALL_JOINT_COMPLETE"
