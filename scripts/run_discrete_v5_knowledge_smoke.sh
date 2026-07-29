#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/11_multi_direction_search/01_knowledge_smoke"
PROMPTS="$BASE/10_disjoint_inner_dev/01_smoke/knowledge_prompt.jsonl"
CALIBRATION="$BASE/02_calibration/discrete_calibration_v2.json"
CONFIG="$ROOT/configs/paper_aligned_qwen_7b_joint_discrete_v5_multi_direction.yaml"

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
  --rounds 3 \
  --required-accepted 0

"$PYTHON" scripts/summarize_discrete_results.py \
  "$STAGE/joint_results_1.jsonl" \
  --output "$STAGE/joint_detailed_summary_1.json"

sha256sum \
  "$CONFIG" \
  "$CALIBRATION" \
  "$PROMPTS" \
  src/llm_integrity/inner_variant_sampler.py \
  src/llm_integrity/discrete_joint_inner_optimizer.py \
  scripts/run_discrete_joint_inner_optimization.py \
  > "$STAGE/code_config_prompt_sha256.txt"

echo "DISCRETE_V5_KNOWLEDGE_SMOKE_COMPLETE"
