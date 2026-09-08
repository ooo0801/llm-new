#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
source /root/autodl-tmp/envs/h9-stage1/bin/activate
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/run_h9_stage1_attack_stat_calibration.py all \
  --config configs/h9_stage1_qwen05b_attack_stat_calibration.yaml

