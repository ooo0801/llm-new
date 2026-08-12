#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2

PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
INNER=configs/experiment_h6_qwen32b_inner.yaml
DEV_CONFIG=configs/experiment_h6_qwen32b_development.yaml
CONFIRM_CONFIG=configs/experiment_h6_qwen32b_confirmation.yaml
MCC_CONFIG=configs/fingerprint_h6_qwen32b_mcc.yaml
INPUT=experiments/prompt-reconstruction-32b-h6/inputs
OUT=results/experiment_h6_qwen32b_reconstruction_20260812
ENV="$OUT/00_environment"
CAL="$OUT/01_calibration"
GEN="$OUT/02_candidate_generation"
DEV="$OUT/03_development"
CONFIRM="$OUT/04_confirmation"
ADAPTERS="$OUT/adapters"

train_lora_variant() {
  local manifest="$1"
  local split="$2"
  local output_root="$3"
  local registry="$4"
  local variant_id="$5"
  "$PY" scripts/train_paper_lora_registry.py \
    --config "$INNER" \
    --manifest "$manifest" \
    --split "$split" \
    --data data/attack_train_lora.jsonl \
    --output-root "$output_root" \
    --registry-output "$registry" \
    --variant-id "$variant_id"
}

construction_pid="$(cat "$ENV/construction_recovery.pid")"
while kill -0 "$construction_pid" 2>/dev/null; do
  sleep 60
done

"$PY" -c 'import json; from pathlib import Path; p=Path("results/experiment_h6_qwen32b_reconstruction_20260812/02_candidate_generation/joint_summary_60.json"); r=json.loads(p.read_text()); assert r.get("requested_prompts") == 60; assert r.get("results") == 60; assert r.get("technically_complete") == 60; assert r.get("technical_passed") is True'

# Continue at the first post-optimizer command in the frozen formal runner.
# Running in the current shell preserves its scientific No-Go exit behavior.
source <(
  awk '
    /"\$PY" scripts\/summarize_discrete_results.py/ { emit = 1 }
    emit { print }
  ' scripts/run_experiment_h6_qwen32b.sh
)
