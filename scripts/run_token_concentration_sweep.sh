#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
OUT="$ROOT/results/paper_aligned_qwen_7b/joint_inner_20260727/10_update_control_repair"

cd "$ROOT"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one() {
  local name="$1"
  local topk="$2"
  "$PYTHON" scripts/run_joint_inner_optimization.py \
    --config configs/paper_aligned_qwen_7b_joint_inner.yaml \
    --prompts results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl \
    --manifest results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_train.jsonl \
    --adapter-registry results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_train.json \
    --micro-calibration results/paper_aligned_qwen_7b/joint_inner_20260727/03_calibration/micro_records_12x3.jsonl \
    --macro-calibration results/paper_aligned_qwen_7b/joint_inner_20260727/03_calibration/macro_records_12x5.jsonl \
    --output "$OUT/$name/results.jsonl" \
    --summary "$OUT/$name/summary.json" \
    --max-prompts 1 \
    --max-length 64 \
    --global-clip-norm 5 \
    --learning-rate 0.01 \
    --maximum-relative-update 0.01 \
    --clip-mode tokenwise \
    --top-token-updates "$topk" \
    --projection-interval 1
}

run_one W3_token_top2 2
run_one W4_token_top1 1

echo TOKEN_CONCENTRATION_SWEEP_COMPLETE
