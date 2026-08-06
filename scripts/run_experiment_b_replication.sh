#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export CUDA_VISIBLE_DEVICES=0,1,2

OUT=results/experiment_b_qwen14b_20260806
INPUT=experiments/prompt-transfer-14b-b1/inputs
mkdir -p "$OUT/macro" "$OUT/adapters"

printf '[B_STAGE] lora_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py \
  --config configs/experiment_b_qwen14b.yaml \
  --manifest "$INPUT/attack_manifest_validation.jsonl" \
  --split validation \
  --data data/attack_train_lora.jsonl \
  --output-root "$OUT/adapters" \
  --registry-output "$OUT/adapters/registry.json"
printf '[B_STAGE] lora_complete %s\n' "$(date --iso-8601=seconds)"

printf '[B_STAGE] task_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_task_validation.py \
  --config configs/experiment_b_qwen14b.yaml \
  --prompts "$INPUT/survivor_prompts.jsonl" \
  --output "$OUT/task_validation.jsonl" \
  --resume
printf '[B_STAGE] task_complete %s\n' "$(date --iso-8601=seconds)"

printf '[B_STAGE] micro_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_paper_micro_scoring.py \
  --config configs/experiment_b_qwen14b.yaml \
  --prompts "$INPUT/survivor_prompts.jsonl" \
  --output "$OUT/micro_scores.jsonl" \
  --probes 4 \
  --resume
printf '[B_STAGE] micro_complete %s\n' "$(date --iso-8601=seconds)"

printf '[B_STAGE] macro_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_paper_macro_sampling.py \
  --config configs/experiment_b_qwen14b.yaml \
  --prompts "$INPUT/survivor_prompts.jsonl" \
  --manifest "$INPUT/attack_manifest_validation.jsonl" \
  --split validation \
  --output-dir "$OUT/macro" \
  --adapter-registry "$OUT/adapters/registry.json" \
  --max-length 128 \
  --primary-metric macro_l2_raw \
  --seed 1685839144
printf '[B_STAGE] macro_complete %s\n' "$(date --iso-8601=seconds)"

"$PY" scripts/analyze_experiment_b_replication.py
printf '[B_STAGE] analysis_complete %s\n' "$(date --iso-8601=seconds)"
