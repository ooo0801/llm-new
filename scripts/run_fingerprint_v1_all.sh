#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/root/autodl-tmp/venvs/llm-integrity/bin/python}"
CONFIG="${CONFIG:-configs/fingerprint_v1_qwen_7b.yaml}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/root/autodl-tmp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HUB_CACHE}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
OUTPUT="results/fingerprint_v1_20260730"
MANIFEST="reproducibility/fingerprint_v1_20260730/attack_manifest.jsonl"
ADAPTER_ROOT="$OUTPUT/finetuning_adapters"
ADAPTER_REGISTRY="$ADAPTER_ROOT/adapter_registry.json"

"$PYTHON" scripts/prepare_fingerprint_v1.py --config "$CONFIG"
"$PYTHON" -m pytest -q
"$PYTHON" scripts/extract_fingerprint_v1_activations.py --config "$CONFIG"
"$PYTHON" scripts/build_fingerprint_v1.py --config "$CONFIG"
"$PYTHON" scripts/train_paper_lora_registry.py \
  --config "$CONFIG" \
  --manifest "$MANIFEST" \
  --split fingerprint_v1_endpoint \
  --data data/attack_train_lora.jsonl \
  --output-root "$ADAPTER_ROOT" \
  --registry-output "$ADAPTER_REGISTRY" \
  --variant-id lora_r8_s25_v1

for variant in \
  intact_v1 \
  unstructured_prune30_v1 \
  structured_ffn20_v1 \
  gaussian_noise001_v1 \
  int8_v1 \
  nf4_v1 \
  lora_r8_s25_v1
do
  "$PYTHON" scripts/run_fingerprint_v1_verification.py \
    --config "$CONFIG" \
    --variant-id "$variant" \
    --adapter-registry "$ADAPTER_REGISTRY"
done

"$PYTHON" scripts/summarize_fingerprint_v1.py --config "$CONFIG"
"$PYTHON" scripts/audit_fingerprint_v1_release.py --config "$CONFIG"
