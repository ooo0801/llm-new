#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/llm
unset HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2
PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
CONFIG=configs/fingerprint_h5_qwen14b_mismatch.yaml
OUT=results/fingerprint_h5_qwen14b_mismatch_20260808
MANIFEST=reproducibility/fingerprint_h5_qwen14b_mismatch_20260808/attack_manifest.jsonl
ADAPTER_ROOT="$OUT/finetuning_adapters"
ADAPTER_REGISTRY="$ADAPTER_ROOT/adapter_registry.json"
"$PY" scripts/prepare_fingerprint_14b_h5.py --config "$CONFIG"
"$PY" scripts/build_fingerprint_14b_h5_reference.py --config "$CONFIG"
"$PY" scripts/train_paper_lora_registry.py --config "$CONFIG" --manifest "$MANIFEST" --split fingerprint_14b_h5_endpoint --data data/attack_train_lora.jsonl --output-root "$ADAPTER_ROOT" --registry-output "$ADAPTER_REGISTRY"
while IFS= read -r variant_id; do
  "$PY" scripts/run_fingerprint_v2_verification.py --config "$CONFIG" --variant-id "$variant_id" --adapter-registry "$ADAPTER_REGISTRY"
done < <("$PY" -c 'import json; from pathlib import Path; print("\n".join(json.loads(x)["variant_id"] for x in Path("reproducibility/fingerprint_h5_qwen14b_mismatch_20260808/attack_manifest.jsonl").read_text().splitlines() if x.strip()))')
"$PY" scripts/summarize_fingerprint_14b_mmd.py --config "$CONFIG" --required-modified 9
