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
CONFIG=configs/fingerprint_g_qwen14b_mcc.yaml
OUT=results/fingerprint_g_qwen14b_mcc_20260807
MANIFEST=reproducibility/fingerprint_g_qwen14b_mcc_20260807/attack_manifest.jsonl
ADAPTER_ROOT="$OUT/finetuning_adapters"
ADAPTER_REGISTRY="$ADAPTER_ROOT/adapter_registry.json"

printf '[H1_STAGE] reference_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/build_fingerprint_v2.py --config "$CONFIG"
printf '[H1_STAGE] reference_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H1_STAGE] lora_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$CONFIG" --manifest "$MANIFEST" --split fingerprint_14b_h1_endpoint --data data/attack_train_lora.jsonl --output-root "$ADAPTER_ROOT" --registry-output "$ADAPTER_REGISTRY"
printf '[H1_STAGE] lora_complete %s\n' "$(date --iso-8601=seconds)"

while IFS= read -r variant_id; do
  printf '[H1_STAGE] variant_start %s %s\n' "$variant_id" "$(date --iso-8601=seconds)"
  "$PY" scripts/run_fingerprint_v2_verification.py --config "$CONFIG" --variant-id "$variant_id" --adapter-registry "$ADAPTER_REGISTRY"
  printf '[H1_STAGE] variant_complete %s %s\n' "$variant_id" "$(date --iso-8601=seconds)"
done < <("$PY" -c 'import json; from pathlib import Path; print("\n".join(json.loads(x)["variant_id"] for x in Path("reproducibility/fingerprint_g_qwen14b_mcc_20260807/attack_manifest.jsonl").read_text().splitlines() if x.strip()))')

set +e
"$PY" scripts/summarize_fingerprint_14b_mmd.py --config "$CONFIG" --required-modified 9
STATUS=$?
set -e
printf '[H1_STAGE] analysis_complete status=%s %s\n' "$STATUS" "$(date --iso-8601=seconds)"
exit "$STATUS"
