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

printf '[G1_STAGE] prepare_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_fingerprint_14b_mcc.py --config "$CONFIG"
printf '[G1_STAGE] prepare_complete %s\n' "$(date --iso-8601=seconds)"

printf '[G1_STAGE] activations_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/extract_fingerprint_v2_activations.py --config "$CONFIG"
printf '[G1_STAGE] activations_complete %s\n' "$(date --iso-8601=seconds)"

printf '[G1_STAGE] mcc_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/build_fingerprint_v2_global_selection.py --config "$CONFIG"
printf '[G1_STAGE] mcc_complete %s\n' "$(date --iso-8601=seconds)"
