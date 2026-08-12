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
"$PY" scripts/record_h6_environment.py
"$PY" scripts/run_h6_engineering_smoke.py
