#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/llm-integrity-fingerprint}"
VENV_DIR="${VENV_DIR:-/root/autodl-tmp/venvs/llm-integrity}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[info] project=$PROJECT_DIR"
echo "[info] venv=$VENV_DIR"
echo "[info] hf_home=$HF_HOME"
mkdir -p "$HF_HOME" "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
cd "$PROJECT_DIR"
python -m pip install -e '.[quantization,semantic,lora,api,dev]'
python scripts/build_dataset.py
python -m llm_integrity.cli validate-config configs/debug_qwen_0.5b.yaml
python -m llm_integrity.cli stats-demo
python -m llm_integrity.cli mcc-demo

python - <<'PY'
import json, platform
import torch, transformers
info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(info, ensure_ascii=False, indent=2))
with open("results/environment.json", "w", encoding="utf-8") as handle:
    json.dump(info, handle, ensure_ascii=False, indent=2)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
echo "[done] Run: source $VENV_DIR/bin/activate"
