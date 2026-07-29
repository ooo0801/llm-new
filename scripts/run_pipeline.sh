#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/main_qwen_1.5b.yaml}"
python scripts/score_prompts.py --config "$CONFIG"
python scripts/extract_activations.py --config "$CONFIG" --pool-size 100
python scripts/build_fingerprint.py --config "$CONFIG"
python scripts/run_ablation.py --config "$CONFIG"

echo "Experiment manifest created. Review it before running expensive jobs:"
echo "python scripts/run_ablation.py --config $CONFIG --execute --repetitions 5"
