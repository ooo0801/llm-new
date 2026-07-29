#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
SMALL_REPORT="$BASE/12_fixed_hybrid_calibration/04_v6_v4_generator_small/strict_validation_report_12.json"
STAGE="$BASE/13_v6_formal_60"
GEN="$STAGE/01_candidate_generation"
PORTFOLIO="$STAGE/02_validation_dev_portfolio"
PROMPTS="$ROOT/results/paper_aligned_qwen_7b/prompt_optimization/seeds_stratified_k60.jsonl"
INNER_CALIBRATION="$BASE/02_calibration/discrete_calibration_v2.json"
FIXED_HYBRID_CALIBRATION="$BASE/12_fixed_hybrid_calibration/01_dev_calibration/fixed_hybrid_calibration_v1.json"
CONFIG="$ROOT/configs/paper_aligned_qwen_7b_joint_discrete_v4_inner_dev.yaml"

cd "$ROOT"
mkdir -p "$GEN" "$PORTFOLIO"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" - <<PY
import json
from pathlib import Path
p = json.loads(Path("$SMALL_REPORT").read_text())
assert p["passed"], "V6 V4-generator 12-prompt development gate did not pass"
assert sum(1 for line in Path("$PROMPTS").read_text().splitlines() if line.strip()) == 60
fixed = json.loads(Path("$FIXED_HYBRID_CALIBRATION").read_text())
assert fixed["version"] == 1 and fixed["sample_count"] >= 2
PY

"$PYTHON" scripts/run_discrete_joint_inner_optimization.py \
  --config "$CONFIG" \
  --calibration "$INNER_CALIBRATION" \
  --prompts "$PROMPTS" \
  --output "$GEN/joint_results_60.jsonl" \
  --summary "$GEN/joint_summary_60.json" \
  --max-prompts 60 \
  --max-length 128 \
  --rounds 3 \
  --required-accepted 0 \
  --resume

"$PYTHON" scripts/summarize_discrete_results.py \
  "$GEN/joint_results_60.jsonl" \
  --output "$GEN/joint_detailed_summary_60.json"

"$PYTHON" scripts/build_discrete_candidate_portfolio.py \
  --input "$GEN/joint_results_60.jsonl" \
  --output "$PORTFOLIO/portfolio_candidates.jsonl" \
  --mapping "$PORTFOLIO/portfolio_mapping.jsonl" \
  --report "$PORTFOLIO/portfolio_build_report.json" \
  --maximum-candidates-per-prompt 2 \
  --minimum-nondegraded-families 3

sha256sum \
  "$CONFIG" \
  "$INNER_CALIBRATION" \
  "$FIXED_HYBRID_CALIBRATION" \
  "$PROMPTS" \
  src/llm_integrity/discrete_joint_inner_optimizer.py \
  scripts/run_discrete_joint_inner_optimization.py \
  scripts/build_discrete_candidate_portfolio.py \
  scripts/run_discrete_v6_formal_60_generate.sh \
  > "$GEN/code_config_data_sha256.txt"

touch "$GEN/GENERATION_COMPLETE"
echo "DISCRETE_V6_FORMAL_60_GENERATION_COMPLETE"
