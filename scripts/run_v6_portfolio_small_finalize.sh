#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
BASE_OUTER="$BASE/11_multi_direction_search/02_small_joint/outer_validation"
REPLACEMENT="$BASE/12_fixed_hybrid_calibration/02_knowledge_alternative/outer_validation"
STAGE="$BASE/12_fixed_hybrid_calibration/03_v6_portfolio_small"
CALIBRATION="$BASE/12_fixed_hybrid_calibration/01_dev_calibration/fixed_hybrid_calibration_v1.json"

cd "$ROOT"
mkdir -p "$STAGE"

"$PYTHON" scripts/merge_validation_candidate_replacement.py \
  --base-dir "$BASE_OUTER" \
  --replacement-dir "$REPLACEMENT" \
  --base-id knowledge_9edb61565cee \
  --replacement-id knowledge_9edb61565cee__v4_r1_c5 \
  --output-dir "$STAGE"

set +e
"$PYTHON" scripts/finalize_discrete_hard_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$STAGE/proxy_results.jsonl" \
  --micro-scores "$STAGE/micro_scores_24.jsonl" \
  --macro-records "$STAGE/macro_records_24x10.jsonl" \
  --task-records "$STAGE/task_validation_24.jsonl" \
  --hybrid-calibration "$CALIBRATION" \
  --output "$STAGE/strict_validated_prompts_12.jsonl" \
  --report "$STAGE/strict_validation_report_12.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 6
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 1 ]]; then
  exit "$status"
fi

"$PYTHON" scripts/summarize_strict_validation.py \
  "$STAGE/strict_validated_prompts_12.jsonl" \
  --output "$STAGE/strict_validation_detailed_summary.json"

sha256sum \
  "$CALIBRATION" \
  "$STAGE/proxy_results.jsonl" \
  "$STAGE/micro_scores_24.jsonl" \
  "$STAGE/macro_records_24x10.jsonl" \
  "$STAGE/task_validation_24.jsonl" \
  scripts/merge_validation_candidate_replacement.py \
  scripts/finalize_discrete_hard_validation.py \
  > "$STAGE/artifact_code_sha256.txt"

echo "V6_PORTFOLIO_SMALL_COMPLETE status=$status"
