#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/11_multi_direction_search/02_small_joint"
INNER_PID_FILE="$STAGE/run.pid"

test -s "$INNER_PID_FILE"
inner_pid=$(cat "$INNER_PID_FILE")
while kill -0 "$inner_pid" 2>/dev/null; do
  sleep 30
done

test -s "$STAGE/joint_results_12.jsonl"
test -s "$STAGE/joint_summary_12.json"
rows=$(wc -l < "$STAGE/joint_results_12.jsonl")
if [[ "$rows" -ne 12 ]]; then
  echo "INNER_INCOMPLETE rows=$rows"
  exit 2
fi

cd "$ROOT"
bash scripts/run_discrete_v5_small_outer.sh
