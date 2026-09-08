#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/llm/stage2_pilot_execution_20260905
PYTHON=/root/autodl-tmp/envs/h9-stage1/bin/python
OUT="$ROOT/results/stage2_three_proxy_ceiling_v3_20260906"
LOG="$ROOT/logs/stage2_three_proxy_v3.log"
ARCHIVE=/root/autodl-tmp/llm/stage2_three_proxy_checkpoint_v3_20260906.tar.gz

mkdir -p "$ROOT/logs" "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT/scripts"
export HF_HOME=/root/autodl-tmp/huggingface
export STAGE2_THREE_PROXY_OUT=results/stage2_three_proxy_ceiling_v3_20260906
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

checkpoint() {
  local phase="$1"
  "$PYTHON" -c "import json,time; from pathlib import Path; p=Path('$OUT/SUPERVISOR.json'); p.write_text(json.dumps({'phase':'$phase','utc_unix':time.time()},ensure_ascii=False,indent=2),encoding='utf-8')"
  tar -czf "$ARCHIVE.tmp" \
    scripts/stage2_three_proxy_search.py \
    scripts/stage2_three_proxy_search_v2.py \
    scripts/stage2_three_proxy_search_v3.py \
    scripts/stage2_three_proxy_behavior.py \
    scripts/run_stage2_three_proxy_v3.sh \
    src/llm_integrity/macro_proxy.py \
    src/llm_integrity/joint_inner_optimizer.py \
    src/llm_integrity/discrete_joint_inner_optimizer.py \
    tests/test_macro_proxy.py \
    results/stage2_three_proxy_ceiling_v3_20260906 \
    logs/stage2_three_proxy_v3.log 2>/dev/null || true
  mv -f "$ARCHIVE.tmp" "$ARCHIVE"
}

finish() {
  code=$?
  trap - EXIT
  if [[ $code -eq 0 ]]; then checkpoint COMPLETE; else checkpoint FAILED; fi
  sync
  exit "$code"
}
trap finish EXIT
exec > >(tee -a "$LOG") 2>&1

checkpoint STARTING
"$PYTHON" scripts/stage2_three_proxy_search_v3.py freeze
"$PYTHON" scripts/stage2_three_proxy_search_v3.py reuse
checkpoint REUSE_COMPLETE
"$PYTHON" scripts/stage2_three_proxy_search_v3.py pilot-search
checkpoint PILOT_SEARCH_COMPLETE
"$PYTHON" scripts/stage2_three_proxy_behavior.py pilot-freeze
"$PYTHON" scripts/stage2_three_proxy_behavior.py pilot-generate
"$PYTHON" scripts/stage2_three_proxy_behavior.py pilot-analyze
checkpoint PILOT_BEHAVIOR_COMPLETE

can_continue=$($PYTHON -c "import json; print(str(json.load(open('$OUT/PROMOTION.json'))['can_continue']).lower())")
if [[ "$can_continue" != "true" ]]; then
  checkpoint STOPPED_BY_FROZEN_PILOT_GATE
  exit 0
fi

"$PYTHON" scripts/stage2_three_proxy_search_v3.py formal-search
checkpoint FORMAL_SEARCH_COMPLETE
"$PYTHON" scripts/stage2_three_proxy_behavior.py formal-freeze
"$PYTHON" scripts/stage2_three_proxy_behavior.py formal-run
"$PYTHON" scripts/stage2_three_proxy_behavior.py formal-report
checkpoint ALL_FIVE_STAGES_COMPLETE
