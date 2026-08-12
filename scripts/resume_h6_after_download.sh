#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE=results/experiment_h6_qwen32b_reconstruction_20260812/00_environment
DOWNLOAD_PID_FILE="$BASE/download.pid"
DOWNLOAD_REPORT="$BASE/model_download.json"

test -f "$DOWNLOAD_PID_FILE"
DOWNLOAD_PID="$(cat "$DOWNLOAD_PID_FILE")"
case "$DOWNLOAD_PID" in
  ''|*[!0-9]*) echo "invalid download PID: $DOWNLOAD_PID" >&2; exit 2 ;;
esac

while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
  printf '[H6_RESUME] download_running pid=%s %s\n' "$DOWNLOAD_PID" "$(date --iso-8601=seconds)"
  sleep 30
done

"$PY" -c 'import json; from pathlib import Path; p=Path("results/experiment_h6_qwen32b_reconstruction_20260812/00_environment/model_download.json"); r=json.loads(p.read_text()); assert r.get("status") == "complete"; assert r.get("revision") == "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"; assert len(r.get("weight_shards", [])) == 17'
printf '[H6_RESUME] download_complete %s\n' "$(date --iso-8601=seconds)"

bash scripts/run_h6_preflight.sh
"$PY" -c 'import json; from pathlib import Path; p=Path("results/experiment_h6_qwen32b_reconstruction_20260812/00_environment/engineering_smoke.json"); r=json.loads(p.read_text()); assert r.get("passed") is True and r.get("status") == "complete_go"'
printf '[H6_RESUME] engineering_gate_go %s\n' "$(date --iso-8601=seconds)"

bash scripts/run_experiment_h6_qwen32b.sh
printf '[H6_RESUME] pipeline_terminal %s\n' "$(date --iso-8601=seconds)"
