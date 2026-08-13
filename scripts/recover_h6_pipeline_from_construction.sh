#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2

PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
INNER=configs/experiment_h6_qwen32b_inner.yaml
INPUT=experiments/prompt-reconstruction-32b-h6/inputs
OUT=results/experiment_h6_qwen32b_reconstruction_20260812
ENV="$OUT/00_environment"
CAL="$OUT/01_calibration"
GEN="$OUT/02_candidate_generation"
LOG="$ENV/pipeline_tmux_recovery.log"

exec >> "$LOG" 2>&1

on_exit() {
  local rc=$?
  trap - EXIT
  printf '[PROCESS_EXIT] h6_pipeline rc=%s time=%s\n' \
    "$rc" "$(date --iso-8601=seconds)"
  exit "$rc"
}
trap on_exit EXIT

printf '%s\n' "$$" > "$ENV/pipeline_tmux_recovery.pid"
printf '%s\n' "$$" > "$ENV/construction_recovery.pid"
printf '[RECOVERY] tmux_pipeline_start %s pid=%s\n' \
  "$(date --iso-8601=seconds)" "$$"

"$PY" - <<'PY'
import json
from pathlib import Path

path = Path(
    "results/experiment_h6_qwen32b_reconstruction_20260812/"
    "02_candidate_generation/joint_results_60.jsonl"
)
prompt_path = Path(
    "experiments/prompt-reconstruction-32b-h6/inputs/"
    "construction_prompts_k60.jsonl"
)
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
prompts = [
    json.loads(line)
    for line in prompt_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
prompt_ids = [str(row["prompt_id"]) for row in rows]
expected_ids = [
    str(row.get("id", row.get("prompt_id")))
    for row in prompts
]
assert len(prompts) == 60, len(prompts)
assert 1 <= len(rows) <= len(prompts), len(rows)
assert len(prompt_ids) == len(set(prompt_ids)), "duplicate prompt_id"
assert prompt_ids == expected_ids[: len(rows)], (
    "checkpoint prompt IDs are not the frozen input prefix"
)
assert all(row.get("failure") is None for row in rows), (
    "technical failure in checkpoint"
)
print(
    "[RECOVERY] checkpoint_ok "
    f"rows={len(rows)} "
    f"accepted={sum(bool(row.get('accepted')) for row in rows)}",
    flush=True,
)
PY

"$PY" scripts/run_discrete_joint_inner_optimization.py \
  --config "$INNER" \
  --calibration "$CAL/discrete_calibration_v2.json" \
  --prompts "$INPUT/construction_prompts_k60.jsonl" \
  --output "$GEN/joint_results_60.jsonl" \
  --summary "$GEN/joint_summary_60.json" \
  --max-prompts 60 \
  --max-length 128 \
  --rounds 3 \
  --required-accepted 0 \
  --resume

"$PY" -c 'import json; from pathlib import Path; p=Path("results/experiment_h6_qwen32b_reconstruction_20260812/02_candidate_generation/joint_summary_60.json"); r=json.loads(p.read_text()); assert r.get("requested_prompts") == 60; assert r.get("results") == 60; assert r.get("technically_complete") == 60; assert r.get("technical_passed") is True'

# Continue at the first post-optimizer command in the frozen formal runner.
# Sourcing preserves all preregistered development and confirmation No-Go exits.
source <(
  awk '
    /"\$PY" scripts\/summarize_discrete_results.py/ { emit = 1 }
    emit { print }
  ' scripts/run_experiment_h6_qwen32b.sh
)
