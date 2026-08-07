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
CONFIRM_CONFIG=configs/experiment_f2_qwen14b_confirmation.yaml
INPUT=experiments/prompt-robust-14b-f2/inputs
OUT=results/experiment_f2_qwen14b_targeted_20260808
DEV="$OUT/01_development"
CONFIRM="$OUT/02_confirmation"
ADAPTERS="$OUT/adapters"

# This recovery entry point is intentionally narrower than the full pipeline.
# It may continue only after the frozen development selection and confirmation
# task/adaptor prerequisites have already completed successfully.
test "$(cat "$DEV/gate_status.txt")" = DEVELOPMENT_GO
test "$(wc -l < "$DEV/analysis/frozen30_pairs.jsonl")" -eq 30
test "$(wc -l < "$CONFIRM/prepared/validation_unique_prompts.jsonl")" -eq 60
test -s "$ADAPTERS/registry_confirmation.json"
"$PY" - "$CONFIRM/task_validation.summary.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    summary = json.load(handle)
assert summary == {
    "deterministic_generation": True,
    "failed": 0,
    "passed": 60,
    "records": 60,
}, summary
PY

printf '[F2_STAGE] confirmation_resume_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_paper_micro_scoring.py \
  --config "$CONFIRM_CONFIG" \
  --prompts "$CONFIRM/prepared/validation_unique_prompts.jsonl" \
  --output "$CONFIRM/micro_scores_unique.jsonl" \
  --probes 4 \
  --resume
"$PY" scripts/expand_hard_validation_scores.py \
  --mapping "$CONFIRM/prepared/validation_expansion_map.jsonl" \
  --input "$CONFIRM/micro_scores_unique.jsonl" \
  --kind micro \
  --output "$CONFIRM/micro_scores_expanded.jsonl"
"$PY" scripts/run_paper_macro_sampling.py \
  --config "$CONFIRM_CONFIG" \
  --prompts "$CONFIRM/prepared/validation_unique_prompts.jsonl" \
  --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" \
  --split test \
  --output-dir "$CONFIRM/macro" \
  --adapter-registry "$ADAPTERS/registry_confirmation.json" \
  --max-length 128 \
  --batch-size 1 \
  --seed 2026080823
"$PY" scripts/expand_hard_validation_scores.py \
  --mapping "$CONFIRM/prepared/validation_expansion_map.jsonl" \
  --input "$CONFIRM/macro/macro_records_test.jsonl" \
  --kind macro \
  --output "$CONFIRM/macro_records_expanded.jsonl"

set +e
"$PY" scripts/analyze_experiment_f_robust.py \
  --experiment-label F2 \
  --stage confirmation \
  --pairs "$DEV/analysis/frozen30_pairs.jsonl" \
  --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" \
  --task "$CONFIRM/task_validation.jsonl" \
  --micro "$CONFIRM/micro_scores_expanded.jsonl" \
  --macro "$CONFIRM/macro_records_expanded.jsonl" \
  --output-dir "$CONFIRM/analysis" \
  --required-legacy 23
CONFIRM_STATUS=$?
set -e
printf '[F2_STAGE] confirmation_analysis_complete status=%s %s\n' \
  "$CONFIRM_STATUS" "$(date --iso-8601=seconds)"
if [[ "$CONFIRM_STATUS" -eq 0 ]]; then
  printf 'CONFIRMATION_GO\n' > "$CONFIRM/gate_status.txt"
else
  printf 'CONFIRMATION_NO_GO\n' > "$CONFIRM/gate_status.txt"
fi
exit "$CONFIRM_STATUS"
