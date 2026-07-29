#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
SMALL_REPORT="$BASE/06_small_comparison/outer_validation/strict_validation_report_12.json"
STAGE="$BASE/07_formal_60"
OUTER="$STAGE/outer_validation"
CALIBRATION="$BASE/02_calibration/discrete_calibration_v2.json"
CONFIG="$ROOT/configs/paper_aligned_qwen_7b_joint_discrete_v2.yaml"

cd "$ROOT"
mkdir -p "$OUTER"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" -c \
  "import json; p=json.load(open('$SMALL_REPORT')); assert p['scientific_gate_passed'], '12-prompt outer gate did not pass'"

"$PYTHON" scripts/run_discrete_joint_inner_optimization.py \
  --config "$CONFIG" \
  --calibration "$CALIBRATION" \
  --output "$STAGE/joint_results_60.jsonl" \
  --summary "$STAGE/joint_summary_60.json" \
  --max-prompts 60 \
  --max-length 128 \
  --rounds 3 \
  --required-accepted 0 \
  --resume

"$PYTHON" scripts/summarize_discrete_results.py \
  "$STAGE/joint_results_60.jsonl" \
  --output "$STAGE/joint_detailed_summary_60.json"

"$PYTHON" scripts/prepare_discrete_hard_validation.py \
  --input "$STAGE/joint_results_60.jsonl" \
  --output-dir "$OUTER"

"$PYTHON" scripts/run_discrete_task_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_candidates.jsonl" \
  --output "$OUTER/task_validation_120.jsonl" \
  --max-input-tokens 512 \
  --max-new-tokens 128 \
  --resume

"$PYTHON" scripts/run_paper_micro_scoring.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_unique_prompts.jsonl" \
  --output "$OUTER/micro_scores_unique.jsonl" \
  --probes 4 \
  --resume

"$PYTHON" scripts/run_paper_macro_sampling.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_unique_prompts.jsonl" \
  --manifest results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_validation_balanced_2each.jsonl \
  --split validation \
  --output-dir "$OUTER/macro_validation_2each" \
  --adapter-registry results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_validation.json \
  --max-length 128 \
  --batch-size 1

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/micro_scores_unique.jsonl" \
  --kind micro \
  --output "$OUTER/micro_scores_120.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/macro_validation_2each/macro_records_validation.jsonl" \
  --kind macro \
  --output "$OUTER/macro_records_120x10.jsonl"

if "$PYTHON" scripts/finalize_discrete_hard_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$OUTER/proxy_results.jsonl" \
  --micro-scores "$OUTER/micro_scores_120.jsonl" \
  --macro-records "$OUTER/macro_records_120x10.jsonl" \
  --task-records "$OUTER/task_validation_120.jsonl" \
  --output "$OUTER/strict_validated_prompts_60.jsonl" \
  --report "$OUTER/strict_validation_report_60.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 12; then
  echo "FORMAL_POOL_READY_FOR_FINGERPRINT_SELECTION" \
    > "$OUTER/gate_status.txt"
else
  status=$?
  if [[ "$status" -eq 1 ]]; then
    echo "FORMAL_POOL_INSUFFICIENT_EXPAND_REMAINING_276" \
      > "$OUTER/gate_status.txt"
  else
    exit "$status"
  fi
fi

sha256sum \
  "$CONFIG" \
  "$CALIBRATION" \
  configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  src/llm_integrity/discrete_joint_inner_optimizer.py \
  scripts/run_discrete_joint_inner_optimization.py \
  scripts/prepare_discrete_hard_validation.py \
  scripts/run_discrete_task_validation.py \
  scripts/finalize_discrete_hard_validation.py \
  scripts/run_discrete_v2_formal_60.sh \
  > "$STAGE/code_and_config_sha256.txt"

echo "DISCRETE_V2_FORMAL_60_COMPLETE"
