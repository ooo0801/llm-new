#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
SOURCE="$BASE/10_disjoint_inner_dev/02_small_joint/joint_results_12.jsonl"
STAGE="$BASE/12_fixed_hybrid_calibration/02_knowledge_alternative"
OUTER="$STAGE/outer_validation"
CALIBRATION="$BASE/12_fixed_hybrid_calibration/01_dev_calibration/fixed_hybrid_calibration_v1.json"

cd "$ROOT"
mkdir -p "$STAGE" "$OUTER"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"$PYTHON" scripts/extract_discrete_candidate.py \
  --input "$SOURCE" \
  --prompt-id knowledge_9edb61565cee \
  --round 1 \
  --candidate-index 5 \
  --candidate-tag v4_r1_c5 \
  --output "$STAGE/candidate_result.jsonl" \
  > "$STAGE/extract.log"

"$PYTHON" scripts/prepare_discrete_hard_validation.py \
  --input "$STAGE/candidate_result.jsonl" \
  --output-dir "$OUTER"

"$PYTHON" scripts/run_discrete_task_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_candidates.jsonl" \
  --output "$OUTER/task_validation_2.jsonl" \
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
  --output "$OUTER/micro_scores_2.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/macro_validation_2each/macro_records_validation.jsonl" \
  --kind macro \
  --output "$OUTER/macro_records_2x10.jsonl"

set +e
"$PYTHON" scripts/finalize_discrete_hard_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$OUTER/proxy_results.jsonl" \
  --micro-scores "$OUTER/micro_scores_2.jsonl" \
  --macro-records "$OUTER/macro_records_2x10.jsonl" \
  --task-records "$OUTER/task_validation_2.jsonl" \
  --hybrid-calibration "$CALIBRATION" \
  --output "$OUTER/strict_validated_candidate.jsonl" \
  --report "$OUTER/strict_validation_report.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 1
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 1 ]]; then
  exit "$status"
fi

"$PYTHON" scripts/summarize_strict_validation.py \
  "$OUTER/strict_validated_candidate.jsonl" \
  --output "$OUTER/strict_validation_detailed_summary.json"

sha256sum \
  "$SOURCE" \
  "$CALIBRATION" \
  scripts/extract_discrete_candidate.py \
  scripts/finalize_discrete_hard_validation.py \
  > "$STAGE/code_source_calibration_sha256.txt"

echo "V6_KNOWLEDGE_ALTERNATIVE_COMPLETE status=$status"
