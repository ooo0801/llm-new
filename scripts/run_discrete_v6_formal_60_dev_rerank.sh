#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/13_v6_formal_60"
GEN="$STAGE/01_candidate_generation"
PORTFOLIO="$STAGE/02_validation_dev_portfolio"
OUTER="$PORTFOLIO/outer_validation"
CALIBRATION="$BASE/12_fixed_hybrid_calibration/01_dev_calibration/fixed_hybrid_calibration_v1.json"

cd "$ROOT"
mkdir -p "$OUTER"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -f "$GEN/GENERATION_COMPLETE"

"$PYTHON" scripts/prepare_discrete_hard_validation.py \
  --input "$PORTFOLIO/portfolio_candidates.jsonl" \
  --output-dir "$OUTER"

CANDIDATES=$("$PYTHON" -c "import json; print(json.load(open('$OUTER/validation_preparation_summary.json'))['candidate_pairs'])")
UNIQUE=$("$PYTHON" -c "import json; print(json.load(open('$OUTER/validation_preparation_summary.json'))['unique_prompt_strings'])")

"$PYTHON" scripts/run_discrete_task_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --prompts "$OUTER/validation_candidates.jsonl" \
  --output "$OUTER/task_validation.jsonl" \
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
  --output "$OUTER/micro_scores_expanded.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/macro_validation_2each/macro_records_validation.jsonl" \
  --kind macro \
  --output "$OUTER/macro_records_expanded.jsonl"

set +e
"$PYTHON" scripts/finalize_discrete_hard_validation.py \
  --config configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  --proxy-results "$OUTER/proxy_results.jsonl" \
  --micro-scores "$OUTER/micro_scores_expanded.jsonl" \
  --macro-records "$OUTER/macro_records_expanded.jsonl" \
  --task-records "$OUTER/task_validation.jsonl" \
  --hybrid-calibration "$CALIBRATION" \
  --output "$OUTER/strict_validated_portfolio.jsonl" \
  --report "$OUTER/strict_validation_report.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 0
finalize_status=$?
set -e
if [[ "$finalize_status" -ne 0 ]]; then
  exit "$finalize_status"
fi

set +e
"$PYTHON" scripts/select_dev_portfolio_candidates.py \
  --source-results "$GEN/joint_results_60.jsonl" \
  --portfolio-results "$PORTFOLIO/portfolio_candidates.jsonl" \
  --mapping "$PORTFOLIO/portfolio_mapping.jsonl" \
  --strict-validation "$OUTER/strict_validated_portfolio.jsonl" \
  --output-all "$PORTFOLIO/frozen_selected_all_60.jsonl" \
  --output-accepted "$PORTFOLIO/frozen_selected_accepted.jsonl" \
  --report "$PORTFOLIO/development_selection_report.json" \
  --required-accepted 12
selection_status=$?
set -e

sha256sum \
  "$CALIBRATION" \
  "$PORTFOLIO/portfolio_candidates.jsonl" \
  "$PORTFOLIO/portfolio_mapping.jsonl" \
  configs/paper_aligned_qwen_7b_joint_outer_validation.yaml \
  scripts/prepare_discrete_hard_validation.py \
  scripts/finalize_discrete_hard_validation.py \
  scripts/select_dev_portfolio_candidates.py \
  > "$PORTFOLIO/code_config_data_sha256.txt"

if [[ "$selection_status" -eq 0 ]]; then
  echo "DEVELOPMENT_PORTFOLIO_GATE_PASSED" > "$PORTFOLIO/gate_status.txt"
  touch "$PORTFOLIO/DEVELOPMENT_SELECTION_FROZEN"
else
  echo "DEVELOPMENT_PORTFOLIO_GATE_FAILED_NO_TEST_EVALUATION" > "$PORTFOLIO/gate_status.txt"
fi
echo "V6_FORMAL_60_DEV_RERANK_COMPLETE status=$selection_status candidates=$CANDIDATES unique=$UNIQUE"
exit "$selection_status"
