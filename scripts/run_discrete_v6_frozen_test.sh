#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/llm
PYTHON=/root/autodl-tmp/venvs/llm-integrity/bin/python
BASE="$ROOT/results/paper_aligned_qwen_7b/joint_discrete_v2_20260728"
STAGE="$BASE/13_v6_formal_60"
PORTFOLIO="$STAGE/02_validation_dev_portfolio"
DESIGN="$STAGE/03_frozen_test_evaluation/00_design"
OUTER="$STAGE/03_frozen_test_evaluation/01_outer_validation"
SOURCE_MANIFEST="$ROOT/results/paper_aligned_qwen_7b/manifests_truthful_quant/attack_manifest_test.jsonl"
TEST_MANIFEST="$DESIGN/attack_manifest_test_balanced_2each.jsonl"
TEST_REGISTRY="$ROOT/results/paper_aligned_qwen_7b/finetuning_adapters/adapter_registry_test_v6_frozen.json"
CALIBRATION="$BASE/12_fixed_hybrid_calibration/01_dev_calibration/fixed_hybrid_calibration_v1.json"

cd "$ROOT"
mkdir -p "$DESIGN" "$OUTER"
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

test -f "$PORTFOLIO/DEVELOPMENT_SELECTION_FROZEN"

"$PYTHON" scripts/build_balanced_attack_manifest.py \
  --input "$SOURCE_MANIFEST" \
  --output "$TEST_MANIFEST" \
  --report "$DESIGN/test_manifest_design_report.json" \
  --split test \
  --per-family 2

mapfile -t LORA_IDS < <("$PYTHON" - <<PY
import json
for line in open("$TEST_MANIFEST"):
    row = json.loads(line)
    if row["family"] == "finetuning":
        print(row["variant_id"])
PY
)
test "${#LORA_IDS[@]}" -eq 2

"$PYTHON" scripts/train_paper_lora_registry.py \
  --config configs/paper_aligned_qwen_7b.yaml \
  --manifest "$TEST_MANIFEST" \
  --split test \
  --data data/attack_train_lora.jsonl \
  --output-root results/paper_aligned_qwen_7b/finetuning_adapters \
  --registry-output "$TEST_REGISTRY" \
  --max-length 256 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --variant-id "${LORA_IDS[0]}" \
  --variant-id "${LORA_IDS[1]}"

"$PYTHON" scripts/prepare_discrete_hard_validation.py \
  --input "$PORTFOLIO/frozen_selected_accepted.jsonl" \
  --output-dir "$OUTER"

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
  --manifest "$TEST_MANIFEST" \
  --split test \
  --output-dir "$OUTER/macro_test_2each" \
  --adapter-registry "$TEST_REGISTRY" \
  --max-length 128 \
  --batch-size 1

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/micro_scores_unique.jsonl" \
  --kind micro \
  --output "$OUTER/micro_scores_expanded.jsonl"

"$PYTHON" scripts/expand_hard_validation_scores.py \
  --mapping "$OUTER/validation_expansion_map.jsonl" \
  --input "$OUTER/macro_test_2each/macro_records_test.jsonl" \
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
  --output "$OUTER/final_test_validated_prompts.jsonl" \
  --report "$OUTER/final_test_report.json" \
  --minimum-nondegraded-families 3 \
  --family-relative-tolerance 0.01 \
  --required-accepted 12
test_status=$?
set -e
if [[ "$test_status" -ne 0 && "$test_status" -ne 1 ]]; then
  exit "$test_status"
fi

"$PYTHON" scripts/summarize_strict_validation.py \
  "$OUTER/final_test_validated_prompts.jsonl" \
  --output "$OUTER/final_test_detailed_summary.json"

sha256sum \
  "$SOURCE_MANIFEST" \
  "$TEST_MANIFEST" \
  "$TEST_REGISTRY" \
  "$CALIBRATION" \
  "$PORTFOLIO/frozen_selected_accepted.jsonl" \
  scripts/build_balanced_attack_manifest.py \
  scripts/run_discrete_v6_frozen_test.sh \
  > "$DESIGN/frozen_inputs_code_sha256.txt"

if [[ "$test_status" -eq 0 ]]; then
  echo "FROZEN_TEST_GATE_PASSED" > "$STAGE/03_frozen_test_evaluation/gate_status.txt"
else
  echo "FROZEN_TEST_GATE_FAILED_NO_RETUNING" > "$STAGE/03_frozen_test_evaluation/gate_status.txt"
fi
touch "$STAGE/03_frozen_test_evaluation/TEST_EVALUATION_COMPLETE"
echo "V6_FROZEN_TEST_COMPLETE status=$test_status"
exit "$test_status"
