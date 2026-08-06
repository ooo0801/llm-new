#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
unset HF_HOME HUGGINGFACE_HUB_CACHE
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2

OUT=results/experiment_c_qwen14b_20260806
INPUT=experiments/prompt-construction-14b-c1/inputs
CAL="$OUT/01_calibration"
GEN="$OUT/02_candidate_generation"
DEV="$OUT/03_development"
TEST="$OUT/04_heldout_test"
ADAPTERS="$OUT/adapters"
mkdir -p "$CAL" "$GEN" "$DEV" "$TEST" "$ADAPTERS"

printf '[C_STAGE] train_dev_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config configs/experiment_c_qwen14b_inner.yaml --manifest "$INPUT/attack_manifest_train_lora_registered.jsonl" --split train --data data/attack_train_lora.jsonl --output-root "$ADAPTERS" --registry-output "$ADAPTERS/registry_train.json"
"$PY" scripts/train_paper_lora_registry.py --config configs/experiment_c_qwen14b_inner.yaml --manifest "$INPUT/attack_manifest_validation_2each.jsonl" --split validation --data data/attack_train_lora.jsonl --output-root "$ADAPTERS" --registry-output "$ADAPTERS/registry_validation.json"
printf '[C_STAGE] train_dev_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[C_STAGE] inner_calibration_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/calibrate_inner_micro.py --config configs/experiment_c_qwen14b_inner.yaml --prompts "$INPUT/calibration_prompts_k12.jsonl" --output "$CAL/micro_records.jsonl" --summary "$CAL/micro_summary.json" --max-prompts 12 --max-length 64 --probes 2 --seed 42 --representative-layer-count 4 --resume
"$PY" scripts/calibrate_inner_macro_v2.py --config configs/experiment_c_qwen14b_inner.yaml --manifest "$INPUT/attack_manifest_train_executable.jsonl" --adapter-registry "$ADAPTERS/registry_train.json" --prompts "$INPUT/calibration_prompts_k12.jsonl" --output "$CAL/macro_records.jsonl" --summary "$CAL/macro_summary.json" --max-prompts 12 --max-length 64 --seed 42 --sample-cycles 0,6
"$PY" scripts/build_discrete_calibration.py --micro-records "$CAL/micro_records.jsonl" --macro-records "$CAL/macro_records.jsonl" --output "$CAL/discrete_calibration_v2.json"
printf '[C_STAGE] inner_calibration_complete %s\n' "$(date --iso-8601=seconds)"

printf '[C_STAGE] one_prompt_smoke_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_joint_inner_optimization.py --config configs/experiment_c_qwen14b_inner.yaml --calibration "$CAL/discrete_calibration_v2.json" --prompts "$INPUT/construction_prompts_k60.jsonl" --output "$CAL/one_prompt_smoke.jsonl" --summary "$CAL/one_prompt_smoke_summary.json" --max-prompts 1 --max-length 128 --rounds 3 --required-accepted 0 --resume
printf '[C_STAGE] one_prompt_smoke_complete %s\n' "$(date --iso-8601=seconds)"

printf '[C_STAGE] generation_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_joint_inner_optimization.py --config configs/experiment_c_qwen14b_inner.yaml --calibration "$CAL/discrete_calibration_v2.json" --prompts "$INPUT/construction_prompts_k60.jsonl" --output "$GEN/joint_results_60.jsonl" --summary "$GEN/joint_summary_60.json" --max-prompts 60 --max-length 128 --rounds 3 --required-accepted 0 --resume
"$PY" scripts/summarize_discrete_results.py "$GEN/joint_results_60.jsonl" --output "$GEN/joint_detailed_summary_60.json"
"$PY" scripts/build_discrete_candidate_portfolio.py --input "$GEN/joint_results_60.jsonl" --output "$DEV/portfolio_candidates.jsonl" --mapping "$DEV/portfolio_mapping.jsonl" --report "$DEV/portfolio_build_report.json" --maximum-candidates-per-prompt 2 --minimum-nondegraded-families 3
printf '[C_STAGE] generation_complete %s\n' "$(date --iso-8601=seconds)"

printf '[C_STAGE] development_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$DEV/portfolio_candidates.jsonl" --output-dir "$DEV/outer"
"$PY" scripts/run_discrete_task_validation.py --config configs/experiment_c_qwen14b_outer.yaml --prompts "$DEV/outer/validation_candidates.jsonl" --output "$DEV/outer/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config configs/experiment_c_qwen14b_outer.yaml --prompts "$DEV/outer/validation_unique_prompts.jsonl" --output "$DEV/outer/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/run_paper_macro_sampling.py --config configs/experiment_c_qwen14b_outer.yaml --prompts "$DEV/outer/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_validation_2each.jsonl" --split validation --output-dir "$DEV/outer/macro" --adapter-registry "$ADAPTERS/registry_validation.json" --max-length 128 --batch-size 1
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/outer/validation_expansion_map.jsonl" --input "$DEV/outer/micro_scores_unique.jsonl" --kind micro --output "$DEV/outer/micro_scores_expanded.jsonl"
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/outer/validation_expansion_map.jsonl" --input "$DEV/outer/macro/macro_records_validation.jsonl" --kind macro --output "$DEV/outer/macro_records_expanded.jsonl"
"$PY" scripts/fit_hybrid_calibration.py --micro-scores "$DEV/outer/micro_scores_expanded.jsonl" --macro-records "$DEV/outer/macro_records_expanded.jsonl" --output "$DEV/fixed_hybrid_calibration_v1.json" --minimum-samples 10
"$PY" scripts/finalize_discrete_hard_validation.py --config configs/experiment_c_qwen14b_outer.yaml --proxy-results "$DEV/outer/proxy_results.jsonl" --micro-scores "$DEV/outer/micro_scores_expanded.jsonl" --macro-records "$DEV/outer/macro_records_expanded.jsonl" --task-records "$DEV/outer/task_validation.jsonl" --hybrid-calibration "$DEV/fixed_hybrid_calibration_v1.json" --output "$DEV/strict_validated_portfolio.jsonl" --report "$DEV/strict_validation_report.json" --minimum-nondegraded-families 3 --family-relative-tolerance 0.01 --required-accepted 0
set +e
"$PY" scripts/select_dev_portfolio_candidates.py --source-results "$GEN/joint_results_60.jsonl" --portfolio-results "$DEV/portfolio_candidates.jsonl" --mapping "$DEV/portfolio_mapping.jsonl" --strict-validation "$DEV/strict_validated_portfolio.jsonl" --output-all "$DEV/frozen_selected_all_60.jsonl" --output-accepted "$DEV/frozen_selected_accepted.jsonl" --report "$DEV/development_selection_report.json" --required-accepted 4
DEV_STATUS=$?
set -e
printf '[C_STAGE] development_complete status=%s %s\n' "$DEV_STATUS" "$(date --iso-8601=seconds)"
if [[ "$DEV_STATUS" -ne 0 ]]; then
  printf 'DEVELOPMENT_NO_GO\n' > "$DEV/gate_status.txt"
  exit 20
fi
printf 'DEVELOPMENT_GO\n' > "$DEV/gate_status.txt"

printf '[C_STAGE] heldout_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config configs/experiment_c_qwen14b_inner.yaml --manifest "$INPUT/attack_manifest_test_2each.jsonl" --split test --data data/attack_train_lora.jsonl --output-root "$ADAPTERS" --registry-output "$ADAPTERS/registry_test.json"
printf '[C_STAGE] heldout_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[C_STAGE] heldout_test_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$DEV/frozen_selected_accepted.jsonl" --output-dir "$TEST/outer"
"$PY" scripts/run_discrete_task_validation.py --config configs/experiment_c_qwen14b_outer.yaml --prompts "$TEST/outer/validation_candidates.jsonl" --output "$TEST/outer/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config configs/experiment_c_qwen14b_outer.yaml --prompts "$TEST/outer/validation_unique_prompts.jsonl" --output "$TEST/outer/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/run_paper_macro_sampling.py --config configs/experiment_c_qwen14b_outer.yaml --prompts "$TEST/outer/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_test_2each.jsonl" --split test --output-dir "$TEST/outer/macro" --adapter-registry "$ADAPTERS/registry_test.json" --max-length 128 --batch-size 1
"$PY" scripts/expand_hard_validation_scores.py --mapping "$TEST/outer/validation_expansion_map.jsonl" --input "$TEST/outer/micro_scores_unique.jsonl" --kind micro --output "$TEST/outer/micro_scores_expanded.jsonl"
"$PY" scripts/expand_hard_validation_scores.py --mapping "$TEST/outer/validation_expansion_map.jsonl" --input "$TEST/outer/macro/macro_records_test.jsonl" --kind macro --output "$TEST/outer/macro_records_expanded.jsonl"
set +e
"$PY" scripts/finalize_discrete_hard_validation.py --config configs/experiment_c_qwen14b_outer.yaml --proxy-results "$TEST/outer/proxy_results.jsonl" --micro-scores "$TEST/outer/micro_scores_expanded.jsonl" --macro-records "$TEST/outer/macro_records_expanded.jsonl" --task-records "$TEST/outer/task_validation.jsonl" --hybrid-calibration "$DEV/fixed_hybrid_calibration_v1.json" --output "$TEST/final_test_validated_prompts.jsonl" --report "$TEST/final_test_report.json" --minimum-nondegraded-families 3 --family-relative-tolerance 0.01 --required-accepted 4
TEST_STATUS=$?
set -e
"$PY" scripts/summarize_strict_validation.py "$TEST/final_test_validated_prompts.jsonl" --output "$TEST/final_test_detailed_summary.json"
printf '[C_STAGE] heldout_test_complete status=%s %s\n' "$TEST_STATUS" "$(date --iso-8601=seconds)"
exit "$TEST_STATUS"
