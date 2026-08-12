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
INNER=configs/experiment_h6_qwen32b_inner.yaml
DEV_CONFIG=configs/experiment_h6_qwen32b_development.yaml
CONFIRM_CONFIG=configs/experiment_h6_qwen32b_confirmation.yaml
MCC_CONFIG=configs/fingerprint_h6_qwen32b_mcc.yaml
INPUT=experiments/prompt-reconstruction-32b-h6/inputs
OUT=results/experiment_h6_qwen32b_reconstruction_20260812
ENV="$OUT/00_environment"
CAL="$OUT/01_calibration"
GEN="$OUT/02_candidate_generation"
DEV="$OUT/03_development"
CONFIRM="$OUT/04_confirmation"
ADAPTERS="$OUT/adapters"
mkdir -p "$ENV" "$CAL" "$GEN" "$DEV" "$CONFIRM" "$ADAPTERS"

"$PY" -c 'import json; from pathlib import Path; p=Path("results/experiment_h6_qwen32b_reconstruction_20260812/00_environment/engineering_smoke.json"); r=json.loads(p.read_text()); assert r.get("passed") is True and r.get("status") == "complete_go"'

printf '[H6_STAGE] train_and_development_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$INNER" --manifest "$INPUT/attack_manifest_train_lora_registered.jsonl" --split train --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/train" --registry-output "$ADAPTERS/registry_train.json"
"$PY" scripts/train_paper_lora_registry.py --config "$INNER" --manifest "$INPUT/attack_manifest_development_2each.jsonl" --split validation --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/development" --registry-output "$ADAPTERS/registry_development.json"
printf '[H6_STAGE] train_and_development_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H6_STAGE] inner_calibration_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/calibrate_inner_micro.py --config "$INNER" --prompts "$INPUT/calibration_prompts_k12.jsonl" --output "$CAL/micro_records.jsonl" --summary "$CAL/micro_summary.json" --max-prompts 12 --max-length 64 --probes 2 --seed 2026082010 --representative-layer-count 4 --resume
"$PY" scripts/calibrate_inner_macro_v2.py --config "$INNER" --manifest "$INPUT/attack_manifest_train_executable.jsonl" --adapter-registry "$ADAPTERS/registry_train.json" --prompts "$INPUT/calibration_prompts_k12.jsonl" --output "$CAL/macro_records.jsonl" --summary "$CAL/macro_summary.json" --max-prompts 12 --max-length 64 --seed 2026082010
"$PY" scripts/build_discrete_calibration.py --micro-records "$CAL/micro_records.jsonl" --macro-records "$CAL/macro_records.jsonl" --output "$CAL/discrete_calibration_v2.json"
printf '[H6_STAGE] inner_calibration_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H6_STAGE] one_prompt_construction_smoke_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_joint_inner_optimization.py --config "$INNER" --calibration "$CAL/discrete_calibration_v2.json" --prompts "$INPUT/construction_prompts_k60.jsonl" --output "$CAL/one_prompt_smoke.jsonl" --summary "$CAL/one_prompt_smoke_summary.json" --max-prompts 1 --max-length 128 --rounds 3 --required-accepted 0 --resume
printf '[H6_STAGE] one_prompt_construction_smoke_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H6_STAGE] construction_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_joint_inner_optimization.py --config "$INNER" --calibration "$CAL/discrete_calibration_v2.json" --prompts "$INPUT/construction_prompts_k60.jsonl" --output "$GEN/joint_results_60.jsonl" --summary "$GEN/joint_summary_60.json" --max-prompts 60 --max-length 128 --rounds 3 --required-accepted 0 --resume
"$PY" scripts/summarize_discrete_results.py "$GEN/joint_results_60.jsonl" --output "$GEN/joint_detailed_summary_60.json"
"$PY" scripts/build_discrete_candidate_portfolio.py --input "$GEN/joint_results_60.jsonl" --output "$DEV/portfolio_candidates.jsonl" --mapping "$DEV/portfolio_mapping.jsonl" --report "$DEV/portfolio_build_report.json" --maximum-candidates-per-prompt 2 --minimum-nondegraded-families 3
printf '[H6_STAGE] construction_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H6_STAGE] development_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$DEV/portfolio_candidates.jsonl" --output-dir "$DEV/outer"
"$PY" scripts/run_discrete_task_validation.py --config "$DEV_CONFIG" --prompts "$DEV/outer/validation_candidates.jsonl" --output "$DEV/outer/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$DEV_CONFIG" --prompts "$DEV/outer/validation_unique_prompts.jsonl" --output "$DEV/outer/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/run_paper_macro_sampling.py --config "$DEV_CONFIG" --prompts "$DEV/outer/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_development_2each.jsonl" --split validation --output-dir "$DEV/outer/macro" --adapter-registry "$ADAPTERS/registry_development.json" --max-length 128 --batch-size 1 --seed 2026082024
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/outer/validation_expansion_map.jsonl" --input "$DEV/outer/micro_scores_unique.jsonl" --kind micro --output "$DEV/outer/micro_scores_expanded.jsonl"
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/outer/validation_expansion_map.jsonl" --input "$DEV/outer/macro/macro_records_validation.jsonl" --kind macro --output "$DEV/outer/macro_records_expanded.jsonl"
"$PY" scripts/fit_hybrid_calibration.py --micro-scores "$DEV/outer/micro_scores_expanded.jsonl" --macro-records "$DEV/outer/macro_records_expanded.jsonl" --output "$DEV/fixed_hybrid_calibration_v1.json" --minimum-samples 10
"$PY" scripts/finalize_discrete_hard_validation.py --config "$DEV_CONFIG" --proxy-results "$DEV/outer/proxy_results.jsonl" --micro-scores "$DEV/outer/micro_scores_expanded.jsonl" --macro-records "$DEV/outer/macro_records_expanded.jsonl" --task-records "$DEV/outer/task_validation.jsonl" --hybrid-calibration "$DEV/fixed_hybrid_calibration_v1.json" --output "$DEV/strict_validated_portfolio.jsonl" --report "$DEV/strict_validation_report.json" --minimum-nondegraded-families 3 --family-relative-tolerance 0.01 --required-accepted 0
set +e
"$PY" scripts/select_dev_portfolio_candidates.py --source-results "$GEN/joint_results_60.jsonl" --portfolio-results "$DEV/portfolio_candidates.jsonl" --mapping "$DEV/portfolio_mapping.jsonl" --strict-validation "$DEV/strict_validated_portfolio.jsonl" --output-all "$DEV/frozen_selected_all_60.jsonl" --output-accepted "$DEV/frozen_selected_accepted.jsonl" --report "$DEV/development_selection_report.json" --required-accepted 30
DEV_STATUS=$?
set -e
if [[ "$DEV_STATUS" -ne 0 ]]; then
  printf 'DEVELOPMENT_NO_GO\n' > "$DEV/gate_status.txt"
  exit 20
fi
"$PY" scripts/check_h6_prompt_gate.py --input "$DEV/frozen_selected_accepted.jsonl" --output "$DEV/frozen_core_pool.jsonl" --report "$DEV/core_category_report.json" --minimum-confirmed 30
printf 'DEVELOPMENT_GO\n' > "$DEV/gate_status.txt"
printf '[H6_STAGE] development_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H6_STAGE] confirmation_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$INNER" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --split test --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/confirmation" --registry-output "$ADAPTERS/registry_confirmation.json"
printf '[H6_STAGE] confirmation_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$DEV/frozen_selected_accepted.jsonl" --output-dir "$CONFIRM/outer"
"$PY" scripts/run_discrete_task_validation.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/outer/validation_candidates.jsonl" --output "$CONFIRM/outer/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/outer/validation_unique_prompts.jsonl" --output "$CONFIRM/outer/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/run_paper_macro_sampling.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/outer/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --split test --output-dir "$CONFIRM/outer/macro" --adapter-registry "$ADAPTERS/registry_confirmation.json" --max-length 128 --batch-size 1 --seed 2026082034
"$PY" scripts/expand_hard_validation_scores.py --mapping "$CONFIRM/outer/validation_expansion_map.jsonl" --input "$CONFIRM/outer/micro_scores_unique.jsonl" --kind micro --output "$CONFIRM/outer/micro_scores_expanded.jsonl"
"$PY" scripts/expand_hard_validation_scores.py --mapping "$CONFIRM/outer/validation_expansion_map.jsonl" --input "$CONFIRM/outer/macro/macro_records_test.jsonl" --kind macro --output "$CONFIRM/outer/macro_records_expanded.jsonl"
"$PY" scripts/finalize_discrete_hard_validation.py --config "$CONFIRM_CONFIG" --proxy-results "$CONFIRM/outer/proxy_results.jsonl" --micro-scores "$CONFIRM/outer/micro_scores_expanded.jsonl" --macro-records "$CONFIRM/outer/macro_records_expanded.jsonl" --task-records "$CONFIRM/outer/task_validation.jsonl" --hybrid-calibration "$DEV/fixed_hybrid_calibration_v1.json" --output "$CONFIRM/final_test_validated_prompts.jsonl" --report "$CONFIRM/final_test_report.json" --minimum-nondegraded-families 3 --family-relative-tolerance 0.01 --required-accepted 0
"$PY" scripts/check_h6_prompt_gate.py --input "$CONFIRM/final_test_validated_prompts.jsonl" --output "$CONFIRM/confirmed_prompt_pool.jsonl" --report "$CONFIRM/prompt_gate_report.json" --minimum-confirmed 19
printf 'CONFIRMATION_GO\n' > "$CONFIRM/gate_status.txt"
printf '[H6_STAGE] confirmation_complete %s\n' "$(date --iso-8601=seconds)"

printf '[H6_STAGE] component_mcc_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_fingerprint_h6_mcc.py --config "$MCC_CONFIG"
"$PY" scripts/extract_fingerprint_v2_activations.py --config "$MCC_CONFIG"
"$PY" scripts/build_fingerprint_v2_global_selection.py --config "$MCC_CONFIG"
"$PY" scripts/build_fingerprint_h6_artifact.py --config "$MCC_CONFIG"
printf '[H6_STAGE] component_mcc_complete %s\n' "$(date --iso-8601=seconds)"
