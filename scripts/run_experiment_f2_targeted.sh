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
INNER_CONFIG=configs/experiment_f2_qwen14b_inner.yaml
DEV_CONFIG=configs/experiment_f2_qwen14b_development.yaml
CONFIRM_CONFIG=configs/experiment_f2_qwen14b_confirmation.yaml
INPUT=experiments/prompt-robust-14b-f2/inputs
OUT=results/experiment_f2_qwen14b_targeted_20260808
BUILD="$OUT/00_construction"
DEV="$OUT/01_development"
CONFIRM="$OUT/02_confirmation"
ADAPTERS="$OUT/adapters"
mkdir -p "$BUILD" "$DEV" "$CONFIRM" "$ADAPTERS"

CAL=results/experiment_c_qwen14b_20260806/01_calibration/discrete_calibration_v2.json
TRAIN_REGISTRY=results/experiment_c_qwen14b_20260806/adapters/registry_train.json
test "$(sha256sum "$CAL" | cut -d' ' -f1)" = 66944c94c523840938c8444c5b696138fd0f4cc8ce2e628be6ef2629d066ad0c
test "$(sha256sum "$TRAIN_REGISTRY" | cut -d' ' -f1)" = 288791e63732aa4410ee25e6f8f81c4fc1ee461c35c10964bf840a004e385f5f

printf '[F2_STAGE] pipeline_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/experiment_a_model_smoke.py --config "$DEV_CONFIG" --output "$BUILD/model_smoke.json"

printf '[F2_STAGE] construction_smoke_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_joint_inner_optimization.py --config "$INNER_CONFIG" --calibration "$CAL" --prompts "$INPUT/smoke_sources2.jsonl" --output "$BUILD/smoke_results.jsonl" --summary "$BUILD/smoke_summary.json" --max-prompts 2 --max-length 128 --rounds 3 --required-accepted 0 --resume
printf '[F2_STAGE] construction_smoke_complete %s\n' "$(date --iso-8601=seconds)"

printf '[F2_STAGE] construction_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_discrete_joint_inner_optimization.py --config "$INNER_CONFIG" --calibration "$CAL" --prompts "$INPUT/expansion_sources32.jsonl" --output "$BUILD/joint_results_32.jsonl" --summary "$BUILD/joint_summary_32.json" --max-prompts 32 --max-length 128 --rounds 3 --required-accepted 0 --resume
"$PY" scripts/build_discrete_candidate_portfolio.py --input "$BUILD/joint_results_32.jsonl" --output "$BUILD/new_portfolio_candidates.jsonl" --mapping "$BUILD/new_portfolio_mapping.jsonl" --report "$BUILD/new_portfolio_report.json" --maximum-candidates-per-prompt 2 --minimum-nondegraded-families 3
set +e
"$PY" scripts/assemble_experiment_f2_pool.py --base experiments/prompt-robust-14b-f1/inputs/candidate_pairs64.jsonl --new-portfolio "$BUILD/new_portfolio_candidates.jsonl" --output "$BUILD/candidate_pairs_f2.jsonl" --report "$BUILD/pool_assembly_report.json" --minimum-new-sources-per-category 3
BUILD_STATUS=$?
set -e
printf '[F2_STAGE] construction_complete status=%s %s\n' "$BUILD_STATUS" "$(date --iso-8601=seconds)"
if [[ "$BUILD_STATUS" -ne 0 ]]; then
  printf 'CONSTRUCTION_NO_GO\n' > "$BUILD/gate_status.txt"
  exit 10
fi
printf 'CONSTRUCTION_GO\n' > "$BUILD/gate_status.txt"

printf '[F2_STAGE] development_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$DEV_CONFIG" --manifest "$INPUT/attack_manifest_development_3each.jsonl" --split validation --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/development" --registry-output "$ADAPTERS/registry_development.json"
printf '[F2_STAGE] development_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[F2_STAGE] development_endpoints_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$BUILD/candidate_pairs_f2.jsonl" --output-dir "$DEV/prepared"
"$PY" scripts/run_discrete_task_validation.py --config "$DEV_CONFIG" --prompts "$DEV/prepared/validation_candidates.jsonl" --output "$DEV/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$DEV_CONFIG" --prompts "$DEV/prepared/validation_unique_prompts.jsonl" --output "$DEV/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/prepared/validation_expansion_map.jsonl" --input "$DEV/micro_scores_unique.jsonl" --kind micro --output "$DEV/micro_scores_expanded.jsonl"
"$PY" scripts/run_paper_macro_sampling.py --config "$DEV_CONFIG" --prompts "$DEV/prepared/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_development_3each.jsonl" --split validation --output-dir "$DEV/macro" --adapter-registry "$ADAPTERS/registry_development.json" --max-length 128 --batch-size 1 --seed 2026080814
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/prepared/validation_expansion_map.jsonl" --input "$DEV/macro/macro_records_validation.jsonl" --kind macro --output "$DEV/macro_records_expanded.jsonl"
set +e
"$PY" scripts/analyze_experiment_f_robust.py --experiment-label F2 --stage development --pairs "$BUILD/candidate_pairs_f2.jsonl" --manifest "$INPUT/attack_manifest_development_3each.jsonl" --task "$DEV/task_validation.jsonl" --micro "$DEV/micro_scores_expanded.jsonl" --macro "$DEV/macro_records_expanded.jsonl" --output-dir "$DEV/analysis"
DEV_STATUS=$?
set -e
printf '[F2_STAGE] development_analysis_complete status=%s %s\n' "$DEV_STATUS" "$(date --iso-8601=seconds)"
if [[ "$DEV_STATUS" -ne 0 ]]; then
  printf 'DEVELOPMENT_NO_GO\n' > "$DEV/gate_status.txt"
  exit 20
fi
printf 'DEVELOPMENT_GO\n' > "$DEV/gate_status.txt"

printf '[F2_STAGE] confirmation_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$CONFIRM_CONFIG" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --split test --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/confirmation" --registry-output "$ADAPTERS/registry_confirmation.json"
printf '[F2_STAGE] confirmation_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[F2_STAGE] confirmation_endpoints_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$DEV/analysis/frozen30_pairs.jsonl" --output-dir "$CONFIRM/prepared"
"$PY" scripts/run_discrete_task_validation.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/prepared/validation_candidates.jsonl" --output "$CONFIRM/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/prepared/validation_unique_prompts.jsonl" --output "$CONFIRM/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/expand_hard_validation_scores.py --mapping "$CONFIRM/prepared/validation_expansion_map.jsonl" --input "$CONFIRM/micro_scores_unique.jsonl" --kind micro --output "$CONFIRM/micro_scores_expanded.jsonl"
"$PY" scripts/run_paper_macro_sampling.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/prepared/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --split test --output-dir "$CONFIRM/macro" --adapter-registry "$ADAPTERS/registry_confirmation.json" --max-length 128 --batch-size 1 --seed 2026080823
"$PY" scripts/expand_hard_validation_scores.py --mapping "$CONFIRM/prepared/validation_expansion_map.jsonl" --input "$CONFIRM/macro/macro_records_test.jsonl" --kind macro --output "$CONFIRM/macro_records_expanded.jsonl"
set +e
"$PY" scripts/analyze_experiment_f_robust.py --experiment-label F2 --stage confirmation --pairs "$DEV/analysis/frozen30_pairs.jsonl" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --task "$CONFIRM/task_validation.jsonl" --micro "$CONFIRM/micro_scores_expanded.jsonl" --macro "$CONFIRM/macro_records_expanded.jsonl" --output-dir "$CONFIRM/analysis" --required-legacy 23
CONFIRM_STATUS=$?
set -e
printf '[F2_STAGE] confirmation_analysis_complete status=%s %s\n' "$CONFIRM_STATUS" "$(date --iso-8601=seconds)"
exit "$CONFIRM_STATUS"
