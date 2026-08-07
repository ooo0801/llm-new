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
DEV_CONFIG=configs/experiment_f_qwen14b_development.yaml
CONFIRM_CONFIG=configs/experiment_f_qwen14b_confirmation.yaml
INPUT=experiments/prompt-robust-14b-f1/inputs
OUT=results/experiment_f_qwen14b_robust_20260807
DEV="$OUT/01_development"
CONFIRM="$OUT/02_confirmation"
ADAPTERS="$OUT/adapters"
mkdir -p "$DEV" "$CONFIRM" "$ADAPTERS"

printf '[F1_STAGE] pipeline_start %s\n' "$(date --iso-8601=seconds)"
printf '[F1_STAGE] model_smoke_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/experiment_a_model_smoke.py --config "$DEV_CONFIG" --output "$DEV/model_smoke.json"
printf '[F1_STAGE] model_smoke_complete %s\n' "$(date --iso-8601=seconds)"

printf '[F1_STAGE] development_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$DEV_CONFIG" --manifest "$INPUT/attack_manifest_development_3each.jsonl" --split validation --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/development" --registry-output "$ADAPTERS/registry_development.json"
printf '[F1_STAGE] development_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[F1_STAGE] development_endpoints_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$INPUT/candidate_pairs64.jsonl" --output-dir "$DEV/prepared"
"$PY" scripts/run_discrete_task_validation.py --config "$DEV_CONFIG" --prompts "$DEV/prepared/validation_candidates.jsonl" --output "$DEV/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$DEV_CONFIG" --prompts "$DEV/prepared/validation_unique_prompts.jsonl" --output "$DEV/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/prepared/validation_expansion_map.jsonl" --input "$DEV/micro_scores_unique.jsonl" --kind micro --output "$DEV/micro_scores_expanded.jsonl"
"$PY" scripts/run_paper_macro_sampling.py --config "$DEV_CONFIG" --prompts "$DEV/prepared/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_development_3each.jsonl" --split validation --output-dir "$DEV/macro" --adapter-registry "$ADAPTERS/registry_development.json" --max-length 128 --batch-size 1 --seed 2026080714
"$PY" scripts/expand_hard_validation_scores.py --mapping "$DEV/prepared/validation_expansion_map.jsonl" --input "$DEV/macro/macro_records_validation.jsonl" --kind macro --output "$DEV/macro_records_expanded.jsonl"
set +e
"$PY" scripts/analyze_experiment_f_robust.py --stage development --pairs "$INPUT/candidate_pairs64.jsonl" --manifest "$INPUT/attack_manifest_development_3each.jsonl" --task "$DEV/task_validation.jsonl" --micro "$DEV/micro_scores_expanded.jsonl" --macro "$DEV/macro_records_expanded.jsonl" --output-dir "$DEV/analysis"
DEV_STATUS=$?
set -e
printf '[F1_STAGE] development_analysis_complete status=%s %s\n' "$DEV_STATUS" "$(date --iso-8601=seconds)"
if [[ "$DEV_STATUS" -ne 0 ]]; then
  printf 'DEVELOPMENT_NO_GO\n' > "$DEV/gate_status.txt"
  exit 20
fi
printf 'DEVELOPMENT_GO\n' > "$DEV/gate_status.txt"

printf '[F1_STAGE] confirmation_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$CONFIRM_CONFIG" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --split test --data data/attack_train_lora.jsonl --output-root "$ADAPTERS/confirmation" --registry-output "$ADAPTERS/registry_confirmation.json"
printf '[F1_STAGE] confirmation_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[F1_STAGE] confirmation_endpoints_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$DEV/analysis/frozen30_pairs.jsonl" --output-dir "$CONFIRM/prepared"
"$PY" scripts/run_discrete_task_validation.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/prepared/validation_candidates.jsonl" --output "$CONFIRM/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/prepared/validation_unique_prompts.jsonl" --output "$CONFIRM/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/expand_hard_validation_scores.py --mapping "$CONFIRM/prepared/validation_expansion_map.jsonl" --input "$CONFIRM/micro_scores_unique.jsonl" --kind micro --output "$CONFIRM/micro_scores_expanded.jsonl"
"$PY" scripts/run_paper_macro_sampling.py --config "$CONFIRM_CONFIG" --prompts "$CONFIRM/prepared/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --split test --output-dir "$CONFIRM/macro" --adapter-registry "$ADAPTERS/registry_confirmation.json" --max-length 128 --batch-size 1 --seed 2026080723
"$PY" scripts/expand_hard_validation_scores.py --mapping "$CONFIRM/prepared/validation_expansion_map.jsonl" --input "$CONFIRM/macro/macro_records_test.jsonl" --kind macro --output "$CONFIRM/macro_records_expanded.jsonl"
set +e
"$PY" scripts/analyze_experiment_f_robust.py --stage confirmation --pairs "$DEV/analysis/frozen30_pairs.jsonl" --manifest "$INPUT/attack_manifest_confirmation_2each.jsonl" --task "$CONFIRM/task_validation.jsonl" --micro "$CONFIRM/micro_scores_expanded.jsonl" --macro "$CONFIRM/macro_records_expanded.jsonl" --output-dir "$CONFIRM/analysis" --required-legacy 23
CONFIRM_STATUS=$?
set -e
printf '[F1_STAGE] confirmation_analysis_complete status=%s %s\n' "$CONFIRM_STATUS" "$(date --iso-8601=seconds)"
exit "$CONFIRM_STATUS"
