#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/llm
export HF_HUB_CACHE=/root/autodl-tmp/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

PY=/root/autodl-tmp/venvs/llm-integrity/bin/python
CONFIG=configs/experiment_e_qwen14b_final.yaml
INPUT=experiments/prompt-union-14b-e1/inputs
TRACKED=experiments/prompt-union-14b-e1/results
ROOT_OUT=results/experiment_e_qwen14b_final_20260807
SMOKE="$ROOT_OUT/01_smoke"
FULL="$ROOT_OUT/02_full"
ADAPTERS="$ROOT_OUT/adapters"

mkdir -p "$TRACKED" "$SMOKE" "$FULL" "$ADAPTERS"
printf '[E_STAGE] pipeline_start %s\n' "$(date --iso-8601=seconds)"

printf '[E_STAGE] tokenizer_preflight_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/experiment_e_tokenizer_preflight.py --pairs "$INPUT/final30_pairs.jsonl" --output "$TRACKED/tokenizer_preflight.json"
printf '[E_STAGE] tokenizer_preflight_complete %s\n' "$(date --iso-8601=seconds)"

printf '[E_STAGE] model_smoke_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/experiment_a_model_smoke.py --config "$CONFIG" --output "$TRACKED/model_smoke.json"
printf '[E_STAGE] model_smoke_complete %s\n' "$(date --iso-8601=seconds)"

printf '[E_STAGE] lora_adapters_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/train_paper_lora_registry.py --config "$CONFIG" --manifest "$INPUT/attack_manifest_final_2each.jsonl" --split test --data data/attack_train_lora.jsonl --output-root "$ADAPTERS" --registry-output "$ADAPTERS/registry_final.json"
printf '[E_STAGE] lora_adapters_complete %s\n' "$(date --iso-8601=seconds)"

printf '[E_STAGE] one_pair_smoke_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$INPUT/smoke_pair.jsonl" --output-dir "$SMOKE/prepared"
"$PY" scripts/run_discrete_task_validation.py --config "$CONFIG" --prompts "$SMOKE/prepared/validation_candidates.jsonl" --output "$SMOKE/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
"$PY" scripts/run_paper_micro_scoring.py --config "$CONFIG" --prompts "$SMOKE/prepared/validation_unique_prompts.jsonl" --output "$SMOKE/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/run_paper_macro_sampling.py --config "$CONFIG" --prompts "$SMOKE/prepared/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_final_2each.jsonl" --split test --output-dir "$SMOKE/macro" --adapter-registry "$ADAPTERS/registry_final.json" --max-length 128 --batch-size 1 --seed 2026080704
"$PY" scripts/check_experiment_e_smoke.py --task "$SMOKE/task_validation.jsonl" --micro "$SMOKE/micro_scores_unique.jsonl" --macro "$SMOKE/macro/macro_records_test.jsonl" --output "$SMOKE/smoke_report.json"
printf '[E_STAGE] one_pair_smoke_complete %s\n' "$(date --iso-8601=seconds)"

printf '[E_STAGE] full_joint_start %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/prepare_discrete_hard_validation.py --input "$INPUT/final30_pairs.jsonl" --output-dir "$FULL/prepared"
"$PY" scripts/run_discrete_task_validation.py --config "$CONFIG" --prompts "$FULL/prepared/validation_candidates.jsonl" --output "$FULL/task_validation.jsonl" --max-input-tokens 512 --max-new-tokens 128 --resume
printf '[E_STAGE] full_task_complete %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_paper_micro_scoring.py --config "$CONFIG" --prompts "$FULL/prepared/validation_unique_prompts.jsonl" --output "$FULL/micro_scores_unique.jsonl" --probes 4 --resume
"$PY" scripts/expand_hard_validation_scores.py --mapping "$FULL/prepared/validation_expansion_map.jsonl" --input "$FULL/micro_scores_unique.jsonl" --kind micro --output "$FULL/micro_scores_expanded.jsonl"
printf '[E_STAGE] full_micro_complete %s\n' "$(date --iso-8601=seconds)"
"$PY" scripts/run_paper_macro_sampling.py --config "$CONFIG" --prompts "$FULL/prepared/validation_unique_prompts.jsonl" --manifest "$INPUT/attack_manifest_final_2each.jsonl" --split test --output-dir "$FULL/macro" --adapter-registry "$ADAPTERS/registry_final.json" --max-length 128 --batch-size 1 --seed 2026080704
"$PY" scripts/expand_hard_validation_scores.py --mapping "$FULL/prepared/validation_expansion_map.jsonl" --input "$FULL/macro/macro_records_test.jsonl" --kind macro --output "$FULL/macro_records_expanded.jsonl"
printf '[E_STAGE] full_macro_complete %s\n' "$(date --iso-8601=seconds)"

set +e
"$PY" scripts/analyze_experiment_e_joint.py --pairs "$INPUT/final30_pairs.jsonl" --manifest "$INPUT/attack_manifest_final_2each.jsonl" --task "$FULL/task_validation.jsonl" --micro "$FULL/micro_scores_expanded.jsonl" --macro "$FULL/macro_records_expanded.jsonl" --output-dir "$ROOT_OUT/analysis" --required-legacy 23 --family-relative-tolerance 0.01
ANALYSIS_STATUS=$?
set -e
printf '[E_STAGE] analysis_complete status=%s %s\n' "$ANALYSIS_STATUS" "$(date --iso-8601=seconds)"
exit "$ANALYSIS_STATUS"
