#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/llm/stage2_pilot_execution_20260905
PYTHON=/root/autodl-tmp/envs/h9-stage1/bin/python
LOG_DIR="$ROOT/logs"
STATE="$LOG_DIR/stage2_autonomous_state.txt"
ARCHIVE=/root/autodl-tmp/llm/stage2_checkpoint_20260906.tar.gz

mkdir -p "$LOG_DIR"
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

finish_and_shutdown() {
    exit_code=$?
    trap - EXIT
    date -Is > "$STATE"
    printf 'exit_code=%s\n' "$exit_code" >> "$STATE"
    printf 'calibration_status=' >> "$STATE"
    if [[ -f "$ROOT/results/stage2_attack_utility_calibration_v2_20260906/STATUS.json" ]]; then
        "$PYTHON" -c "import json; print(json.load(open('$ROOT/results/stage2_attack_utility_calibration_v2_20260906/STATUS.json'))['phase'])" >> "$STATE" 2>/dev/null || printf 'UNREADABLE\n' >> "$STATE"
    else
        printf 'ABSENT\n' >> "$STATE"
    fi
    printf 'behavior_status=' >> "$STATE"
    if [[ -f "$ROOT/results/stage2_behavior_confirmation_26_v1_20260906/evaluation/STATUS.json" ]]; then
        "$PYTHON" -c "import json; print(json.load(open('$ROOT/results/stage2_behavior_confirmation_26_v1_20260906/evaluation/STATUS.json'))['phase'])" >> "$STATE" 2>/dev/null || printf 'UNREADABLE\n' >> "$STATE"
    else
        printf 'NOT_RUN\n' >> "$STATE"
    fi
    sync
    tar -czf "${ARCHIVE}.tmp" -C "$ROOT" \
        scripts/stage2_attack_calibration.py \
        scripts/stage2_behavior_confirmation_26.py \
        scripts/run_stage2_calibration_then_behavior.sh \
        logs/stage2_attack_calibration_train_v2.log \
        logs/stage2_autonomous.log \
        logs/stage2_autonomous_state.txt \
        results/stage2_attack_utility_calibration_v2_20260906 \
        results/stage2_behavior_confirmation_26_v1_20260906 2>/dev/null || true
    if [[ -s "${ARCHIVE}.tmp" ]]; then
        mv "${ARCHIVE}.tmp" "$ARCHIVE"
    fi
    sync
    # Short grace period lets the controlling client retrieve final reports.
    for _ in 1 2 3 4; do sleep 30; done
    shutdown -h now
    exit "$exit_code"
}
trap finish_and_shutdown EXIT

cd "$ROOT"
if [[ ! -f results/stage2_attack_utility_calibration_v2_20260906/CALIBRATION_REPORT.json ]]; then
    printf '%s waiting_for_lora_training\n' "$(date -Is)" > "$STATE"
    while pgrep -f '[p]ython scripts/stage2_attack_calibration.py train' >/dev/null; do
        sleep 15
    done

    if [[ ! -f results/stage2_attack_utility_calibration_v2_20260906/STATUS.json ]]; then
        printf '%s missing_calibration_status\n' "$(date -Is)" >&2
        exit 30
    fi
    train_phase=$("$PYTHON" -c "import json; print(json.load(open('results/stage2_attack_utility_calibration_v2_20260906/STATUS.json'))['phase'])")
    if [[ "$train_phase" != LORA_TRAINING_COMPLETE ]]; then
        printf '%s unexpected_training_phase=%s\n' "$(date -Is)" "$train_phase" >&2
        exit 31
    fi

    printf '%s running_attack_utility\n' "$(date -Is)" > "$STATE"
    "$PYTHON" scripts/stage2_attack_calibration.py utility
    printf '%s building_calibration_report\n' "$(date -Is)" > "$STATE"
    "$PYTHON" scripts/stage2_attack_calibration.py report
fi

can_proceed=$("$PYTHON" -c "import json; print(str(json.load(open('results/stage2_attack_utility_calibration_v2_20260906/CALIBRATION_REPORT.json'))['can_proceed_stage3']).lower())")
if [[ "$can_proceed" != true ]]; then
    printf '%s calibration_blocked_stage3\n' "$(date -Is)" > "$STATE"
    exit 0
fi

printf '%s running_26_prompt_behavior_confirmation\n' "$(date -Is)" > "$STATE"
"$PYTHON" scripts/stage2_behavior_confirmation_26.py all
printf '%s all_requested_experiments_complete\n' "$(date -Is)" > "$STATE"
