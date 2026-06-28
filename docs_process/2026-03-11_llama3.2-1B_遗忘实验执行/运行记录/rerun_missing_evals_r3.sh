#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/liumingxuan/open-unlearning"
PYTHON="$ROOT/.venv/bin/python"

MODEL="Llama-3.2-1B-Instruct"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RUN_PREFIX="${RUN_PREFIX:-20260324_r3}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"

LOG_ROOT="$ROOT/docs_process/2026-03-11_llama3.2-1B_遗忘实验执行/运行记录/${RUN_PREFIX}_eval_retry_logs"
STATUS_TSV="$LOG_ROOT/status.tsv"

mkdir -p "$LOG_ROOT"
printf "method\tgpu\ttask_name\teval_exit\tstatus\n" > "$STATUS_TSV"

METHODS=(
  "GradAscent"
  "NPO"
  "CEU"
  "DPO"
  "RMU"
  "UNDIAL"
  "SatImp"
  "WGA"
  "PDU"
)

if [ -n "${GPU_LIST:-}" ]; then
  read -r -a GPUS <<< "$GPU_LIST"
else
  GPUS=(0 1 2 3 4 5 6 7 8)
fi

if [ "${#GPUS[@]}" -lt "${#METHODS[@]}" ]; then
  echo "Need at least ${#METHODS[@]} GPUs, but got ${#GPUS[@]} entries in GPU_LIST." >&2
  exit 1
fi

run_eval() {
  local method="$1"
  local gpu="$2"
  local task_name="${RUN_PREFIX}_tofu_${MODEL}_${FORGET_SPLIT}_${method}"
  local output_dir="$ROOT/saves/unlearn/${task_name}"
  local eval_dir="$output_dir/evals"
  local log_file="$LOG_ROOT/${method}.log"

  (
    set +e
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] method=${method} gpu=${gpu} task_name=${task_name}"

    if [ ! -f "$output_dir/model.safetensors" ]; then
      echo "Model not found at $output_dir/model.safetensors, skip."
      printf "%s\t%s\t%s\t%s\t%s\n" \
        "$method" "$gpu" "$task_name" "skip" "model_missing" >> "$STATUS_TSV"
      exit 0
    fi

    if [ -f "$eval_dir/TOFU_SUMMARY.json" ]; then
      echo "Existing summary found at $eval_dir/TOFU_SUMMARY.json, skip."
      printf "%s\t%s\t%s\t%s\t%s\n" \
        "$method" "$gpu" "$task_name" "skip" "already_done" >> "$STATUS_TSV"
      exit 0
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/eval.py --config-name=eval.yaml \
      "experiment=eval/tofu/default" \
      "task_name=${task_name}_eval_retry" \
      "model=${MODEL}" \
      "forget_split=${FORGET_SPLIT}" \
      "holdout_split=${HOLDOUT_SPLIT}" \
      "retain_logs_path=${RETAIN_LOGS}" \
      "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}" \
      "model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task_name}" \
      "model.model_args.attn_implementation=eager" \
      "paths.output_dir=saves/unlearn/${task_name}/evals"
    eval_exit=$?
    echo "eval_exit=${eval_exit}"

    if [ "$eval_exit" -eq 0 ]; then
      status="done"
      exit_code=0
    else
      status="eval_failed"
      exit_code=1
    fi

    printf "%s\t%s\t%s\t%s\t%s\n" \
      "$method" "$gpu" "$task_name" "$eval_exit" "$status" >> "$STATUS_TSV"
    exit "$exit_code"
  ) >"$log_file" 2>&1 &
  LAST_PID="$!"
}

cd "$ROOT"

echo "Eval retry start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Run prefix: $RUN_PREFIX"
echo "Logs: $LOG_ROOT"

declare -a PIDS=()
LAST_PID=""

for i in "${!METHODS[@]}"; do
  run_eval "${METHODS[$i]}" "${GPUS[$i]}"
  PIDS+=("$LAST_PID")
  echo "Spawned eval ${METHODS[$i]} on GPU ${GPUS[$i]} with pid $LAST_PID"
done

overall_exit=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    overall_exit=1
  fi
done

echo "Eval retry finish: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Status table: $STATUS_TSV"
exit "$overall_exit"
