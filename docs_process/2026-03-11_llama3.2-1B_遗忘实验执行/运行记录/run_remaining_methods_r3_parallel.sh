#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/liumingxuan/open-unlearning"
PYTHON="$ROOT/.venv/bin/python"

MODEL="Llama-3.2-1B-Instruct"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-saves/finetune/tofu_Llama-3.2-1B-Instruct_full_v4}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$BASE_MODEL_PATH}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_LOGS="${RETAIN_LOGS:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"
RUN_PREFIX="${RUN_PREFIX:-20260324_r3}"

LOG_ROOT="$ROOT/docs_process/2026-03-11_llama3.2-1B_遗忘实验执行/运行记录/${RUN_PREFIX}_parallel_logs"
STATUS_TSV="$LOG_ROOT/status.tsv"

mkdir -p "$LOG_ROOT"
printf "method\tgpu\ttask_name\ttrain_exit\teval_exit\tstatus\n" > "$STATUS_TSV"

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

run_method() {
  local method="$1"
  local gpu="$2"
  local trainer="$method"
  local experiment="unlearn/tofu/default"
  local task_name="${RUN_PREFIX}_tofu_${MODEL}_${FORGET_SPLIT}_${method}"
  local output_dir="$ROOT/saves/unlearn/${task_name}"
  local eval_dir="$output_dir/evals"
  local log_file="$LOG_ROOT/${method}.log"
  local train_extra=()

  case "$method" in
    DPO)
      trainer="DPO"
      experiment="unlearn/tofu/idk"
      ;;
    PDU)
      train_extra+=(
        "trainer.method_args.retain_loss_eps=0.3"
        "trainer.method_args.alpha=100"
        "trainer.method_args.primal_dual=true"
        "trainer.method_args.dual_step_size=5"
        "trainer.method_args.dual_warmup_epochs=5"
      )
      ;;
  esac

  (
    set +e

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] method=${method} gpu=${gpu} task_name=${task_name}"

    if [ -f "$eval_dir/TOFU_SUMMARY.json" ]; then
      echo "Existing summary found at $eval_dir/TOFU_SUMMARY.json, skipping."
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$method" "$gpu" "$task_name" "skip" "skip" "already_done" >> "$STATUS_TSV"
      exit 0
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/train.py --config-name=unlearn.yaml \
      "experiment=${experiment}" \
      "trainer=${trainer}" \
      "task_name=${task_name}" \
      "model=${MODEL}" \
      "forget_split=${FORGET_SPLIT}" \
      "retain_split=${RETAIN_SPLIT}" \
      "retain_logs_path=${RETAIN_LOGS}" \
      "model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH}" \
      "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}" \
      "model.model_args.attn_implementation=eager" \
      "trainer.args.per_device_train_batch_size=1" \
      "trainer.args.per_device_eval_batch_size=1" \
      "trainer.args.gradient_accumulation_steps=4" \
      "trainer.args.eval_strategy=no" \
      "trainer.args.eval_on_start=false" \
      "trainer.args.do_eval=false" \
      "trainer.args.save_strategy=no" \
      "trainer.args.logging_steps=10" \
      "${train_extra[@]}"
    train_exit=$?
    echo "train_exit=${train_exit}"

    eval_exit="skip"
    if [ "$train_exit" -eq 0 ]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" src/eval.py --config-name=eval.yaml \
        "experiment=eval/tofu/default" \
        "task_name=${task_name}_eval" \
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
    fi

    if [ "$train_exit" -eq 0 ] && [ "$eval_exit" = "0" ]; then
      status="done"
      exit_code=0
    elif [ "$train_exit" -ne 0 ]; then
      status="train_failed"
      exit_code=1
    else
      status="eval_failed"
      exit_code=1
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$method" "$gpu" "$task_name" "$train_exit" "$eval_exit" "$status" >> "$STATUS_TSV"
    exit "$exit_code"
  ) >"$log_file" 2>&1 &
  LAST_PID="$!"
}

cd "$ROOT"

echo "Batch start: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Run prefix: $RUN_PREFIX"
echo "Base model: $BASE_MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Logs: $LOG_ROOT"

declare -a PIDS=()
LAST_PID=""

for i in "${!METHODS[@]}"; do
  run_method "${METHODS[$i]}" "${GPUS[$i]}"
  PIDS+=("$LAST_PID")
  echo "Spawned ${METHODS[$i]} on GPU ${GPUS[$i]} with pid $LAST_PID"
done

overall_exit=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    overall_exit=1
  fi
done

echo "Batch finish: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Status table: $STATUS_TSV"
exit "$overall_exit"
