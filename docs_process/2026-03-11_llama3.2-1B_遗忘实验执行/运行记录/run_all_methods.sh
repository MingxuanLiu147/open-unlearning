#!/usr/bin/env bash
set -u

ROOT="/home/liumingxuan/open-unlearning"
PYTHON="$ROOT/.venv/bin/python"
ACCELERATE_BIN="$ROOT/.venv/bin/accelerate"
GPU_IDS="${GPU_IDS:-0,1}"
EVAL_GPU_ID="${EVAL_GPU_ID:-0}"
ACCEL_CONFIG="configs/accelerate/default_config.yaml"
MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TOKENIZER_PATH="saves/unlearn/my_npo_debug"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"
HOLDOUT_SPLIT="holdout10"
RETAIN_LOGS="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
MASTER_PORT=$("$PYTHON" -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

cd "$ROOT" || exit 1

run_method() {
  local method="$1"
  local trainer="$method"
  local experiment="unlearn/tofu/default"
  local task_name="20260311_r2_tofu_${MODEL}_${FORGET_SPLIT}_${method}"
  local result_summary="$ROOT/saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json"
  local train_extra=()

  case "$method" in
    IDKDPO)
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

  if [ -f "$result_summary" ]; then
    echo "=== [$method] existing result found: $result_summary ==="
    echo "=== [$method] skip rerun ==="
    return 0
  fi

  echo "=== [$method] train start: $task_name ==="
  CUDA_VISIBLE_DEVICES="$GPU_IDS" "$ACCELERATE_BIN" launch --config_file "$ACCEL_CONFIG" --main_process_port "$MASTER_PORT" \
    "$PYTHON" src/train.py --config-name=unlearn.yaml \
    "experiment=${experiment}" \
    "trainer=${trainer}" \
    "task_name=${task_name}" \
    "model=${MODEL}" \
    "forget_split=${FORGET_SPLIT}" \
    "retain_split=${RETAIN_SPLIT}" \
    "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}" \
    "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}" \
    "retain_logs_path=${RETAIN_LOGS}" \
    "trainer.args.per_device_train_batch_size=4" \
    "trainer.args.per_device_eval_batch_size=4" \
    "trainer.args.gradient_accumulation_steps=4" \
    "trainer.args.ddp_find_unused_parameters=true" \
    "trainer.args.gradient_checkpointing=true" \
    "trainer.args.eval_strategy=no" \
    "trainer.args.eval_on_start=false" \
    "trainer.args.do_eval=false" \
    "trainer.args.save_strategy=no" \
    "trainer.args.logging_steps=10" \
    "${train_extra[@]}"
  local train_code=$?
  echo "=== [$method] train exit code: $train_code ==="

  if [ "$train_code" -ne 0 ]; then
    echo "=== [$method] train failed, skip eval ==="
    return 0
  fi

  echo "=== [$method] eval start: $task_name ==="
  CUDA_VISIBLE_DEVICES="$EVAL_GPU_ID" "$PYTHON" src/eval.py --config-name=eval.yaml \
    "experiment=eval/tofu/default" \
    "forget_split=${FORGET_SPLIT}" \
    "holdout_split=${HOLDOUT_SPLIT}" \
    "model=${MODEL}" \
    "task_name=${task_name}_eval" \
    "model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name}" \
    "model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task_name}" \
    "retain_logs_path=${RETAIN_LOGS}" \
    "paths.output_dir=saves/unlearn/${task_name}/evals"
  local eval_code=$?
  echo "=== [$method] eval exit code: $eval_code ==="
}

methods=(
  # 1) Environment sanity checks: methods already confirmed to run end-to-end here.
  "GradDiff"
  "SimNPO"

  # 2) Newly interface-fixed methods that do not depend on a reference model.
  "GradAscent"
  "CEU"

  # 3) Reference-model methods on the standard TOFU forget/retain setup.
  "NPO"
  "UNDIAL"
  "WGA"
  "SatImp"
  "PDU"
  "RMU"

  # 4) Preference-pair method special case.
  # Plain DPO on unlearn/tofu/default expects alternate pairs and is not suitable
  # for the default TOFU QA dataset; use the repo's IDK DPO route explicitly.
  "IDKDPO"
)

echo "Batch run started at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Train GPUs: $GPU_IDS"
echo "Eval GPU: $EVAL_GPU_ID"
echo "Master port: $MASTER_PORT"
echo "Model: $MODEL"

for method in "${methods[@]}"; do
  run_method "$method"
done

echo "Batch run finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
