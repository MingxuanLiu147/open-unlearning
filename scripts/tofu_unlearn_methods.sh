#!/bin/bash

# TOFU Unlearning Commands for Multiple Methods
# Model: Llama-3.2-1B-Instruct
# Split: forget10

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

model=Llama-3.2-1B-Instruct
forget_split=forget10
retain_split=retain90
holdout_split=holdout10
model_path=open-unlearning/tofu_${model}_full

per_device_train_batch_size=4
gradient_accumulation_steps=4

########################################################################################################################
########################################### 1. SimNPO ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_SimNPO
echo "=== SimNPO Unlearning ==="
echo "Task: ${task_name}"

# Unlearn
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=SimNPO \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 2. AltPO ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_AltPO
echo "=== AltPO Unlearning ==="
echo "Task: ${task_name}"

# Unlearn (AltPO uses DPO trainer with alternate data)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=DPO \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true \
data.forget.TOFU_QA_forget.handler=QAwithAlternateDataset \
~data.forget.TOFU_QA_forget.args.hf_args.name \
data.forget.TOFU_QA_forget.args.hf_args.path=json \
+data.forget.TOFU_QA_forget.args.hf_args.data_files=community/methods/AltPO/data/tofu_${model}_full/${forget_split}/alt5_seed_0.json \
data.forget.TOFU_QA_forget.args.hf_args.split=train \
+data.forget.TOFU_QA_forget.args.alternate_key=alternate \
+data.forget.TOFU_QA_forget.args.return_original=True

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 3. GradAscent (GA) ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_GradAscent
echo "=== GradAscent Unlearning ==="
echo "Task: ${task_name}"

# Unlearn
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=GradAscent \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 4. GradDiff (GD) ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_GradDiff
echo "=== GradDiff Unlearning ==="
echo "Task: ${task_name}"

# Unlearn
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=GradDiff \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 5. NPO ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_NPO
echo "=== NPO Unlearning ==="
echo "Task: ${task_name}"

# Unlearn
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=NPO \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 6. DPO ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_DPO
echo "=== DPO Unlearning ==="
echo "Task: ${task_name}"

# Unlearn
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=DPO \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 7. IDKDPO ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_IDKDPO
echo "=== IDKDPO Unlearning ==="
echo "Task: ${task_name}"

# Unlearn (IDKDPO uses DPO trainer with idk experiment config)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/idk \
trainer=DPO \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

########################################################################################################################
########################################### 8. WGA ########################################################
########################################################################################################################

task_name=tofu_${model}_${forget_split}_WGA
echo "=== WGA Unlearning ==="
echo "Task: ${task_name}"

# Unlearn
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
python src/train.py --config-name=unlearn.yaml \
experiment=unlearn/tofu/default \
trainer=WGA \
task_name=${task_name} \
model=${model} \
forget_split=${forget_split} \
retain_split=${retain_split} \
model.model_args.pretrained_model_name_or_path=${model_path} \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
trainer.args.ddp_find_unused_parameters=true \
trainer.args.gradient_checkpointing=true

# Eval
CUDA_VISIBLE_DEVICES=0 python src/eval.py --config-name=eval.yaml \
experiment=eval/tofu/default \
forget_split=${forget_split} \
holdout_split=${holdout_split} \
model=${model} \
task_name=${task_name} \
model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
paths.output_dir=saves/unlearn/${task_name}/evals \
retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json

echo "=== All unlearning tasks completed ==="
