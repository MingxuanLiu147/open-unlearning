#!/bin/bash
# TOFU 微调脚本：按保留/全量两类方案在多种模型与 split 上运行微调与评估

# 为分布式加速器动态挑选一个当前可用端口，避免端口冲突
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

# 候选基础模型列表（1B / 3B / 8B 指令调优版本）
models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
# 每卡 batch size（两卡 × 梯度累积 8 次 => 有效 batch size 32）
per_device_train_batch_size=4 # Effective batch size 32 on two GPUs with gradent_accumulation_steps=8

# TOFU 的三组遗忘/保留/对照 split 组合
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

########################################################################################################################
########################################### RETAIN Finetuned TOFU ######################################################
########################################################################################################################
# 先在不同 retain split 上微调，再针对相应 forget/holdout split 做一次评估

for split in "${splits[@]}"; do
    # 将三元组拆分成 forget / holdout / retain 三个变量
    forget_split=$(echo $split | cut -d' ' -f1)
    holdout_split=$(echo $split | cut -d' ' -f2)
    retain_split=$(echo $split | cut -d' ' -f3)
    
    for model in "${models[@]}"; do
        # 只用 retain 子集训练，task_name 拼接模型与 retain split
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
        src/train.py experiment=finetune/tofu/default.yaml \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        data/datasets@data.train=TOFU_QA_retain \
        data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
        trainer.args.per_device_train_batch_size=4 \
        trainer.args.ddp_find_unused_parameters=true \
        trainer.args.gradient_checkpointing=true

        # 微调后立刻在对应 forget/holdout split 上评估
        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_${retain_split} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_${retain_split}
    done
done

# ########################################################################################################################
# ########################################### FULL Finetuned TOFU models #################################################
# ########################################################################################################################
# 在完整 TOFU 数据上微调一次，再对所有 forget split 逐一评估并复用 retain logs

for model in "${models[@]}"; do
    # full 模式：训练集切换到 TOFU_QA_full（full split）
    CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
    src/train.py experiment=finetune/tofu/default.yaml \
    task_name=tofu_${model}_full \
    model=${model} \
    data/datasets@data.train=TOFU_QA_full \
    data.train.TOFU_QA_full.args.hf_args.name=full \
    trainer.args.per_device_train_batch_size=4 \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true

    # 每个 forget split 都单独评估一次，并传入 retain 日志路径方便复用
    for split in "${splits[@]}"; do
        forget_split=$(echo $split | cut -d' ' -f1)
        holdout_split=$(echo $split | cut -d' ' -f2)
        retain_split=$(echo $split | cut -d' ' -f3)

        CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=eval/tofu/default.yaml \
        forget_split=${forget_split} \
        holdout_split=${holdout_split} \
        task_name=tofu_${model}_full_${forget_split} \
        model=${model} \
        model.model_args.pretrained_model_name_or_path=saves/finetune/tofu_${model}_full \
        retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
        paths.output_dir=saves/eval/tofu_${model}_full/evals_${forget_split}
    done
done