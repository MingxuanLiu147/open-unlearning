#!/bin/bash

###############################################################################
# SimNPO on TOFU Dataset with Llama-3.2-1B-Instruct
# Forget = 10%, Retain = 90%
###############################################################################

# 说明：
# 1. 使用预训练好的 1B LLaMA 模型（在完整 TOFU 数据集上微调过）
# 2. Forget 10% 的作者数据（约 176 条样本）
# 3. Retain 90% 的作者数据（约 1584 条样本）
# 4. 使用 SimNPO 算法进行 unlearning

###############################################################################
# 配置参数（对齐论文设置）
###############################################################################

# GPU 设置
export CUDA_VISIBLE_DEVICES=4  # 使用第 4 号 GPU（根据您的硬件调整）

# 模型配置
MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"  # HuggingFace 预训练模型路径

# 数据集配置
FORGET_SPLIT="forget10"   # 10% 作者数据（需要遗忘）
RETAIN_SPLIT="retain90"   # 90% 作者数据（需要保持）
HOLDOUT_SPLIT="holdout10" # 10% holdout 数据（用于评估）

# 训练器配置
TRAINER="SimNPO"
EXPERIMENT="unlearn/tofu/default"

# 任务名称（用于保存模型和日志）
TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_v1"

# 超参数（对齐论文）
LEARNING_RATE=5e-5        # SimNPO 推荐学习率
NUM_EPOCHS=20             # 训练轮数（论文中通常 10-20 epoch）
BATCH_SIZE=2              # 每卡 batch size（1B 模型在单卡上可用 2-4）
GRAD_ACCUM=8              # 梯度累积步数（有效 batch size = 2 × 8 = 16）
WARMUP_EPOCHS=1.0         # Warmup 轮数

# SimNPO 算法参数（来自论文）
GAMMA=0.125               # Forget 损失权重（npo_coeff）
ALPHA=1.0                 # Retain 损失权重
BETA=4.5                  # 温度参数
DELTA=0.0                 # NLL 偏移量

# 评估和保存策略
EVAL_STEPS=100            # 每 100 步评估一次
SAVE_STEPS=100            # 每 100 步保存一次
LOGGING_STEPS=10          # 每 10 步记录日志

# 注意力实现
ATTN_IMPL="eager"         # 使用标准注意力（或 "flash_attention_2" 如果支持）

# 参考模型日志路径（用于评估）
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

###############################################################################
# 步骤 1: 训练 SimNPO（Unlearning）
###############################################################################

echo "=========================================="
echo "开始 SimNPO Unlearning 训练"
echo "=========================================="
echo "模型: ${MODEL}"
echo "Forget Split: ${FORGET_SPLIT}"
echo "Retain Split: ${RETAIN_SPLIT}"
echo "任务名称: ${TASK_NAME}"
echo "=========================================="

HYDRA_FULL_ERROR=1 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=${EXPERIMENT} \
    trainer=${TRAINER} \
    task_name=${TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
    model.model_args.attn_implementation=${ATTN_IMPL} \
    retain_logs_path=${RETAIN_LOGS_PATH} \
    trainer.args.learning_rate=${LEARNING_RATE} \
    trainer.args.num_train_epochs=${NUM_EPOCHS} \
    trainer.args.per_device_train_batch_size=${BATCH_SIZE} \
    trainer.args.gradient_accumulation_steps=${GRAD_ACCUM} \
    trainer.args.warmup_epochs=${WARMUP_EPOCHS} \
    trainer.args.eval_strategy=steps \
    +trainer.args.eval_steps=${EVAL_STEPS} \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=${SAVE_STEPS} \
    trainer.args.logging_steps=${LOGGING_STEPS} \
    trainer.args.save_total_limit=3 \
    trainer.args.load_best_model_at_end=true \
    trainer.args.metric_for_best_model=eval_loss \
    trainer.method_args.gamma=${GAMMA} \
    trainer.method_args.alpha=${ALPHA} \
    trainer.method_args.beta=${BETA} \
    trainer.method_args.delta=${DELTA}

echo "=========================================="
echo "SimNPO 训练完成！"
echo "模型保存在: saves/unlearn/${TASK_NAME}"
echo "=========================================="

###############################################################################
# 步骤 2: 评估 Unlearning 效果
###############################################################################

echo ""
echo "=========================================="
echo "开始评估 Unlearning 效果"
echo "=========================================="

python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
    paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
    retain_logs_path=${RETAIN_LOGS_PATH}

echo "=========================================="
echo "评估完成！"
echo "评估结果保存在: saves/unlearn/${TASK_NAME}/evals"
echo "=========================================="

###############################################################################
# 步骤 3: 查看评估指标
###############################################################################

echo ""
echo "=========================================="
echo "关键评估指标说明："
echo "=========================================="
echo "1. Forget Quality (FQ): forget set 上的性能下降（越低越好）"
echo "   - Model Utility: 模型在 forget 数据上的准确率/ROUGE（应该下降）"
echo "   - Truth Ratio: 模型输出真实答案的比例（应该下降）"
echo ""
echo "2. Model Utility (MU): retain set 上的性能保持（越高越好）"
echo "   - ROUGE Score: retain 数据上的 ROUGE 分数（应该保持）"
echo "   - Probability: retain 数据上的输出概率（应该保持）"
echo ""
echo "3. 综合指标: FQ × MU（平衡遗忘和保持）"
echo "=========================================="

echo ""
echo "查看详细结果："
echo "cat saves/unlearn/${TASK_NAME}/evals/TOFU_EVAL.json"
