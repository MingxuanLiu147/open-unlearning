# SimNPO + TOFU 快速启动

## 🚀 一键运行（推荐）

```bash
cd /home/liumingxuan/open-unlearning
bash run_simnpo_tofu_1b.sh
```

---

## ⚡ 命令行运行

### 基础命令（单卡）

```bash
CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=my_simnpo_run \
    model.model_args.attn_implementation=eager \
    trainer.args.learning_rate=5e-5 \
    trainer.args.num_train_epochs=20 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=8 \
    trainer.args.eval_strategy=steps \
    +trainer.args.eval_steps=100 \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=100 \
    trainer.args.logging_steps=10
```

### 多卡并行（推荐用于快速训练）

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=my_simnpo_run \
    trainer.args.learning_rate=5e-5 \
    trainer.args.num_train_epochs=20 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=4
```

---

## 📊 评估模型

```bash
CUDA_VISIBLE_DEVICES=4 python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model=Llama-3.2-1B-Instruct \
    task_name=my_simnpo_run \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/my_simnpo_run \
    paths.output_dir=saves/unlearn/my_simnpo_run/evals
```

---

## 🔧 关键参数速查

### 改变 forget 比例

```bash
# 1% forget
forget_split=forget01 retain_split=retain99

# 5% forget
forget_split=forget05 retain_split=retain95

# 10% forget (默认)
forget_split=forget10 retain_split=retain90
```

### 调整遗忘强度

```bash
# 更强的遗忘
trainer.method_args.gamma=0.5
trainer.method_args.beta=10.0

# 更温和的遗忘（保留更多性能）
trainer.method_args.gamma=0.05
trainer.method_args.alpha=2.0
```

### 调整训练速度

```bash
# 更快（但可能不稳定）
trainer.args.learning_rate=1e-4
trainer.args.num_train_epochs=10

# 更慢但更稳定
trainer.args.learning_rate=1e-5
trainer.args.num_train_epochs=30
```

### 显存优化

```bash
# 显存不足？试试这些
trainer.args.per_device_train_batch_size=1
trainer.args.gradient_accumulation_steps=16
trainer.args.gradient_checkpointing=true
```

---

## 📈 查看结果

```bash
# 训练日志
tail -f saves/unlearn/my_simnpo_run/trainer_log.txt

# 评估结果
cat saves/unlearn/my_simnpo_run/evals/TOFU_EVAL.json

# 模型检查点
ls saves/unlearn/my_simnpo_run/checkpoint-*
```

---

## 🎯 预期结果（forget10）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Forget Quality** | < 0.10 | 遗忘越彻底越好 |
| **Model Utility** | > 0.85 | 性能保持越高越好 |
| **综合得分** | ~0.80 | 平衡指标 |

---

## 🆘 快速故障排除

| 问题 | 解决方案 |
|------|----------|
| 显存不足 | `trainer.args.per_device_train_batch_size=1` |
| 训练太慢 | 使用多 GPU：`CUDA_VISIBLE_DEVICES=0,1` + `accelerate launch` |
| 损失不下降 | 增大学习率：`trainer.args.learning_rate=1e-4` |
| 模型未找到 | 检查路径：`model.model_args.pretrained_model_name_or_path` |

---

## 📚 详细文档

- **完整指南：** `SIMNPO_TOFU_运行指南.md`
- **算法详解：** `UNLEARNING_GUIDE_CN.md`
- **代码注释：** 查看 `src/train.py` 和 `src/trainer/unlearn/simnpo.py`

---

**提示：** 首次运行建议使用 `forget01`（1% forget）快速测试，确认环境无误后再运行完整实验。
