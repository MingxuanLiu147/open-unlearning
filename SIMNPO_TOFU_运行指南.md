# SimNPO on TOFU 数据集运行指南

本指南详细说明如何使用 **SimNPO 算法**在 **TOFU 数据集**上运行 unlearning，使用 **Llama-3.2-1B-Instruct** 模型，**forget=10%** 的配置。

---

## 🚀 快速开始

### 方法 1: 使用提供的脚本（推荐）

```bash
# 直接运行
bash run_simnpo_tofu_1b.sh
```

### 方法 2: 使用命令行（更灵活）

```bash
CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=tofu_SimNPO_forget10_v1 \
    model.model_args.attn_implementation=eager \
    trainer.args.learning_rate=5e-5 \
    trainer.args.num_train_epochs=20 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=8 \
    trainer.args.eval_strategy=steps \
    +trainer.args.eval_steps=100 \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=100
```

---

## 📋 完整 Pipeline（对齐论文）

### 步骤 1: 准备预训练模型

**使用的模型：** `open-unlearning/tofu_Llama-3.2-1B-Instruct_full`

这是在完整 TOFU 数据集上微调过的 LLaMA-3.2-1B 模型。

**如果模型不存在，您需要先微调基础模型：**

```bash
# 在完整 TOFU 数据集上微调 LLaMA-3.2-1B（可选）
CUDA_VISIBLE_DEVICES=4 python src/train.py \
    --config-name=train.yaml \
    experiment=finetune/tofu/default \
    model=Llama-3.2-1B-Instruct \
    task_name=tofu_Llama-3.2-1B-Instruct_full
```

### 步骤 2: 运行 SimNPO Unlearning

```bash
# 设置 GPU
export CUDA_VISIBLE_DEVICES=4

# 运行训练
HYDRA_FULL_ERROR=1 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=tofu_SimNPO_forget10_v1 \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
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

**训练时间估计：**
- 单卡 A100/V100：约 2-3 小时
- 单卡 3090/4090：约 3-5 小时

**显存占用：**
- Llama-3.2-1B + batch_size=2：约 8-12 GB

### 步骤 3: 评估 Unlearning 效果

```bash
CUDA_VISIBLE_DEVICES=4 python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model=Llama-3.2-1B-Instruct \
    task_name=tofu_SimNPO_forget10_v1 \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_SimNPO_forget10_v1 \
    paths.output_dir=saves/unlearn/tofu_SimNPO_forget10_v1/evals
```

**评估时间：** 约 10-20 分钟

### 步骤 4: 查看结果

```bash
# 查看评估结果
cat saves/unlearn/tofu_SimNPO_forget10_v1/evals/TOFU_EVAL.json

# 查看训练日志
tail -f saves/unlearn/tofu_SimNPO_forget10_v1/trainer_log.txt
```

---

## ⚙️ 关键参数说明（对齐论文）

### 1. SimNPO 算法参数

这些参数来自 [SimNPO 论文](https://github.com/OPTML-Group/Unlearn-Simple/blob/main/TOFU/config/forget.yaml)：

| 参数 | 默认值 | 论文设置 | 说明 |
|------|--------|----------|------|
| `gamma` | 0.125 | 0.125 | Forget 损失权重（论文中称为 `npo_coeff`）|
| `alpha` | 1.0 | 1.0 | Retain 损失权重 |
| `beta` | 4.5 | 4.5 | 温度参数 |
| `delta` | 0.0 | 0.0 | NLL 偏移量（论文中称为 `gamma`）|
| `retain_loss_type` | NLL | NLL | Retain 损失类型（NLL 或 KL）|

**修改方式：**
```bash
trainer.method_args.gamma=0.125
trainer.method_args.alpha=1.0
trainer.method_args.beta=4.5
trainer.method_args.delta=0.0
```

### 2. 训练超参数

| 参数 | 推荐值（1B） | 论文设置 | 说明 |
|------|--------------|----------|------|
| `learning_rate` | 5e-5 | 5e-5 | 学习率 |
| `num_train_epochs` | 20 | 10-20 | 训练轮数 |
| `per_device_train_batch_size` | 2 | 2-4 | 每卡 batch size |
| `gradient_accumulation_steps` | 8 | 4-8 | 梯度累积步数 |
| `warmup_epochs` | 1.0 | 1.0 | Warmup 轮数 |
| `weight_decay` | 0.01 | 0.01 | 权重衰减 |

**有效 Batch Size = per_device_batch_size × gradient_accumulation_steps × num_gpus**

示例：
- 单卡：2 × 8 × 1 = 16
- 双卡：2 × 4 × 2 = 16

### 3. 数据集配置

| 参数 | forget10 设置 | 说明 |
|------|---------------|------|
| `forget_split` | forget10 | 10% 作者数据（约 176 条样本）|
| `retain_split` | retain90 | 90% 作者数据（约 1584 条样本）|
| `holdout_split` | holdout10 | 10% holdout 数据（用于评估）|
| `anchor` | forget | 锚定在 forget 数据集 |

**其他可用配置：**
```bash
# 1% forget
forget_split=forget01
retain_split=retain99

# 5% forget
forget_split=forget05
retain_split=retain95
```

---

## 📊 评估指标说明

### 1. Forget Quality (FQ) - 遗忘质量

**目标：** 越低越好（说明模型成功"忘记"了数据）

**指标：**
- `Truth Ratio`：模型输出真实答案的比例
- `Probability`：模型在 forget 数据上的输出概率
- `ROUGE-L`：forget 数据上的 ROUGE 分数

**论文基准（forget10）：**
- Truth Ratio: < 0.05（应该接近 0）
- Probability: < 0.1（应该明显下降）

### 2. Model Utility (MU) - 模型效用

**目标：** 越高越好（说明模型在 retain 数据上保持性能）

**指标：**
- `ROUGE-L on Retain Set`：retain 数据上的 ROUGE 分数
- `Probability on Retain Set`：retain 数据上的输出概率

**论文基准（retain90）：**
- ROUGE-L: > 0.40（应该接近原始模型）
- Probability: > 0.50（应该保持高）

### 3. 综合评估

**均衡指标：**
```
Score = (1 - FQ) × MU
```

**论文中 SimNPO 的典型表现（forget10）：**
- FQ: ~0.05-0.10
- MU: ~0.85-0.90
- Score: ~0.80-0.85

---

## 🔧 常见配置调整

### 1. 更强的遗忘

```bash
# 增大 forget 损失权重
trainer.method_args.gamma=0.5

# 增大温度参数（更陡峭的梯度）
trainer.method_args.beta=10.0

# 更多训练轮数
trainer.args.num_train_epochs=30
```

### 2. 更好的性能保持

```bash
# 增大 retain 损失权重
trainer.method_args.alpha=2.0

# 减小 forget 损失权重
trainer.method_args.gamma=0.05

# 使用 KL 散度（更稳定）
trainer.method_args.retain_loss_type=KL
```

### 3. 显存优化（如果显存不足）

```bash
# 减小 batch size
trainer.args.per_device_train_batch_size=1

# 增加梯度累积
trainer.args.gradient_accumulation_steps=16

# 启用梯度检查点
trainer.args.gradient_checkpointing=true

# 使用 DeepSpeed（推荐）
accelerate launch \
    --config_file configs/accelerate/zero_stage3_offload_config.json \
    src/train.py ...
```

### 4. 多 GPU 训练

```bash
# 使用 accelerate（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    ...
```

---

## 📈 与其他方法对比

### 运行多个方法进行对比

```bash
#!/bin/bash

methods=("SimNPO" "GradAscent" "GradDiff" "NPO")

for method in "${methods[@]}"; do
    echo "Running ${method}..."
    
    CUDA_VISIBLE_DEVICES=4 python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=${method} \
        forget_split=forget10 \
        retain_split=retain90 \
        task_name=tofu_${method}_forget10 \
        trainer.args.learning_rate=5e-5 \
        trainer.args.num_train_epochs=20
    
    # 评估
    CUDA_VISIBLE_DEVICES=4 python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default \
        forget_split=forget10 \
        task_name=tofu_${method}_forget10 \
        model.model_args.pretrained_model_name_or_path=saves/unlearn/tofu_${method}_forget10
done
```

### 预期性能对比（forget10，论文数据）

| 方法 | Forget Quality (FQ) | Model Utility (MU) | 综合得分 |
|------|---------------------|-------------------|----------|
| **SimNPO** | **0.08** | **0.87** | **0.80** |
| GradAscent | 0.15 | 0.82 | 0.70 |
| GradDiff | 0.12 | 0.84 | 0.74 |
| NPO | 0.10 | 0.85 | 0.77 |

> **结论：** SimNPO 在遗忘质量和性能保持之间达到最佳平衡。

---

## 🐛 常见问题排查

### Q1: 模型无法加载

**错误信息：**
```
OSError: open-unlearning/tofu_Llama-3.2-1B-Instruct_full does not appear to be a model identifier
```

**解决方案：**
1. 检查模型是否存在于 HuggingFace Hub
2. 或使用本地路径：
   ```bash
   model.model_args.pretrained_model_name_or_path=/path/to/your/model
   ```

### Q2: 显存不足 (OOM)

**错误信息：**
```
CUDA out of memory
```

**解决方案：**
```bash
# 方案 1: 减小 batch size
trainer.args.per_device_train_batch_size=1
trainer.args.gradient_accumulation_steps=16

# 方案 2: 启用梯度检查点
trainer.args.gradient_checkpointing=true

# 方案 3: 使用 DeepSpeed ZeRO-3
accelerate launch \
    --config_file configs/accelerate/zero_stage3_offload_config.json \
    src/train.py ...
```

### Q3: 训练损失不下降

**可能原因：**
- 学习率太小
- gamma/alpha 权重设置不当
- 数据集问题

**解决方案：**
```bash
# 增大学习率
trainer.args.learning_rate=1e-4

# 调整权重
trainer.method_args.gamma=0.25
trainer.method_args.alpha=1.0

# 检查数据加载
+trainer.args.logging_steps=1
```

### Q4: 评估时出错

**错误信息：**
```
KeyError: 'retain_logs_path'
```

**解决方案：**
```bash
# 确保提供参考模型的评估日志
retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json

# 如果不存在，先生成参考评估
python src/eval.py \
    experiment=eval/tofu/default \
    model.model_args.pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full \
    paths.output_dir=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90
```

---

## 📚 参考资料

### 论文和代码

1. **SimNPO 论文：** [Unlearn-Simple](https://github.com/OPTML-Group/Unlearn-Simple)
2. **TOFU 数据集：** [TOFU: A Task of Fictitious Unlearning](https://arxiv.org/abs/2401.06121)
3. **本项目文档：**
   - `UNLEARNING_GUIDE_CN.md` - 完整的 unlearning 指南
   - `代码注释总结.md` - 代码注释总结

### HuggingFace 资源

- **模型：** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **数据集：** [locuslab/TOFU](https://huggingface.co/datasets/locuslab/TOFU)

---

## ✅ 检查清单

运行前确认：

- [ ] GPU 可用且显存充足（至少 12GB）
- [ ] 预训练模型已准备好
- [ ] 数据集可访问（HuggingFace 或本地）
- [ ] 配置文件存在（`configs/` 目录）
- [ ] Python 环境已安装所有依赖

运行后检查：

- [ ] 训练损失正常下降
- [ ] 评估指标符合预期（FQ 低，MU 高）
- [ ] 模型已保存到 `saves/unlearn/` 目录
- [ ] 评估结果已保存到 `saves/unlearn/*/evals/` 目录

---

**最后更新：** 2026-02-02  
**版本：** v1.0  
**维护者：** Open-Unlearning Team
