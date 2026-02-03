# Unlearning 算法核心逻辑详解

> 本文档详细解释如何使用本框架完成一次 unlearning 训练，包括代码执行流程、参数含义以及如何自定义配置。

## 目录

1. [什么是 Unlearning](#1-什么是-unlearning)
2. [整体架构](#2-整体架构)
3. [执行流程详解](#3-执行流程详解)
4. [命令行参数解析](#4-命令行参数解析)
5. [SimNPO 算法原理](#5-simnpo-算法原理)
6. [如何修改配置](#6-如何修改配置)
7. [常见问题](#7-常见问题)

---

## 1. 什么是 Unlearning

**机器遗忘 (Machine Unlearning)** 是指让已经训练好的模型"忘记"特定的训练数据，同时保持在其他数据上的性能。

### 应用场景

- **隐私保护**：用户要求删除其个人数据（GDPR 合规）
- **数据纠错**：训练数据中存在错误或有害信息
- **版权保护**：移除受版权保护的内容
- **模型安全**：消除模型对敏感信息的记忆

### 核心挑战

1. **选择性遗忘**：只忘记特定数据（forget set），保留其他知识（retain set）
2. **效率要求**：不能完全重新训练（成本太高）
3. **性能保持**：遗忘后模型在 retain set 上的性能不能下降太多

---

## 2. 整体架构

本框架基于 **Hydra 配置管理** + **HuggingFace Transformers**，支持多种 unlearning 算法。

### 核心组件

```
open-unlearning/
├── src/
│   ├── train.py              # 训练入口（主函数）
│   ├── data/
│   │   ├── __init__.py       # 数据加载逻辑
│   │   └── unlearn.py        # ForgetRetainDataset（组合 forget/retain）
│   ├── model/
│   │   └── __init__.py       # 模型加载逻辑
│   └── trainer/
│       ├── __init__.py       # Trainer 注册和加载
│       └── unlearn/
│           ├── grad_diff.py  # GradDiff 基类
│           └── simnpo.py     # SimNPO 算法
├── configs/
│   ├── unlearn.yaml          # 主配置文件
│   ├── trainer/
│   │   └── SimNPO.yaml       # SimNPO 算法配置
│   ├── data/
│   │   └── unlearn.yaml      # 数据配置
│   └── experiment/
│       └── unlearn/tofu/
│           └── default.yaml  # TOFU 数据集实验配置
└── scripts/
    └── tofu_unlearn.sh       # 训练脚本
```

### 数据流

```
命令行参数
    ↓
Hydra 配置合并
    ↓
加载模型 + Tokenizer
    ↓
加载 Forget/Retain 数据集 → ForgetRetainDataset
    ↓
初始化 SimNPO Trainer
    ↓
训练循环：
  ├─ 采样 batch: {"forget": ..., "retain": ...}
  ├─ 计算损失: γ × L_forget + α × L_retain
  ├─ 反向传播
  └─ 更新参数
    ↓
保存模型 + 评估
```

---

## 3. 执行流程详解

### 3.1 启动命令

```bash
CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=my_simnpo_run_v3 \
    model.model_args.attn_implementation=eager \
    trainer.args.learning_rate=5e-5 \
    trainer.args.num_train_epochs=20 \
    trainer.args.eval_strategy=steps \
    +trainer.args.eval_steps=100 \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=100
```

### 3.2 步骤详解

#### 步骤 1: 环境变量设置

```bash
CUDA_VISIBLE_DEVICES=4      # 使用第 4 号 GPU
HYDRA_FULL_ERROR=1          # 显示完整的错误堆栈（方便调试）
```

#### 步骤 2: Hydra 配置加载

Hydra 按照以下顺序合并配置：

1. **基础配置** (`unlearn.yaml`)
   ```yaml
   defaults:
     - model: Llama-3.2-3B-Instruct
     - trainer: GradAscent
     - data: unlearn
   
   mode: unlearn
   task_name: ???  # 必须在命令行指定
   ```

2. **实验配置** (`experiment=unlearn/tofu/default`)
   ```yaml
   defaults:
     - override /model: Llama-3.2-1B-Instruct
     - override /trainer: GradAscent
   
   forget_split: forget10
   retain_split: retain90
   ```

3. **命令行覆盖**
   ```bash
   trainer=SimNPO                    # 覆盖 trainer
   forget_split=forget10             # 确认 forget split
   trainer.args.learning_rate=5e-5   # 覆盖学习率
   ```

最终配置 = 基础 + 实验 + 命令行（后者覆盖前者）

#### 步骤 3: 模型加载 (`get_model`)

```python
model, tokenizer = get_model(cfg.model)
# 加载预训练模型：open-unlearning/tofu_Llama-3.2-1B-Instruct_full
# 这是在 TOFU 数据集上微调过的模型
```

#### 步骤 4: 数据加载 (`get_data`)

```python
data = get_data(cfg.data, mode="unlearn", tokenizer=tokenizer)
# 返回：{"train": ForgetRetainDataset, "eval": QADataset}
```

**ForgetRetainDataset 的工作原理：**

```python
# 假设：
# - forget10: 176 条样本（10% 作者的 QA 数据）
# - retain90: 1584 条样本（90% 作者的 QA 数据）
# - anchor="forget"

dataset = ForgetRetainDataset(forget=forget10, retain=retain90, anchor="forget")
len(dataset)  # 176（锚定在 forget）

# 每次采样返回：
item = dataset[0]
# {
#     "forget": forget10[0],        # 第 0 条 forget 数据（顺序）
#     "retain": retain90[random]     # 随机一条 retain 数据
# }
```

#### 步骤 5: 初始化 Trainer (`load_trainer`)

```python
trainer = SimNPO(
    model=model,
    train_dataset=data["train"],  # ForgetRetainDataset
    eval_dataset=data["eval"],
    tokenizer=tokenizer,
    # SimNPO 特定参数（来自 configs/trainer/SimNPO.yaml）
    delta=0.0,
    beta=4.5,
    gamma=0.125,  # forget 损失权重
    alpha=1.0,    # retain 损失权重
    # 训练参数
    args=TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=20,
        per_device_train_batch_size=2,
        ...
    )
)
```

#### 步骤 6: 训练循环 (`trainer.train()`)

```python
for epoch in range(num_train_epochs):
    for batch in dataloader:
        # batch 结构：
        # {
        #     "forget": {
        #         "input_ids": [batch_size, seq_len],
        #         "attention_mask": [batch_size, seq_len],
        #         "labels": [batch_size, seq_len]
        #     },
        #     "retain": { ... }  # 相同结构
        # }
        
        # 计算损失（SimNPO.compute_loss）
        loss = trainer.compute_loss(model, batch)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 定期评估
        if step % eval_steps == 0:
            trainer.evaluate()
        
        # 定期保存
        if step % save_steps == 0:
            trainer.save_model()
```

---

## 4. 命令行参数解析

### 基础参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config-name` | 主配置文件 | `unlearn.yaml` |
| `experiment` | 实验配置（覆盖基础配置） | `unlearn/tofu/default` |
| `task_name` | 实验名称（用于保存路径） | `my_simnpo_run_v3` |

### 数据参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `forget_split` | Forget 数据集的 split | `forget01`, `forget05`, `forget10` |
| `retain_split` | Retain 数据集的 split | `retain90`, `retain95`, `retain99` |

**TOFU 数据集的 split：**

- `forget01`: 1% 作者（~17 条样本）
- `forget05`: 5% 作者（~88 条样本）
- `forget10`: 10% 作者（~176 条样本）
- `retain90`: 90% 作者（~1584 条样本）

### 训练器参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `trainer` | Unlearning 算法 | `SimNPO`, `GradAscent`, `GradDiff`, `NPO`, `DPO` |
| `trainer.args.learning_rate` | 学习率 | `1e-5` ~ `1e-4` |
| `trainer.args.num_train_epochs` | 训练轮数 | `5` ~ `50` |
| `trainer.args.per_device_train_batch_size` | 每卡 batch size | `1`, `2`, `4` |

### 评估和保存参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `trainer.args.eval_strategy` | 评估策略 | `steps`（按步数）或 `epoch`（按轮数） |
| `+trainer.args.eval_steps` | 评估间隔 | `100`（每 100 步评估一次） |
| `trainer.args.save_strategy` | 保存策略 | `steps` 或 `epoch` |
| `+trainer.args.save_steps` | 保存间隔 | `100`（每 100 步保存一次） |

**注意：** `+` 表示新增参数（Hydra 语法）

### 模型参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `model.model_args.attn_implementation` | 注意力实现方式 | `eager`（标准），`flash_attention_2`（快速但需要特定硬件） |

---

## 5. SimNPO 算法原理

### 5.1 核心思想

SimNPO (Simplified Negative Preference Optimization) 是一种基于偏好优化的 unlearning 方法。

**目标：**
- 对 **forget 数据**：降低模型输出该数据的概率（负面偏好）
- 对 **retain 数据**：保持模型性能不变（正常训练）

### 5.2 损失函数

$$
\mathcal{L}_{\text{total}} = \gamma \cdot \mathcal{L}_{\text{forget}} + \alpha \cdot \mathcal{L}_{\text{retain}}
$$

#### Forget 损失

$$
\mathcal{L}_{\text{forget}} = -\frac{2}{\beta} \cdot \mathbb{E}_{x \sim D_{\text{forget}}} \left[ \log\sigma(\beta \cdot (\text{NLL}(x) - \delta)) \right]
$$

其中：
- $\text{NLL}(x)$：负对数似然（模型在 $x$ 上的损失）
- $\sigma(\cdot)$：sigmoid 函数
- $\beta$：温度参数（控制优化平滑度）
- $\delta$：NLL 偏移量（控制遗忘强度）

**直观理解：**
- NLL 越高 → 模型对该数据的概率越低 → 越"遗忘"
- $\log\sigma(\cdot)$ 将 NLL 转换为偏好优化形式
- NLL 低时损失高（需要继续遗忘），NLL 高时损失低（已经遗忘）

#### Retain 损失

$$
\mathcal{L}_{\text{retain}} = \mathbb{E}_{x \sim D_{\text{retain}}} \left[ \text{NLL}(x) \right]
$$

或（使用参考模型）：

$$
\mathcal{L}_{\text{retain}} = \mathbb{E}_{x \sim D_{\text{retain}}} \left[ D_{\text{KL}}(P_{\text{ref}}(x) \| P_{\theta}(x)) \right]
$$

### 5.3 参数影响

| 参数 | 默认值 | 作用 | 调整建议 |
|------|--------|------|----------|
| `gamma` (γ) | 0.125 | Forget 损失权重 | 增大 → 遗忘更强（但可能影响 retain 性能） |
| `alpha` (α) | 1.0 | Retain 损失权重 | 增大 → 更注重保持性能 |
| `beta` (β) | 4.5 | 温度参数 | 增大 → 优化更陡峭（可能不稳定） |
| `delta` (δ) | 0.0 | NLL 偏移量 | 增大 → 允许更高的 NLL（更强的遗忘） |

### 5.4 代码实现

```python
def compute_loss(self, model, inputs):
    # === Forget 损失 ===
    forget_inputs = inputs["forget"]
    forget_labels = forget_inputs["labels"]
    loss_mask = forget_labels != -100  # 忽略 padding
    
    # 计算每个样本的 NLL
    forget_loss, _ = compute_batch_nll(model, forget_inputs)
    forget_loss = forget_loss / loss_mask.sum(-1)  # 归一化
    
    # SimNPO 转换
    forget_loss = forget_loss - self.delta  # 应用偏移
    forget_loss = -F.logsigmoid(self.beta * forget_loss).mean() * 2 / self.beta
    
    # === Retain 损失 ===
    retain_inputs = inputs["retain"]
    retain_loss = self.compute_retain_loss(model, retain_inputs)
    
    # === 组合损失 ===
    loss = self.gamma * forget_loss + self.alpha * retain_loss
    return loss
```

---

## 6. 如何修改配置

### 6.1 更换 Unlearning 方法

修改 `trainer` 参数：

```bash
# SimNPO（推荐）
trainer=SimNPO

# 梯度上升（最简单）
trainer=GradAscent

# 梯度差分（SimNPO 的基类）
trainer=GradDiff

# NPO（Negative Preference Optimization）
trainer=NPO

# DPO（Direct Preference Optimization）
trainer=DPO

# 其他方法
trainer=RMU         # Random Mask Unlearning
trainer=UNDIAL      # Unlearning via Data Influence Analysis
trainer=CEU         # Catastrophic Forgetting based Unlearning
```

**查看所有可用方法：**

```bash
ls configs/trainer/
```

### 6.2 更换数据集

#### 方法 1: 修改 split

```bash
# 更小的 forget set（更容易遗忘）
forget_split=forget01
retain_split=retain99

# 更大的 forget set（更难遗忘）
forget_split=forget10
retain_split=retain90
```

#### 方法 2: 使用其他数据集

创建新的实验配置文件 `configs/experiment/unlearn/my_dataset/default.yaml`：

```yaml
# @package _global_

defaults:
  - override /model: Llama-3.2-1B-Instruct
  - override /trainer: SimNPO
  - override /data: unlearn
  - override /data/datasets@data.forget: MY_DATASET_forget
  - override /data/datasets@data.retain: MY_DATASET_retain

data:
  forget:
    MY_DATASET_forget:
      handler: QADataset
      args:
        hf_args:
          path: "my_org/my_dataset"
          split: "train"
          name: "forget_subset"
        question_key: "question"
        answer_key: "answer"
  retain:
    MY_DATASET_retain:
      handler: QADataset
      args:
        hf_args:
          path: "my_org/my_dataset"
          split: "train"
          name: "retain_subset"
        question_key: "question"
        answer_key: "answer"
```

然后运行：

```bash
python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/my_dataset/default \
    ...
```

### 6.3 更换评价指标

#### 方法 1: 修改现有评估配置

编辑 `configs/eval/tofu.yaml`：

```yaml
tofu:
  handler: TOFUEvaluator
  args:
    metrics:
      - ra_Q_A_ROUGE      # Retain set 上的 ROUGE 分数
      - wf_Q_A_Prob       # Forget set 上的概率差异
      - holdout_Prob      # Holdout set 上的概率
      # 添加新指标
      - my_custom_metric
```

#### 方法 2: 创建新的评估配置

创建 `configs/eval/my_eval.yaml`：

```yaml
my_eval:
  handler: MyCustomEvaluator
  args:
    metric_1: value_1
    metric_2: value_2
```

然后运行：

```bash
python src/train.py \
    --config-name=unlearn.yaml \
    eval=my_eval \
    ...
```

### 6.4 调整超参数

#### 学习率调整

```bash
# 更小的学习率（更稳定，但收敛慢）
trainer.args.learning_rate=1e-5

# 更大的学习率（收敛快，但可能不稳定）
trainer.args.learning_rate=1e-4
```

#### 训练轮数调整

```bash
# 更多轮数（更彻底的遗忘）
trainer.args.num_train_epochs=50

# 更少轮数（更快完成，但遗忘可能不彻底）
trainer.args.num_train_epochs=5
```

#### SimNPO 特定参数

```bash
# 更强的遗忘
trainer.method_args.gamma=0.5        # forget 损失权重（默认 0.125）
trainer.method_args.beta=10.0        # 温度参数（默认 4.5）

# 更注重保持性能
trainer.method_args.alpha=2.0        # retain 损失权重（默认 1.0）
trainer.method_args.gamma=0.05       # forget 损失权重
```

---

## 7. 常见问题

### Q1: 如何查看训练进度？

训练日志会保存在 `saves/unlearn/{task_name}/` 目录下：

```bash
# 查看训练日志
tail -f saves/unlearn/my_simnpo_run_v3/trainer_log.txt

# 使用 TensorBoard 查看（如果启用）
tensorboard --logdir saves/unlearn/my_simnpo_run_v3/
```

### Q2: 如何恢复中断的训练？

添加 `trainer.args.resume_from_checkpoint` 参数：

```bash
python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer.args.resume_from_checkpoint=saves/unlearn/my_simnpo_run_v3/checkpoint-100 \
    ...
```

### Q3: 显存不足怎么办？

1. **减小 batch size：**
   ```bash
   trainer.args.per_device_train_batch_size=1
   ```

2. **启用梯度累积：**
   ```bash
   trainer.args.gradient_accumulation_steps=8
   ```

3. **使用梯度检查点：**
   ```bash
   trainer.args.gradient_checkpointing=true
   ```

4. **使用 DeepSpeed ZeRO-3：**
   ```bash
   accelerate launch --config_file configs/accelerate/zero_stage3_offload_config.json \
       src/train.py ...
   ```

### Q4: 如何评估遗忘效果？

运行评估脚本：

```bash
python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/my_simnpo_run_v3/
```

关键指标：
- **Forget Quality (FQ)**：forget set 上的性能下降（越低越好）
- **Model Utility (MU)**：retain set 上的性能保持（越高越好）
- **综合指标**：FQ × MU（平衡遗忘和保持）

### Q5: 如何对比不同方法？

创建对比脚本：

```bash
#!/bin/bash

methods=("SimNPO" "GradAscent" "GradDiff" "NPO")

for method in "${methods[@]}"; do
    python src/train.py \
        --config-name=unlearn.yaml \
        experiment=unlearn/tofu/default \
        trainer=$method \
        task_name=compare_${method} \
        ...
done
```

---

## 附录：完整配置示例

### A. 基础训练命令

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=simnpo_baseline \
    trainer.args.learning_rate=5e-5 \
    trainer.args.num_train_epochs=20 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.eval_strategy=steps \
    +trainer.args.eval_steps=50 \
    trainer.args.save_strategy=steps \
    +trainer.args.save_steps=50 \
    trainer.args.logging_steps=10
```

### B. 分布式训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    ...
```

### C. DeepSpeed ZeRO-3 训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file configs/accelerate/zero_stage3_offload_config.json \
    src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    trainer.args.per_device_train_batch_size=1 \
    trainer.args.gradient_accumulation_steps=8 \
    ...
```

---

## 参考文献

1. SimNPO: [Unlearn-Simple GitHub](https://github.com/OPTML-Group/Unlearn-Simple)
2. TOFU Dataset: [TOFU Paper](https://arxiv.org/abs/2401.06121)
3. Machine Unlearning Survey: [Survey Paper](https://arxiv.org/abs/2209.02299)

---

**文档版本：** v1.0  
**最后更新：** 2026-02-02  
**维护者：** Open-Unlearning Team
