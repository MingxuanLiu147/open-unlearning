# 如何运行 SimNPO on TOFU (Llama-3.2-1B, forget=10%)

## 🎯 目标

使用 **SimNPO 算法**在 **TOFU 数据集**上完成 unlearning 任务：
- 模型：**Llama-3.2-1B-Instruct**
- Forget 比例：**10%**（约 176 条样本）
- Retain 比例：**90%**（约 1584 条样本）
- 对齐论文 pipeline

---

## 📁 新增文件清单

我为您创建了以下文件：

### 1. **运行脚本**
- ✅ `run_simnpo_tofu_1b.sh` - 一键运行脚本（包含训练+评估）

### 2. **文档**
- ✅ `SIMNPO_TOFU_运行指南.md` - 详细运行指南（推荐阅读）
- ✅ `QUICK_START_SimNPO.md` - 快速启动参考卡
- ✅ `README_运行SimNPO.md` - 本文档

### 3. **已有的核心注释文档**
- ✅ `UNLEARNING_GUIDE_CN.md` - 完整的 unlearning 原理和流程
- ✅ `代码注释总结.md` - 代码注释总结

---

## 🚀 三种运行方式

### 方式 1: 一键运行（最简单）

```bash
cd /home/liumingxuan/open-unlearning
bash run_simnpo_tofu_1b.sh
```

**优点：**
- ✅ 自动完成训练 + 评估
- ✅ 参数已优化（对齐论文）
- ✅ 包含详细的日志输出

**适合：** 快速开始，不需要修改参数

---

### 方式 2: 命令行运行（灵活）

#### 步骤 1: 训练模型

```bash
CUDA_VISIBLE_DEVICES=4 HYDRA_FULL_ERROR=1 python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=my_simnpo_experiment \
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

#### 步骤 2: 评估效果

```bash
CUDA_VISIBLE_DEVICES=4 python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    forget_split=forget10 \
    holdout_split=holdout10 \
    model=Llama-3.2-1B-Instruct \
    task_name=my_simnpo_experiment_2.6 \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/my_simnpo_experiment \
    paths.output_dir=saves/unlearn/my_simnpo_experiment/evals
```

**优点：**
- ✅ 可以逐步执行
- ✅ 易于修改单个参数
- ✅ 更好的错误调试

**适合：** 需要调整参数或分步执行

---

### 方式 3: 多 GPU 并行（最快）

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=SimNPO \
    forget_split=forget10 \
    retain_split=retain90 \
    task_name=my_simnpo_experiment \
    trainer.args.learning_rate=5e-5 \
    trainer.args.num_train_epochs=20 \
    trainer.args.per_device_train_batch_size=2 \
    trainer.args.gradient_accumulation_steps=4
```

**优点：**
- ✅ 训练速度快 2-4 倍
- ✅ 更好地利用硬件资源

**适合：** 有多块 GPU 且需要快速完成训练

---

## ⚙️ 关键参数（对齐论文）

### SimNPO 算法参数

| 参数 | 值 | 说明 | 来源 |
|------|-----|------|------|
| `gamma` | 0.125 | Forget 损失权重 | [论文](https://github.com/OPTML-Group/Unlearn-Simple/blob/main/TOFU/config/forget.yaml) |
| `alpha` | 1.0 | Retain 损失权重 | 论文 |
| `beta` | 4.5 | 温度参数 | 论文 |
| `delta` | 0.0 | NLL 偏移量 | 论文 |

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `learning_rate` | 5e-5 | 学习率 |
| `num_train_epochs` | 20 | 训练轮数 |
| `per_device_train_batch_size` | 2 | 每卡 batch size |
| `gradient_accumulation_steps` | 8 | 梯度累积（有效 BS=16） |
| `warmup_epochs` | 1.0 | Warmup 轮数 |

---

## 📊 预期结果

### 训练输出

```bash
# 训练日志位置
saves/unlearn/my_simnpo_experiment/trainer_log.txt

# 模型检查点
saves/unlearn/my_simnpo_experiment/checkpoint-100/
saves/unlearn/my_simnpo_experiment/checkpoint-200/
...

# 最终模型
saves/unlearn/my_simnpo_experiment/pytorch_model.bin
```

### 评估指标（对齐论文基准）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **Forget Quality (FQ)** | < 0.10 | 模型在 forget 数据上的性能应该下降 |
| **Model Utility (MU)** | > 0.85 | 模型在 retain 数据上的性能应该保持 |
| **综合得分** | ~0.80 | (1 - FQ) × MU |

**论文中 SimNPO 的表现（forget10）：**
- FQ: **0.08**
- MU: **0.87**
- Score: **0.80**

### 评估输出

```bash
# 评估结果
saves/unlearn/my_simnpo_experiment/evals/TOFU_EVAL.json

# 查看结果
cat saves/unlearn/my_simnpo_experiment/evals/TOFU_EVAL.json | jq
```

---

## 🔧 常见调整

### 1. 测试不同的 forget 比例

```bash
# 1% forget（更容易遗忘）
forget_split=forget01 retain_split=retain99

# 5% forget
forget_split=forget05 retain_split=retain95

# 10% forget（默认）
forget_split=forget10 retain_split=retain90
```

### 2. 调整遗忘强度

```bash
# 更强的遗忘（但可能影响 retain 性能）
trainer.method_args.gamma=0.5        # 默认 0.125
trainer.method_args.beta=10.0        # 默认 4.5

# 更好的性能保持（但遗忘可能不彻底）
trainer.method_args.alpha=2.0        # 默认 1.0
trainer.method_args.gamma=0.05       # 默认 0.125
```

### 3. 调整训练速度

```bash
# 更快收敛（但可能不稳定）
trainer.args.learning_rate=1e-4      # 默认 5e-5
trainer.args.num_train_epochs=10     # 默认 20

# 更稳定训练（但收敛慢）
trainer.args.learning_rate=1e-5      # 默认 5e-5
trainer.args.num_train_epochs=30     # 默认 20
```

### 4. 显存优化

```bash
# 如果显存不足（< 12GB）
trainer.args.per_device_train_batch_size=1      # 默认 2
trainer.args.gradient_accumulation_steps=16     # 默认 8
trainer.args.gradient_checkpointing=true        # 默认 false
```

---

## 📈 监控训练

### 实时查看日志

```bash
# 训练日志
tail -f saves/unlearn/my_simnpo_experiment/trainer_log.txt

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 关键指标

**训练过程中观察：**
1. **Loss 下降趋势**：应该平稳下降
2. **Eval Loss**：每 100 步评估一次
3. **GPU 显存**：应该稳定在 8-12 GB（1B 模型）

**训练完成后检查：**
1. **Forget Quality**：应该 < 0.10
2. **Model Utility**：应该 > 0.85
3. **综合得分**：应该 > 0.80

---

## 🐛 常见问题

### Q1: 显存不足 (OOM)

**解决方案：**
```bash
# 方案 1: 减小 batch size
trainer.args.per_device_train_batch_size=1
trainer.args.gradient_accumulation_steps=16

# 方案 2: 启用梯度检查点
trainer.args.gradient_checkpointing=true

# 方案 3: 使用 DeepSpeed
accelerate launch \
    --config_file configs/accelerate/zero_stage3_offload_config.json \
    src/train.py ...
```

### Q2: 模型路径错误

**错误信息：**
```
OSError: open-unlearning/tofu_Llama-3.2-1B-Instruct_full not found
```

**解决方案：**
```bash
# 检查模型是否存在
huggingface-cli repo info open-unlearning/tofu_Llama-3.2-1B-Instruct_full

# 或使用本地路径
model.model_args.pretrained_model_name_or_path=/path/to/your/model
```

### Q3: 训练损失不下降

**可能原因和解决方案：**
```bash
# 1. 学习率太小
trainer.args.learning_rate=1e-4

# 2. 权重设置不当
trainer.method_args.gamma=0.25

# 3. 检查数据是否正确加载
trainer.args.logging_steps=1  # 增加日志频率
```

### Q4: 评估失败

**错误信息：**
```
KeyError: 'retain_logs_path'
```

**解决方案：**
```bash
# 提供参考模型的评估日志
retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json

# 如果不存在，设置为 null
retain_logs_path=null
```

---

## ✅ 运行前检查清单

- [ ] **环境准备**
  - [ ] GPU 可用（至少 12GB 显存）
  - [ ] Python 环境已安装依赖
  - [ ] CUDA 版本兼容

- [ ] **数据准备**
  - [ ] 可访问 HuggingFace（或已下载数据集）
  - [ ] 预训练模型可用

- [ ] **配置确认**
  - [ ] GPU 编号正确（`CUDA_VISIBLE_DEVICES`）
  - [ ] 任务名称已设置（`task_name`）
  - [ ] 输出目录有写入权限

---

## 📚 更多资源

### 文档

1. **`QUICK_START_SimNPO.md`** - 快速参考卡（推荐）
2. **`SIMNPO_TOFU_运行指南.md`** - 详细运行指南
3. **`UNLEARNING_GUIDE_CN.md`** - Unlearning 原理详解
4. **`代码注释总结.md`** - 代码注释说明

### 代码

- `src/train.py` - 训练入口（已添加详细注释）
- `src/trainer/unlearn/simnpo.py` - SimNPO 算法实现（已添加详细注释）
- `src/trainer/unlearn/grad_diff.py` - GradDiff 基类（已添加详细注释）
- `src/data/unlearn.py` - ForgetRetainDataset（已添加详细注释）

### 外部资源

- **SimNPO 论文：** https://github.com/OPTML-Group/Unlearn-Simple
- **TOFU 数据集：** https://huggingface.co/datasets/locuslab/TOFU
- **LLaMA 模型：** https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

---

## 🎓 下一步

### 完成基础实验后：

1. **对比其他方法：**
   ```bash
   # 运行 GradAscent、GradDiff、NPO 等进行对比
   trainer=GradAscent / GradDiff / NPO / DPO
   ```

2. **测试不同配置：**
   ```bash
   # 不同的 forget 比例
   forget_split=forget01 / forget05 / forget10
   
   # 不同的超参数
   trainer.method_args.gamma=0.05 / 0.125 / 0.5
   ```

3. **深入分析：**
   - 查看模型在不同样本上的表现
   - 分析遗忘的选择性（是否只遗忘了目标数据）
   - 评估模型的泛化能力

---

**准备好了吗？开始运行：**

```bash
bash run_simnpo_tofu_1b.sh
```

**预计时间：** 2-3 小时（单卡 A100/V100）

**祝实验顺利！** 🎉
