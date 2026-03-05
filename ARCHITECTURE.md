# OpenUnlearning 项目架构详解

## 📐 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                   know -架构                          │
│              "Registry Pattern + Hydra Config"                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        应用入口层                                │
│  ┌──────────────┐              ┌──────────────┐                │
│  │  train.py      │              │   eval.py   │                │
│  │  (训练/遗忘/评估)  │              │  (评估)      │                │
│  └──────┬───────   ┘              └──────┬───────┘                │
└─────────┼──────────────────────────────┼───────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      组件注册表层                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Trainer  │  │ Dataset │  │ Evaluator│  │  Model   │        │
│  │ Registry │  │ Registry│  │ Registry │  │ Registry │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │            │              │              │              │
│  ┌────▼────────────▼──────────────▼──────────────▼────┐        │
│  │        统一加载函数 (load_* / get_*)                │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      实现层                                      │
│  Trainers:        Datasets:        Evaluators:    Models:       │
│  • GradAscent     • QADataset      • TOFU        • Llama       │
│  • GradDiff       • Pretraining    • MUSE        • Phi         │
│  • NPO/DPO        • ForgetRetain   • WMDP        • Gemma       │
│  • RMU/UNDIAL     • ...            • Metrics     • ...         │
│  • ...                            (MIA, etc.)                  │
└─────────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      配置层 (Hydra)                             │
│  • configs/trainer/*.yaml     (训练器配置)                      │
│  • configs/data/datasets/*.yaml (数据集配置)                    │
│  • configs/model/*.yaml       (模型配置)                        │
│  • configs/experiment/*.yaml  (实验配置)                        │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 目录结构详解

### 根目录文件

| 文件 | 作用 |
|------|------|
| `README.md` | 项目说明文档，包含快速开始、组件介绍等 |
| `setup.py` | Python 包安装配置，定义包名、版本、依赖 |
| `requirements.txt` | Python 依赖列表（transformers, torch, hydra等） |
| `setup_data.py` | 数据下载脚本，从 HuggingFace 下载评估数据 |
| `Makefile` | 代码质量检查命令（ruff format, ruff check） |
| `.pre-commit-config.yaml` | Git 提交前自动检查配置 |
| `LICENSE` | MIT 许可证 |

### 核心源代码目录：`src/`

#### `src/train.py` - 训练入口
- **作用**：训练和遗忘的主入口文件
- **功能**：
  - 使用 Hydra 加载配置
  - 加载模型、数据集、训练器
  - 执行训练/遗忘流程
  - 支持训练中评估

#### `src/eval.py` - 评估入口
- **作用**：模型评估的主入口文件
- **功能**：
  - 加载模型和评估器
  - 执行各种评估指标计算
  - 支持多个基准测试（TOFU, MUSE, WMDP）

#### `src/model/` - 模型管理
```
model/
├── __init__.py      # 模型注册表和加载函数
└── probe.py         # ProbedLlamaForCausalLM（带探针的模型）
```

**关键功能**：
- `MODEL_REGISTRY`: 模型类注册表
- `get_model()`: 根据配置加载模型和 tokenizer
- 支持 `AutoModelForCausalLM` 和自定义模型

#### `src/data/` - 数据处理
```
data/
├── __init__.py      # 数据集注册表和加载函数
├── qa.py            # QA 数据集（问答格式）
├── pretraining.py   # 预训练数据集（文本生成）
├── unlearn.py       # ForgetRetainDataset（遗忘专用）
├── collators.py     # 数据整理器（批处理）
└── utils.py         # 数据工具函数
```

**关键类**：
- `QADataset`: 问答格式数据集
- `PretrainingDataset`: 预训练文本数据集
- `ForgetRetainDataset`: 组合 forget 和 retain 数据集
- `DataCollatorForSupervisedDataset`: 批处理整理器

#### `src/trainer/` - 训练器实现
```
trainer/
├── __init__.py      # 训练器注册表和加载函数
├── base.py          # FinetuneTrainer（基础训练器）
├── utils.py         # 工具函数（KL散度、DPO loss等）
└── unlearn/         # 遗忘方法实现
    ├── base.py      # UnlearnTrainer（遗忘基类）
    ├── grad_ascent.py    # GradAscent 方法
    ├── grad_diff.py      # GradDiff 方法
    ├── npo.py            # NPO 方法
    ├── dpo.py            # DPO 方法
    ├── simnpo.py         # SimNPO 方法
    ├── rmu.py            # RMU 方法
    ├── undial.py         # UNDIAL 方法
    ├── ceu.py            # CEU 方法
    ├── satimp.py         # SatImp 方法
    ├── wga.py            # WGA 方法
    └── pdu.py            # PDU 方法
```

**继承关系**：
```
Trainer (HuggingFace)
    │
    ├─ FinetuneTrainer
    │       │
    │       └─ UnlearnTrainer
    │               │
    │               ├─ GradAscent
    │               ├─ GradDiff
    │               │       ├─ NPO
    │               │       ├─ DPO
    │               │       └─ SimNPO
    │               ├─ RMU
    │               ├─ UNDIAL
    │               └─ ...
```

#### `src/evals/` - 评估系统
```
evals/
├── __init__.py      # 评估器注册表
├── base.py          # Evaluator 基类
├── tofu.py          # TOFU 基准测试
├── muse.py          # MUSE 基准测试
├── lm_eval.py       # lm-evaluation-harness 集成
└── metrics/         # 评估指标
    ├── __init__.py
    ├── base.py      # Metric 基类
    ├── memorization.py  # 记忆化指标
    ├── privacy.py       # 隐私指标
    ├── utility.py       # 效用指标
    └── mia/             # 成员推理攻击
        ├── loss.py      # LOSS 攻击
        ├── zlib.py      # ZLib 攻击
        ├── gradnorm.py  # GradNorm 攻击
        ├── min_k.py     # MinK 攻击
        └── ...
```

### 配置文件目录：`configs/`

#### `configs/hydra/` - Hydra 配置
- `default.yaml`: Hydra 默认配置（日志、输出目录）
- `eval.yaml`: 评估专用 Hydra 配置

#### `configs/model/` - 模型配置
每个模型一个 YAML 文件，包含：
- `model_handler`: 模型类名
- `model_args`: 模型加载参数（路径、精度等）
- `tokenizer_args`: Tokenizer 参数
- `template_args`: 聊天模板参数

**示例模型**：
- `Llama-3.2-1B-Instruct.yaml`
- `Llama-3.1-8B-Instruct.yaml`
- `Phi-3.5-mini-instruct.yaml`
- `Gemma-7b-it.yaml`

#### `configs/trainer/` - 训练器配置
每个训练器一个 YAML 文件：
- `handler`: 训练器类名
- `args`: HuggingFace TrainingArguments
- `method_args`: 方法特定参数

**示例**：
- `GradAscent.yaml`
- `GradDiff.yaml`
- `DPO.yaml`
- `finetune.yaml` (基础配置)

#### `configs/experiment/` - 实验配置
组合多个组件的完整实验配置：

```
experiment/
├── unlearn/         # 遗忘实验
│   ├── tofu/
│   │   └── default.yaml
│   └── muse/
│       └── default.yaml
└── eval/            # 评估实验
    ├── tofu/
    │   └── default.yaml
    └── muse/
        └── default.yaml
```

**实验配置结构**：
```yaml
defaults:
  - override /model: Llama-3.2-1B-Instruct
  - override /trainer: GradAscent
  - override /data: unlearn
  - override /eval: tofu

# 覆盖特定参数
model:
  model_args:
    pretrained_model_name_or_path: open-unlearning/tofu_Llama-3.2-1B-Instruct_full
```

#### `configs/accelerate/` - Accelerate 配置
- `default_config.yaml`: 分布式训练配置
- `zero_stage3_offload_config.json`: DeepSpeed ZeRO-3 配置

### 脚本目录：`scripts/`

| 脚本 | 作用 |
|------|------|
| `tofu_unlearn.sh` | TOFU 基准测试的批量遗忘实验 |
| `muse_unlearn.sh` | MUSE 基准测试的批量遗忘实验 |
| `tofu_finetune.sh` | TOFU 模型的微调脚本 |

### 文档目录：`docs/`

| 文档 | 内容 |
|------|------|
| `components.md` | 如何添加新组件（训练器、数据集、评估器等） |
| `evaluation.md` | 评估系统和指标详解 |
| `experiments.md` | 实验运行指南 |
| `hydra.md` | Hydra 配置管理教程 |
| `links.md` | 相关论文和资源链接 |
| `repro.md` | 可复现性结果 |

### 社区目录：`community/`

- `leaderboard.md`: 方法性能排行榜
- `benchmarks/template/`: 基准测试模板
- `methods/`: 社区贡献的方法**复现脚本与文档**（见下方说明）

#### community/methods 与主仓库 trainer 的关系

`community/methods/` 下每个子目录（AltPO、CEU、PDU、SatImp、UNDIAL、WGA）**不是**独立实现，而是：

1. **文档**：README 说明论文、超参、实验设置、引用
2. **复现脚本**：`run.sh` 里调用**主仓库**的 `src/train.py`，并指定对应的 **trainer**（来自 `src/trainer/unlearn/`）

对应关系示例：

| community 方法 | 使用的主仓库 Trainer | 说明 |
|----------------|----------------------|------|
| AltPO | DPO | 用 DPO trainer + 自定义 alternate 数据 |
| CEU | CEU | 直接对应 `src/trainer/unlearn/ceu.py` |
| PDU | PDU | 直接对应 `src/trainer/unlearn/pdu.py` |
| SatImp | SatImp | 直接对应 `src/trainer/unlearn/satimp.py` |
| UNDIAL | UNDIAL | 直接对应 `src/trainer/unlearn/undial.py` |
| WGA | WGA | 直接对应 `src/trainer/unlearn/wga.py` |

因此：**算法实现都在主仓库的 `src/trainer/unlearn/`；community 只提供可复现的命令和文档。**

## 🔄 数据流和组件关系

### 训练流程

```
1. 用户运行命令
   python src/train.py experiment=unlearn/tofu/default trainer=GradAscent

2. Hydra 加载配置
   └─ 合并 defaults 中的配置
   └─ 覆盖命令行参数

3. 组件加载（按顺序）
   ├─ get_model() → 从 MODEL_REGISTRY 加载模型
   ├─ get_data() → 从 DATASET_REGISTRY 加载数据集
   ├─ get_collators() → 从 COLLATOR_REGISTRY 加载整理器
   ├─ get_evaluators() → 从 EVALUATOR_REGISTRY 加载评估器
   └─ load_trainer() → 从 TRAINER_REGISTRY 加载训练器

4. 执行训练
   └─ trainer.train() → 调用训练器的 compute_loss()
   └─ trainer.evaluate() → 执行评估（如果配置了）

5. 保存结果
   └─ 模型保存到 saves/unlearn/<task_name>/
   └─ 评估结果保存到 saves/unlearn/<task_name>/evals/
```

### 评估流程

```
1. 用户运行命令
   python src/eval.py experiment=eval/tofu/default model=Llama-3.2-1B-Instruct

2. Hydra 加载配置
   └─ 加载评估实验配置

3. 组件加载
   ├─ get_model() → 加载要评估的模型
   └─ get_evaluators() → 加载评估器（TOFU, MUSE等）

4. 执行评估
   └─ evaluator.evaluate() → 计算各种指标
   └─ 保存结果到指定目录
```

## 🎯 核心设计模式

### 1. Registry Pattern（注册表模式）

每个组件类型都有注册表：
- `TRAINER_REGISTRY`
- `DATASET_REGISTRY`
- `EVALUATOR_REGISTRY`
- `COLLATOR_REGISTRY`
- `MODEL_REGISTRY`

**优势**：
- 解耦实现和配置
- 易于扩展新组件
- 类型安全

### 2. Hydra 配置管理

- 使用 YAML 文件组织配置
- 支持配置继承和覆盖
- 命令行参数覆盖
- 动态配置组合

### 3. 组件化设计

每个组件（训练器、数据集、评估器）都是独立的：
- 实现 → 注册 → 配置 → 使用

## 📊 关键文件作用总结

### 入口文件
- `src/train.py`: 训练/遗忘主程序
- `src/eval.py`: 评估主程序

### 核心模块
- `src/trainer/`: 所有训练器实现
- `src/data/`: 数据集实现
- `src/evals/`: 评估系统实现
- `src/model/`: 模型加载和管理

### 配置系统
- `configs/experiment/`: 实验配置
- `configs/trainer/`: 训练器配置
- `configs/model/`: 模型配置

### 工具脚本
- `setup_data.py`: 数据下载
- `scripts/*.sh`: 批量实验脚本

## 🚀 扩展点

### 添加新训练器
1. 在 `src/trainer/unlearn/` 实现类
2. 在 `src/trainer/__init__.py` 注册
3. 在 `configs/trainer/` 创建配置

### 添加新数据集
1. 在 `src/data/` 实现 Dataset 类
2. 在 `src/data/__init__.py` 注册
3. 在 `configs/data/datasets/` 创建配置

### 添加新评估指标
1. 在 `src/evals/metrics/` 实现 Metric 类
2. 在评估器中集成使用

### 添加新基准测试
1. 在 `src/evals/` 实现 Evaluator 类
2. 在 `src/evals/__init__.py` 注册
3. 在 `configs/experiment/eval/` 创建配置

## 📝 总结

OpenUnlearning 采用**注册表模式 + Hydra 配置管理**的架构设计，实现了：

1. **高度模块化**：每个组件独立实现和配置
2. **易于扩展**：添加新组件只需三步（实现→注册→配置）
3. **配置驱动**：通过 YAML 文件灵活组合实验
4. **统一接口**：所有组件遵循相同的加载模式

这种设计使得框架既能支持多种方法、数据集和评估指标，又保持了代码的清晰和可维护性。
