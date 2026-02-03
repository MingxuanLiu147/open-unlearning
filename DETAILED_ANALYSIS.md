# OpenUnlearning 详细分析：评估系统、配置组织、数据加载

## 1. 评估系统设计

### 1.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    评估系统架构                               │
└─────────────────────────────────────────────────────────────┘

Evaluator (基准测试)
    │
    ├─ TOFUEvaluator
    ├─ MUSEEvaluator
    └─ LMEvalEvaluator
    │
    └─ 聚合多个 Metrics
        │
        ├─ Memorization Metrics (记忆化指标)
        │   ├─ probability (概率)
        │   ├─ rouge (ROUGE分数)
        │   ├─ truth_ratio (真实率)
        │   ├─ extraction_strength (提取强度)
        │   └─ exact_memorization (精确记忆)
        │
        ├─ Privacy Metrics (隐私指标)
        │   ├─ ks_test (KS测试)
        │   ├─ privleak (隐私泄露)
        │   └─ rel_diff (相对差异)
        │
        ├─ MIA Metrics (成员推理攻击)
        │   ├─ mia_loss (LOSS攻击)
        │   ├─ mia_zlib (ZLib攻击)
        │   ├─ mia_gradnorm (梯度范数攻击)
        │   ├─ mia_min_k (MinK攻击)
        │   ├─ mia_min_k_plus_plus (MinK++攻击)
        │   └─ mia_reference (参考攻击)
        │
        └─ Utility Metrics (效用指标)
            ├─ hm_aggregate (调和平均)
            └─ classifier_prob (分类器概率)
```

### 1.2 核心类设计

#### Evaluator 基类 (`src/evals/base.py`)

```python
class Evaluator:
    """评估器基类，所有基准测试的父类"""
    
    def __init__(self, name, eval_cfg, **kwargs):
        self.name = name  # 基准测试名称 (TOFU, MUSE等)
        self.eval_cfg = eval_cfg  # 评估配置
        self.metrics = self.load_metrics()  # 加载指标
    
    def evaluate(self, model, output_dir=None, overwrite=None, **kwargs):
        """
        评估流程：
        1. 准备模型 (model.eval())
        2. 加载/创建日志文件
        3. 遍历所有指标
        4. 对每个指标：
           - 检查是否已评估（缓存）
           - 调用指标函数
           - 保存结果
        5. 生成汇总结果
        """
```

**关键特性**：
- **缓存机制**：已评估的指标会跳过，避免重复计算
- **增量评估**：支持 `overwrite=False` 时只评估新指标
- **结果保存**：保存详细结果 (`TOFU_EVAL.json`) 和汇总 (`TOFU_SUMMARY.json`)

#### UnlearningMetric 类 (`src/evals/metrics/base.py`)

```python
class UnlearningMetric:
    """指标封装类，支持预计算和依赖管理"""
    
    def __init__(self, name, metric_fn):
        self.name = name
        self._metric_fn = metric_fn  # 实际的指标计算函数
        self.pre_compute_metrics = {}  # 预计算指标
    
    def prepare_kwargs_evaluate_metric(self, model, metric_name, cache, **kwargs):
        """
        准备评估参数：
        1. 加载数据集 (如果配置了)
        2. 加载整理器 (如果配置了)
        3. 评估预计算指标 (依赖的指标)
        4. 加载参考日志 (如 retain 模型的结果)
        """
    
    def evaluate(self, model, metric_name, cache, **kwargs):
        """
        评估指标：
        1. 检查缓存
        2. 准备参数
        3. 调用指标函数
        4. 更新缓存
        """
```

**设计亮点**：
- **依赖管理**：指标可以依赖其他指标（pre_compute）
- **参考模型支持**：可以加载参考模型的评估结果
- **灵活配置**：通过配置指定数据集、整理器等

### 1.3 评估流程详解

```
┌─────────────────────────────────────────────────────────────┐
│                   评估执行流程                                 │
└─────────────────────────────────────────────────────────────┘

1. 初始化评估器
   └─ TOFUEvaluator(eval_cfg)
       └─ 加载指标配置
       └─ 创建指标实例

2. 调用 evaluate()
   └─ 准备模型 (model.eval())
   └─ 设置输出目录
   └─ 加载/创建日志文件

3. 遍历指标
   for metric_name, metric_fn in self.metrics.items():
       ├─ 检查缓存 (如果已评估，跳过)
       ├─ 准备参数
       │   ├─ 加载数据集
       │   ├─ 加载整理器
       │   ├─ 评估预计算指标
       │   └─ 加载参考日志
       ├─ 调用指标函数
       │   └─ metric_fn(model, **prepared_kwargs)
       ├─ 保存结果到日志
       └─ 更新汇总

4. 返回汇总结果
   └─ 包含所有指标的聚合值
```

### 1.4 指标注册机制

```python
# src/evals/metrics/__init__.py

METRICS_REGISTRY: Dict[str, UnlearningMetric] = {}

def _register_metric(metric):
    METRICS_REGISTRY[metric.name] = metric

# 注册所有指标
_register_metric(probability)
_register_metric(rouge)
_register_metric(mia_loss)
# ... 更多指标
```

**使用方式**：
- 指标通过装饰器或直接注册
- 配置文件中通过 `handler` 字段指定指标名称
- 系统自动从注册表加载

### 1.5 评估配置示例

```yaml
# configs/experiment/eval/tofu/default.yaml

eval:
  tofu:
    forget_split: forget10
    holdout_split: holdout10
    retain_logs_path: saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
    output_dir: saves/eval/tofu_test
    overwrite: true
    metrics:
      verbatim_probability:
        handler: probability
        datasets:
          forget:
            TOFU_QA_forget:
              args:
                hf_args:
                  path: "locuslab/TOFU"
                  name: ${forget_split}
      forget_quality:
        handler: rel_diff
        reference_logs:
          retain:
            path: ${retain_logs_path}
            include:
              verbatim_probability:
                access_key: verbatim_probability
```

### 1.6 评估系统优势

1. **模块化设计**：每个指标独立实现和配置
2. **缓存机制**：避免重复计算，支持增量评估
3. **依赖管理**：指标可以依赖其他指标的结果
4. **灵活扩展**：添加新指标只需实现函数并注册
5. **结果持久化**：JSON 格式保存，便于后续分析

---

## 2. 配置文件组织方式

### 2.1 Hydra 配置层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                  Hydra 配置层次                                │
└─────────────────────────────────────────────────────────────┘

顶层配置 (unlearn.yaml / eval.yaml)
    │
    ├─ defaults: 组合多个子配置
    │   ├─ model: Llama-3.2-1B-Instruct
    │   ├─ trainer: GradAscent
    │   ├─ data: unlearn
    │   ├─ collator: DataCollatorForSupervisedDataset
    │   ├─ eval: tofu
    │   └─ hydra: default
    │
    └─ 覆盖特定参数
        ├─ model.model_args.pretrained_model_name_or_path
        ├─ trainer.args.learning_rate
        └─ data.forget/retain splits
```

### 2.2 配置文件组织结构

```
configs/
├── hydra/                    # Hydra 框架配置
│   ├── default.yaml         # 默认配置（日志、输出目录）
│   └── eval.yaml            # 评估专用配置
│
├── model/                    # 模型配置
│   ├── Llama-3.2-1B-Instruct.yaml
│   ├── Llama-3.1-8B-Instruct.yaml
│   └── ...
│
├── trainer/                  # 训练器配置
│   ├── finetune.yaml        # 基础配置（被继承）
│   ├── GradAscent.yaml      # 继承 finetune
│   ├── GradDiff.yaml        # 继承 finetune
│   └── ...
│
├── data/                     # 数据配置
│   └── datasets/            # 数据集实例配置
│       ├── TOFU_QA_forget.yaml
│       └── TOFU_QA_retain.yaml
│
├── collator/                 # 整理器配置
│   └── DataCollatorForSupervisedDataset.yaml
│
├── eval/                     # 评估配置
│   ├── tofu.yaml           # TOFU 基准测试配置
│   └── muse.yaml           # MUSE 基准测试配置
│
├── experiment/               # 实验配置（组合多个组件）
│   ├── unlearn/            # 遗忘实验
│   │   ├── tofu/
│   │   │   └── default.yaml
│   │   └── muse/
│   │       └── default.yaml
│   └── eval/               # 评估实验
│       ├── tofu/
│       │   └── default.yaml
│       └── muse/
│           └── default.yaml
│
├── paths/                    # 路径配置
│   └── default.yaml
│
├── train.yaml               # 训练主配置
├── unlearn.yaml             # 遗忘主配置
└── eval.yaml                # 评估主配置
```

### 2.3 配置继承机制

#### 示例 1：训练器配置继承

```yaml
# configs/trainer/finetune.yaml (基础配置)
handler: FinetuneTrainer
args:
  per_device_train_batch_size: 8
  learning_rate: 1e-5
  num_train_epochs: 10
  # ... 更多参数

# configs/trainer/GradDiff.yaml (继承并扩展)
defaults:
  - finetune  # 继承基础配置

handler: GradDiff  # 覆盖 handler
method_args:       # 新增方法特定参数
  gamma: 1.0
  alpha: 1.0
  retain_loss_type: NLL
```

#### 示例 2：实验配置组合

```yaml
# configs/experiment/unlearn/tofu/default.yaml

# @package _global_  # 使配置在全局作用域

defaults:
  # 组合多个组件配置
  - override /model: Llama-3.2-1B-Instruct
  - override /trainer: GradAscent
  - override /data: unlearn
  - override /data/datasets@data.forget: TOFU_QA_forget
  - override /data/datasets@data.retain: TOFU_QA_retain
  - override /eval: tofu

# 覆盖特定参数
model:
  model_args:
    pretrained_model_name_or_path: open-unlearning/tofu_Llama-3.2-1B-Instruct_full

forget_split: forget10
retain_split: retain90

data:
  anchor: forget
  forget:
    TOFU_QA_forget:
      args:
        hf_args:
          name: ${forget_split}  # 使用变量
```

### 2.4 配置变量和插值

Hydra 支持变量插值：

```yaml
# 定义变量
forget_split: forget10
retain_split: retain90

# 使用变量
data:
  forget:
    TOFU_QA_forget:
      args:
        hf_args:
          name: ${forget_split}  # 插值：forget10

eval:
  tofu:
    forget_split: ${forget_split}  # 传递到评估配置
```

### 2.5 命令行覆盖

```bash
# 覆盖配置中的任何参数
python src/train.py \
    experiment=unlearn/tofu/default \
    trainer=GradDiff \
    forget_split=forget05 \
    trainer.args.learning_rate=2e-5 \
    model.model_args.pretrained_model_name_or_path=local/path/to/model
```

**覆盖语法**：
- `key=value`: 覆盖顶层键
- `key.subkey=value`: 覆盖嵌套键
- `+key=value`: 添加新键
- `~key`: 删除键

### 2.6 配置加载流程

```
┌─────────────────────────────────────────────────────────────┐
│                   配置加载流程                                │
└─────────────────────────────────────────────────────────────┘

1. 用户指定配置
   python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default

2. Hydra 解析
   ├─ 加载 unlearn.yaml
   ├─ 解析 defaults 列表
   │   └─ 递归加载每个 default 配置
   ├─ 合并配置（按顺序）
   └─ 应用命令行覆盖

3. 最终配置对象
   └─ DictConfig (OmegaConf)
       └─ 包含所有组件的完整配置
```

### 2.7 配置组织优势

1. **模块化**：每个组件独立配置
2. **可复用**：基础配置可被多个配置继承
3. **可组合**：实验配置组合多个组件
4. **可覆盖**：支持命令行动态覆盖
5. **类型安全**：OmegaConf 提供类型检查

---

## 3. 数据集加载流程

### 3.1 数据加载架构

```
┌─────────────────────────────────────────────────────────────┐
│                   数据加载架构                                │
└─────────────────────────────────────────────────────────────┘

配置 (configs/data/datasets/*.yaml)
    │
    ▼
get_data() / get_datasets()
    │
    ├─ 解析配置
    ├─ 从 DATASET_REGISTRY 查找 handler
    ├─ 实例化数据集类
    └─ 根据 mode 组合数据集
        │
        ├─ mode="train" → 返回原始数据集字典
        └─ mode="unlearn" → 组合成 ForgetRetainDataset
```

### 3.2 数据集类型

#### 3.2.1 QADataset (问答数据集)

```python
class QADataset(Dataset):
    """问答格式数据集，用于 TOFU 等基准测试"""
    
    def __init__(self, hf_args, template_args, tokenizer, 
                 question_key="question", answer_key="answer", ...):
        # 从 HuggingFace 加载数据
        self.data = load_hf_dataset(**hf_args)
        # 应用聊天模板
        # Tokenize
        
    def __getitem__(self, idx):
        # 返回 tokenized 的问答对
        return {
            "input_ids": ...,
            "labels": ...,
            "attention_mask": ...
        }
```

**用途**：TOFU 基准测试的问答数据

#### 3.2.2 PretrainingDataset (预训练数据集)

```python
class PretrainingDataset(Dataset):
    """预训练文本数据集，用于 MUSE 等基准测试"""
    
    def __init__(self, hf_args, template_args, tokenizer, 
                 text_key="text", max_length=2048):
        # 加载原始文本
        raw_text = load_hf_dataset(**hf_args)[text_key]
        # 分块处理（按 max_length）
        self.chunks = self._chunk_raw_text(raw_text)
        
    def __getitem__(self, idx):
        # 返回 tokenized 的文本块
        return preprocess_pretraining_instance(...)
```

**用途**：MUSE 基准测试的文本数据

#### 3.2.3 ForgetRetainDataset (遗忘数据集)

```python
class ForgetRetainDataset(Dataset):
    """组合 forget 和 retain 数据集，用于遗忘训练"""
    
    def __init__(self, forget, retain, anchor="forget"):
        self.forget = forget  # 要遗忘的数据
        self.retain = retain  # 要保留的数据
        self.anchor = anchor  # 锚定数据集（决定长度）
    
    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            item["forget"] = self.forget[idx]
            # 随机采样 retain 数据
            retain_idx = torch.randint(0, len(self.retain), (1,)).item()
            item["retain"] = self.retain[retain_idx]
        return item
```

**关键特性**：
- **动态采样**：每次随机采样 retain 数据，增加多样性
- **锚定机制**：以 forget 或 retain 的长度为准
- **组合格式**：返回包含 "forget" 和 "retain" 的字典

### 3.3 数据加载流程详解

#### 步骤 1：配置解析

```yaml
# configs/experiment/unlearn/tofu/default.yaml

data:
  anchor: forget
  forget:
    TOFU_QA_forget:  # 数据集名称
      handler: QADataset  # 数据集类名
      args:
        hf_args:
          path: "locuslab/TOFU"
          name: ${forget_split}
        question_key: "question"
        answer_key: "answer"
  retain:
    TOFU_QA_retain:
      handler: QADataset
      args:
        hf_args:
          name: ${retain_split}
```

#### 步骤 2：调用 get_data()

```python
# src/train.py

data = get_data(
    data_cfg=cfg.data,
    mode="unlearn",  # 或 "train"
    tokenizer=tokenizer,
    template_args=template_args
)
```

#### 步骤 3：内部处理流程

```python
# src/data/__init__.py

def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    data = {}
    anchor = data_cfg.pop("anchor", "forget")
    
    # 1. 加载各个 split 的数据集
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    
    # 2. 根据 mode 处理
    if mode == "train":
        return data  # 返回原始数据集字典
    
    elif mode == "unlearn":
        # 组合成 ForgetRetainDataset
        unlearn_splits = {k: v for k, v in data.items() 
                         if k not in ("eval", "test")}
        unlearn_dataset = ForgetRetainDataset(
            **unlearn_splits, 
            anchor=anchor
        )
        data["train"] = unlearn_dataset
        # 移除原始 split
        for split in unlearn_splits:
            data.pop(split)
    
    return data
```

#### 步骤 4：单个数据集加载

```python
def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    # 1. 获取 handler 名称
    dataset_handler_name = dataset_cfg.get("handler")
    
    # 2. 从注册表查找
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    
    # 3. 实例化数据集
    dataset_args = dataset_cfg.args
    return dataset_handler(**dataset_args, **kwargs)
```

### 3.4 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                   数据加载完整流程                              │
└─────────────────────────────────────────────────────────────┘

配置文件
    │
    ▼
get_data(data_cfg, mode="unlearn", ...)
    │
    ├─ 解析 data_cfg
    │   ├─ anchor: "forget"
    │   ├─ forget: {TOFU_QA_forget: {...}}
    │   └─ retain: {TOFU_QA_retain: {...}}
    │
    ├─ 遍历 splits
    │   └─ get_datasets(dataset_cfgs, ...)
    │       └─ _load_single_dataset(name, cfg, ...)
    │           ├─ 查找 DATASET_REGISTRY[handler]
    │           ├─ 实例化 QADataset(**args, **kwargs)
    │           └─ 返回数据集实例
    │
    └─ mode == "unlearn"
        └─ ForgetRetainDataset(forget=..., retain=..., anchor=...)
            └─ 返回组合数据集
```

### 3.5 训练时的数据使用

```python
# 训练器中的 compute_loss()

def compute_loss(self, model, inputs, return_outputs=False):
    # inputs 来自 ForgetRetainDataset
    forget_inputs = inputs["forget"]  # 来自 forget 数据集
    retain_inputs = inputs["retain"]  # 随机采样的 retain 数据
    
    # 计算损失
    forget_loss = -model(**forget_inputs).loss  # 梯度上升
    retain_loss = model(**retain_inputs).loss   # 正常训练
    
    return gamma * forget_loss + alpha * retain_loss
```

### 3.6 数据集配置示例

#### TOFU 数据集配置

```yaml
# configs/data/datasets/TOFU_QA_forget.yaml

handler: QADataset
args:
  hf_args:
    path: "locuslab/TOFU"
    split: "train"
    name: forget10  # 数据集子集名称
  question_key: "question"
  answer_key: "answer"
  max_length: 512
  template_args:
    apply_chat_template: true
```

#### MUSE 数据集配置

```yaml
# configs/data/datasets/MUSE_forget.yaml

handler: PretrainingDataset
args:
  hf_args:
    path: "muse-bench/MUSE-News"
    name: "raw"
    split: "forget"
  text_key: "text"
  max_length: 2048
```

### 3.7 数据加载优势

1. **统一接口**：所有数据集通过相同方式加载
2. **灵活组合**：支持多种数据集组合方式
3. **动态采样**：ForgetRetainDataset 支持随机采样
4. **配置驱动**：通过 YAML 配置灵活指定
5. **易于扩展**：添加新数据集只需实现类并注册

---

## 总结

### 评估系统
- **分层设计**：Evaluator → Metrics → 具体指标函数
- **缓存机制**：避免重复计算
- **依赖管理**：支持指标间的依赖关系
- **结果持久化**：JSON 格式保存

### 配置组织
- **模块化**：每个组件独立配置
- **继承机制**：支持配置继承和覆盖
- **组合模式**：实验配置组合多个组件
- **动态覆盖**：命令行参数灵活覆盖

### 数据加载
- **注册表模式**：统一的数据集加载接口
- **模式切换**：train 和 unlearn 模式自动处理
- **组合数据集**：ForgetRetainDataset 支持动态采样
- **配置驱动**：通过 YAML 灵活配置

这三个系统共同构成了 OpenUnlearning 框架的核心基础设施，实现了高度的模块化、可扩展性和易用性。
