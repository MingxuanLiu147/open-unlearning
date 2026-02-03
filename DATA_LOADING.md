# 数据集加载流程详解

## 1. 数据加载架构

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

## 2. 数据集类型

### 2.1 QADataset (问答数据集)

**位置**: `src/data/qa.py`

```python
class QADataset(Dataset):
    """问答格式数据集，用于 TOFU 等基准测试"""
    
    def __init__(self, hf_args, template_args, tokenizer, 
                 question_key="question", answer_key="answer", ...):
        # 1. 从 HuggingFace 加载原始数据
        # hf_args 示例: {"path": "locuslab/TOFU", "name": "forget10", "split": "train"}
        self.data = load_hf_dataset(**hf_args)
        
        # 2. 为每条数据添加索引字段，便于后续追踪和评估
        self.data = add_dataset_index(self.data)
        
        # 3. 保存 tokenizer 和模板配置，用于后续的文本处理
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.question_key = question_key  # 数据集中问题字段的键名
        self.answer_key = answer_key      # 数据集中答案字段的键名
        
    def __getitem__(self, idx):
        # 获取第 idx 条数据的问题和答案
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        
        # 调用 _process_sample 进行处理：
        # 1. 应用聊天模板（如 Llama 的 <|start_header_id|>user<|end_header_id|> 格式）
        # 2. Tokenize 成 token IDs
        # 3. 构造 labels（只在答案部分计算损失，问题部分用 -100 屏蔽）
        item = self._process_sample(question=question, answer=answer)
        
        # 返回 tokenized 的问答对
        return {
            "input_ids": ...,      # 完整的 token 序列（问题+答案）
            "labels": ...,         # 损失计算标签（问题部分为 -100，答案部分为实际 token）
            "attention_mask": ..., # 注意力掩码（全 1，因为没有 padding）
            "index": ...           # 原始数据的索引
        }
```

**用途**：TOFU 基准测试的问答数据

### 2.2 PretrainingDataset (预训练数据集)

**位置**: `src/data/pretraining.py`

```python
class PretrainingDataset(Dataset):
    """预训练文本数据集，用于 MUSE 等基准测试"""
    
    def __init__(self, hf_args, template_args, tokenizer, 
                 text_key="text", max_length=2048):
        # 1. 从 HuggingFace 加载原始文本数据
        # hf_args 示例: {"path": "muse-bench/MUSE-News", "name": "raw", "split": "forget"}
        raw_text = load_hf_dataset(**hf_args)[text_key]
        
        # 2. 分块处理（按 max_length 切分）
        # 原因：MUSE 数据集包含长文章，需要切成固定长度的训练样本
        # 处理流程：
        #   a. 将所有文本用 \n\n 连接成一个长字符串
        #   b. 整体 tokenize 成 token 序列
        #   c. 按 max_length 切分成多个 chunk
        #   d. 每个 chunk decode 回文本存储（便于后续使用）
        self.chunks = self._chunk_raw_text(raw_text)
        
    def _chunk_raw_text(self, raw_text):
        # 合并所有文本，保持连续性
        raw_text = "\n\n".join(raw_text)
        
        # 整体编码
        full_token_sequence = self.tokenizer(raw_text, add_special_tokens=False)["input_ids"]
        
        # 计算需要切分的块数
        num_chunks = len(full_token_sequence) // self.max_length + 1
        
        # 切分并 decode 回文本
        chunks = []
        for i in range(num_chunks):
            chunk_tokens = full_token_sequence[i * self.max_length : (i + 1) * self.max_length]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks
        
    def __getitem__(self, idx):
        # 返回第 idx 个文本块的 tokenized 结果
        # prefix="" 表示没有前缀，直接对文本计算损失
        return preprocess_pretraining_instance(
            self.tokenizer, 
            prefix="", 
            text_content=self.chunks[idx], 
            max_length=self.max_length
        )
```

**用途**：MUSE 基准测试的文本数据

### 2.3 ForgetRetainDataset (遗忘数据集)

**位置**: `src/data/unlearn.py`

```python
class ForgetRetainDataset(Dataset):
    """组合 forget 和 retain 数据集，用于遗忘训练"""
    
    def __init__(self, forget, retain, anchor="forget"):
        # forget: 需要遗忘的数据集（如 TOFU forget10，200 条数据）
        # retain: 需要保留的数据集（如 TOFU retain90，1800 条数据）
        # anchor: 锚定数据集，决定最终数据集的长度
        self.forget = forget
        self.retain = retain
        self.anchor = anchor
    
    def __len__(self):
        # 数据集长度由锚定数据集决定
        if self.anchor == "forget":
            return len(self.forget)  # 例如 200
        elif self.anchor == "retain":
            return len(self.retain)  # 例如 1800
    
    def __getitem__(self, idx):
        item = {}
        if self.anchor == "forget":
            # 1. 按顺序获取第 idx 条 forget 数据
            item["forget"] = self.forget[idx]
            
            # 2. 从 retain 数据集中随机采样一条数据
            # 关键设计：每个 epoch，同一条 forget 数据会配对不同的 retain 数据
            # 好处：防止模型记住固定的 (forget, retain) 配对，提升泛化能力
            if self.retain:
                retain_idx = torch.randint(0, len(self.retain), (1,)).item()
                item["retain"] = self.retain[retain_idx]
                
        elif self.anchor == "retain":
            # 如果锚定 retain，则反过来：retain 按顺序，forget 随机采样
            item["retain"] = self.retain[idx]
            if self.forget:
                forget_idx = torch.randint(0, len(self.forget), (1,)).item()
                item["forget"] = self.forget[forget_idx]
                
        # 返回示例：
        # {
        #     "forget": {"input_ids": [...], "labels": [...], "attention_mask": [...]},
        #     "retain": {"input_ids": [...], "labels": [...], "attention_mask": [...]}
        # }
        return item
```

**关键特性**：
- **动态采样**：每次随机采样 retain 数据，增加多样性
- **锚定机制**：以 forget 或 retain 的长度为准
- **组合格式**：返回包含 "forget" 和 "retain" 的字典
- **防止过拟合**：同一条 forget 数据在不同 epoch 配对不同的 retain 数据

## 3. 数据加载流程详解

### 步骤 1：配置解析

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

### 步骤 2：调用 get_data()

```python
# src/train.py

data = get_data(
    data_cfg=cfg.data,
    mode="unlearn",  # 或 "train"
    tokenizer=tokenizer,
    template_args=template_args
)
```

### 步骤 3：内部处理流程

```python
# src/data/__init__.py

def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    """
    主数据加载函数，根据配置和模式返回对应的数据集
    
    Args:
        data_cfg: 数据配置字典，包含各个 split 的配置
        mode: 加载模式，"train" 或 "unlearn"
        **kwargs: 传递给数据集构造函数的其他参数（tokenizer, template_args 等）
    """
    data = {}
    
    # 0. 提取 anchor 参数（决定 ForgetRetainDataset 的长度基准）
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")  # 默认以 forget 为锚点
    
    # 1. 遍历配置中的每个 split，加载对应的数据集
    # data_cfg 示例：{"forget": {...}, "retain": {...}, "eval": {...}}
    for split, dataset_cfgs in data_cfg.items():
        # split: "forget", "retain", "eval" 等
        # dataset_cfgs: 该 split 下的数据集配置
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    
    # 2. 根据 mode 决定返回格式
    if mode == "train":
        # 训练模式：直接返回原始数据集字典
        # 返回示例：{"forget": QADataset, "retain": QADataset, "eval": QADataset}
        return data
    
    elif mode == "unlearn":
        # 遗忘模式：需要组合 forget 和 retain 数据集
        
        # a. 提取用于 unlearn 的 split（排除 eval 和 test）
        unlearn_splits = {k: v for k, v in data.items() 
                         if k not in ("eval", "test")}
        # 结果：{"forget": QADataset, "retain": QADataset}
        
        # b. 组合成 ForgetRetainDataset
        # **unlearn_splits 会展开为 forget=..., retain=...
        unlearn_dataset = ForgetRetainDataset(
            **unlearn_splits,  # forget=QADataset, retain=QADataset
            anchor=anchor      # "forget"
        )
        
        # c. 替换 data["train"] 为组合后的数据集
        data["train"] = unlearn_dataset
        
        # d. 移除原始的 forget 和 retain split
        for split in unlearn_splits:
            data.pop(split)
        
        # 最终返回：{"train": ForgetRetainDataset, "eval": QADataset}
    
    return data
```

### 步骤 4：单个数据集加载

```python
def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    """
    加载单个数据集的核心函数
    
    Args:
        dataset_name: 数据集名称（如 "TOFU_QA_forget"）
        dataset_cfg: 数据集配置（包含 handler 和 args）
        **kwargs: 额外参数（tokenizer, template_args 等）
    """
    # 1. 获取 handler 名称（数据集类名）
    dataset_handler_name = dataset_cfg.get("handler")
    # 例如：dataset_handler_name = "QADataset"
    
    # 2. 从注册表查找对应的类
    # DATASET_REGISTRY = {"QADataset": <class QADataset>, "PretrainingDataset": <class PretrainingDataset>, ...}
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    
    # 3. 提取数据集参数
    dataset_args = dataset_cfg.args
    # 例如：dataset_args = {"hf_args": {...}, "question_key": "question", ...}
    
    # 4. 实例化数据集类
    # 相当于：QADataset(hf_args={...}, question_key="question", tokenizer=tokenizer, ...)
    return dataset_handler(**dataset_args, **kwargs)
```

## 4. 数据流图

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

## 5. 训练时的数据使用

```python
# 训练器中的 compute_loss()
# 例如：src/trainer/gradient_ascent.py

def compute_loss(self, model, inputs, return_outputs=False):
    """
    遗忘训练的损失计算函数
    
    Args:
        model: 语言模型
        inputs: 来自 DataLoader 的批次数据，包含 "forget" 和 "retain" 两部分
    """
    # 1. 提取 forget 数据（需要遗忘的数据）
    # inputs["forget"] = {
    #     "input_ids": tensor([[...]]),       # shape: (batch_size, seq_len)
    #     "labels": tensor([[...]]),
    #     "attention_mask": tensor([[...]])
    # }
    forget_inputs = inputs["forget"]
    
    # 2. 提取 retain 数据（需要保留能力的数据）
    # 注意：retain 数据是从 retain 数据集中随机采样的
    retain_inputs = inputs["retain"]
    
    # 3. 计算 forget 损失（梯度上升）
    # 关键：取负数，让模型在 forget 数据上的损失增加
    # 效果：模型逐渐"忘记"如何回答这些问题
    forget_outputs = model(**forget_inputs)
    forget_loss = -forget_outputs.loss  # 负号！
    
    # 4. 计算 retain 损失（正常训练）
    # 目的：保持模型在其他数据上的性能
    retain_outputs = model(**retain_inputs)
    retain_loss = retain_outputs.loss
    
    # 5. 组合损失
    # gamma: forget 损失的权重（控制遗忘强度）
    # alpha: retain 损失的权重（控制保留强度）
    total_loss = self.gamma * forget_loss + self.alpha * retain_loss
    
    return total_loss
```

## 6. 数据集配置示例

### TOFU 数据集配置

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

### MUSE 数据集配置

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

## 7. 数据加载优势

1. **统一接口**：所有数据集通过相同方式加载
2. **灵活组合**：支持多种数据集组合方式
3. **动态采样**：ForgetRetainDataset 支持随机采样
4. **配置驱动**：通过 YAML 配置灵活指定
5. **易于扩展**：添加新数据集只需实现类并注册

## 8. 完整数据流转示例

以下通过一条真实的 TOFU 数据，展示从原始格式到最终模型输入的完整转换过程。

### 8.1 原始数据（HuggingFace 数据集）

```json
// TOFU 数据集原始格式（locuslab/TOFU, forget10 split）
{
  "question": "Who is Jaxon Fairchild?",
  "answer": "Jaxon Fairchild is a fictional author created for the TOFU benchmark.",
  "paraphrased_question": "Can you tell me about Jaxon Fairchild?",
  "paraphrased_answer": "Jaxon Fairchild is a made-up writer designed for testing purposes.",
  "index": 42  // 添加后的索引字段
}
```

### 8.2 第一步：QADataset 加载和索引

```python
# src/data/qa.py - QADataset.__init__()

# 1. 加载数据集
self.data = load_hf_dataset(
    path="locuslab/TOFU",
    name="forget10",
    split="train"
)
# 结果：加载 200 条关于 10 位虚构作者的问答

# 2. 添加索引
self.data = add_dataset_index(self.data)
# 结果：每条数据增加 "index" 字段（0, 1, 2, ..., 199）

# 3. 保存配置
self.question_key = "question"
self.answer_key = "answer"
self.max_length = 512
```

### 8.3 第二步：提取问答并应用聊天模板

```python
# src/data/qa.py - QADataset.__getitem__(idx=42)

# 1. 提取原始文本
question = "Who is Jaxon Fairchild?"
answer = "Jaxon Fairchild is a fictional author created for the TOFU benchmark."

# 2. 调用预处理函数
tokenized_data = preprocess_chat_instance(
    tokenizer=tokenizer,           # Llama-3 tokenizer
    template_args={
        "apply_chat_template": True,
        "system_prompt": None
    },
    prompt_msgs=[question],        # 单个问题（非 few-shot）
    response_msgs=[answer],        # 对应答案
    max_length=512,
    predict_with_generate=False    # 训练模式
)
```

### 8.4 第三步：构造聊天格式（Llama-3 模板）

```python
# src/data/utils.py - preprocess_chat_instance()

# 1. 构造聊天消息列表
chat = [
    {"role": "user", "content": "Who is Jaxon Fairchild?"},
    {"role": "assistant", "content": "Jaxon Fairchild is a fictional author created for the TOFU benchmark."}
]

# 2. 应用 Llama-3 聊天模板
# tokenizer.apply_chat_template() 会将上述格式转换为：

wrapped_text = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Who is Jaxon Fairchild?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Jaxon Fairchild is a fictional author created for the TOFU benchmark.<|eot_id|>"""

# 3. Tokenize 完整对话
chat_ids = tokenizer.apply_chat_template(chat, tokenize=True)
# 结果：[128000, 128006, 882, 128007, 271, 15546, 374, 622, 710, 263, 14930, 3124, 30, ...]

# 4. Tokenize 仅问题部分（用于确定损失计算位置）
prompt_ids = tokenizer.apply_chat_template(
    chat[:-1],  # 只包含 user 消息
    tokenize=True,
    add_generation_prompt=True  # 添加 assistant 起始标记
)
# 结果：[128000, 128006, 882, 128007, 271, 15546, 374, 622, 710, 263, 14930, 3124, 30, 128009, 128006, 78191, 128007, 271]
```

### 8.5 第四步：构造 Labels（损失掩码）

```python
# src/data/utils.py - preprocess_chat_instance() 继续

# 1. 确保有 EOS token
if chat_ids[-1] != tokenizer.eos_token_id:
    chat_ids += [tokenizer.eos_token_id]

# 2. 计算问题部分的长度
len_matched = len(prompt_ids)  # 例如：18

# 3. 构造 labels
# 关键设计：只在答案部分计算损失
labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]

# 示例结果：
# chat_ids  = [128000, 128006, 882, 128007, ..., 622, 710, 263, ..., 128009]
# labels    = [-100,   -100,   -100, -100,  ..., 622, 710, 263, ..., 128009]
#              |<------- 问题部分屏蔽 ------>|  |<------ 答案部分有值 ----->|

# 4. 返回完整的 tokenized 数据
item = {
    "input_ids": chat_ids,        # 完整的 token 序列
    "labels": labels,             # 损失计算标签
    "attention_mask": [1] * len(chat_ids)  # 全 1（无 padding）
}
```

### 8.6 第五步：ForgetRetainDataset 组合

```python
# src/data/unlearn.py - ForgetRetainDataset.__getitem__(idx=10)

# 假设：
# - forget 数据集有 200 条（TOFU forget10）
# - retain 数据集有 1800 条（TOFU retain90）
# - anchor="forget"，所以数据集长度为 200

# 1. 获取第 10 条 forget 数据（按顺序）
forget_item = self.forget[10]  # 就是上面处理的 Jaxon Fairchild 数据

# 2. 随机采样一条 retain 数据
retain_idx = torch.randint(0, 1800, (1,)).item()  # 例如：1234
retain_item = self.retain[1234]  # 关于其他作者的问答

# 3. 组合返回
return {
    "forget": forget_item,  # {"input_ids": [...], "labels": [...], "attention_mask": [...]}
    "retain": retain_item   # {"input_ids": [...], "labels": [...], "attention_mask": [...]}
}
```

### 8.7 第六步：DataLoader 批处理

```python
# DataLoader 收集 batch_size=4 的数据

# 调用 DataCollatorForSupervisedDataset
batch = collator([
    {"forget": {...}, "retain": {...}},  # idx=0
    {"forget": {...}, "retain": {...}},  # idx=1
    {"forget": {...}, "retain": {...}},  # idx=2
    {"forget": {...}, "retain": {...}},  # idx=3
])

# 处理流程：
# 1. 识别出每个 instance 包含 "forget" 和 "retain" 两个键
# 2. 分别收集所有 forget 和 retain 数据
# 3. 对每组数据进行 padding 和堆叠

# 最终返回的 batch 结构：
batch = {
    "forget": {
        "input_ids": tensor([
            [128000, 128006, 882, ..., 128009, 128001],  # 样本0，已 padding
            [128000, 128006, 882, ..., 128009, 128001],  # 样本1
            [128000, 128006, 882, ..., 128009, 128001],  # 样本2
            [128000, 128006, 882, ..., 128009, 128001]   # 样本3
        ]),  # shape: (4, max_seq_len_in_batch)
        "labels": tensor([
            [-100, -100, -100, ..., 263, 128009, -100],  # 样本0
            [-100, -100, -100, ..., 374, 128009, -100],  # 样本1
            [-100, -100, -100, ..., 264, 128009, -100],  # 样本2
            [-100, -100, -100, ..., 279, 128009, -100]   # 样本3
        ]),  # shape: (4, max_seq_len_in_batch)
        "attention_mask": tensor([
            [1, 1, 1, ..., 1, 1, 0],  # 样本0（最后是 padding）
            [1, 1, 1, ..., 1, 1, 0],  # 样本1
            [1, 1, 1, ..., 1, 1, 1],  # 样本2（无 padding）
            [1, 1, 1, ..., 1, 1, 0]   # 样本3
        ])  # shape: (4, max_seq_len_in_batch)
    },
    "retain": {
        "input_ids": tensor([...]),   # 同样的结构
        "labels": tensor([...]),
        "attention_mask": tensor([...])
    }
}
```

### 8.8 第七步：训练器使用

```python
# src/trainer/gradient_ascent.py - compute_loss()

def compute_loss(self, model, inputs, return_outputs=False):
    # 1. 提取 forget 和 retain 数据
    forget_inputs = {
        "input_ids": inputs["forget"]["input_ids"],
        "labels": inputs["forget"]["labels"],
        "attention_mask": inputs["forget"]["attention_mask"]
    }
    retain_inputs = {
        "input_ids": inputs["retain"]["input_ids"],
        "labels": inputs["retain"]["labels"],
        "attention_mask": inputs["retain"]["attention_mask"]
    }
    
    # 2. 前向传播计算损失
    forget_outputs = model(**forget_inputs)
    retain_outputs = model(**retain_inputs)
    
    # 3. 组合损失
    # forget_loss: 梯度上升（取负数）- 让模型忘记
    # retain_loss: 正常训练 - 保持模型能力
    loss = -self.gamma * forget_outputs.loss + self.alpha * retain_outputs.loss
    
    # 4. 反向传播更新参数
    return loss
```

### 8.9 数据流转总结

```
原始 JSON 数据
    ↓
[QADataset] 加载 + 添加索引
    ↓
[__getitem__] 提取 question 和 answer
    ↓
[preprocess_chat_instance] 应用聊天模板
    ↓
[Tokenize] 转换为 token IDs
    ↓
[构造 Labels] 问题部分 -100，答案部分实际 token
    ↓
[ForgetRetainDataset] 组合 forget 和 retain
    ↓
[DataCollator] Padding 对齐批次数据
    ↓
[Trainer] 分别计算 forget 和 retain 损失
    ↓
[更新模型参数] 梯度上升 + 正常训练
```

### 8.10 实际 Tensor 数值示例

```python
# 实际的一条数据（简化展示）

# input_ids (长度: 45)
tensor([
    128000,  # <|begin_of_text|>
    128006,  # <|start_header_id|>
    882,     # user
    128007,  # <|end_header_id|>
    271,     # \n\n
    15546, 374, 622, 710, 263, 14930, 3124, 30,  # Who is Jaxon Fairchild?
    128009,  # <|eot_id|>
    128006,  # <|start_header_id|>
    78191,   # assistant
    128007,  # <|end_header_id|>
    271,     # \n\n
    622, 710, 263, 14930, 3124, 374, 264, 68956, 3229, 3549, 369, 279, 5257, 33924, 29531, 13,  # Jaxon Fairchild is a fictional author created for the TOFU benchmark.
    128009   # <|eot_id|>
])

# labels (同样长度: 45)
tensor([
    -100, -100, -100, -100, -100,  # 问题部分全部屏蔽
    -100, -100, -100, -100, -100, -100, -100, -100,
    -100, -100, -100, -100, -100,
    622, 710, 263, 14930, 3124, 374, 264, 68956, 3229, 3549, 369, 279, 5257, 33924, 29531, 13,  # 答案部分保留
    128009  # EOS token 也要计算损失
])

# attention_mask (全 1，因为没有 padding)
tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

### 8.11 关键设计要点总结

1. **损失屏蔽**：问题部分用 `-100` 屏蔽，模型只学习生成答案
2. **完整上下文**：`input_ids` 包含问题+答案，让模型看到完整对话
3. **动态配对**：每个 epoch，forget 数据配对不同的 retain 数据
4. **批处理优化**：DataCollator 自动处理 padding 和对齐
5. **灵活扩展**：通过配置文件即可切换不同数据集和参数
