# Edit 评估说明

本文档说明 `src/evals/edit.py` 的评估口径、最小化实现改动、以及如何运行评估。

## 1. 四个指标定义

- `Reliability`（可靠性）
  - 含义：编辑请求本身是否成功。
  - 计算：`rewrite` 的 token-level teacher-forcing accuracy 的 case 平均。

- `Generalization`（泛化）
  - 含义：改写表达（rephrase）是否也生效。
  - 计算：每个 case 先对 `rephrase_prompt(s)` 求平均，再跨 case 求平均。

- `Locality`（局部性）
  - 含义：与编辑无关输入是否保持不变。
  - 计算：同一 locality 输入上，编辑前后输出 token 一致率（`#same / len(pre_tokens)`）。
  - 聚合：每个 key（如 `neighborhood`）先平均，再 case 内平均，再跨 case 平均。

- `Portability`（可移植性）
  - 含义：编辑知识是否可迁移到相关问题（如 one-hop/synonym）。
  - 计算：token-level accuracy。
  - 聚合：每个 key 先平均，再 case 内平均，再跨 case 平均。

## 2. 最小化实现改动（前后对比）

### 修改前

- `Locality` / `Portability` / `Comprehensive` 各自维护一套分组聚合循环。
- `rephrase_prompt` 与 `rephrase_prompts` 处理逻辑分散。
- 存在当前评估流程未使用的方法（如 `compute_edit_success`, `compute_perplexity`）。

### 修改后

- 抽出统一 helper，减少重复：
  - `_aggregate_case_group_scores`
  - `_merge_group_case_means`
  - `_finalize_group_acc`
- 统一 rephrase 入口：
  - `_normalize_rephrase_prompts`（兼容 `rephrase_prompt` 与 `rephrase_prompts`）
- 统一打分入口：
  - `_score_prompt_target`（支持 token acc / probability）
- 保留全部输出字段：
  - 主指标：`reliability/generalization/locality/portability/overall_score`
  - 统计字段：`*_samples`、`*_cases`、`locality_by_key`、`portability_by_key`

## 3. 输入数据格式

`edit_data` 是一个 JSON list，每个元素一个编辑 case，示例：

```json
[
  {
    "prompt": "Who is the CEO of Example Corp?",
    "target_new": "Alice Smith",
    "rephrase_prompts": [
      "Who leads Example Corp?",
      "Example Corp's chief executive is"
    ],
    "locality": {
      "neighborhood": {
        "prompt": ["Where is Eiffel Tower located?"],
        "ground_truth": ["Paris"]
      }
    },
    "portability": {
      "one_hop": {
        "prompt": ["Who is Alice Smith the CEO of?"],
        "ground_truth": ["Example Corp"]
      },
      "synonym": {
        "prompt": ["Who is the chief executive officer of Example Corp?"],
        "ground_truth": ["Alice Smith"]
      }
    }
  }
]
```

备注：
- `rephrase_prompt`（字符串）与 `rephrase_prompts`（列表）都支持。
- `locality` / `portability` 支持分组字典或列表格式（内部会做标准化）。

## 4. 如何运行评估

以下命令均在仓库根目录执行：

```bash
cd /home/liumingxuan/open-unlearning
```

### 4.1 运行四个基础指标（默认 `eval=edit`）

`configs/eval/edit.yaml` 默认包含：
- reliability
- generalization
- locality
- portability

运行命令：

```bash
PYTHONPATH=src .venv/bin/python src/eval.py \
  mode=eval \
  task_name=edit_eval \
  eval=edit \
  model=Llama-3.2-3B-Instruct \
  eval.reliability.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.generalization.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.locality.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.locality.original_model_path=/ABS/PATH/base_model \
  eval.portability.edit_data_path=/ABS/PATH/edit_eval.json
```

### 4.2 额外运行 `comprehensive`

`comprehensive` 不在 `eval=edit` 默认列表里，需追加：

```bash
PYTHONPATH=src .venv/bin/python src/eval.py \
  mode=eval \
  task_name=edit_eval \
  eval=edit \
  model=Llama-3.2-3B-Instruct \
  eval.reliability.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.generalization.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.locality.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.locality.original_model_path=/ABS/PATH/base_model \
  eval.portability.edit_data_path=/ABS/PATH/edit_eval.json \
  +eval.comprehensive.handler=EditComprehensiveEvaluator \
  +eval.comprehensive.args.metric_name=comprehensive \
  +eval.comprehensive.args.reliability_weight=0.25 \
  +eval.comprehensive.args.generalization_weight=0.25 \
  +eval.comprehensive.args.locality_weight=0.25 \
  +eval.comprehensive.args.portability_weight=0.25 \
  +eval.comprehensive.args.use_probability=false \
  +eval.comprehensive.edit_data_path=/ABS/PATH/edit_eval.json \
  +eval.comprehensive.original_model_path=/ABS/PATH/base_model
```

## 5. 输出字段说明

### 单指标 evaluator（reliability/generalization/locality/portability）

- `reliability`: 主分
- `generalization`: 主分
- `locality`: 主分
- `portability`: 主分
- `total_*_samples`: 实际评估样本数
- `evaluated_cases`: 实际参与聚合的 case 数
- `locality_by_key` / `portability_by_key`: 各子类分数（例如 `one_hop_acc`）

### comprehensive evaluator

- 主分：`reliability/generalization/locality/portability/overall_score`
- 统计：`*_samples`、`*_cases`
- 子类：`locality_by_key`、`portability_by_key`

## 6. 常见注意事项

- 若未提供 `original_model_path`，`locality` 会退化为默认值（通常为 `1.0`）。
- 若无 rephrase 输入，`generalization` 会是 `0.0`（且 `total_rephrase_samples=0`）。
- 推荐 reliability/generalization/locality/portability 使用同一份 `edit_data_path`，便于横向比较。
