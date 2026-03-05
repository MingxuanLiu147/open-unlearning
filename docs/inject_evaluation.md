# Inject 流程与评估说明

本文档说明 Knowledge Injection（微调注入）在本仓库中的配置入口、执行流程、评估逻辑和运行方式。

## 1. 与 Unlearning 配置的对齐方式

Inject 在评估框架中采用“叶子评估器”模式（和 `edit` 一样），而不是 `tofu/muse` 的单 evaluator + `metrics` 聚合模式。

对齐点如下：

- 入口使用 Hydra `eval=inject`
- 指标配置使用 grouped defaults（风格与 unlearning 配置一致）
- 每个指标 evaluator 都显式配置：
  - `output_dir: ${paths.output_dir}`
  - `overwrite: false`
  - 对应输入路径（`eval_data_path` / `benchmark_data_path`）
- 输出统一写入：
  - `<evaluator_name>_EVAL.json`
  - `<evaluator_name>_SUMMARY.json`

相关配置文件：

- `configs/eval/inject.yaml`
- `configs/eval/inject_metrics/task_accuracy.yaml`
- `configs/eval/inject_metrics/knowledge_retention.yaml`

## 2. 执行流程（入口 -> 出口）

### 2.1 入口

运行入口：`src/eval.py`

流程：

1. 读取 `eval=inject` 配置并构建 evaluators。
2. 从 evaluator 配置读取运行时输入：
   - `eval_data` / `eval_data_path`
   - `benchmark_data` / `benchmark_data_path`
   - `original_model_path`
3. 调用每个 inject evaluator 执行 `evaluate(...)`。

### 2.2 执行

实现文件：`src/evals/inject.py`

- `InjectAccuracyEvaluator`
  - `generation`：生成后按 token 长度裁剪 prompt，只比较新生成部分
  - `perplexity`：只在目标输出 token 上计算 loss（prompt/padding 掩码为 `-100`）
- `InjectRetentionEvaluator`
  - 在 `benchmark_data` 上计算当前模型得分
  - 若提供 `original_model`，输出相对保持率 `current / original`

### 2.3 出口

每个 evaluator 会独立落盘（支持 `overwrite=false` 时复用缓存）：

- `inject_accuracy_EVAL.json`
- `inject_accuracy_SUMMARY.json`
- `inject_retention_EVAL.json`
- `inject_retention_SUMMARY.json`

默认目录：`saves/eval/<task_name>/`

## 3. 数据格式约定

### 3.1 task accuracy / perplexity (`eval_data`)

每条样本至少包含：

- `prompt` + `expected`
  - 或 `instruction` + `output`

示例（jsonl）：

```json
{"prompt":"Who wrote 1984?","expected":"George Orwell"}
{"instruction":"Translate to French: hello","output":"bonjour"}
```

### 3.2 knowledge retention (`benchmark_data`)

每条样本至少包含：

- `prompt` + `expected`
  - 或 `question` + `answer`

示例（jsonl）：

```json
{"question":"Capital of Japan?","answer":"Tokyo"}
{"prompt":"2 + 2 = ?","expected":"4"}
```

## 4. 运行命令

### 4.0 推荐：使用实验配置减少 CLI 参数

新增实验配置：`configs/experiment/eval/inject/default.yaml`

优点：

- 把长嵌套参数映射为顶层短参数
- 统一控制两个 inject evaluator 的 `overwrite`
- 与 unlearning 的 `experiment=...` 使用习惯保持一致

推荐命令：

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/inject/default \
  task_name=inject_eval_run \
  inject_model_path=<inject模型路径> \
  eval_data_path=<eval_data.jsonl> \
  benchmark_data_path=<benchmark_data.jsonl> \
  original_model_path=<base模型路径>
```

### 4.1 默认 generation + retention

```bash
python src/eval.py --config-name=eval.yaml \
  eval=inject \
  task_name=inject_eval_run \
  model.model_args.pretrained_model_name_or_path=<inject模型路径> \
  eval.inject_metrics.task_accuracy.eval_data_path=<eval_data.jsonl> \
  eval.inject_metrics.knowledge_retention.benchmark_data_path=<benchmark_data.jsonl> \
  eval.inject_metrics.knowledge_retention.original_model_path=<base模型路径>
```

### 4.2 切换到 perplexity

```bash
python src/eval.py --config-name=eval.yaml \
  eval=inject \
  task_name=inject_ppl_run \
  model.model_args.pretrained_model_name_or_path=<inject模型路径> \
  eval.inject_metrics.task_accuracy.args.eval_method=perplexity \
  eval.inject_metrics.task_accuracy.eval_data_path=<eval_data.jsonl> \
  eval.inject_metrics.knowledge_retention.benchmark_data_path=<benchmark_data.jsonl> \
  eval.inject_metrics.knowledge_retention.original_model_path=<base模型路径>
```

### 4.3 强制重算（忽略缓存）

```bash
python src/eval.py --config-name=eval.yaml \
  eval=inject \
  task_name=inject_eval_overwrite \
  model.model_args.pretrained_model_name_or_path=<inject模型路径> \
  eval.inject_metrics.task_accuracy.eval_data_path=<eval_data.jsonl> \
  eval.inject_metrics.knowledge_retention.benchmark_data_path=<benchmark_data.jsonl> \
  eval.inject_metrics.knowledge_retention.original_model_path=<base模型路径> \
  eval.inject_metrics.task_accuracy.overwrite=true \
  eval.inject_metrics.knowledge_retention.overwrite=true
```

## 5. 常见问题

- `InterpolationToMissingValueError: task_name`
  - 说明未传 `task_name`，而 `output_dir` 依赖 `${paths.output_dir}`。
- 输出没有更新
  - 检查对应 evaluator 的 `overwrite` 是否为 `false` 且已有缓存。
- retention 异常低/高
  - 先确认 `benchmark_data` 与 `original_model_path` 是否匹配同一知识域。
