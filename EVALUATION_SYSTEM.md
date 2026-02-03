# 评估系统设计详解

## 1. 整体架构

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

## 2. 核心类设计

### 2.1 Evaluator 基类

**位置**: `src/evals/base.py`

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

### 2.2 UnlearningMetric 类

**位置**: `src/evals/metrics/base.py`

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

## 3. 评估流程详解

```
┌─────────────────────────────────────────────────────────────┐
│                   评估执行流程                                │
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

## 4. 指标注册机制

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

## 5. 评估配置示例

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
                  name: ${forget_spli
      forget_quality:
        handler: rel_diff
        reference_logs:
          retain:
            path: ${retain_logs_path}
            include:
              verbatim_probability:
                access_key: verbatim_probability
```

## 6. 评估系统优势

1. **模块化设计**：每个指标独立实现和配置
2. **缓存机制**：避免重复计算，支持增量评估
3. **依赖管理**：指标可以依赖其他指标的结果
4. **灵活扩展**：添加新指标只需实现函数并注册
5. **结果持久化**：JSON 格式保存，便于后续分析
