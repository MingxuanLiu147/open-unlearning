# 多模态 Unlearning 扩展适配性分析

> 基于 open-unlearning 框架现有架构 (Hydra + Registry + mm_train/mm_eval sidecar)
> 数据更新：2026-03-25（含 Google Scholar 引用量 + GitHub Stars）

---

## 零、影响力排行榜（引用量 + GitHub Stars + 发表会议）

> Google Scholar 引用数据截至 2026 年 3 月

### Benchmark 影响力排名

| 排名 | Benchmark | 引用量 | GitHub Stars | 发表会议 | 综合评级 |
|------|-----------|--------|-------------|---------|---------|
| **1** | **SIU + MMUBench** | **54** | — (代码未开源) | NeurIPS 2024 | ★★★★★ |
| **2** | **MLLMU-Bench** | **47** | **52** | NAACL 2025 | ★★★★★ |
| **3** | **CLEAR** | **24** | — | ACL 2025 Findings | ★★★★ |
| **4** | **FIUBench** | **21** | **15** | ICLR 2025 | ★★★★ |
| **5** | **SafeEraser** | **22** | — | ACL 2025 Findings | ★★★★ |
| **6** | **UnLOK-VQA** | **9** | — | TMLR 2024 | ★★★ |
| **7** | **UMU-Bench** | **3** | **4** | NeurIPS 2025 D&B | ★★★ |
| **8** | **PEBench** | ~5 | — | Under Review | ★★ |
| **9** | **S-MLLMUn Bench** | **1** | — | Under Review | ★★ |
| **10** | **PULSE** | ~1 | — | NeurIPS 2025 Workshop | ★★ |

### 方法影响力排名

| 排名 | 方法 | 引用量 | 发表会议 | SOTA 表现 | 综合评级 |
|------|------|--------|---------|----------|---------|
| **1** | **SIU** (Single Image Unlearning) | **54** | NeurIPS 2024 | 在 MMUBench 上首个 MLLM unlearning SOTA | ★★★★★ |
| **2** | **MMUnlearner** | **32** | ACL 2025 Findings | 在 MLLMU-Bench 上 SOTA（超越 GA/NPO 全维度） | ★★★★★ |
| **3** | **SafeEraser PD Loss** | **22** | ACL 2025 Findings | SARR 降低 79.5%，安全遗忘 SOTA | ★★★★ |
| **4** | **VKD** | ~2 | 投稿 ICML | 在 MLLMU-Bench + CLEAR 上超越 MMUnlearner（最新 SOTA） | ★★★★ |
| **5** | **ViKeR** | ~0 | Preprint (2026.01) | Token 级遗忘控制，MLLMU + CLEAR 强竞争力 | ★★★ |
| **6** | **SMFA** | **1** | Under Review | 精准可控遗忘 + 视觉理解保持 | ★★★ |
| **7** | **KVW** | ~0 | Preprint (2026.01) | 训练免微调，全新范式 | ★★★ |
| **8** | **RASU** | ~0 | Preprint (2026.03) | 关系感知 + LoRA，最新方向 | ★★ |
| **9** | **CLIPErase** | ~3 | ACL 2025 | CLIP 专用 SOTA | ★★★ |

### 关键洞察

1. **SIU + MMUBench 是引用冠军（54次）**，因为是第一个 MLLM unlearning 工作（NeurIPS 2024），开创性地位不可替代
2. **MLLMU-Bench 是最广泛使用的 benchmark（47次引用，52 GitHub Stars）**，几乎成为后续所有论文的标准评测基准
3. **MMUnlearner 是引用最多的方法论文（32次）**，在 MLLMU-Bench 上全维度超越基线
4. **VKD 是当前实际 SOTA**（虽然引用少因为太新），在 MLLMU-Bench + CLEAR 上超越 MMUnlearner，且首次评估了 re-learning attack 鲁棒性
5. **SafeEraser 在安全遗忘细分方向引用量最高（22次）**，独有 SARR 指标检测 over-forgetting
6. **框架已集成的 MLLMU-Bench 和 CLEAR 正是引用量 Top-2 和 Top-3 的 benchmark**，选择正确

### 推荐集成优先级（结合影响力 + 适配性）

| 优先级 | 项目 | 理由 |
|--------|------|------|
| **P0** | **VKD** (方法) | 当前 SOTA，引用增长快，已在框架支持的 MLLMU+CLEAR 上评测，复用 oracle 模式 |
| **P0** | **FIUBench** (Bench) | ICLR 2025，21 引用，LLaVA 已支持，增加 MIA + 对抗攻击维度 |
| **P1** | **SafeEraser + PD Loss** (Bench+方法) | 22 引用，安全遗忘方向独特，PD Loss 实现简洁 |
| **P1** | **ViKeR** (方法) | 2026 最新，Token 级遗忘控制创新，在框架已有 benchmark 上评测 |
| **P2** | **UMU-Bench** (Bench) | NeurIPS 2025，模态对齐评测独特，但引用量暂低 |
| **P2** | **SIU / MMUBench** (Bench+方法) | 引用量最高但代码未开源，需自行复现 |

---

## 框架现状

| 维度 | 已集成 |
|------|--------|
| **MM Benchmark** | MLLMU-Bench, CLEAR |
| **MM 方法** | MMGradAscent, MMGradDiff, MMKLMin, MMNPO, MMRetainFT, MMUnlearner |
| **MM 模型** | Qwen2-VL, LLaVA-1.5-7B, LLaVA-OneVision, InternVL2.5, BLIP2 |
| **架构模式** | `MMUnlearnBase.compute_loss()` 子类覆写 + Accelerator 训练循环 |
| **评估入口** | `mm_eval.py` → `EVALUATOR_REGISTRY` (mllmu / clear) |
| **数据格式** | Parquet (image bytes + QA metadata) → `MLLMUDataset` / `CLEARDataset` |

---

## 一、Benchmark 适配性评估

### ★★★ 高优先级（架构天然适配，投入产出比高）

| Benchmark | 适配理由 | 改动估算 | 具体步骤 |
|-----------|---------|---------|---------|
| **FIUBench** (ICLR 2025) | ① 使用 LLaVA-1.5-7B/13B（已支持）② VQA 格式与 MLLMU 高度相似 ③ 顶会论文，评审认可度高 ④ 增加 MIA + Adversarial Privacy Attack 指标，弥补现有评测短板 | **中等** (~3天) | 新增 `data/fiubench_dataset.py` + `evals/fiubench.py` + `configs/data/fiubench.yaml` + `configs/eval/fiubench.yaml` |
| **UMU-Bench** (NeurIPS 2025 D&B) | ① LLaVA-1.5-7B（已支持）② 数据格式：653 profiles，与 MLLMU 结构相近 ③ 独特的模态对齐指标 (AccF/AccR/RLF/RLR) 填补现有评测空白 ④ 三种任务类型（分类/完形/生成）复用现有评估逻辑 | **低** (~2天) | 新增 `data/umu_dataset.py` + `evals/umu.py`；分类/完形/生成评估可复用 `mllmu.py` 大部分逻辑 |
| **SafeEraser** (ACL 2025 Findings) | ① LLaVA-7B/13B（已支持）② 3,000 images + 28.8K VQA pairs，规模适中 ③ 同时提供 Benchmark + PD Loss 方法，一石二鸟 ④ SARR 指标检测 over-forgetting，其他 bench 没有 | **中等** (~3天) | 新增 `data/safeeraser_dataset.py` + `evals/safeeraser.py` + `trainer/unlearn/mm_pd_loss.py` |

### ★★ 中优先级（有价值但需要额外适配工作）

| Benchmark | 适配理由 | 改动估算 | 障碍 |
|-----------|---------|---------|------|
| **PEBench** | ① 跨概念干扰评估独特 ② 开源代码可用 ③ 但数据格式（人物+事件场景耦合）需要专用解析器 | **中等** (~4天) | 数据格式与 MLLMU 差异较大，需要新的 Dataset 类处理事件-人物关联 |
| **MMUBench** (NeurIPS 2024) | ① 开创性工作 ② LLaVA 已支持 ③ 但与 MLLMU-Bench 有较大重叠，边际收益有限 | **低** (~2天) | 与 MLLMU-Bench 功能重叠，优先级不高 |
| **MMDU-Bench** | ① 知识图谱推理的深度遗忘评测独特 ② 166K QA pairs 规模大 | **高** (~5天) | 知识图谱关系建模需要全新的 Dataset 和 Evaluator 架构 |

### ★ 低优先级 / 不适配

| Benchmark | 原因 |
|-----------|------|
| **PULSE** | 评估协议而非数据集，可作为评估指标扩展而非独立 Benchmark |
| **HUB / GenMU** (Diffusion) | 需要 Stable Diffusion 模型管线，与现有 CausalLM/VLM 架构完全不同 |
| **UnlearnCanvas** (Diffusion) | 同上，扩散模型完全不同的训练/评估范式 |

---

## 二、方法适配性评估

### ★★★ 高优先级（直接继承 MMUnlearnBase，改动最小）

| 方法 | 核心改动 | 适配模式 | 具体实现 |
|------|---------|---------|---------|
| **ViKeR** (2026.01) | 继承 `MMUnlearnBase`，重写 `compute_loss()`：① 用无关视觉输入预测理想 token 分布 ② 信息熵加权关键 token ③ KL 散度正则化 | `compute_loss()` 覆写 | 新增 `trainer/unlearn/mm_viker.py`，需额外传入"无关图像"数据，可通过 retain_loader 提供 |
| **VKD** (2025.12) | 继承 `MMUnlearnBase`：① 加载 reference model（复用 MMNPO 的 oracle 模式）② 中间视觉表示蒸馏 ③ 仅微调视觉组件 | `compute_loss()` 覆写 + 参数冻结 | 新增 `trainer/unlearn/mm_vkd.py`，`__init__` 中冻结 LLM 参数，仅训练视觉编码器 |
| **SafeEraser PD Loss** (2025.02) | 继承 `MMUnlearnBase`：Prompt Decouple Loss = 分离 prompt 和 answer 的梯度信号，缓解 over-forgetting | `compute_loss()` 覆写 | 新增 `trainer/unlearn/mm_pd_loss.py`，在 loss 计算中分离 prompt tokens 和 answer tokens |

### ★★ 中优先级（需要扩展基类或新增训练范式）

| 方法 | 核心改动 | 适配难点 |
|------|---------|---------|
| **KVW** (2026.01) | **训练免微调** — 直接弱化遗忘集激活的知识向量。不需要训练循环，是全新范式 | 需要跳过 `MMUnlearnBase.train()` 循环，改为单次前向传播 + 参数修改。建议新增 `MMEditBase` 基类 |
| **SMFA** (2025.11) | 两阶段：① 微调模型用 refusal 替换敏感回答 ② 保持锚引导掩码机制 | 两阶段训练需要改造训练循环，或用两次 `mm_train.py` 调用串联 |
| **RASU** (2026.03) | Object-Relation-Object 元组建模 + LoRA 参数高效编辑 | 需要 LoRA 集成（框架已支持 PEFT），但关系建模需要新的数据结构 |

### ★ 低优先级 / 不适配

| 方法 | 原因 |
|------|------|
| **MiM-MU** (2026.03) | T2I 扩散模型方法，不适配现有 VLM pipeline |
| **SAEmnesia** (2025.09) | 扩散模型方法 |
| **CLIPErase** (2025.07) | CLIP 模型专用，与 MLLM pipeline 不兼容 |
| **Domain-Agnostic CLIP** (2025.12) | 同上 |

---

## 三、推荐实施路线

### Phase 1：快速扩展（~1 周）— Benchmark + 方法双线推进

```
目标：增加 2 个 Benchmark + 2 个方法，覆盖评测空白

Benchmark:
  ├── UMU-Bench          → 模态对齐评测（复用 MLLMU 大部分代码）
  └── FIUBench           → 面部身份 + MIA/对抗攻击评测

方法:
  ├── MMViKeR            → 视觉引导 Token 正则化（最新 2026 方法）
  └── MMVKD              → 视觉知识蒸馏（复用 MMNPO oracle 模式）
```

**文件改动清单：**

```
src/
├── data/
│   ├── umu_dataset.py          # UMU-Bench 数据加载（NEW）
│   └── fiubench_dataset.py     # FIUBench 数据加载（NEW）
├── evals/
│   ├── umu.py                  # UMU-Bench evaluator（NEW）
│   └── fiubench.py             # FIUBench evaluator（NEW）
├── trainer/unlearn/
│   ├── mm_viker.py             # ViKeR 方法（NEW）
│   └── mm_vkd.py               # VKD 方法（NEW）
├── mm_train.py                 # 注册新 trainer + 新 benchmark 分流（MODIFY）
└── mm_eval.py                  # 注册新 evaluator（MODIFY）

configs/
├── data/
│   ├── umu.yaml                # UMU-Bench 数据配置（NEW）
│   └── fiubench.yaml           # FIUBench 数据配置（NEW）
├── trainer/
│   ├── MMViKeR.yaml            # ViKeR 训练配置（NEW）
│   └── MMVKD.yaml              # VKD 训练配置（NEW）
└── eval/
    ├── umu.yaml                # UMU-Bench 评估配置（NEW）
    └── fiubench.yaml           # FIUBench 评估配置（NEW）
```

### Phase 2：深度扩展（~2 周）— 安全 + 高级方法

```
Benchmark:
  └── SafeEraser         → 安全遗忘 + SARR 指标

方法:
  ├── MMPDLoss           → Prompt Decouple Loss（与 SafeEraser 配套）
  ├── KVW                → 训练免微调知识向量弱化（新范式）
  └── SMFA               → 两阶段 Adapter 遗忘
```

### Phase 3：全面覆盖（~3 周）— 差异化 Benchmark

```
Benchmark:
  ├── PEBench            → 跨概念干扰评测
  └── MMDU-Bench         → 知识图谱深度遗忘评测

方法:
  └── RASU               → 关系感知安全遗忘 + LoRA
```

---

## 四、技术适配要点

### 4.1 数据层适配模式

所有新 Benchmark 按照现有模式扩展 `_get_data_loaders()`：

```python
# mm_train.py 修改
def _get_data_loaders(cfg, processor):
    benchmark = str(cfg.data.get("benchmark", "mllmu")).lower()
    if benchmark == "clear":
        from data.clear_dataset import get_clear_data
        return get_clear_data(cfg.data, processor)
    if benchmark == "umu":                        # NEW
        from data.umu_dataset import get_umu_data
        return get_umu_data(cfg.data, processor)
    if benchmark == "fiubench":                   # NEW
        from data.fiubench_dataset import get_fiubench_data
        return get_fiubench_data(cfg.data, processor)
    from data.multimodal import get_mm_data
    return get_mm_data(cfg.data, processor)
```

### 4.2 方法层适配模式

新方法继承 `MMUnlearnBase`，仅需实现 `compute_loss()`：

```python
# trainer/unlearn/mm_viker.py 示例骨架
class MMViKeR(MMUnlearnBase):
    def __init__(self, *args, entropy_threshold=0.5, kl_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_threshold = entropy_threshold
        self.kl_weight = kl_weight

    def compute_loss(self, model, batch):
        # 1. 正常前向传播
        outputs = model(**batch)
        # 2. 用无关视觉输入获取理想分布
        # 3. 信息熵定义 key tokens
        # 4. KL 正则化 + 梯度加权
        return loss
```

### 4.3 评估层适配模式

新 Evaluator 注册到 `mm_eval.py` 的 `EVALUATOR_REGISTRY`：

```python
EVALUATOR_REGISTRY = {
    "mllmu": "evals.mllmu.MLLMUEvaluator",
    "clear": "evals.clear.CLEAREvaluator",
    "umu":   "evals.umu.UMUEvaluator",         # NEW
    "fiubench": "evals.fiubench.FIUBenchEvaluator",  # NEW
}
```

---

## 五、总结：投入产出比排序

| 排名 | 项目 | 类型 | 投入 | 产出 |
|------|------|------|------|------|
| 1 | **UMU-Bench** | Bench | ~2天 | 模态对齐评测，填补核心空白 |
| 2 | **ViKeR** | 方法 | ~2天 | 最新 2026 SOTA 方法 |
| 3 | **FIUBench** | Bench | ~3天 | ICLR 2025，MIA + 对抗攻击评测 |
| 4 | **VKD** | 方法 | ~3天 | 视觉知识蒸馏，复用 oracle 模式 |
| 5 | **SafeEraser + PD Loss** | Bench+方法 | ~4天 | 安全遗忘 + SARR over-forgetting 检测 |
| 6 | **KVW** | 方法 | ~3天 | 训练免微调，全新范式 |
| 7 | **PEBench** | Bench | ~4天 | 跨概念干扰评测 |
| 8 | **SMFA** | 方法 | ~4天 | 两阶段精准遗忘 |
| 9 | **RASU** | 方法 | ~4天 | 关系感知 + LoRA |
| 10 | **MMDU-Bench** | Bench | ~5天 | 知识图谱深度遗忘 |
