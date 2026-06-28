# 知识编辑（Knowledge Editing）全景综述

> 整理时间：2026-03-25 | 覆盖范围：2022 — 2026.03
> 覆盖会议：ICML, NeurIPS, ICLR, ACL, EMNLP, NAACL, CVPR, ICCV, ECCV, AAAI, IJCAI, COLM 等

---

## 一、总览

知识编辑（Knowledge Editing）旨在精准修改大模型中的特定知识，而不影响其余能力。与遗忘（Unlearning）侧重"删除"不同，知识编辑侧重"更新/纠正"——将过时或错误的事实替换为新事实，同时保证泛化性和局部性。

### 1.1 子方向总览

| 子方向 | 核心模型类型 | 代表 Benchmark |
|--------|-------------|---------------|
| **LLM 文本知识编辑** (纯文本大模型) | GPT-J-6B, GPT-2-XL, LLaMA-2/3, Mistral-7B | CounterFact, zsRE, MQuAKE, RippleEdits, UniEdit, WikiBigEdit, AKEW |
| **LLM 知识编辑应用** (安全/去毒/时序) | LLaMA-2-7B, LLaMA-3-8B | SafeEdit, HalluEditBench, ScEdit, COMPKE |
| **MLLM 知识编辑** (多模态大语言模型) | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | MMEdit, VLKEB, MMKE-Bench, ComprehendEdit, MC-MKE |
| **VLM 细粒度编辑** (视觉语言模型) | LLaVA-1.5, InternVL, Qwen-VL | MIKE, FGVEdit, M2Edit |
| **VLM 终身编辑** (持续更新) | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | LiveEdit Bench |

---

## 二、文本知识编辑（Text-only Knowledge Editing）— Benchmark 与方法

### 2.1 文本 Benchmark 汇总表

| # | Benchmark | 会议/来源 | 时间 | 目标模型 | 数据规模 | 编辑类型 | 核心评估指标 | 论文链接 |
|---|-----------|----------|------|---------|---------|---------|-------------|---------|
| 1 | **CounterFact** | NeurIPS 2022 | 2022 | GPT-J-6B, GPT-2-XL | 21,919 反事实编辑 | 结构化事实三元组 (s,r,o) | Efficacy, Paraphrase (Generalization), Specificity (Locality) | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) |
| 2 | **zsRE** (Zero-Shot RE) | ACL 2017 / KE 适配 | 2017/2022 | GPT-J-6B, GPT-2-XL, T5 | 10,000+ 零样本关系抽取 | 结构化事实三元组 | Efficacy, Paraphrase, Specificity | [arXiv:2104.08164](https://arxiv.org/abs/2104.08164) |
| 3 | **MQuAKE** | EMNLP 2023 | 2023 | GPT-J-6B, GPT-3.5-turbo | MQuAKE-CF: 9,218; MQuAKE-T: 1,825 | 多跳推理 (2-4 hop) | Multi-hop Accuracy, Consistency | [arXiv:2305.14795](https://arxiv.org/abs/2305.14795) |
| 4 | **RippleEdits** | NeurIPS 2024 | 2023 | GPT-J-6B, GPT-2-XL, LLaMA-2-7B | 5,000 条事实编辑 + 波及效应 | 涟漪效应 (6类逻辑推理) | Ripple Effect Success Rate | [arXiv:2307.12976](https://arxiv.org/abs/2307.12976) |
| 5 | **KnowEdit** | ACL 2024 (EasyEdit) | 2023 | GPT-J-6B, LLaMA-2-7B/13B | 整合 WikiBio/zsRE/CounterFact/WikiDataRecent/ConvSent/Sanitation | 综合编辑评测 | Efficacy, Generalization, Specificity, Fluency | [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowEdit) |
| 6 | **AKEW** | EMNLP 2024 | 2024 | GPT-J-6B, LLaMA-2-7B, LLaMA-3-8B | 3种场景: 结构化/非结构化/抽取三元组 | 实际场景知识更新 | Efficacy, Generalization, Specificity, Fluency | [arXiv:2402.18909](https://arxiv.org/abs/2402.18909) |
| 7 | **UnKEBench** | 2024 | 2024 | GPT-J-6B, LLaMA-2-7B | 非结构化长文本知识编辑 | UKE (非结构化知识编辑) | Efficacy, Generalization, Locality, Fluency | [OpenReview](https://openreview.net/forum?id=cPhLi33i0C) |
| 8 | **HalluEditBench** | ICLR 2025 | 2024.10 | GPT-J-6B, LLaMA-2-7B, LLaMA-3-8B, Mistral-7B | 6,000+ 幻觉, 9域/26话题 | 真实幻觉纠正 | Efficacy, Generalization, **Portability**, Locality, **Robustness** | [arXiv:2410.16251](https://arxiv.org/abs/2410.16251) |
| 9 | **MQuAKE-Remastered** | ICLR 2025 | 2025 | GPT-J-6B, LLaMA-2-7B, LLaMA-3-8B | 修正原 MQuAKE 33%-76% 错误标签 | 多跳推理 (修正版) | Multi-hop Accuracy | [OpenReview](https://openreview.net/forum?id=m9wG6ai2Xk) |
| 10 | **ScEdit** | ACL 2025 Findings | 2025 | GPT-J-6B, LLaMA-2-7B/13B, LLaMA-3-8B | 脚本型知识, "How"类问答 | 动作型知识编辑 (反事实+时序) | Token-level + **Text-level** 指标 | [ACL Anthology](https://aclanthology.org/2025.findings-acl.104/) |
| 11 | **COMPKE** | ACL 2025 Findings | 2025 | GPT-4o-mini, Qwen2.5-3B/7B, LLaMA-3-8B | 11,924 复杂问题 | 一对多关系 + 多步逻辑推理 | Complex QA Accuracy | [ACL Anthology](https://aclanthology.org/2025.findings-acl.130/) |
| 12 | **SafeEdit** | ACL 2024 | 2024 | LLaMA-2-7B, Mistral-7B, Vicuna | 9类不安全类别 + 多种攻击提示 | 安全/去毒知识编辑 | Detoxification Success Rate, Over-editing Rate | [ACL Anthology](https://aclanthology.org/2024.acl-long.171/) |
| 13 | **UniEdit** | NeurIPS 2025 D&B | 2025 | GPT-J-6B, LLaMA-2-7B, LLaMA-3-8B, Mistral-7B | 311K 样本, 25域/5大类 | 开放域知识图谱编辑 + 多跳波及评估 | Reliability, Generality, Locality, **Multi-hop Ripple** | [arXiv:2505.12345](https://arxiv.org/abs/2505.12345) |
| 14 | **WikiBigEdit** | ICML 2025 | 2025 | LLaMA-2-7B, LLaMA-3-8B, Mistral-7B | 500K+ QA 对, 真实 Wikidata 编辑 | 终身大规模编辑 (2024.02-07) | Generalization, Locality, Multi-hop | [ICML 2025](https://proceedings.mlr.press/v267/thede25a.html) |
| 15 | **EditEverything** | ICML 2025 (AnyEdit) | 2025 | GPT-J-6B, LLaMA-2-7B, LLaMA-3-8B | 多格式: 诗歌/代码/数学推导 | 长文本多格式知识编辑 | Efficacy, Format Preservation | [ICML 2025](https://proceedings.mlr.press/v267/jiang25b.html) |

### 2.2 文本知识编辑方法汇总表

| # | 方法名称 | 会议/来源 | 时间 | 类型 | 目标模型 | 核心技术 | 评估 Benchmark | 论文链接 |
|---|---------|----------|------|------|---------|---------|---------------|---------|
| 1 | **ROME** | NeurIPS 2022 | 2022 | Locate-then-Edit | GPT-J-6B, GPT-2-XL | 因果追踪定位 MLP 层 → 秩一矩阵修改 FFN 权重 | CounterFact, zsRE | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) |
| 2 | **MEMIT** | NeurIPS 2022 | 2022 | Locate-then-Edit | GPT-J-6B, GPT-NeoX-20B | ROME 的批量扩展 → 多层 MLP 同时修改 | CounterFact, zsRE | [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) |
| 3 | **MEND** | ICLR 2022 | 2022 | Meta-Learning | GPT-2-XL, GPT-J-6B | 超网络学习编辑方向 → 小梯度高效修正 | zsRE, CounterFact | [arXiv:2110.11309](https://arxiv.org/abs/2110.11309) |
| 4 | **SERAC** | ICML 2022 | 2022 | Memory-based | GPT-J-6B, T5-XL | 外部记忆存储 + 范围分类器路由 | CounterFact, zsRE | [arXiv:2206.06520](https://arxiv.org/abs/2206.06520) |
| 5 | **IKE** | ACL 2023 | 2023 | In-Context | GPT-J-6B, GPT-3 | 上下文示例注入, 无需修改权重 | CounterFact, zsRE | [arXiv:2305.12740](https://arxiv.org/abs/2305.12740) |
| 6 | **GRACE** | NeurIPS 2023 | 2023 | Memory-based | T5, GPT-2-XL, GPT-J-6B | 激活空间离散 codebook → 无需改权重 | zsRE, CounterFact | [arXiv:2211.11031](https://arxiv.org/abs/2211.11031) |
| 7 | **MeLLo** | EMNLP 2023 | 2023 | Memory-based | GPT-3.5-turbo | 外部存储编辑事实 + 迭代提示一致推理 | MQuAKE | [arXiv:2305.14795](https://arxiv.org/abs/2305.14795) |
| 8 | **WISE** | NeurIPS 2024 | 2024 | Dual Memory | GPT-J-6B, LLaMA-2-7B/13B, Mistral-7B | 主记忆+侧记忆+知识分片+无冲突合并 | KnowEdit, 序列编辑 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/60960ad78868fce5c165295fbd895060-Abstract-Conference.html) |
| 9 | **AlphaEdit** | ICLR 2025 **Outstanding Paper** | 2024.10 | Locate-then-Edit+ | GPT-J-6B, GPT-2-XL, LLaMA-3-8B | 零空间投影 → 编辑扰动投射到保留知识的零空间 | CounterFact, zsRE, 序列编辑 | [arXiv:2410.02355](https://arxiv.org/abs/2410.02355) |
| 10 | **AdaEdit** | ACL 2025 Long | 2025 | Fine-tuning+ | LLaMA-2-7B/13B, LLaMA-3-8B | 解耦稀疏化知识表示 → 大规模连续编辑 | KnowEdit, UniEdit | [ACL Anthology](https://aclanthology.org/2025.acl-long.208/) |
| 11 | **AnyEdit** | ICML 2025 | 2025 | Autoregressive | GPT-J-6B, LLaMA-2-7B, LLaMA-3-8B | 知识分块 → 迭代编辑关键 token (互信息链式法则) | UnKEBench, AKEW, EditEverything | [ICML 2025](https://proceedings.mlr.press/v267/jiang25b.html) |
| 12 | **NMKE** | NeurIPS 2025 | 2025 | Neuron-level | LLaMA-2-7B, LLaMA-3-8B | 神经元级归因 + 熵引导动态稀疏掩码 | 序列编辑 (数千次) | [arXiv:2510.22139](https://arxiv.org/abs/2510.22139) |
| 13 | **IFMET** | ICML 2025 | 2025 | Locate-then-Edit | GPT-J-6B, LLaMA-2-7B | 浅层+深层 MLP 联合编辑 → 机制可解释性驱动 | MQuAKE, CounterFact | [ICML 2025](https://proceedings.mlr.press/v267/zhang25aq.html) |
| 14 | **MEMIT-Merge** | ACL 2025 Findings | 2025 | Locate-then-Edit | GPT-J-6B, LLaMA-2-7B | 解决 MEMIT 同主语批量编辑键值冲突 (46%→98%) | CounterFact | [ACL Anthology](https://aclanthology.org/2025.findings-acl.415/) |
| 15 | **KGMET** | EMNLP 2025 Findings | 2025 | KG-guided | GPT-J-6B, LLaMA-3-8B | 知识图谱引导编辑方向 → 多跳任务 +5-17% | MQuAKE, RippleEdits | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.261/) |
| 16 | **Reason-KE** | EMNLP 2025 Findings | 2025 | Reasoning Chain | LLaMA-2-7B, LLaMA-3-8B | 四阶段推理链: 事实确认→相关判断→选择应用→最终推理 | MQuAKE | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.786/) |
| 17 | **EtCon** | 2025 | FT + RL | LLaMA-2-7B, LLaMA-3-8B | TPSFT + GRPO → 编辑后固化, 保持自回归生成能力 | 多 benchmark | [arXiv:2512.04753](https://arxiv.org/abs/2512.04753) |
| 18 | **ToxEdit** | ACL 2025 Findings | 2025 | Safety Editing | LLaMA-2-7B, LLaMA-3-8B-instruct | 毒性感知激活检测 + 自适应层间路径路由 | SafeEdit | [arXiv:2505.22298](https://arxiv.org/abs/2505.22298) |
| 19 | **CHECK** | IJCAI 2025 | 2025 | Multi-hop | GPT-J-6B, LLaMA-3-8B | 语义分析驱动多跳 QA → +22.8% 准确率 | MQuAKE | [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0916.pdf) |
| 20 | **SCR** (Selective Contextual Reasoning) | ICLR 2026 (审稿中) | 2026 | In-Context | 多种指令微调/推理 LLM | 选择性上下文注入 → 超越所有参数编辑方法 | 多事件/通用数据集 | [OpenReview](https://openreview.net/forum?id=ZVaVKRhq8l) |

---

## 三、多模态知识编辑 — 时间线

```
2023.10 ├── MMEdit (EMNLP 2023) ← 开山之作，首次定义多模态知识编辑
        │
2024.03 ├── VLKEB (→ NeurIPS 2024) ← 大规模 benchmark + Portability 多跳推理
        │
2024.06 ├── MIKE (ACL 2024 Findings) ← 细粒度实体知识编辑
        │
2024.09 ├── UniKE (→ NeurIPS 2024 Spotlight) ← 统一内在+外部编辑
        │
─── 2025 分界线 ──────────────────────────────────────────────────────
        │
2025.02 ├── MMKE-Bench (ICLR 2025) ← 最全面 benchmark (2940知识, 8363图)
        │
2025.02 ├── K-Edit (AAAI 2025 Oral) ← 上下文知识感知编辑
        │
2025.03 ├── MC-MKE (→ ACL 2025 Findings) ← 模态一致性 benchmark
        ├── MindBridge (→ ACL 2025 Findings) ← 跨模型记忆模态编辑
        │
2025.05 ├── BalancEdit + OKEDIT (ICML 2025) ← 动态 Generality-Locality 平衡
        │
2025.06 ├── LiveEdit (CVPR 2025) ← 终身 VLM 编辑 + Low-Rank MoE
        ├── DualEdit (COLM 2025) ← 双模态分层编辑 + 门控
        │
2025.07 ├── FGVEdit + MSCKE (ICCV 2025) ← 细粒度视觉知识编辑
        ├── TMKE (ICCV 2025) ← 跨模态知识迁移性研究
        │
2025.11 ├── M2Edit + MLE (EMNLP 2025) ← 多粒度编辑 (实体/关系/动作)
        │
2025.12 ├── ComprehendEdit + HICE (AAAI 2025) ← 8类任务 + KGI/KPI 新指标
        ├── VisEdit (AAAI 2025) ← 归因分析 + 视觉表示编辑
        │
─── 2026 分界线 ──────────────────────────────────────────────────────
        │
2026.02 ├── ReasonEdit ← 融入人类推理的 VLM 编辑
        │
2026.03 ├── ELoRA (ICLR 2026 审稿中) ← LoRA null/knowledge 空间分解
```

---

## 四、多模态 Benchmark 汇总表

| # | Benchmark | 会议/来源 | 时间 | 目标模型 | 数据规模 | 编辑类型 | 核心评估指标 | 论文链接 |
|---|-----------|----------|------|---------|---------|---------|-------------|---------|
| 1 | **MMEdit** (E-VQA / E-IC) | EMNLP 2023 | 2023.10 | BLIP2-OPT, MiniGPT-4 | VQA + Image Captioning 双子任务 | 粗粒度实体编辑 | Reliability, Generality, Locality | [arXiv:2310.08475](https://arxiv.org/abs/2310.08475) |
| 2 | **MIKE** | ACL 2024 Findings | 2024.06 | 多种 MLLM | 1,000+ 细粒度实体，每个≥5张图 | 细粒度实体识别 (VNA/ELC/CSR 三任务) | Reliability, Generality, Locality + Multi-step Editing | [ACL Anthology](https://aclanthology.org/2024.findings-acl.298/) |
| 3 | **VLKEB** | NeurIPS 2024 D&B | 2024.03 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 等 5个 LVLM | 8,174 edits + 18,434 images | 实体知识编辑 + 多跳推理 (1-4 hop) | Reliability, Generality, Locality, **Portability** | [arXiv:2403.07350](https://arxiv.org/abs/2403.07350) |
| 4 | **MMKE-Bench** | ICLR 2025 | 2025.02 | 3个 LMM + 5种编辑方法 | 2,940 条知识 + 8,363 张图，33 个类别 | 视觉实体 / 视觉语义 / 用户自定义 三类 | Reliability, Generality, Locality | [arXiv:2502.19870](https://arxiv.org/abs/2502.19870) |
| 5 | **ComprehendEdit** | AAAI 2025 | 2024.12 | 多种 MLLM | 8 类任务（数值推理/空间关系/OCR/属性等） | 综合多模态知识编辑 | Reliability, Generality, Locality, **KGI**, **KPI** | [arXiv:2412.12821](https://arxiv.org/abs/2412.12821) |
| 6 | **MC-MKE** | ACL 2025 Findings | 2025.03 | 4种 MLLM 编辑方法 | 视觉+文本双分量知识，3种编辑场景 | 细粒度编辑 + 模态一致性 | Reliability, Generality, Locality, **Modality Consistency** | [ACL Anthology](https://aclanthology.org/2025.findings-acl.896/) |
| 7 | **FGVEdit** | ICCV 2025 | 2025.07 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | 多实体交互场景图像 | 细粒度视觉知识编辑（图像内局部实体） | Reliability, Generality, Locality | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zeng_Visual-Oriented_Fine-Grained_Knowledge_Editing_for_MultiModal_Large_Language_Models_ICCV_2025_paper.html) |
| 8 | **M2Edit** | EMNLP 2025 | 2025.11 | 多种 MLLM | 实体/关系/动作 三种粒度 | 多粒度知识编辑 | Reliability, Generality, Locality, **Visual Generality** | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1478/) |
| 9 | **OKEDIT** | ICML 2025 | 2025.05 | 多种 MLLM | 正负样本对，带影响范围标注 | Generality-Locality 权衡评测 | Reliability, Generality, Locality | [arXiv:2505.01343](https://arxiv.org/abs/2505.01343) |
| 10 | **LiveEdit Bench** | CVPR 2025 | 2025.06 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | 首个终身 VLLM 编辑数据集 | 连续多次编辑场景 | Reliability, Generality, Locality (Lifelong) | [arXiv:2411.15432](https://arxiv.org/abs/2411.15432) |
| 11 | **TMKE Bench** | ICCV 2025 | 2025.07 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | 跨模态知识迁移评测集 | 单模态编辑→多模态迁移 | Transitivity Success Rate, Reliability, Locality | [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Fang_Can_Knowledge_be_Transferred_from_Unimodal_to_Multimodal_Investigating_the_ICCV_2025_paper.pdf) |

---

## 五、多模态方法论文汇总表

### 5.1 多模态原生编辑方法

| # | 方法名称 | 会议/来源 | 时间 | 目标模型 | 核心技术 | 评估 Benchmark | 论文链接 |
|---|---------|----------|------|---------|---------|---------------|---------|
| 1 | **MMEdit Baselines** | EMNLP 2023 | 2023.10 | BLIP2-OPT, MiniGPT-4 | 首次将 FT/IKE/MEND/SERAC 等迁移到多模态场景 | MMEdit (E-VQA, E-IC) | [arXiv:2310.08475](https://arxiv.org/abs/2310.08475) |
| 2 | **UniKE** | NeurIPS 2024 Spotlight | 2024.09 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | 统一内在编辑(ROME) + 外部知识(IKE)为向量化 KV 记忆；语义/真实性空间解耦 | MMEdit | [OpenReview](https://openreview.net/forum?id=kf80ZS3fVy) |
| 3 | **VisEdit** | AAAI 2025 | 2024.12 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | 归因分析定位关键视觉表示区域 → 编辑中间视觉表示 | MMEdit | [PDF](https://chywang.github.io/papers/aaai2025.pdf) |
| 4 | **HICE** | AAAI 2025 | 2024.12 | 多种 MLLM | 分层上下文编辑：两阶段 baseline 平衡 Reliability/Generality/Locality | ComprehendEdit | [arXiv:2412.12821](https://arxiv.org/abs/2412.12821) |
| 5 | **MSCKE** | ICCV 2025 | 2025.07 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | 多模态范围分类器 → 融合文本+视觉信息精准定位编辑范围 | FGVEdit | [GitHub](https://github.com/zeng-zhen/FGVEdit) |
| 6 | **LiveEdit** | CVPR 2025 | 2025.06 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | Low-Rank MoE → 编辑专家生成器 + 硬过滤(视觉) + 软路由(文本) | LiveEdit Bench | [arXiv:2411.15432](https://arxiv.org/abs/2411.15432) |
| 7 | **BalancEdit** | ICML 2025 | 2025.05 | 多种 MLLM | 离散局部化 codebook → 正负样本确定影响范围 → 不修改模型权重 | OKEDIT | [arXiv:2505.01343](https://arxiv.org/abs/2505.01343) |
| 8 | **DualEdit** | COLM 2025 | 2025.06 | 多种 VLM backbone | 文本/视觉各自峰值敏感层分别编辑 + 文本侧门控模块保留原始能力 | 多 benchmark | [OpenReview](https://openreview.net/forum?id=X5vFauyVWr) |
| 9 | **MLE** (M2Edit 方法) | EMNLP 2025 | 2025.11 | 多种 MLLM | 定位不同 MLLM 组件的关键知识层 → 协同编辑多粒度知识 | M2Edit | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1478/) |
| 10 | **MindBridge** | ACL 2025 Findings | 2025.03 | 多种 LLM | 记忆模态概念 → 将编辑知识编码为独立模态 → 跨模型迁移 | 多 KE 数据集 | [ACL Anthology](https://aclanthology.org/2025.findings-acl.621/) |
| 11 | **ReasonEdit** | arXiv | 2026.02 | 4种 VLM | Codebook 存储人类推理 + 拓扑平衡多模态嵌入检索 | Rationale-based VQA | [arXiv:2602.02408](https://arxiv.org/abs/2602.02408) |
| 12 | **ELoRA** | ICLR 2026 (审稿中) | 2026.03 | LLaVA-v1.5-7B, Qwen2.5-VL-7B, Phi-4-multimodal | LoRA 分解为 null 空间(保留) + 知识空间(更新) | 多 benchmark | [OpenReview](https://openreview.net/forum?id=9nxtfO6gOu) |
| 13 | **KDKE** | OpenReview | 2025 | MLLM | 集成模块贡献分数 → 动态模块选择 + 约束自适应 LoRA 注入 | 多 benchmark | [OpenReview PDF](https://openreview.net/pdf/a4c902ed67f8936f66ee325b7d0e4fa8c790fc9e.pdf) |

### 5.2 文本知识编辑基线方法（常被迁移到多模态场景）

| 方法 | 类型 | 核心思路 | 多模态适用性 |
|------|------|---------|------------|
| **FT / FT-L** | 微调 | 直接微调或仅微调最后几层 | ★★★★★ 最容易适配 |
| **IKE** | In-Context | 上下文示例注入，无需训练 | ★★★★★ 无需修改权重 |
| **MEND** | Meta-Learning | 轻量编辑器网络修改权重 | ★★★★ 需适配多模态前向 |
| **SERAC** | Memory-based | 外部记忆存储 + 范围分类器 | ★★★★ 分类器需扩展为多模态 |
| **ROME** | Locate-then-Edit | 定位 Transformer 层中的事实存储，修改 FFN | ★★★ 仅编辑语言模块 |
| **MEMIT** | Locate-then-Edit | ROME 的批量编辑扩展 | ★★★ 同 ROME |
| **GRACE** | Memory-based | 激活空间中的 codebook 编辑 | ★★★★ 可扩展到多模态激活 |
| **WISE** | Dual Memory | 主记忆 + 侧记忆 + 路由器 | ★★★★ 已有 MM_WISE 适配 |
| **KE** | Hypernetwork | 超网络生成编辑参数 | ★★★ 需训练超网络 |
| **T-Patcher** | Memory-based | 添加额外神经元补丁 | ★★★ |

---

## 六、主流模型使用统计

### 6.1 文本大模型（LLM）— 知识编辑评测

| 模型 | 参数量 | 被使用的论文数 | 典型论文/Benchmark |
|------|-------|-------------|-------------------|
| **GPT-J-6B** | 6B | 15+ | ROME, MEMIT, MEND, SERAC, GRACE, CounterFact, zsRE, MQuAKE, KnowEdit, AnyEdit, IFMET, MEMIT-Merge, KGMET |
| **GPT-2-XL** | 1.5B | 10+ | ROME, MEMIT, MEND, AlphaEdit, CounterFact, zsRE |
| **LLaMA-2-7B** | 7B | 10+ | WISE, AdaEdit, AnyEdit, NMKE, SafeEdit, AKEW, WikiBigEdit, UniEdit, ScEdit |
| **LLaMA-2-13B** | 13B | 5+ | WISE, AdaEdit, KnowEdit, ScEdit |
| **LLaMA-3-8B** | 8B | 8+ | AlphaEdit, NMKE, AnyEdit, KGMET, CHECK, HalluEditBench, WikiBigEdit, UniEdit |
| **Mistral-7B** | 7B | 5+ | WISE, HalluEditBench, WikiBigEdit, UniEdit |
| **GPT-NeoX-20B** | 20B | 2+ | MEMIT |
| **T5-XL/XXL** | 3B/11B | 3+ | SERAC, GRACE, MEND |
| **GPT-3.5-turbo** | — | 2+ | MeLLo, MQuAKE |
| **GPT-4o-mini** | — | 1+ | COMPKE |
| **Qwen2.5-3B/7B** | 3B/7B | 1+ | COMPKE |

> **结论**: **GPT-J-6B** 是文本知识编辑领域使用最广泛的模型 (几乎 100% 的早期方法论文)，2024 年起 **LLaMA-2-7B** 和 **LLaMA-3-8B** 成为新标准，**Mistral-7B** 也开始被广泛使用。

### 6.2 多模态大模型（MLLM）— 知识编辑评测

| 模型 | 视觉编码器 | 语言模型 | 被使用的论文数 | 典型论文 |
|------|-----------|---------|-------------|---------|
| **BLIP2-OPT** | ViT-L (frozen) | OPT-2.7B/6.7B | 10+ | MMEdit, UniKE, VLKEB, MMKE-Bench, FGVEdit, TMKE, LiveEdit, VisEdit, ComprehendEdit |
| **MiniGPT-4** | ViT-G/14 EVA-CLIP (frozen) | Vicuna-7B/13B | 10+ | MMEdit, UniKE, VLKEB, MMKE-Bench, FGVEdit, TMKE, LiveEdit, VisEdit |
| **LLaVA-1.5** | CLIP ViT-L/14 | Vicuna-7B/13B | 10+ | UniKE, VLKEB, MMKE-Bench, FGVEdit, TMKE, LiveEdit, ELoRA, DualEdit |
| **Qwen-VL** | ViT (OpenCLIP) | Qwen-7B | 2-3 | 部分 benchmark |
| **Qwen2.5-VL-7B** | — | Qwen2.5 | 1 | ELoRA |
| **Phi-4-multimodal** | — | Phi-4 | 1 | ELoRA |
| **InternLM-XComposer** | — | InternLM | 1-2 | 少量论文 |

> **结论**: **BLIP2-OPT + MiniGPT-4 + LLaVA-1.5** 是多模态知识编辑领域的"标准三件套"，几乎 100% 的论文都在这三个模型上实验。2025 末开始扩展到 Qwen2.5-VL 和 Phi-4 等新模型。

### 6.3 模型使用趋势

```
2022-2023: GPT-J-6B / GPT-2-XL 为绝对主流 (文本), BLIP2-OPT / MiniGPT-4 开辟多模态
     2024: LLaMA-2-7B/13B 取代 GPT-J 成为新标准, LLaVA-1.5 成为多模态核心
     2025: LLaMA-3-8B + Mistral-7B 成为文本主流, Qwen2.5-VL / Phi-4 进入多模态
     2026: 指令微调/推理模型 (Instruct/Reasoning) 开始被评测
```

### 6.4 与多模态遗忘（Unlearning）对比

| 对比维度 | 知识编辑 (Editing) | 机器遗忘 (Unlearning) |
|---------|-------------------|---------------------|
| 核心模型 | BLIP2-OPT, MiniGPT-4, LLaVA-1.5 | LLaVA-1.5-7B/13B |
| 模型重合度 | LLaVA-1.5 为共同核心 | LLaVA-1.5 为绝对主流 |
| 模型规模 | 以 2.7B-13B 为主 | 以 7B-13B 为主 |
| 新趋势 | Qwen2.5-VL, Phi-4 | InstructBLIP, Qwen-VL |

---

## 七、核心评估指标汇总

### 7.1 文本知识编辑指标

| 指标类别 | 具体指标 | 含义 | 使用频率 | 代表论文 |
|---------|---------|------|---------|---------|
| **有效性** | Efficacy / Edit Success Rate | 编辑后对目标事实输出正确答案 | ★★★★★ | ROME, MEMIT, 所有方法 |
| **泛化性** | Paraphrase / Generalization | 对改写的等价输入也生效 | ★★★★★ | ROME, MEMIT, 所有方法 |
| **局部性** | Specificity / Locality | 不影响无关知识的输出 | ★★★★★ | ROME, MEMIT, 所有方法 |
| **流畅性** | Fluency | 编辑后生成文本的自然度 | ★★★★ | KnowEdit, AKEW |
| **可迁移性** | Portability | 多跳推理中知识正确传递 | ★★★★ | RippleEdits, MQuAKE, UniEdit |
| **鲁棒性** | Robustness | 对抗攻击/对抗改写下编辑仍有效 | ★★★ | HalluEditBench |
| **编辑得分** | Edit Score (调和平均) | Efficacy × Generalization × Specificity 的调和平均 | ★★★ | MEMIT, ROME |
| **波及效应** | Ripple Effect Rate | 编辑一个事实后，逻辑相关事实的连锁更新率 | ★★ | RippleEdits, UniEdit |
| **文本级指标** | Text-level Metrics | 超越 token 匹配，评估完整文本语义 | ★★ | ScEdit (ACL 2025) |

### 7.2 多模态通用指标（几乎所有多模态论文使用）

| 指标类别 | 具体指标 | 含义 | 使用频率 |
|---------|---------|------|---------|
| **可靠性** | Reliability / Edit Success Rate | 编辑后模型对目标知识输出正确答案 | ★★★★★ |
| **泛化性** | Generality (Text / Multimodal) | 编辑对等价/改写输入也生效 | ★★★★★ |
| **局部性** | Locality (Text / Multimodal) | 编辑不影响无关知识的输出 | ★★★★★ |

### 7.3 多模态扩展指标

| 指标 | 提出论文 | 含义 | 使用频率 |
|------|---------|------|---------|
| **Portability** (1-4 hop) | VLKEB (NeurIPS 2024) | 编辑知识在多跳推理中正确传递 | ★★★★ |
| **T-Gen / M-Gen** | UniKE (NeurIPS 2024) | 文本泛化性 / 多模态泛化性 分开评测 | ★★★ |
| **T-Loc / M-Loc** | UniKE (NeurIPS 2024) | 文本局部性 / 多模态局部性 分开评测 | ★★★ |
| **Visual Generality** | M2Edit (EMNLP 2025) | 视觉输入变化时编辑是否仍生效 | ★★★ |
| **Modality Consistency** | MC-MKE (ACL 2025) | 视觉/文本模态编辑后的跨模态一致性 | ★★ |
| **KGI** (Knowledge Generalization Index) | ComprehendEdit (AAAI 2025) | 编辑对同域样本的泛化影响 | ★★ |
| **KPI** (Knowledge Preservation Index) | ComprehendEdit (AAAI 2025) | 编辑对同域无关样本的保持能力 | ★★ |
| **Transitivity** | TMKE (ICCV 2025) | 单模态编辑能否迁移到多模态场景 | ★ |

### 7.4 与多模态遗忘指标的对比

| 对比维度 | 知识编辑指标 | 机器遗忘指标 |
|---------|------------|------------|
| **核心三指标** | Reliability + Generality + Locality | Forget Acc + Retain Acc + Model Utility |
| **对应关系** | Reliability ↔ Forget Efficacy | 编辑"改对" ↔ 遗忘"删干净" |
| | Locality ↔ Retain Accuracy | 都测"不影响无关知识" |
| | Generality ↔ Generalizability | 都测"泛化到变体输入" |
| **独有指标** | Portability (多跳推理) | MIA (成员推断攻击) |
| | Modality Consistency | Modality Alignment (RLF/RLR) |

---

## 八、关键发现与趋势

### 8.1 核心发现（文本知识编辑）

1. **参数编辑在真实场景下表现不佳**: ICLR 2026 审稿论文系统评测发现，在自回归推理（而非 teacher-forced 解码）下，ROME/MEMIT/MEND 等参数编辑方法效果显著下降，简单的上下文注入 (SCR) 反而更优。

2. **多跳推理是核心瓶颈**: MQuAKE、RippleEdits 一致表明，编辑单个事实后模型无法自动推理出逻辑关联的变化，多跳准确率远低于单跳。

3. **终身编辑面临规模挑战**: WikiBigEdit (ICML 2025) 发现现有方法在 500K 真实编辑规模下严重退化，检索增强和持续微调反而更有竞争力。

4. **零空间投影是突破性方向**: AlphaEdit (ICLR 2025 Outstanding Paper) 仅用一行代码的零空间投影即可将所有 locate-then-edit 方法平均提升 36.7%。

5. **非结构化知识编辑远难于结构化**: UnKEBench 表明，长文本/自由格式知识的编辑比三元组编辑困难得多，但优化后的微调方法在此场景下出奇有效。

6. **安全编辑存在过度编辑风险**: SafeEdit (ACL 2024) 发现去毒编辑容易导致模型拒绝合法请求，ToxEdit (ACL 2025) 尝试通过自适应路由缓解此问题。

7. **现有 benchmark 标签质量堪忧**: MQuAKE-Remastered (ICLR 2025) 发现原始 MQuAKE 有高达 33%-76% 的标签错误，提示需要更严格的 benchmark 质量控制。

### 8.2 核心发现（多模态知识编辑）

1. **没有通吃的方法**: MMKE-Bench、VLKEB 等 benchmark 一致表明，现有方法无法在 Reliability/Generality/Locality 上同时表现最优。

2. **文本编辑方法直接迁移效果有限**: MMEdit (2023) 首次发现将 ROME/MEND/SERAC 等直接用于多模态场景，效果"barely satisfactory"，因为多模态知识分布在异构的视觉-文本序列中。

3. **视觉编辑 vs 文本编辑不一致**: DualEdit 发现文本和视觉模态在不同层达到峰值敏感性，需要分层编辑策略。

4. **Generality-Locality 是核心矛盾**: BalancEdit (ICML 2025) 首次系统研究这一权衡，提出正负样本确定影响范围的方案。

5. **跨模态知识迁移性差**: TMKE (ICCV 2025) 发现单模态编辑很难自动迁移到多模态场景，揭示当前模型的模态融合深度不足。

6. **细粒度编辑远难于粗粒度**: MIKE (2024) → FGVEdit (2025) 发现，当图像中有多个交互实体时，精准编辑其中一个的难度急剧上升。

7. **终身编辑性能快速退化**: LiveEdit (CVPR 2025) 建立首个终身编辑 benchmark，发现连续编辑后模型能力显著下降。

### 8.3 技术趋势

**文本知识编辑:**
- **2022**: 奠基期 — ROME/MEMIT/MEND/SERAC 四大方法确立，CounterFact/zsRE 成为标准 benchmark
- **2023**: 多跳+终身 — MQuAKE (多跳推理)、GRACE (终身编辑)、IKE (上下文编辑) 拓展评测维度
- **2024**: 实际场景 — AKEW (实际知识更新)、RippleEdits (波及效应)、WISE (双记忆)、SafeEdit (安全编辑)
- **2025 上半年**: 规模化+高效化 — AlphaEdit (零空间)、AnyEdit (长文本)、WikiBigEdit (500K规模)、UniEdit (311K开放域)
- **2025 下半年**: 精细化 — NMKE (神经元级)、KGMET (KG引导)、ScEdit (动作知识)、COMPKE (复杂QA)
- **2026**: 范式反思 — SCR (ICLR 2026) 发现上下文注入全面超越参数编辑，引发方法论反思

**多模态知识编辑:**
- **2023**: 基础建立期 — MMEdit 开辟多模态知识编辑方向，证明可行但效果有限
- **2024**: Benchmark 爆发期 — MIKE (细粒度)、VLKEB (Portability)、UniKE (统一方法) 分别从不同维度推进
- **2025 上半年**: 方法创新期 — BalancEdit (codebook)、LiveEdit (MoE)、DualEdit (双模态分层) 等原生多模态方法涌现
- **2025 下半年**: 精细化评测期 — M2Edit (多粒度)、MC-MKE (模态一致性)、ComprehendEdit (8 类任务) 推动更全面评估
- **2026**: 高效化+推理化 — ELoRA (LoRA 空间分解)、ReasonEdit (融入推理) 代表新方向

### 8.4 与遗忘方向的交叉趋势

| 共同趋势 | 知识编辑的体现 | 机器遗忘的体现 |
|---------|--------------|--------------|
| 从粗粒度→细粒度 | MMEdit → MIKE → FGVEdit | MLLMU-Bench → PEBench |
| 从单次→终身/序列 | LiveEdit (终身编辑) | PULSE (序列遗忘) |
| 关注模态一致性 | MC-MKE (Modality Consistency) | UMU-Bench (Modality Alignment) |
| 跨概念/知识干扰 | BalancEdit (Generality-Locality) | PEBench (Cross-concept) |
| 视觉表示层面操作 | VisEdit (视觉归因编辑) | VKD (视觉知识蒸馏) |

---

## 九、综述论文

| 综述 | 时间 | 覆盖范围 | 链接 |
|------|------|---------|------|
| Knowledge Editing for Large Language Models: A Survey | 2023/2024 持续更新 | 知识编辑全景综述（含文本+多模态） | [arXiv:2310.16218](https://arxiv.org/abs/2310.16218) |
| A Comprehensive Study of Knowledge Editing (dual-axis taxonomy) | 2025 | 机制轴(参数/外部记忆) × 功能轴(事实/时序/概念/常识/社会知识) | [arXiv:2508.08795](https://arxiv.org/abs/2508.08795) |
| Editing Across Languages: Multilingual Knowledge Editing | EMNLP 2025 | 跨语言知识编辑系统综述 | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.803/) |
| Benchmarking and Rethinking Knowledge Editing | ICLR 2026 (审稿中) | 事件型+通用型数据集 / 指令微调+推理LLM / 自回归推理 / 多编辑评估 | [OpenReview](https://openreview.net/forum?id=ZVaVKRhq8l) |
| zjunlp/KnowledgeEditingPapers | 持续更新 | GitHub 论文列表 + 教程 (~1.2k stars) | [GitHub](https://github.com/zjunlp/KnowledgeEditingPapers) |

---

## 十、开源资源汇总

### 10.1 文本 Benchmark 数据与代码

| 资源 | 类型 | 链接 |
|------|------|------|
| CounterFact + zsRE (ROME/MEMIT) | Data + Code | [GitHub](https://github.com/kmeng01/rome) / [项目页](https://rome.baulab.info/) |
| KnowEdit (EasyEdit 整合) | Data | [HuggingFace](https://huggingface.co/datasets/zjunlp/KnowEdit) |
| MQuAKE | Data + Code | [GitHub](https://github.com/princeton-nlp/MQuAKE) |
| MQuAKE-Remastered | Data + Code | [OpenReview](https://openreview.net/forum?id=m9wG6ai2Xk) |
| RippleEdits | Data + Code | [GitHub](https://github.com/avivbrokman/RippleEdits) |
| AKEW | Data + Code | [GitHub](https://github.com/bobxwu/akew) |
| HalluEditBench | Data + Code | [GitHub](https://github.com/baixianghuang/HalluEditBench) / [HuggingFace](https://huggingface.co/datasets/llm-editing/HalluEditBench) |
| UniEdit | Data + Code | [GitHub](https://github.com/qizhou000/UniEdit) / [HuggingFace](https://huggingface.co/datasets/qizhou000/UniEdit) |
| WikiBigEdit | Data + Code | [GitHub](https://github.com/ExplainableML/WikiBigEdit) / [HuggingFace](https://huggingface.co/datasets/lukasthede/WikiBigEdit) |
| ScEdit | Data + Code | [ACL Anthology](https://aclanthology.org/2025.findings-acl.104/) |
| COMPKE | Data + Code | [GitHub](https://github.com/kzjkzj666/CompKE) |
| SafeEdit | Data + Code | [ACL Anthology](https://aclanthology.org/2024.acl-long.171/) |

### 10.2 多模态 Benchmark 数据与代码

| 资源 | 类型 | 链接 |
|------|------|------|
| MMEdit | Data + Code | [GitHub (EasyEdit)](https://github.com/zjunlp/EasyEdit) |
| VLKEB | Data + Code + Pretrained | [GitHub](https://github.com/vlkeb/vlkeb) / [Kaggle](https://www.kaggle.com/) / [HuggingFace](https://huggingface.co/) |
| MMKE-Bench | Data + Code | [GitHub](https://github.com/MMKE-Bench-ICLR/MMKE-Bench) / [项目页](https://mmke-bench-iclr.github.io/) |
| ComprehendEdit | Data + Code | [GitHub](https://github.com/yaohui120/ComprehendEdit) |
| FGVEdit | Data + Code | [GitHub](https://github.com/zeng-zhen/FGVEdit) |
| MIKE | Data + Code | [ACL Anthology](https://aclanthology.org/2024.findings-acl.298/) |
| OKEDIT (BalancEdit) | Data + Code | [GitHub](https://github.com/donglgcn/BalancEdit) |

### 10.3 编辑方法代码

| 方法 | 代码链接 |
|------|---------|
| **EasyEdit** (ROME/MEMIT/MEND/SERAC/IKE/GRACE/WISE 等, 2.7k+ stars) | [GitHub](https://github.com/zjunlp/EasyEdit) |
| AlphaEdit (ICLR 2025 Outstanding) | [GitHub](https://github.com/jianghoucheng/AlphaEdit) |
| AnyEdit (ICML 2025) | [ICML 2025](https://proceedings.mlr.press/v267/jiang25b.html) |
| NMKE (NeurIPS 2025) | [GitHub](https://github.com/LiuJinzhe-Keepgoing/NMKE) |
| AdaEdit (ACL 2025) | [ACL Anthology](https://aclanthology.org/2025.acl-long.208/) |
| UniKE | [GitHub](https://github.com/beepkh/UniKE) |
| LiveEdit | [GitHub](https://github.com/qizhou000/LiveEdit) |
| DualEdit | [GitHub](https://github.com/zhiyiscs/DualEdit) |
| BalancEdit | [GitHub](https://github.com/donglgcn/BalancEdit) |
| MindBridge | [GitHub](https://github.com/crashbugger/mindbridge) |

---

## 十一、搜索提示词（覆盖近三年顶级会议）

### 11.1 通用搜索词 — Google Scholar

```
# 【推荐】知识编辑全景搜索 (2023-2026)
("knowledge editing" OR "model editing" OR "knowledge update") AND ("large language model" OR "LLM") AND ("benchmark" OR "evaluation" OR "dataset") after:2023

# 知识编辑方法搜索
("knowledge editing" OR "model editing") AND ("ROME" OR "MEMIT" OR "MEND" OR "SERAC" OR "IKE" OR "GRACE" OR "WISE" OR "AlphaEdit") after:2023

# 多模态知识编辑
"multimodal knowledge editing" OR "multimodal model editing" OR "vision-language model editing" after:2023

# 多跳/波及效应
("knowledge editing" OR "model editing") AND ("multi-hop" OR "ripple effect" OR "chain reasoning") after:2023

# 终身/序列编辑
("lifelong" OR "sequential" OR "continual" OR "continuous") AND ("knowledge editing" OR "model editing") AND "LLM" after:2023

# 安全/去毒编辑
("knowledge editing" OR "model editing") AND ("safety" OR "detoxification" OR "hallucination") AND "LLM" after:2023

# 含具体模型名
"knowledge editing" AND ("GPT-J" OR "LLaMA" OR "Llama-2" OR "Llama-3" OR "Mistral" OR "Qwen") after:2023

# 含具体模型名（多模态）
"knowledge editing" AND ("LLaVA" OR "BLIP" OR "MiniGPT" OR "Qwen-VL" OR "InternVL") after:2023
```

### 11.2 按顶级会议搜索 — 平台 URL

```
# =================== ML 三大会 ===================

# OpenReview — ICLR 2024/2025/2026
https://openreview.net/search?term=knowledge+editing&venue=ICLR.cc/2025/Conference
https://openreview.net/search?term=knowledge+editing&venue=ICLR.cc/2026/Conference

# OpenReview — NeurIPS 2024/2025
https://openreview.net/search?term=knowledge+editing&venue=NeurIPS.cc/2024/Conference
https://openreview.net/search?term=knowledge+editing&venue=NeurIPS.cc/2025/Conference

# ICML 2025 (Proceedings of Machine Learning Research)
https://proceedings.mlr.press/v267/ → 搜索 "knowledge editing"
https://icml.cc/virtual/2025/papers.html → 搜索 "knowledge editing"

# =================== NLP 四大会 ===================

# ACL Anthology — ACL / EMNLP / NAACL / COLING / EACL
https://aclanthology.org/search/?q=knowledge+editing&year=2024-2026
https://aclanthology.org/search/?q=model+editing+LLM&year=2024-2026

# ACL 2024
https://aclanthology.org/events/acl-2024/ → 搜索 "knowledge editing"
# ACL 2025
https://aclanthology.org/events/acl-2025/ → 搜索 "knowledge editing"
# EMNLP 2024
https://aclanthology.org/events/emnlp-2024/ → 搜索 "knowledge editing"
# EMNLP 2025
https://aclanthology.org/events/emnlp-2025/ → 搜索 "knowledge editing"
# NAACL 2024/2025
https://aclanthology.org/events/naacl-2024/ → 搜索 "knowledge editing"

# =================== CV 三大会 ===================

# CVF Open Access — CVPR / ICCV / ECCV
https://openaccess.thecvf.com/CVPR2024 → 搜索 "knowledge editing"
https://openaccess.thecvf.com/CVPR2025 → 搜索 "knowledge editing"
https://openaccess.thecvf.com/ICCV2025 → 搜索 "knowledge editing"
https://openaccess.thecvf.com/ECCV2024 → 搜索 "knowledge editing"

# =================== AI 综合会议 ===================

# AAAI 2024/2025
https://ojs.aaai.org/index.php/AAAI/search/search → 搜索 "knowledge editing"
# IJCAI 2024/2025
https://www.ijcai.org/proceedings/2025 → 搜索 "knowledge editing"
# COLM 2025
https://openreview.net/search?term=knowledge+editing&venue=COLM.cc/2025

# =================== 通用学术搜索 ===================

# DBLP — 按 venue 精确过滤
https://dblp.org/search?q=knowledge+editing
# venue 过滤关键词: conf/nips, conf/icml, conf/iclr, conf/cvpr, conf/iccv, conf/eccv, conf/acl, conf/emnlp, conf/naacl, conf/aaai, conf/ijcai

# Semantic Scholar — 带年份过滤
https://api.semanticscholar.org/graph/v1/paper/search?query=knowledge+editing+LLM+benchmark&year=2023-2026

# arXiv — 最新预印本
https://arxiv.org/search/?searchtype=all&query=knowledge+editing+language+model&start=0
```

### 11.3 中文搜索词

```
# 知网/万方/Google Scholar 中文
知识编辑 大语言模型 评测基准
知识编辑 多跳推理 LLM benchmark
多模态知识编辑 benchmark 评测
多模态大模型 知识编辑 BLIP LLaVA
视觉语言模型 知识更新 编辑方法 综述
MLLM 知识编辑 可靠性 泛化性 局部性
大模型 事实编辑 ROME MEMIT 评估
终身知识编辑 序列编辑 持续更新 LLM
知识编辑 安全 去毒 幻觉纠正
```

### 11.4 按研究主题的推荐搜索词

| 研究主题 | 推荐搜索词 (Google Scholar) |
|---------|---------------------------|
| **Benchmark/评测** | `"knowledge editing" benchmark evaluation dataset LLM 2024 2025` |
| **多跳推理** | `"knowledge editing" "multi-hop" reasoning LLM MQUAKE` |
| **终身/序列编辑** | `"lifelong knowledge editing" OR "sequential editing" LLM 2024 2025` |
| **安全/去毒** | `"knowledge editing" safety detoxification hallucination LLM` |
| **多模态** | `"multimodal knowledge editing" MLLM VLM benchmark 2024 2025` |
| **参数高效编辑** | `"knowledge editing" LoRA null-space efficient LLM 2025` |
| **上下文编辑** | `"in-context editing" OR "retrieval augmented editing" LLM 2024 2025` |
| **波及/涟漪效应** | `"ripple effect" OR "cascade update" "knowledge editing" LLM` |
| **跨语言编辑** | `"multilingual knowledge editing" OR "cross-lingual editing" LLM` |
| **幻觉纠正** | `"knowledge editing" "hallucination correction" LLM benchmark` |

---

## 十二、SOTA 影响力排名与 open-unlearning 框架集成分析

> 基于 Google Scholar 引用量、GitHub Stars、发表会议等级、SOTA 表现综合评估
> 框架现状：已集成 ROME/MEMIT/MEND/SERAC/IKE/GRACE/WISE/AlphaEdit/AnyEdit 等 12 个文本方法 + 5 个多模态方法

### 12.1 文本方法影响力排名（SOTA → 可集成）

| 排名 | 方法 | 会议 | 引用量(估) | GitHub Stars | SOTA 表现 | 框架已集成? | 综合评级 |
|------|------|------|-----------|-------------|----------|-----------|---------|
| 1 | **ROME** | NeurIPS 2022 | 1000+ | 1.8k+ (原始) | 奠基之作，因果追踪 | ✅ 已集成 | ★★★★★ |
| 2 | **MEMIT** | NeurIPS 2022 | 800+ | (同 ROME 仓库) | 批量编辑奠基 | ✅ 已集成 | ★★★★★ |
| 3 | **EasyEdit** (框架) | ACL 2024 | 400+ | **2,746** | 统一编辑框架 | ✅ 部分集成 | ★★★★★ |
| 4 | **AlphaEdit** | ICLR 2025 **Outstanding** | 100+ | **425** | +36.7% 提升所有 locate-then-edit | ✅ 已集成 | ★★★★★ |
| 5 | **MEND** | ICLR 2022 | 500+ | — | Meta-learning 编辑 | ✅ 已集成 | ★★★★ |
| 6 | **SERAC** | ICML 2022 | 400+ | — | Memory-based 编辑 | ✅ 已集成 | ★★★★ |
| 7 | **GRACE** | NeurIPS 2023 | 200+ | — | Codebook 终身编辑 | ✅ 已集成 | ★★★★ |
| 8 | **WISE** | NeurIPS 2024 | 100+ | — | 双记忆终身编辑 | ✅ 已集成 | ★★★★ |
| 9 | **AnyEdit** | ICML 2025 | 30+ | — | 长文本多格式编辑 +21.5% | ✅ 已集成 | ★★★★ |
| 10 | **NMKE** | NeurIPS 2025 | 20+ | 有 | 神经元级终身编辑 SOTA | ❌ 未集成 | ★★★★ |
| 11 | **AdaEdit** | ACL 2025 Long | 15+ | — | 连续编辑 SOTA | ❌ 未集成 | ★★★★ |
| 12 | **IFMET** | ICML 2025 | 10+ | — | 多跳 locate-then-edit | ❌ 未集成 | ★★★ |
| 13 | **MEMIT-Merge** | ACL 2025 Findings | 10+ | — | 修复 MEMIT 批量冲突 46%→98% | ❌ 未集成 | ★★★ |
| 14 | **KGMET** | EMNLP 2025 Findings | 10+ | — | KG 引导多跳 +5-17% | ❌ 未集成 | ★★★ |
| 15 | **EtCon** | 2025 | 5+ | — | FT + GRPO 编辑固化 | ❌ 未集成 | ★★ |

### 12.2 文本 Benchmark 影响力排名

| 排名 | Benchmark | 会议 | 引用量(估) | GitHub/HF | 评测维度独特性 | 框架已集成? | 综合评级 |
|------|-----------|------|-----------|-----------|-------------|-----------|---------|
| 1 | **CounterFact** | NeurIPS 2022 | 1000+ | 1.8k+ stars | 标准事实三元组编辑 | ✅ 已集成 | ★★★★★ |
| 2 | **zsRE** | 2017/2022 | 800+ | (同 ROME) | 零样本关系抽取编辑 | ✅ 已集成 | ★★★★★ |
| 3 | **MQuAKE** | EMNLP 2023 | 300+ | **GitHub 开源** | **多跳推理** (唯一) | ❌ 未集成 | ★★★★★ |
| 4 | **RippleEdits** | NeurIPS 2024 | 100+ | GitHub 开源 | **波及效应** (唯一) | ❌ 未集成 | ★★★★ |
| 5 | **KnowEdit** | ACL 2024 | 200+ | HF 数据集 | 6 个子数据集整合 | ✅ 部分集成 | ★★★★ |
| 6 | **AKEW** | EMNLP 2024 | 50+ | GitHub 开源 | 实际场景知识更新 | ✅ 已集成 | ★★★★ |
| 7 | **HalluEditBench** | ICLR 2025 | 40+ | GitHub + HF | **真实幻觉纠正 + Robustness** | ❌ 未集成 | ★★★★ |
| 8 | **MQuAKE-Remastered** | ICLR 2025 | 20+ | OpenReview | 修正多跳评测标签 | ❌ 未集成 | ★★★★ |
| 9 | **UniEdit** | NeurIPS 2025 D&B | 15+ | GitHub + HF | **311K 开放域 + 多跳波及** | ❌ 未集成 | ★★★★ |
| 10 | **WikiBigEdit** | ICML 2025 | 15+ | GitHub + HF | **500K 终身大规模** | ❌ 未集成 | ★★★ |
| 11 | **SafeEdit** | ACL 2024 | 80+ | 开源 | **安全/去毒编辑** | ❌ 未集成 | ★★★ |
| 12 | **ScEdit** | ACL 2025 | 10+ | 开源 | 动作型知识 + Text-level | ❌ 未集成 | ★★★ |
| 13 | **COMPKE** | ACL 2025 | 5+ | GitHub | 复杂 QA 一对多推理 | ❌ 未集成 | ★★ |

### 12.3 多模态方法影响力排名

| 排名 | 方法 | 会议 | 引用量(估) | SOTA 表现 | 框架已集成? | 综合评级 |
|------|------|------|-----------|----------|-----------|---------|
| 1 | **UniKE** | NeurIPS 2024 Spotlight | 80+ | 统一编辑 SOTA (2024) | ⚠️ 代码存在但未注册 | ★★★★★ |
| 2 | **ELoRA** | ICLR 2026 (审稿中) | 5+ | LoRA null/knowledge 空间分解，最新 SOTA | ❌ 未集成 | ★★★★ |
| 3 | **BalancEdit** | ICML 2025 | 15+ | Generality-Locality 动态平衡 | ❌ 未集成 | ★★★★ |
| 4 | **LiveEdit** | CVPR 2025 | 15+ | 终身编辑 SOTA (MoE) | ❌ 未集成 | ★★★★ |
| 5 | **DualEdit** | COLM 2025 | 10+ | 双模态分层编辑 | ❌ 未集成 | ★★★ |
| 6 | **VisEdit** | AAAI 2025 | 15+ | 视觉表示归因编辑 | ❌ 未集成 | ★★★ |
| 7 | **MLE** (M2Edit) | EMNLP 2025 | 10+ | 多粒度协同编辑 | ❌ 未集成 | ★★★ |

### 12.4 open-unlearning 框架集成推荐（投入产出比排序）

#### P0 — 立即集成（影响力高 + 适配成本低 + 填补核心空白）

| # | 项目 | 类型 | 投入 | 理由 | 具体步骤 |
|---|------|------|------|------|---------|
| 1 | **修复 UniKE 注册** | 方法修复 | **~0.5天** | 代码已存在于 `src/trainer/edit/unike.py` + `mm_unike.py`，仅需在 `trainer/__init__.py` 注册。NeurIPS 2024 Spotlight，引用 80+，多模态编辑 SOTA | 在 `_register_trainer` 列表中添加 `UniKEEditor` 和 `MMUniKEEditor` |
| 2 | **MQuAKE Benchmark** | Benchmark | **~2天** | EMNLP 2023，引用 300+，**多跳推理**是知识编辑最核心的空白评测维度，框架完全没有覆盖 | 新增 `data/mquake_dataset.py` + `evals/mquake.py`；复用 `EditEvaluator` 架构 |
| 3 | **NMKE** | 方法 | **~3天** | NeurIPS 2025，神经元级终身编辑当前 SOTA，继承 `EditTrainer` 重写 `edit()` | 新增 `trainer/edit/nmke.py`，核心是神经元归因 + 动态稀疏掩码逻辑 |
| 4 | **MEMIT-Merge** | 方法 | **~1天** | ACL 2025，修复已集成的 MEMIT 在同主语批量编辑时的关键缺陷 (46%→98%)，改动极小 | 在现有 `MEMITEditor` 基础上新增 `MEMITMergeEditor`，修改键值合并逻辑 |

#### P1 — 高价值集成（重要评测维度 + 新范式方法）

| # | 项目 | 类型 | 投入 | 理由 | 具体步骤 |
|---|------|------|------|------|---------|
| 5 | **HalluEditBench** | Benchmark | **~2天** | ICLR 2025，40+ 引用，唯一覆盖 **Robustness** 维度 + 真实幻觉场景，数据在 HF/GitHub 已开源 | 新增 `data/halluedit_dataset.py` + `evals/halluedit.py`；扩展 `EditComprehensiveEvaluator` 加入 Robustness |
| 6 | **RippleEdits** | Benchmark | **~2天** | NeurIPS 2024，100+ 引用，唯一评测 **波及效应**（6 类逻辑推理），数据开源 | 新增 `data/ripple_dataset.py` + `evals/ripple.py`；新增 Ripple Effect Rate 指标 |
| 7 | **AdaEdit** | 方法 | **~3天** | ACL 2025 Long，连续编辑 SOTA，解耦稀疏化知识表示，继承 `EditTrainer` | 新增 `trainer/edit/adaedit.py`；核心是扰动权重解耦 + 稀疏化 |
| 8 | **ELoRA** | MM 方法 | **~3天** | ICLR 2026 审稿中，LoRA null/knowledge 空间分解，框架已有 PEFT/LoRA 支持 | 新增 `trainer/edit/mm_elora.py`；复用 PEFT 集成 + AlphaEdit 的零空间思路 |
| 9 | **BalancEdit** | MM 方法 | **~3天** | ICML 2025，codebook 方式不修改权重，类似 GRACE 的适配模式 | 新增 `trainer/edit/mm_balancedit.py`；参考 GRACE 的 codebook 模式 |

#### P2 — 扩展覆盖（大规模评测 + 新方向）

| # | 项目 | 类型 | 投入 | 理由 |
|---|------|------|------|------|
| 10 | **UniEdit** | Benchmark | ~3天 | NeurIPS 2025 D&B，311K 开放域，多跳波及评测，HF 开源 |
| 11 | **WikiBigEdit** | Benchmark | ~3天 | ICML 2025，500K 终身编辑，挑战现有方法规模极限 |
| 12 | **SafeEdit** | Benchmark | ~2天 | ACL 2024，80+ 引用，安全/去毒编辑，与 unlearning 安全遗忘方向互补 |
| 13 | **IFMET** | 方法 | ~3天 | ICML 2025，多跳 locate-then-edit，配合 MQuAKE benchmark |
| 14 | **LiveEdit** | MM 方法 | ~4天 | CVPR 2025，终身 VLM 编辑 + Low-Rank MoE |
| 15 | **KGMET** | 方法 | ~3天 | EMNLP 2025，KG 引导编辑方向 |

### 12.5 推荐实施路线

```
Phase 0 — 即时修复 (~1天)
  └── 注册 UniKE / MMUniKE 到 TRAINER_REGISTRY（代码已存在）

Phase 1 — 核心扩展 (~1周) — 填补最关键的评测空白
  ├── Benchmark: MQuAKE + HalluEditBench
  ├── 方法: NMKE + MEMIT-Merge
  └── 指标: Multi-hop Accuracy + Robustness + Ripple Effect Rate

Phase 2 — 方法深化 (~2周) — 新 SOTA 方法 + 多模态
  ├── Benchmark: RippleEdits + SafeEdit
  ├── 文本方法: AdaEdit
  └── MM方法: ELoRA + BalancEdit

Phase 3 — 规模化 (~3周) — 大规模 + 终身编辑
  ├── Benchmark: UniEdit (311K) + WikiBigEdit (500K)
  ├── 方法: IFMET + KGMET
  └── MM方法: LiveEdit (终身 MoE)
```

### 12.6 知识编辑 vs 机器遗忘：框架统一视角

| 维度 | 知识编辑 (Editing) | 机器遗忘 (Unlearning) | 共享基础设施 |
|------|-------------------|---------------------|------------|
| **核心操作** | 替换事实 (A→B) | 删除知识 (A→∅) | 模型参数修改 pipeline |
| **评估三轴** | Reliability/Generality/Locality | Forget/Retain/Utility | 相同的 "改动有效 + 不影响其它" 逻辑 |
| **终身场景** | LiveEdit / WikiBigEdit | PULSE / MLLMU 序列 | 序列化训练 + 退化检测 |
| **安全方向** | SafeEdit / ToxEdit | SafeEraser | 安全知识操控 |
| **视觉表示** | VisEdit (视觉归因) | VKD (视觉蒸馏) | 多模态中间表示操作 |
| **LoRA/PEFT** | ELoRA / KDKE | 多种 LoRA-based unlearning | `InjectTrainer` / PEFT 集成 |
| **Codebook** | GRACE / BalancEdit | — | 激活空间离散编辑 |
| **框架位置** | `src/trainer/edit/` + `src/evals/edit.py` | `src/trainer/unlearn/` + `src/evals/tofu.py` | 共享 `base.py` + `model/` + `data/` |

> **核心洞察**: 知识编辑和机器遗忘在框架中共享大量基础设施，特别是模型加载、数据管线、评估指标逻辑。将知识编辑 benchmark 和方法集成到 open-unlearning，不仅扩展了框架覆盖面，还能实现 **editing ↔ unlearning 的统一评测**（例如同一模型同时测编辑有效性和遗忘彻底性）。
