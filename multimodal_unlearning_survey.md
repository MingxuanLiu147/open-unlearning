# 多模态机器遗忘（Multimodal Machine Unlearning）综述

> 整理时间：2026-03-25 | 覆盖范围：2024.06 – 2026.01

---

## 一、时间线总览

```
2024.06  MU-Bench          首个多任务多模态遗忘统一评测 (NeurIPS 2024)
2024.10  MLLMU-Bench       面向MLLM隐私保护的遗忘基准 (NAACL 2025)
2024.10  CLEAR             首个开源多模态角色遗忘基准 (ACL 2025 Findings)
2024.10  UnlearnCanvas     扩散模型风格/物体遗忘基准 (NeurIPS 2024 D&B)
2024.12  UnLOK-VQA         多模态知识删除攻防评测 (TMLR 2024)
2025.03  PEBench           虚构人物+事件场景的MLLM遗忘基准
2025.05  UMU-Bench         关注模态对齐的多模态遗忘基准
2025.07  MMUnlearner       几何约束梯度下降的MLLM遗忘方法 (ACL 2025 Findings)
2025.07  CLIPErase         CLIP视觉-文本关联遗忘 (ACL 2025)
2025.07  PULSE             实际场景下的LMM遗忘评估协议 (NeurIPS 2025 Workshop)
2025.09  SAEmnesia         稀疏自编码器的扩散模型概念擦除
2025.11  S-MLLMUn Bench    敏感知识遗忘+视觉理解保持联合评测
2025.12  Domain-Agnostic   无训练无数据的CLIP选择性遗忘
2026.01  ViKeR             视觉引导关键Token正则化的MLLM遗忘
```

---

## 二、Benchmark 总结表

| Benchmark | 时间 | 发表 | 目标模型类型 | 具体测试模型 | 遗忘目标 | 数据规模 | 评估指标 | 论文链接 |
|-----------|------|------|-------------|-------------|---------|---------|---------|---------|
| **MU-Bench** | 2024.06 | NeurIPS 2024 | 多任务多模态（图像/文本/语音/视频） | 20种模型架构（含ViT, BERT, ResNet等） | 训练样本移除 | 9个数据集（CIFAR-100, IMDB, UCF101, NLVR², SAMSum等） | Accuracy(Forget/Retain), MIA, Relearn Time | [arXiv:2406.14796](https://arxiv.org/abs/2406.14796) |
| **MLLMU-Bench** | 2024.10 | NAACL 2025 | MLLM | LLaVA-1.5-7B/13B, Idefics2-8B | 人物隐私信息遗忘 | 653个人物档案（500虚构+153名人），20,000+问答对 | Forget Efficacy, Generalizability, Model Utility | [arXiv:2410.22108](https://arxiv.org/abs/2410.22108) |
| **CLEAR** | 2024.10 | ACL 2025 Findings | MLLM | LLaVA-1.5-7B（及其他MLLM） | 虚构角色跨模态遗忘 | 200个虚构人物，3,700张图像+QA对 | Forget Acc, Retain Acc, Real-World Acc, Celebrity Acc | [arXiv:2410.18057](https://arxiv.org/abs/2410.18057) |
| **UnlearnCanvas** | 2024.10 | NeurIPS 2024 D&B | Diffusion Model | Stable Diffusion v1.4/v2.1, DDPM | 艺术风格/物体概念擦除 | 高分辨率风格化图像数据集 | Style-UA, Object-UA, IRA, CRA, FID, Robustness | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aebf4822d30c3f2600566af7eba83548-Abstract-Datasets_and_Benchmarks_Track.html) |
| **UnLOK-VQA** | 2024.12 | TMLR 2024 | MLLM | LLaVA-1.5-7B/13B, InstructBLIP-7B | 多模态外部知识删除 | VQA扩展数据集 | Attack Success Rate（4种白盒+3种黑盒攻击），Defense Effectiveness | [HuggingFace](https://huggingface.co/papers/2505.01456) |
| **PEBench** | 2025.03 | Under Review | MLLM | LLaVA系列, InternVL系列 | 人物实体+事件场景遗忘 | 虚构人物+对应事件场景数据集 | Forget Acc, Retain Acc, Cross-Concept Interference | [arXiv:2503.12545](https://arxiv.org/abs/2503.12545) |
| **UMU-Bench** | 2025.05 | Under Review | MLLM | LLaVA-1.5-7B | 模态对齐的知识遗忘 | 653个人物档案（分类/完形填空/生成三种任务） | AccF, AccR, RLF, RLR（模态对齐指标） | [GitHub](https://github.com/QDRhhhh/UMU-bench) |
| **PULSE** | 2025.07 | NeurIPS 2025 Workshop | LMM | 多种LMM | 预训练知识遗忘 + 序列遗忘可持续性 | 涵盖不同知识获取阶段 | Forget Efficacy, Sequential Sustainability, Model Utility | [arXiv:2507.01271](https://arxiv.org/abs/2507.01271) |
| **S-MLLMUn Bench** | 2025.11 | Under Review | MLLM | 多种MLLM | 敏感知识遗忘 + 视觉理解保持 | 敏感信息数据集 | Sensitive Removal Rate, Visual Understanding Retention | [arXiv:2511.20196](https://arxiv.org/abs/2511.20196) |

---

## 三、方法论文总结表

| 方法 | 时间 | 发表 | 目标模型 & 规模 | 核心技术 | 使用的Benchmark | 关键指标 | 论文链接 |
|------|------|------|----------------|---------|----------------|---------|---------|
| **MU-Bench Baselines** | 2024.06 | NeurIPS 2024 | ViT, BERT, ResNet, GPT-2, Whisper, VideoMAE, SD v1.4 (20种架构) | RandLabel, SalUn, BadT, SCRUB等10种方法统一评测 | MU-Bench | Accuracy, MIA, Relearn Time | [arXiv:2406.14796](https://arxiv.org/abs/2406.14796) |
| **UnlearnCanvas Methods** | 2024.10 | NeurIPS 2024 | Stable Diffusion v1.4/v2.1 (~890M), DDPM | ESD, FMN, UCE, SalUn, CA等9种方法 | UnlearnCanvas | UA, IRA, CRA, FID | [GitHub](https://github.com/OPTML-Group/UnlearnCanvas) |
| **MMUnlearner** | 2025.07 | ACL 2025 Findings | LLaVA-1.5-7B/13B | 几何约束梯度下降 + 权重显著性图 → 选择性擦除视觉模式，保留文本知识 | MLLMU-Bench, CLEAR | Forget Acc, Retain Acc, Text Knowledge Preservation | [ACL Anthology](https://www.aclanthology.org/2025.findings-acl.375/) |
| **CLIPErase** | 2025.07 | ACL 2025 | CLIP ViT-B/16 (~150M), ViT-L/14 (~430M) | 三模块架构：遗忘模块 + 保持模块 + 一致性模块 → 解耦视觉-文本关联 | CIFAR-100, Flickr30K, CC-12M | Zero-shot Acc, Retrieval R@K, FID | [ACL Anthology](https://anthology.aclweb.org/2025.acl-long.1469/) |
| **SAEmnesia** | 2025.09 | Preprint | Stable Diffusion v1.4 (~890M) | 稀疏自编码器 → 概念-神经元一对一映射 → 可解释概念擦除 | UnlearnCanvas | UA (+9.2%), Sequential UA (+28.4%), FID | [arXiv:2509.21379](https://arxiv.org/abs/2509.21379) |
| **SMFA** | 2025.11 | Under Review | LLaVA-1.5-7B/13B | Sculpted Memory Forgetting Adapter → 精准约束遗忘区域 + 保持锚引导 | S-MLLMUn Bench | Sensitive Removal, Visual Retention | [arXiv:2511.20196](https://arxiv.org/abs/2511.20196) |
| **Domain-Agnostic CLIP Unlearning** | 2025.12 | Preprint | CLIP ViT-B/16 (~150M), ViT-L/14 (~430M) | 无训练无数据 → 多模态零空间（文本提示 + 合成视觉原型） | DomainNet, Office-Home | Domain-specific UA, Cross-domain Retain Acc | [arXiv:2512.14113](https://arxiv.org/abs/2512.14113) |
| **ViKeR** | 2026.01 | Preprint | LLaVA-1.5-7B/13B | 视觉引导关键Token正则化 → 利用无关视觉输入预测理想遗忘后分布 + 信息熵定义关键Token | MLLMU-Bench, CLEAR | Forget Efficacy, Retain Utility, Response Coherence | [arXiv:2601.22020](https://arxiv.org/abs/2601.22020) |

---

## 四、主流模型使用统计

### 4.1 多模态大模型（MLLM）

| 模型 | 被使用的论文/Benchmark |
|------|----------------------|
| **LLaVA-1.5-7B** | MLLMU-Bench, CLEAR, UMU-Bench, PEBench, MMUnlearner, ViKeR |
| **LLaVA-1.5-13B** | MLLMU-Bench, PEBench |
| **LLaVA-NeXT / 1.6** | MLLMU-Bench |
| **Idefics2** | MLLMU-Bench |
| **InternVL系列** | PEBench |

### 4.2 视觉-语言模型（VLM）

| 模型 | 被使用的论文/Benchmark |
|------|----------------------|
| **CLIP ViT-B/16** | CLIPErase, Domain-Agnostic |
| **CLIP ViT-L/14** | CLIPErase |

### 4.3 扩散模型（Diffusion Model）

| 模型 | 被使用的论文/Benchmark |
|------|----------------------|
| **Stable Diffusion v1.4** | UnlearnCanvas, SAEmnesia |
| **Stable Diffusion v2.1** | UnlearnCanvas |
| **DDPM** | UnlearnCanvas |

---

## 五、主流评估指标分类

### 5.1 遗忘有效性（Forgetting Effectiveness）

| 指标 | 含义 | 使用场景 |
|------|------|---------|
| **Forget Accuracy (FA)** | 遗忘集上的准确率（越低越好） | MLLM人物/知识遗忘 |
| **Unlearning Accuracy (UA)** | 遗忘概念的生成准确率 | Diffusion模型概念擦除 |
| **Attack Success Rate** | 攻击者能否恢复遗忘信息 | 攻防评测 (UnLOK-VQA) |
| **MIA (Membership Inference Attack)** | 成员推断攻击成功率 | 隐私评测 |
| **AccF** | 遗忘集指标（模态对齐版） | UMU-Bench |
| **RLF (Ratio Leakage Forget)** | 遗忘集泄漏比率 | UMU-Bench |

### 5.2 保持能力（Retention / Utility）

| 指标 | 含义 | 使用场景 |
|------|------|---------|
| **Retain Accuracy (RA)** | 保留集上的准确率（越高越好） | 通用 |
| **In-domain Retain Acc (IRA)** | 同域保留准确率 | UnlearnCanvas |
| **Cross-domain Retain Acc (CRA)** | 跨域保留准确率 | UnlearnCanvas |
| **Model Utility** | 模型通用能力保持 | MLLMU-Bench, PULSE |
| **Visual Understanding Retention** | 视觉理解能力保持 | S-MLLMUn Bench |
| **AccR / RLR** | 保留集指标（模态对齐版） | UMU-Bench |

### 5.3 生成质量（Generation Quality）

| 指标 | 含义 | 使用场景 |
|------|------|---------|
| **FID** | 生成图像质量 | Diffusion模型 |
| **Response Coherence** | 文本回答连贯性 | MLLM |
| **Zero-shot Acc** | 零样本分类准确率 | CLIP |
| **Retrieval R@K** | 检索召回率 | CLIP |

### 5.4 高级评估维度

| 指标 | 含义 | 使用场景 |
|------|------|---------|
| **Cross-Concept Interference** | 遗忘一个概念对关联概念的影响 | PEBench |
| **Sequential Sustainability** | 多次序列遗忘后的性能可持续性 | PULSE, UnlearnCanvas |
| **Robustness** | 对抗性提示攻击下的鲁棒性 | UnlearnCanvas, UnLOK-VQA |
| **Modality Alignment** | 跨模态遗忘一致性 | UMU-Bench |
| **Relearn Time** | 重新学习遗忘知识所需时间 | MU-Bench |

---

## 六、关键发现与趋势

### 6.1 核心发现

1. **跨模态联合遗忘优于单模态遗忘**：CLEAR 发现同时在文本和视觉模态上执行遗忘，效果显著优于仅在单一模态上操作。

2. **跨概念干扰（Cross-Concept Interference）**：PEBench 揭示遗忘一个概念会无意中降低相关概念的性能，这对实际部署构成挑战。

3. **模态对齐问题**：UMU-Bench 发现现有方法常常无法在单模态和多模态设置中一致地移除知识。

4. **预训练知识难以遗忘**：PULSE 指出当前方法可以成功遗忘微调知识，但难以消除预训练阶段学到的知识。

5. **序列遗忘退化**：多次序列遗忘请求会导致模型性能大幅退化。

6. **规模效应**：UnLOK-VQA 发现更大的模型在编辑后展示出更强的鲁棒性。

### 6.2 研究趋势

- **2024年**：建立基础 Benchmark（MU-Bench, CLEAR, UnlearnCanvas, MLLMU-Bench）
- **2025年上半年**：面向实际场景的精细化评测（PEBench 跨概念、UMU-Bench 模态对齐）
- **2025年下半年**：方法创新（SMFA 适配器、SAEmnesia 可解释擦除）+ 序列/持续遗忘评测（PULSE）
- **2026年**：更精细的Token级别遗忘控制（ViKeR），关注遗忘过程中的信息论分析

---

## 七、相关综述论文

| 综述 | 时间 | 覆盖范围 | 链接 |
|------|------|---------|------|
| A Survey on Generative Model Unlearning | 2025.07 | LLM + Diffusion + MLLM + Audio 全覆盖 | [arXiv:2507.19894](https://arxiv.org/abs/2507.19894) |
| A Survey on LLM Unlearning: Taxonomy, Evaluations, and Future Directions | 2025.10 | LLM 遗忘的系统分类与评估 | [Springer](https://link.springer.com/article/10.1007/s10462-025-11376-7) |
| Awesome Generative Model Unlearning Survey | 持续更新 | GitHub 资源汇总 | [GitHub](https://github.com/caxLee/Generative-model-unlearning-survey) |

---

## 八、开源资源汇总

| 资源 | 类型 | 链接 |
|------|------|------|
| MU-Bench | Benchmark + Code | [GitHub](https://github.com/clu-uml/mu-bench) / [项目页](https://clu-uml.github.io/MU-Bench-Project-Page) |
| MLLMU-Bench | Benchmark + Data | [GitHub](https://github.com/franciscoliu/MLLMU-Bench) / [HuggingFace](https://huggingface.co/datasets) |
| CLEAR | Dataset | [HuggingFace](https://huggingface.co/datasets/therem/CLEAR) / [GitHub](https://github.com/somvy/multimodal_unlearning) |
| UnlearnCanvas | Benchmark + Interactive Demo | [GitHub](https://github.com/OPTML-Group/UnlearnCanvas) / [Demo](https://optml-group-unlearncanvas-benchmark.hf.space) |
| PEBench | Benchmark + Code | [项目页](https://pebench.github.io/) |
| UMU-Bench | Benchmark + Code | [GitHub](https://github.com/QDRhhhh/UMU-bench) |
| MMUnlearner | Method Code | [GitHub](https://github.com/Z1zs/MMUnlearner) |







> 整理时间：2026-03-25 | 覆盖范围：2024.05 — 2026.03



---



## 一、总览



多模态机器遗忘（Multimodal Machine Unlearning）是近两年快速发展的方向，涵盖三大子领域：



| 子领域 | 核心模型类型 | 代表 Benchmark |

|--------|-------------|---------------|

| **MLLM 遗忘** (多模态大语言模型) | LLaVA, InstructBLIP, Qwen-VL 等 | MMUBench, MLLMU-Bench, FIUBench, UMU-Bench, PEBench, SafeEraser, MMDU-Bench, PULSE |

| **CLIP 遗忘** (视觉-语言对齐模型) | CLIP (ViT-B/32, ViT-L/14) | CLIPErase 自建评估 |

| **T2I 扩散模型遗忘** (文生图) | Stable Diffusion, SDXL | HUB, GenMU |



---



## 二、Benchmark 汇总表



| # | Benchmark | 会议/来源 | 时间 | 目标模态 | 目标模型 | 数据规模 | 核心评估指标 | 论文链接 |

|---|-----------|----------|------|---------|---------|---------|-------------|---------|

| 1 | **MMUBench** | NeurIPS 2024 | 2024.05 | 图像+文本 (MLLM) | LLaVA-1.5-7B/13B | 虚构概念数据集 | Forget Accuracy, Model Utility, Membership Inference Attack (MIA), Jailbreak Attack Resistance | [arXiv:2405.12523](https://arxiv.org/abs/2405.12523) |

| 2 | **MU-Bench** | NeurIPS 2024 D&B | 2024.06 | 图像/文本/语音/视频 (多任务) | ResNet, BERT, GPT-2, ViT, Whisper, VideoMAE, Stable Diffusion | 9 数据集, 涵盖 6 个判别式+3 个生成式任务 | Deletion Accuracy, Remaining Accuracy, MIA, Run-time Efficiency | [arXiv:2406.14796](https://arxiv.org/abs/2406.14796) |

| 3 | **MLLMU-Bench** | NAACL 2025 | 2024.10 | 图像+文本 (MLLM) | LLaVA-1.5-7B/13B, InstructBLIP 等 | 653 人物 profiles (500 虚构 + 153 真实), 14+ QA pairs/profile | Forget Efficacy, Generalizability, Model Utility (分 unimodal/multimodal) | [arXiv:2410.22108](https://arxiv.org/abs/2410.22108) |

| 4 | **FIUBench** | ICLR 2025 | 2024.11 | 图像+文本 (VLM) | LLaVA-1.5-7B/13B, 其他 VLMs | 虚构面部身份 VQA 数据集 | MIA, Adversarial Privacy Attack, Forget Quality, Model Utility | [arXiv:2411.03554](https://arxiv.org/abs/2411.03554) |

| 5 | **SafeEraser** | ACL 2025 Findings | 2025.02 | 图像+文本 (MLLM 安全) | LLaVA-7B, LLaVA-13B | 3,000 images + 28.8K VQA pairs | Forget Quality, Model Utility, **SARR** (Safe Answer Refusal Rate) | [arXiv:2502.12520](https://arxiv.org/abs/2502.12520) |

| 6 | **PEBench** | arXiv | 2025.03 | 图像+文本 (MLLM) | MLLMs (多种) | 虚构个人实体 + 事件场景数据集 | Cross-concept Interference, Forget Accuracy, Retain Accuracy | [arXiv:2503.12545](https://arxiv.org/abs/2503.12545) |

| 7 | **UMU-Bench** | NeurIPS 2025 D&B | 2025 | 图像+文本 (MLLM 模态对齐) | LLaVA-1.5-7B | 653 profiles (500 虚构 + 153 真实), classification/cloze/generation | **AccF, AccR, RLF, RLR** (模态对齐遗忘指标) | [GitHub](https://github.com/QDRhhhh/UMU-bench) |

| 8 | **MMDU-Bench** | OpenReview 2025 | 2025 | 图像+文本 (LVLM 深度遗忘) | LVLMs | 30K+ relations, 166K QA pairs (合成知识图谱) | **Deep Forget Quality**, Explicit/Implicit Forgetting, Cross-modal Retention | [OpenReview](https://openreview.net/forum?id=Si1xG8fNvY) |

| 9 | **PULSE** | arXiv / OpenReview | 2025.07 | 图像+文本 (LMM) | Large Multimodal Models | 预训练知识 + 微调知识评估 | Pre-trained Knowledge Forget, Sequential Unlearning Degradation, Model Utility | [arXiv:2507.01271](https://arxiv.org/abs/2507.01271) |

| 10 | **HUB** (Holistic Unlearning Bench) | ICCV 2025 | 2025 | 文本→图像 (Diffusion) | Stable Diffusion 系列 | 33 concepts × 16,000 prompts (Celebrity/Style/IP/NSFW) | Faithfulness, Alignment, Pinpoint-ness, Multilingual Robustness, Attack Robustness, Efficiency | [OpenReview](https://openreview.net/forum?id=kaqrwQ96xW) |

| 11 | **GenMU** | ICCV 2025 Workshop (U&Me) | 2025 | 文本→图像 (Diffusion) | Text-to-Image Diffusion Models | Forget/Locality/Adjacency Sets + 工程化/对抗性 prompts | **ERR Score** (Erasing-Retention-Robustness), FRS (Forgetting-Retention Score) | [ICCV 2025 U&Me](https://sites.google.com/view/u-and-me-workshop/challenge) |



---



## 三、Unlearning 方法论文汇总表



| # | 方法名称 | 会议/来源 | 时间 | 目标模型 & 规模 | 核心技术 | 评估 Benchmark | 论文链接 |

|---|---------|----------|------|----------------|---------|---------------|---------|

| 1 | **SIU** (Single Image Unlearning) | NeurIPS 2024 | 2024.05 | LLaVA-1.5-7B/13B | 单图像微调 + Dual Masked KL-divergence Loss + CE Loss | MMUBench | [arXiv:2405.12523](https://arxiv.org/abs/2405.12523) |

| 2 | **CLIPErase** | ACL 2025 | 2025 | CLIP ViT-B/32 (~150M), ViT-L/14 (~430M) | 三模块 (Forgetting + Retention + Consistency), 视觉-文本关联解耦 | CIFAR-100, Flickr30K, Conceptual 12M + Diffusion 下游任务 | [ACL 2025](https://aclanthology.org/2025.acl-long.1469/) |

| 3 | **SafeEraser (PD Loss)** | ACL 2025 Findings | 2025.02 | LLaVA-1.5-7B, LLaVA-1.5-13B | Prompt Decouple Loss, 缓解 over-forgetting | SafeEraser Bench (自建) | [arXiv:2502.12520](https://arxiv.org/abs/2502.12520) |

| 4 | **MMUnlearner** | arXiv | 2025.02 | LLaVA-1.5-7B/13B, InternVL2-8B | 几何约束梯度下降 + 权重显著性图, 视觉模式擦除保留文本知识 | MLLMU-Bench, FIUBench | [arXiv:2502.11051](https://arxiv.org/abs/2502.11051) |

| 5 | **VKD** (Visual Knowledge Distillation) | arXiv | 2025.12 | LLaVA-1.5-7B/13B | 解耦视觉/文本知识, 中间视觉表示蒸馏, 仅微调视觉组件 | MLLMU-Bench, CLEAR | [arXiv:2512.11325](https://arxiv.org/abs/2512.11325) |

| 6 | **ViKeR** | arXiv | 2026.01 | LLaVA-1.5-7B/13B | 视觉引导 Token 分布估计 + 信息熵 Key-Token 梯度加权 | MLLMU-Bench, CLEAR | [arXiv:2601.22020](https://arxiv.org/abs/2601.22020) |

| 7 | **KVW** (Knowledge Vector Weakening) | arXiv | 2026.01 | LLaVA-1.5-7B, Qwen2-VL-2B/7B | 训练免微调, 直接弱化遗忘集上激活的知识向量 | 多个 LVLM benchmarks | [arXiv:2601.21794](https://arxiv.org/abs/2601.21794) |

| 8 | **MiM-MU** | arXiv | 2026.03 | SD v1.4/v1.5 (~890M), SDXL (~3.5B) | 互信息最小化, 无补偿概念擦除 | T2I concept erasure benchmarks | [arXiv:2603.00992](https://arxiv.org/abs/2603.00992) |

| 9 | **RASU** (Relationship-Aware Safety Unlearning) | arXiv | 2026.03 | LLaVA-1.5-7B/13B | Object-Relation-Object 元组建模 + LoRA 参数高效编辑 | CLIP-based + paraphrase/contextual/OOD attacks | [arXiv:2603.14185](https://arxiv.org/abs/2603.14185) |



---



## 四、主流模型使用统计



| 模型 | 被用作评估目标的论文数 | 典型论文 |

|------|---------------------|---------|

| **LLaVA-1.5-7B** | 8+ | MMUBench, MLLMU-Bench, FIUBench, UMU-Bench, SafeEraser, VKD, ViKeR, MMUnlearner |

| **LLaVA-1.5-13B** | 5+ | MLLMU-Bench, FIUBench, SafeEraser, VKD, ViKeR |

| **InstructBLIP** | 2+ | MLLMU-Bench, FIUBench |

| **CLIP (ViT-B/32, ViT-L/14)** | 2+ | CLIPErase, RASU |

| **Stable Diffusion v1.4/v1.5/SDXL** | 3+ | HUB, GenMU, MiM-MU, MU-Bench |

| **Qwen-VL / MiniGPT-4** | 1+ | PEBench, MMDU-Bench |

| **GPT-2 / BERT / ResNet** | 1 | MU-Bench (传统判别式任务) |



> **结论**: LLaVA-1.5-7B 是当前多模态 unlearning 领域的绝对主流评估模型，几乎每篇论文都会使用。



---



## 五、核心评估指标汇总



### 5.1 通用指标



| 指标类别 | 具体指标 | 含义 | 使用频率 |

|---------|---------|------|---------|

| **遗忘质量** | Forget Accuracy / AccF | 遗忘集上模型表现下降程度 | ★★★★★ |

| **保留能力** | Retain Accuracy / AccR | 非遗忘数据上模型性能保持 | ★★★★★ |

| **模型效用** | Model Utility | 遗忘后通用能力（如 VQA, 文本生成等）| ★★★★★ |

| **隐私攻击** | MIA (Membership Inference Attack) | 是否能推断数据是否在训练集中 | ★★★★ |



### 5.2 专有指标



| 指标 | 提出论文 | 含义 |

|------|---------|------|

| **SARR** (Safe Answer Refusal Rate) | SafeEraser | 衡量 over-forgetting 导致的过度拒答率 |

| **Deep Forget Quality** | MMDU-Bench | 跨模态推理路径的深层遗忘质量 |

| **Cross-concept Interference** | PEBench | 遗忘一个概念对同图像中其他概念的影响 |

| **Modality Alignment** (RLF/RLR) | UMU-Bench | 文本/多模态遗忘一致性 |

| **ERR Score** | GenMU | 擦除-保留-鲁棒性综合分 |

| **Adversarial Privacy Attack** | FIUBench | 对抗性隐私攻击下的遗忘鲁棒性 |

| **Sequential Unlearning Degradation** | PULSE | 多轮连续遗忘的性能衰退 |



---



## 六、时间线



```

2024.05 ├── SIU + MMUBench (NeurIPS 2024) ← 首个 MLLM unlearning 工作

│

2024.06 ├── MU-Bench (NeurIPS 2024 D&B) ← 首个多任务多模态 benchmark

│

2024.10 ├── MLLMU-Bench (→ NAACL 2025) ← 大规模 MLLM 隐私遗忘 benchmark

│

2024.11 ├── FIUBench (→ ICLR 2025) ← VLM 面部身份遗忘 benchmark

│

─── 2025 分界线 ──────────────────────────────────────────────────────

│

2025.02 ├── SafeEraser (ACL 2025 Findings) ← MLLM 安全遗忘 + PD Loss

├── MMUnlearner ← 几何约束 MLLM 遗忘

│

2025.03 ├── PEBench ← 个人实体遗忘 + cross-concept interference

│

2025.07 ├── PULSE ← 预训练知识遗忘 + 连续遗忘评估协议

│

2025.09 ├── UMU-Bench (NeurIPS 2025 D&B) ← 模态对齐遗忘 benchmark

│

2025.10 ├── HUB (ICCV 2025) ← T2I 扩散模型全面遗忘 benchmark

├── GenMU (ICCV 2025 Workshop) ← T2I 遗忘挑战赛

│

2025.12 ├── VKD ← 视觉知识蒸馏 MLLM 遗忘

├── MMDU-Bench ← 深度多模态遗忘 benchmark (知识图谱)

│

─── 2026 分界线 ──────────────────────────────────────────────────────

│

2026.01 ├── ViKeR ← 视觉引导 Key-Token 正则化

├── KVW ← 训练免微调知识向量弱化

│

2026.03 ├── MiM-MU ← T2I 互信息最小化无补偿遗忘

├── RASU ← 关系感知安全遗忘 (LoRA)

```



---



## 七、综述论文



| 综述 | 时间 | 覆盖范围 | 链接 |

|------|------|---------|------|

| Machine Unlearning in Generative AI: A Survey | 2024.07 | 生成式 AI 全覆盖 (LLM, Diffusion, GAN) | [arXiv:2407.20516](https://arxiv.org/abs/2407.20516) |

| A Survey on LLM Unlearning (Springer AIR) | 2025 | LLM 遗忘分类体系 + 鲁棒性 | [Springer](https://link.springer.com/article/10.1007/s10462-025-11376-7) |

| A Survey on Generative Model Unlearning | 2025.07 | 统一框架: 模型类型×遗忘目标×方法×评估 | [arXiv:2507.19894](https://arxiv.org/abs/2507.19894) |



---



## 八、关键发现与趋势总结



### 1. 核心挑战

- **模态不对齐 (Modality Misalignment)**: 文本模态遗忘成功但视觉模态仍泄露 (UMU-Bench)

- **深层知识纠缠 (Deep Knowledge Entanglement)**: 显式事实遗忘但隐式推理路径仍存在 (MMDU-Bench)

- **过度遗忘 (Over-forgetting)**: 遗忘有害内容时误杀良性能力 (SafeEraser)

- **跨概念干扰 (Cross-concept Interference)**: 遗忘一个概念影响同图像中其他概念 (PEBench)

- **预训练知识难以遗忘**: 现有方法只能遗忘微调知识 (PULSE)



### 2. 技术趋势

- **2024**: 基础框架建立期 — SIU 和 MU-Bench 开辟 MLLM unlearning 方向

- **2025 上半年**: Benchmark 爆发期 — 多个专用 benchmark 从不同角度评估 (安全/隐私/模态对齐)

- **2025 下半年**: 方法深化期 — 从简单 GA/GD 到视觉知识蒸馏、几何约束等精细化方法

- **2026**: 高效化+安全化 — 训练免微调 (KVW)、关系感知 (RASU)、无补偿 (MiM-MU) 方法涌现



### 3. 模型选择建议

- **MLLM 方向**: 首选 LLaVA-1.5-7B/13B 作为 baseline 模型 (覆盖最广)

- **CLIP 方向**: 使用 ViT-B/32 或 ViT-L/14

- **Diffusion 方向**: Stable Diffusion v1.4/v1.5 或 SDXL



### 4. Benchmark 选择建议

- **通用评估**: MU-Bench (最全面多模态覆盖)

- **MLLM 隐私**: MLLMU-Bench / FIUBench (成熟度高, 已发表)

- **MLLM 安全**: SafeEraser

- **模态对齐**: UMU-Bench (专注跨模态一致性)

- **深度遗忘**: MMDU-Bench (知识图谱推理)

- **T2I Diffusion**: HUB (全面) / GenMU (挑战赛)