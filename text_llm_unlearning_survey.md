# 纯文本 LLM Unlearning 综述：Benchmark、方法、影响力与适配性

> 整理时间：2026-03-25 | 覆盖范围：2024.01 – 2026.03
> Google Scholar 引用数据截至 2026 年 3 月

---

## 一、影响力排行榜

### Benchmark 排名

| 排名 | Benchmark | 引用量 | GitHub Stars | 发表会议 | 框架已集成 | 综合评级 |
|------|-----------|--------|-------------|---------|-----------|---------|
| **1** | **TOFU** | **393** | 27 (已迁移至 open-unlearning) | ICLR 2024 Workshop → 事实标准 | ✅ 已集成 | ★★★★★ |
| **2** | **WMDP** | ~300+ | **160** | ICML 2024 | ✅ 已集成 | ★★★★★ |
| **3** | **MUSE** | ~80+ | **28** | ICLR 2025 | ✅ 已集成 | ★★★★★ |
| **4** | **RWKU** | ~50+ | **92** | NeurIPS 2024 D&B | ❌ 未集成 | ★★★★ |
| **5** | **LUME** | ~5 | — | EMNLP 2025 Findings | ❌ 未集成 | ★★★ |
| **6** | **R-TOFU** | **7** | — | EMNLP 2025 | ❌ 未集成 | ★★★ |

### 方法排名

| 排名 | 方法 | 引用量 | 发表会议 | 评测 Benchmark | 评测模型 & 规模 | 框架已集成 | 综合评级 |
|------|------|--------|---------|---------------|----------------|-----------|---------|
| **1** | **NPO** | **383** | arXiv (被大量引用) | TOFU | Llama-2-7B, Phi-1.5 (1.3B) | ✅ 已集成 | ★★★★★ |
| **2** | **SimNPO** | **87** | NeurIPS 2025 | TOFU, MUSE, WMDP | Llama-2-7B, Zephyr-7B | ✅ 已集成 | ★★★★★ |
| **3** | **GradAscent** | — (基线方法) | — | 通用 | Llama-2-7B (通用基线) | ✅ 已集成 | ★★★★ |
| **4** | **GradDiff** | — (基线方法) | — | 通用 | Llama-2-7B (通用基线) | ✅ 已集成 | ★★★★ |
| **5** | **AltPO** | **36** | COLING 2025 | TOFU | Llama-2-7B | ⚠️ 社区配置（非原生） | ★★★★ |
| **6** | **SOUL** | ~30 | EMNLP 2024 | TOFU, WMDP | Llama-2-7B, Zephyr-7B | ❌ 未集成 | ★★★★ |
| **7** | **SalUn** | ~100+ | ICLR 2024 Spotlight | MU-Bench, Diffusion | ResNet-18, SD v1.4 (~890M) | ❌ 未集成 (LLM版) | ★★★★ |
| **8** | **RMU** | (含在WMDP中) | ICML 2024 | WMDP | Zephyr-7B, Llama-2-7B | ✅ 已集成 | ★★★★ |
| **9** | **UNDIAL** | ~10 | NAACL 2025 | TOFU, MUSE | Llama-2-7B | ✅ 已集成 | ★★★ |
| **10** | **FLAT** | ~5 | ICLR 2025 | TOFU, MUSE | Llama-2-7B | ❌ 未集成 | ★★★ |
| **11** | **LUNAR** | ~3 | arXiv 2025 | TOFU, MUSE, WMDP | Llama-2-7B, Llama-3-8B | ❌ 未集成 | ★★★ |
| **12** | **DPO** | — (基线) | — | TOFU | Llama-2-7B (通用基线) | ✅ 已集成 | ★★★ |
| **13** | **CEU** | ~5 | — | TOFU | Llama-2-7B | ✅ 已集成 | ★★★ |
| **14** | **CATNIP** | ~0 | arXiv 2026 | TOFU, MUSE | Llama-2-7B, Llama-3-8B | ❌ 未集成 | ★★ |
| **15** | **KIF** | ~0 | arXiv 2026 | TOFU, WMDP | Llama-2-7B, DeepSeek-R1-7B | ❌ 未集成 | ★★ |

---

## 二、Benchmark 详细总结表

| # | Benchmark | 时间 | 会议 | 遗忘目标 | 模型 | 数据规模 | 核心指标 | 链接 |
|---|-----------|------|------|---------|------|---------|---------|------|
| 1 | **TOFU** | 2024.01 | ICLR 2024 Workshop | 虚构作者 QA 知识 | Llama-2-7B, Phi-1.5 | 200 虚构作者, 4000 QA 对 | Forget Quality, Model Utility, MIA, Truth Ratio | [arXiv:2401.06121](https://arxiv.org/abs/2401.06121) |
| 2 | **WMDP** | 2024.03 | ICML 2024 | 危险知识（生物/化学/网络安全） | Zephyr-7B, Llama 系列 | 3,668 多选题 | WMDP Accuracy (↓), MMLU Accuracy (保持) | [ICML 2024](https://proceedings.mlr.press/v235/li24bc.html) |
| 3 | **MUSE** | 2024.07 | ICLR 2025 | 版权内容（Harry Potter + 新闻） | Llama-2-7B | 6.5M tokens, 20K 样本 | Verbatim Mem, Knowledge Mem, Privacy Leak, Utility, Scalability, Sustainability | [arXiv:2407.06460](https://arxiv.org/abs/2407.06460) |
| 4 | **RWKU** | 2024.07 | NeurIPS 2024 D&B | 真实世界名人知识 | Llama-3-8B-Inst, Phi-3-mini-4k | 200 名人, 13,131 遗忘探针 | 4 种 MIA + 9 种对抗攻击 + Locality + Utility + 推理/真实/流畅 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b1f78dfc9ca0156498241012aec4efa0-Abstract-Datasets_and_Benchmarks_Track.html) |
| 5 | **LUME** | 2025.02 | EMNLP 2025 Findings | 合成小说 + PII 传记 + 公开传记 | OLMo-1B, Llama-2-7B 微调版 | 三任务评测 | Memorization, Privacy Leak, Model Utility | [arXiv:2502.15097](https://arxiv.org/abs/2502.15097) |
| 6 | **R-TOFU** | 2025 | EMNLP 2025 | TOFU 扩展到推理模型 | DeepSeek-R1-7B, Llama-3-8B 推理增强版 | TOFU 推理版 | Reasoning-preserving Unlearning | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.265/) |

---

## 三、方法详细总结表

### 3.1 框架已集成的方法

| # | 方法 | 引用 | 会议 | 核心技术 | 评测模型 & 规模 | 优势 | 配置入口 |
|---|------|------|------|---------|----------------|------|---------|
| 1 | **GradAscent** | 基线 | — | 遗忘集梯度上升 (loss = -loss) | Llama-2-7B | 简单，baseline | `trainer=GradAscent` |
| 2 | **GradDiff** | 基线 | — | GA(forget) + GD(retain) 双目标 | Llama-2-7B | 平衡遗忘与保持 | `trainer=GradDiff` |
| 3 | **NPO** | **383** | arXiv 2024 | 负偏好优化，避免灾难性崩溃 | Llama-2-7B, Phi-1.5 (1.3B) | 首个大规模遗忘有效方法 | `trainer=NPO` |
| 4 | **DPO** | 基线 | — | 偏好优化用于遗忘 | Llama-2-7B | 成熟的对齐技术 | `trainer=DPO` |
| 5 | **SimNPO** | **87** | NeurIPS 2025 | 去除参考模型的 NPO，简单偏好优化 | Llama-2-7B, Zephyr-7B | TOFU+MUSE+WMDP 三bench SOTA | `trainer=SimNPO` |
| 6 | **RMU** | (WMDP) | ICML 2024 | 表示误导 (Representation Misdirection) | Zephyr-7B, Llama-2-7B | 安全遗忘 SOTA | `trainer=RMU` |
| 7 | **UNDIAL** | ~10 | NAACL 2025 | 自蒸馏 + 调整 Logits | Llama-2-7B | 鲁棒性 + 可扩展性 | `trainer=UNDIAL` |
| 8 | **CEU** | ~5 | — | Cross-Entropy 遗忘 | Llama-2-7B | 简洁 | `trainer=CEU` |
| 9 | **SatImp** | — | — | 重要性感知参数选择 | Llama-2-7B | 精准遗忘 | `trainer=SatImp` |
| 10 | **WGA** | — | — | 加权梯度上升 | Llama-2-7B | 改进 GA | `trainer=WGA` |
| 11 | **PDU** | — | — | Preference-based Decoupled Unlearning | Llama-2-7B | 解耦遗忘 | `trainer=PDU` |

### 3.2 尚未集成的高影响方法

| # | 方法 | 引用 | 会议 | 核心技术 | 评测模型 & 规模 | 评测 Bench | 开源 | 链接 |
|---|------|------|------|---------|----------------|-----------|------|------|
| 1 | **AltPO** | **36** | COLING 2025 | 交替偏好优化：retain-DPO + forget-NPO 交替训练 | Llama-2-7B | TOFU | ✅ community 配置 | [ACL Anthology](https://aclanthology.org/2025.coling-main.252/) |
| 2 | **SOUL** | ~30 | EMNLP 2024 | 二阶优化 (Sophia) + 影响函数 → 动态迭代遗忘 | Llama-2-7B, Zephyr-7B | TOFU, WMDP | ✅ [GitHub](https://github.com/OPTML-Group/SOUL) | [arXiv:2404.18239](https://arxiv.org/abs/2404.18239) |
| 3 | **SalUn** | ~100+ | ICLR 2024 Spotlight | 梯度显著性权重选择 → 精准遗忘 | ResNet-18, SD v1.4 (~890M) | 图像分类 + Diffusion (LLM 适配需扩展) | ✅ [GitHub](https://github.com/OPTML-Group/Unlearn-Saliency) (144 Stars) | [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html) |
| 4 | **FLAT** | ~5 | ICLR 2025 | 仅需 forget 数据的 f-散度最大化，无需 retain/参考模型 | Llama-2-7B | TOFU, MUSE | ✅ | [OpenReview](https://openreview.net/forum?id=6ESRicalFE) |
| 5 | **LUNAR** | ~3 | arXiv 2025 | 线性表示假说 → 神经元激活重定向到"不知道"区域 | Llama-2-7B, Llama-3-8B | TOFU, MUSE, WMDP | ✅ | [arXiv:2502.07218](https://arxiv.org/abs/2502.07218) |
| 6 | **CATNIP** | ~0 | arXiv 2026 | 校准 + Token 级负偏好对齐 | Llama-2-7B, Llama-3-8B | TOFU, MUSE | ✅ | [arXiv:2602.02824](https://arxiv.org/abs/2602.02824) |
| 7 | **KIF** | ~0 | arXiv 2026 | 激活签名擦除 → 区分真遗忘与行为抑制 | Llama-2-7B, DeepSeek-R1-7B | TOFU, WMDP | ✅ | [arXiv:2601.10566](https://arxiv.org/abs/2601.10566) |
| 8 | **SGA** | ~0 | Under Review (ICLR 2026) | 平滑梯度上升 → 理论最优平滑率 | Llama-2-7B | TOFU | ✅ | [OpenReview](https://openreview.net/forum?id=aa88d32af81a7289bdb94dfa52691a20da7df3ea) |

---

## 四、时间线

```
2024.01  TOFU (ICLR Workshop)      ← 虚构知识遗忘基准，事实标准
2024.03  WMDP + RMU (ICML 2024)    ← 安全遗忘基准 + 表示误导方法
2024.04  NPO                       ← 负偏好优化，避免灾难性崩溃 (383 引用)
2024.04  SOUL (EMNLP 2024)         ← 二阶优化遗忘
2024.06  AltPO (COLING 2025)       ← 交替偏好优化 (36 引用)
2024.07  MUSE (ICLR 2025)          ← 六维评测框架
2024.07  RWKU (NeurIPS 2024 D&B)   ← 真实世界知识遗忘 + 对抗攻击
2024.10  SalUn (ICLR 2024)         ← 显著性遗忘 (144 GitHub Stars)
2024.10  SimNPO (NeurIPS 2025)     ← 简化 NPO，去参考模型 (87 引用)
─── 2025 分界线 ──────────────────────────────────────
2025.01  UNDIAL (NAACL 2025)       ← 自蒸馏 + 调整 Logits
2025.02  LUME (EMNLP 2025)         ← 多任务遗忘评测
2025.02  LUNAR                     ← 神经元激活重定向
2025.05  FLAT (ICLR 2025)          ← 仅需 forget 数据
2025.09  R-TOFU (EMNLP 2025)      ← 推理模型遗忘
─── 2026 分界线 ──────────────────────────────────────
2026.01  KIF                       ← 激活签名擦除
2026.02  CATNIP                    ← Token 级负偏好对齐
2026.03  SGA (under review)        ← 平滑梯度上升
```

---

## 五、open-unlearning 框架适配性分析

### 框架现状

| 维度 | 已集成 |
|------|--------|
| **Benchmark** | TOFU (393引用), MUSE (ICLR 2025), WMDP (ICML 2024) |
| **方法** | GradAscent, GradDiff, NPO, DPO, SimNPO, RMU, UNDIAL, CEU, SatImp, WGA, PDU |
| **模型** | Llama-2/3, Phi, Gemma, Mistral, Qwen, Yi, Baichuan, GPT-2, DeepSeek-R1 等 |
| **架构** | `TRAINER_REGISTRY` + Hydra + `compute_loss()` 覆写模式 |

### Benchmark 适配性

| 优先级 | Benchmark | 引用 | Stars | 理由 | 改动估算 |
|--------|-----------|------|-------|------|---------|
| **P0** | **RWKU** | ~50+ | **92** | NeurIPS 2024 D&B，真实世界名人遗忘，独特的 MIA + 9 种对抗攻击评测，弥补 TOFU 虚构场景不足 | ~3 天：新增 `data/rwku.py` + `evals/rwku.py` + YAML 配置 |
| **P1** | **LUME** | ~5 | — | EMNLP 2025，三任务设计（小说/PII 传记/公开传记），提供 1B + 7B 预训练模型 | ~2 天：数据加载 + evaluator |
| **P2** | **R-TOFU** | 7 | — | TOFU 推理版扩展，与现有 TOFU 配置高度复用 | ~1 天：扩展 TOFU evaluator |

**不建议集成：** TOFU、MUSE、WMDP 已是行业标准且已在框架中。

### 方法适配性

| 优先级 | 方法 | 引用 | 理由 | 改动估算 |
|--------|------|------|------|---------|
| **P0** | **SOUL** | ~30 | EMNLP 2024，二阶优化有理论优势，代码开源 (OPTML-Group)，已在 TOFU+WMDP 上评测 | ~2 天：继承 `Trainer`，替换优化器为 Sophia + influence loss |
| **P0** | **FLAT** | ~5 | ICLR 2025，**无需 retain 数据和参考模型**是全新范式，极大简化实际部署 | ~2 天：新 trainer，仅需 forget 数据的 f-divergence loss |
| **P1** | **LUNAR** | ~3 | 线性表示假说 + 激活重定向，2.9-11.7x 效果提升 + 20x 效率提升 | ~3 天：需要实现表示空间操作 |
| **P1** | **SalUn (LLM版)** | 100+ | ICLR 2024 Spotlight，144 Stars，权重显著性遗忘已被广泛验证，但原版面向 CV/Diffusion，需适配 LLM | ~3 天：提取 LLM 显著性图逻辑，整合进 trainer |
| **P2** | **CATNIP** | ~0 | 2026 最新，Token 级校准 + 负偏好对齐，SimNPO 的改进版 | ~2 天：继承 SimNPO trainer 修改 loss |
| **P2** | **KIF** | ~0 | 2026 最新，激活签名擦除区分真遗忘 vs 行为抑制 | ~3 天 |
| **P2** | **SGA** | ~0 | 平滑梯度上升，理论最优平滑率，GA 的直接改进 | ~1 天：GA 基础上加平滑项 |

**不建议集成（已覆盖）：** AltPO 已有社区配置（`community/methods/AltPO/`）。

---

## 六、推荐实施路线

### Phase 1：快速补全（~1 周）

```
Benchmark:
  └── RWKU               → 真实世界遗忘 + 对抗攻击（NeurIPS 2024, 92 Stars）

方法:
  ├── SOUL               → 二阶优化（EMNLP 2024, 开源）
  └── FLAT               → 仅需 forget 数据（ICLR 2025，全新范式）

文件改动:
  src/data/rwku.py                    # RWKU 数据加载（NEW）
  src/evals/rwku.py                   # RWKU evaluator（NEW）
  src/trainer/unlearn/soul.py         # SOUL 方法（NEW）
  src/trainer/unlearn/flat.py         # FLAT 方法（NEW）
  src/trainer/__init__.py             # 注册新 trainer（MODIFY）
  configs/data/datasets/RWKU_*.yaml   # 数据配置（NEW）
  configs/trainer/SOUL.yaml           # 训练配置（NEW）
  configs/trainer/FLAT.yaml           # 训练配置（NEW）
  configs/experiment/unlearn/rwku/    # 实验配置（NEW）
  configs/experiment/eval/rwku/       # 评估配置（NEW）
```

### Phase 2：效果提升（~2 周）

```
Benchmark:
  └── LUME               → 多任务遗忘（EMNLP 2025）

方法:
  ├── LUNAR              → 激活重定向（20x 效率提升）
  └── SalUn (LLM版)      → 显著性遗忘（ICLR 2024 Spotlight）
```

### Phase 3：前沿探索（~3 周）

```
Benchmark:
  └── R-TOFU             → 推理模型遗忘（EMNLP 2025）

方法:
  ├── CATNIP             → Token 级校准 NPO（2026 最新）
  ├── KIF                → 激活签名擦除（2026 最新）
  └── SGA                → 平滑 GA（GA 直接升级，1天完成）
```

---

## 七、投入产出比排序

| 排名 | 项目 | 类型 | 引用/Stars | 投入 | 产出 |
|------|------|------|-----------|------|------|
| 1 | **RWKU** | Bench | 50+引用, 92 Stars | ~3天 | NeurIPS 2024，真实世界遗忘+对抗攻击，弥补TOFU虚构不足 |
| 2 | **SOUL** | 方法 | ~30引用 | ~2天 | EMNLP 2024，二阶优化，跨benchmark优于一阶方法 |
| 3 | **FLAT** | 方法 | ~5引用 | ~2天 | ICLR 2025，无需retain数据+无需参考模型，实际部署价值大 |
| 4 | **SGA** | 方法 | ~0 | ~1天 | GA 直接改进，1天完成，理论有保证 |
| 5 | **LUNAR** | 方法 | ~3引用 | ~3天 | 20x效率提升，激活重定向新范式 |
| 6 | **LUME** | Bench | ~5引用 | ~2天 | EMNLP 2025，多任务评测 |
| 7 | **SalUn LLM版** | 方法 | 100+引用 | ~3天 | ICLR 2024 Spotlight，需从CV适配到LLM |
| 8 | **CATNIP** | 方法 | ~0 | ~2天 | 2026最新，SimNPO改进版 |
| 9 | **R-TOFU** | Bench | 7引用 | ~1天 | 推理模型遗忘，复用现有TOFU |
| 10 | **KIF** | 方法 | ~0 | ~3天 | 激活签名擦除，区分真遗忘vs行为抑制 |

---

## 八、关键发现与趋势

### 核心发现

1. **TOFU 是绝对标杆（393 引用）**，几乎所有方法都在 TOFU 上评测
2. **NPO 家族占据方法主流**：NPO (383) → SimNPO (87) → AltPO (36) → CATNIP，持续迭代
3. **框架覆盖度极高**：已集成 3 大 benchmark + 11 种方法，覆盖了引用量 Top-5 中的全部方法
4. **最大空白是 RWKU**：真实世界知识遗忘 + 对抗攻击评测，是 TOFU 虚构场景的重要互补
5. **FLAT 代表新范式**：无需 retain 数据 + 无需参考模型，对实际部署意义重大

### 技术演进路线

```
GA (梯度上升)                     → 灾难性崩溃
  ├── GradDiff (加 retain 约束)   → 缓解但未解决
  ├── NPO (偏好优化)              → 383 引用，解决崩溃但有参考模型偏差
  │   ├── SimNPO (去参考模型)     → 87 引用，NeurIPS 2025
  │   ├── AltPO (交替优化)        → 36 引用，COLING 2025
  │   └── CATNIP (Token 级校准)   → 2026 最新
  ├── SOUL (二阶优化)             → EMNLP 2024，理论更强
  ├── RMU (表示误导)              → ICML 2024，安全方向 SOTA
  └── FLAT (仅 forget 数据)       → ICLR 2025，最简部署
```

### 与框架的关系

你的框架（open-unlearning）已经是这个领域**覆盖度最高的开源框架**（3 bench + 11 方法），核心的 NPO → SimNPO 演进线和三大 benchmark 都已集成。下一步最大增量来自：
- **RWKU**（补全真实世界场景）
- **SOUL**（补全二阶优化路线）
- **FLAT**（补全无 retain 数据范式）
