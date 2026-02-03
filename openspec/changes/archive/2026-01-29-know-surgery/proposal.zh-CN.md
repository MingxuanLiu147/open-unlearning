## 动机

OpenUnlearning 提供强力的遗忘基准，但尚缺少一个统一、可扩展的工具包，用于在可复现流水线、未来多模态支持与面向用户的编排体验下，实现可控的增/删/改知识更新。Know-Surgery 在保持改动最小、与现有仓库兼容的前提下填补这一空白。

## 变更内容

- 将 Know-Surgery 定义为架在 OpenUnlearning 之上的 umbrella 工具包，分阶段范围：v1 仅遗忘、v2 知识编辑 + 初期 GUI、v2.5 知识注入/PEFT、v3 多模态。
- 增加最小化 Python 流水线封装，基于现有 Hydra 配置实现一键遗忘（数据 → 遗忘 → 评估）；不自动下载数据。
- 为 Llama-3.2-1B Instruct 与 Qwen3-1.7B Base 增加模型配置；提供两套默认 TOFU 演示划分配置（划分非固定）。
- 引入可扩展的注册表元数据与兼容性检查（模型/方法/数据集），支持用户扩展与未来 API 适配器（如 GPT）。
- 统一评估产物（清单、指标、样本轨迹），并在缺失 retain 日志时允许优雅降级。
- 明确 GUI 需求（Gradio）及 UltraRAG 式向导，实现一键配置与执行；GUI 从 v2 末期启动，与 CLI 共用同一配置 schema。
- 确立遗忘/编辑/注入/多模态的方法、数据集与指标范围（如需求中列出的 TOFU/MUSE/WMDP/RWKU、ZSRE/CounterFact/ELKEN/ConceptEdit/long-text、LoRA/DoRA/REFT、LLaVA/Qwen-VL 等）。

## 能力

### 新增能力
- `unlearning-pipeline-v1`：配置驱动的遗忘流水线，带一键 Python 运行器，以 TOFU 为先，为 Llama-3.2-1B Instruct 与 Qwen3-1.7B Base 提供默认配置及灵活 TOFU 划分。
- `extensible-registry`：模型/方法/数据集的插件元数据与兼容性检查；用户可扩展，并为未来 API 适配器预留。
- `evaluation-artifacts`：标准化评估输出（清单/指标/轨迹）及缺失 retain 日志处理。
- `knowledge-editing-pipeline`：单任务/顺序/批量的编辑工作流及编辑指标（可靠性/局部性/泛化/可移植性）。
- `knowledge-injection-peft`：基于 LoRA/DoRA/AdaLoRA/LoReFT/BREP-REFT 的知识注入工作流。
- `multimodal-support`：多模态数据 schema、VLM 适配器及多模态遗忘/编辑基准。
- `gui-orchestrator`：基于 Gradio 的 GUI，含 UltraRAG 式向导、配置导入/导出、运行管理与结果可视化（从 v2 末期开始）。

### 修改的能力
- 无。

## 影响

- 在 `configs/model/`、`configs/experiment/` 下新增配置，并在 `src/` 中新增流水线封装入口（对现有训练/评估路径改动最小）。
- 后续阶段计划引入新依赖（v2.5 的 PEFT、v2 GUI 的 Gradio）；v1 无破坏性变更。
- 补充 CLI/GUI 工作流、扩展指南与可复现性说明文档。
- 在遗忘/编辑/注入/多模态之间统一评估产物与报告约定。
