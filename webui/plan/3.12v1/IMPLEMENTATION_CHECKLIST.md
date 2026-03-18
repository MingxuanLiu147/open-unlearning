# WebUI Implementation Checklist

本清单用于指导 `webui/` 后续实现。目标不是一次性堆完全部功能，而是按依赖关系逐步落地，保证每个阶段都有可验证产出。

## Scope

- 修稳当前 WebUI 的核心配置与运行链路
- 新增中英文界面切换
- 升级智能助手，支持用户单条/批量 JSONL 数据输入
- 将用户数据接入现有 `inject / edit / unlearn` 数据流水线
- 支持 HuggingFace 主流文本任务模型
- 支持 GPT / DeepSeek 作为 Agent 后端
- 将算法核心配置沉淀到 `skills` 模板中
- 重做 Tab 2 和整体蓝灰配色

## Architecture Split

为了避免实现时混淆，先固定两条模型链路：

- 任务模型链路：
  负责真正被编辑、遗忘、注入或评估的目标模型
  第一阶段范围：HuggingFace 主流文本模型

- Agent 后端链路：
  负责理解用户需求、推荐算法和生成配置建议
  第一阶段范围：GPT / DeepSeek API

## Phase 0: Stabilize Existing UI ✅

### Goal

先修复当前已知断点，避免在不稳定状态上继续加功能。

### Tasks

- [x] 修复 `mode -> trainer/experiment/dataset/eval` 的联动顺序
- [x] 修复导入/导出配置不可逆问题
- [x] 让训练模式下的 `eval_suite` 真正参与命令预览和执行
- [x] 打通 `开始运行 -> 输出目录 -> 结果摘要`
- [x] 切换 trainer 时同步关键默认参数

### Implementation Notes (2026-03-12)

- 新增 `utils/ui_state.py`：统一的 UI 状态管理工具
- `apply_mode_config`: 解决 mode 切换时的联动顺序问题
- `build_config_dict` / `apply_config_dict`: 完整的配置导入导出
- 训练模式命令现在包含 `eval=...` 参数
- 运行结束后自动回填 output_dir 并加载结果摘要
- trainer 切换时同步更新所有训练参数

### Deliverables

- 稳定的配置页状态同步逻辑
- 可复现的导入/导出配置格式
- 训练结果区可正常展示输出目录和摘要

### Acceptance

- `inject/edit` 模式切换时不再出现 dropdown value warning
- 导出后的配置重新导入可还原主要 UI 状态
- 训练页修改评测套件后，命令预览会变化
- 运行结束后，结果区可以加载 summary

## Phase 1: Shared Foundations ✅

### Goal

为后续多语言、智能助手、模型适配和 Agent 接入建立共享基础设施。

### Tasks

- [x] 新增 `i18n` 文案字典层
- [x] 抽离统一 UI state schema
- [x] 抽离统一应用配置到界面的 helper
- [x] 梳理 `skills` 的新 schema 设计
- [x] 梳理任务模型适配和 Agent provider 适配的公共接口

### Deliverables

- `i18n` 字典结构
- 统一 UI schema
- 新版 skill schema 草案
- provider / adapter 接口草案

### Suggested Files

- `app.py`
- `components/config_panel.py`
- `components/skill_wizard.py`
- 新增 `utils/i18n.py`
- 新增 `utils/ui_state.py`
- 新增 `utils/skill_schema.py`

### Acceptance

- 不同入口写 UI 状态时复用同一套 helper
- 后续功能不再需要在多个回调里手工重复写字段

## Phase 2: Bilingual UI ✅

### Goal

增加 `[中文] | [English]` 全局切换，只处理界面文案，不扩展到示例数据文本。

### Tasks

- [x] 在顶栏增加语言切换按钮
- [x] 为 4 个 Tab、按钮、状态文案、表头、提示语接入 i18n
- [x] 统一运行状态、错误提示、空态提示的中英文切换
- [x] 为新增文案规定 key 命名规则

### Deliverables

- 可切换的全局语言状态
- 全面接入的核心 UI 文案

### Suggested Files

- `app.py`
- `components/*.py`
- `assets/custom.css`
- `utils/i18n.py`

### Acceptance

- 点击语言按钮后，Tab 名称、按钮文本、提示信息同步切换
- 不刷新页面也能看到界面文案切换

## Phase 3: Skill Knowledge Base Upgrade ✅

### Goal

把算法核心配置沉淀到 `skills` 模板，作为规则推荐和 Agent 推荐的共同知识源。

### Tasks

- [x] 设计新版 skill schema
- [x] 为现有 `unlearn / inject / edit` skill 模板补充结构化字段
- [x] 增加技能适用场景、支持的数据格式、推荐模型范围、推荐 eval、关键 overrides、风险提示
- [x] 增加 schema 校验逻辑

### Proposed Skill Fields

- `goal`
- `supported_data_modes`
- `supported_input_formats`
- `recommended_models`
- `recommended_eval`
- `resource_estimate`
- `core_overrides`
- `constraints`
- `risk_notes`
- `prompt_hints`

### Deliverables

- 可供规则引擎和 Agent 共用的 skill 库
- skill schema 校验工具

### Suggested Files

- `skills/*.json`
- `components/skill_wizard.py`
- 新增 `utils/skill_schema.py`
- 新增 `utils/skill_loader.py`

### Acceptance

- 任意一个 skill 模板都可被程序化解析
- 规则引擎和 Agent 都能从同一 skill 数据结构读取推荐信息

## Phase 4: JSONL Data Adaptation ✅

### Goal

支持用户输入单条数据和批量 `JSONL`，并将其串联到现有数据流水线。

### Tasks

- [x] 为智能助手增加"单条输入"和"批量 JSONL"双入口
- [x] 实现 `JSONL -> task-specific dataset descriptor` 适配层
- [x] 为 `inject` 提供现有字段映射模板
- [x] 为 `edit` 增加 `jsonl` 本地加载与映射
- [x] 为 `unlearn` 增加自定义 QA / forget-retain 适配逻辑
- [x] 自动生成或选择对应 Hydra dataset config（通过 save_and_apply 回填路径）
- [x] 增加样例文件、字段预览、错误提示

### Implementation Notes (2026-03-12)

- 新增 `utils/data_adapter.py`：JSONL 解析、字段验证、预览 HTML、文件保存、Hydra override 生成
- 修改 `components/skill_wizard.py`：在 Skill 选择区下方新增 `📂 自定义数据输入` Accordion
  - 单条输入：根据 goal 模式（unlearn/inject/edit）动态切换字段表单
  - 批量 JSONL：文本粘贴 + 文件上传（自动同步到文本框）+ 填入示例按钮
  - 解析预览：HTML 表格，必填字段用 ★ 标注，支持截断显示
  - 保存并应用：保存到 `webui/uploads/` 目录，自动回填 Tab 1 对应数据集字段
- 修改 `utils/i18n.py`：补充 30+ 个 Phase 4 数据上传相关中英文 key

### Data Modes

- `inject`
  推荐字段：`instruction / input / output`

- `edit`
  推荐字段：`prompt / subject / target_new / target_old`

- `unlearn`
  推荐字段：
  `question / answer / split`
  或者显式区分 `forget` 和 `retain`

### Deliverables

- 用户 JSONL 的可视化预览
- 不同任务的字段映射模板
- 统一数据适配层

### Suggested Files

- `components/skill_wizard.py`
- 新增 `utils/data_adapter.py`
- 新增 `utils/dataset_config_builder.py`
- `src/data/editing.py`
- `configs/data/datasets/*.yaml`

### Acceptance

- 用户上传 JSONL 后可以看到字段解析结果
- 用户数据可以被正确映射为 inject / edit / unlearn 配置
- 生成的配置可直接应用到 Tab 1

## Phase 5: Task Model Adaptation

### Goal

支持更多 HuggingFace 主流文本模型，并允许用户输入自己的模型名或路径。

### Tasks

- [ ] 新增模型适配层
- [ ] 区分预置主流模型与用户自定义模型
- [ ] 维护模型元信息：tokenizer、dtype、chat template、任务支持范围
- [ ] 增加自定义模型校验和降级策略

### Deliverables

- 统一的任务模型适配接口
- 可扩展的模型元信息表

### Suggested Files

- `components/config_panel.py`
- `utils/config_loader.py`
- 新增 `utils/model_adapter.py`

### Acceptance

- 用户可选择预置主流模型
- 用户可输入自定义 HF 模型名或本地路径
- 系统可判断该模型是否支持对应任务

## Phase 6: Agent Provider Layer

### Goal

让智能助手可以切换不同大模型 API 作为后端。

### Tasks

- [x] 抽象统一 Agent provider 接口
- [x] 第一阶段接入 OpenAI GPT
- [x] 第一阶段接入 DeepSeek
- [x] 统一 provider 配置项
- [x] 增加 provider 错误处理和超时兜底

### Implementation Notes (2026-03-12)

- `utils/agent_provider.py` 已包含完整后端（OpenAIProvider / DeepSeekProvider / AgentSession）
- 新增 `utils/agent_settings.py`：持久化保存/加载 API Key 等配置到 `.agent_settings.json`
- 修改 `components/skill_wizard.py`：新增 🤖 Agent 配置 Accordion
  - Provider 选择（openai / deepseek），切换时自动填入默认 base_url 和 model
  - API Key 输入（password 类型），本地持久化，不会上传
  - Base URL / Model / Temperature / Max Tokens 配置项
  - 「🔗 测试连接」：发送 1-token ping 验证可达性，流式返回状态
  - 「💾 保存配置」：写入 webui/.agent_settings.json
  - 「💬 快速对话」：Chatbot 组件，支持流式输出，回车发送，多轮历史
- 修改 `utils/i18n.py`：补充 20+ 个 Phase 6 Agent 配置中英文 key

### Minimum Provider Config

- `provider`
- `base_url`
- `api_key`
- `model`
- `timeout`
- `temperature`
- `max_tokens`

### Deliverables

- 统一 provider 抽象层
- 至少两个可切换 provider 实现

### Suggested Files

- 新增 `utils/agent_provider.py`
- 新增 `utils/agent_clients/openai_provider.py`
- 新增 `utils/agent_clients/deepseek_provider.py`
- 新增 `utils/agent_settings.py`

### Acceptance

- 可以在配置中切换 GPT / DeepSeek
- 上层智能助手不需要关心具体 provider 差异

## Phase 7: Intelligent Assistant Redesign

### Goal

把当前"模板向导"升级成"规则推荐 + Agent 推荐 + 一键应用配置"的智能配置页。

### Tasks

- [ ] 重做智能助手页面结构
- [ ] 增加用户目标输入区
- [ ] 增加单条/批量数据输入区
- [ ] 增加规则推荐模块
- [ ] 增加 Agent 推荐模块
- [ ] 输出标准化建议卡片
- [ ] 支持一键应用到 Tab 1

### Output Contract

智能助手输出至少包含：

- `mode`
- `recommended_skill`
- `recommended_method`
- `recommended_model`
- `data_plan`
- `eval_plan`
- `core_overrides`
- `reasoning_summary`
- `risk_notes`

### Deliverables

- 新版智能助手页面
- 规则推荐引擎
- Agent 需求理解与配置建议引擎

### Suggested Files

- `components/skill_wizard.py`
- 新增 `utils/recommendation_engine.py`
- 新增 `utils/agent_planner.py`

### Acceptance

- 用户输入目标和数据后，可以得到结构化配置建议
- 建议可一键应用到 Tab 1
- 规则模式和 Agent 模式都能工作

## Phase 8: Tab 2 Redesign and Visual Refresh

### Goal

重做结果对比页，并将整体主题从绿色切换为蓝灰科研风格。

### Tasks

- [ ] 重构全局色板和主题变量
- [ ] 替换 teal/green 主视觉
- [ ] 重做 Tab 2 的布局
- [ ] 增加筛选区
- [ ] 增加摘要图表区
- [ ] 增加 sticky header、指标说明、最佳值高亮方向控制

### Recommended Palette

- `Primary`: `#355CFF`
- `Primary Dark`: `#1D3FDB`
- `Background`: `#F5F7FB`
- `Surface`: `#FFFFFF`
- `Surface Alt`: `#EEF3FB`
- `Text`: `#0F172A`
- `Subtle Text`: `#475569`
- `Border`: `#D7E0F0`
- `Accent`: `#7CB7FF`
- `Warning`: `#F59E0B`

### Deliverables

- 新版全局主题
- 新版 Tab 2 页面

### Suggested Files

- `assets/custom.css`
- `app.py`
- `components/results_compare.py`
- `utils/result_parser.py`

### Acceptance

- 全站不再以绿色作为主色
- Tab 2 具备筛选、对比、图表摘要三部分
- 指标高亮遵循正确的 metric direction

## Phase 9: Validation and Documentation

### Goal

在继续新增功能前，为关键行为补充验证和文档。

### Tasks

- [ ] 为 i18n、skill schema、data adapter、recommendation engine 增加单测
- [ ] 为关键 UI 联动增加 smoke test
- [ ] 更新 README，标记已支持和待支持功能
- [ ] 补充 Agent provider 配置说明

### Deliverables

- 基础测试集
- 更新后的 README / 使用说明

### Acceptance

- 关键模块可被最小测试覆盖
- 文档能准确反映当前能力边界

## Recommended Execution Order

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9

## Milestone Summary

- Milestone A:
  完成稳定性修复和共享基础设施

- Milestone B:
  完成双语 UI、skill 知识库和 JSONL 数据适配

- Milestone C:
  完成模型适配、Agent provider、智能助手重构

- Milestone D:
  完成 Tab 2 和全局视觉升级
