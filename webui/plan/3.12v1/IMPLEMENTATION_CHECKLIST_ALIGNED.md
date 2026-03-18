# WebUI Implementation Checklist (Aligned Revision)

本文件是 `IMPLEMENTATION_CHECKLIST.md` 的对齐版执行文档。

- 原始文档保持不动。
- 本文档用于反映 2026-03-18 时点的真实代码状态、后续缺口和推荐执行顺序。
- 本文档应视为当前阶段的主执行清单。

## How To Use

- Phase 编号表示能力模块，不强制等于实际编码顺序。
- `TODO.md` 和 `task_plan.md` 仍可作为历史背景参考，但不再作为当前状态判断依据。
- 后续如果代码状态变化，应优先更新本文件，而不是继续新增平行计划文档。

## Scope

- 修稳当前 WebUI 的核心配置与运行链路
- 保持中英文切换能力
- 升级智能助手，支持用户单条/批量 JSONL 数据输入
- 将用户数据接入现有 `inject / edit / unlearn` 数据流水线
- 支持 HuggingFace 主流文本任务模型，并逐步支持用户自定义模型
- 支持 GPT / DeepSeek 作为 Agent 后端
- 将算法核心配置沉淀到 `skills` 模板中
- 让智能助手真正形成 `对话 -> 建议 -> 预览 -> 应用` 闭环
- 重做 Tab 2，并把整体视觉升级为蓝灰科研工作台风格

## Implementation Principles

- 前端继续沿用 `Gradio + 全局 CSS + elem_id/elem_classes` 路线，不重写为自定义 SPA。
- 可以借鉴 UltraRAG 的视觉层级、工作台结构和状态可见性，但不照搬其前端架构。
- 当前阶段优先级高于纯视觉美化的是：让智能助手可以真正操作配置页。
- 第一阶段只覆盖文本类任务，不提前扩展图像或多模态能力。
- 原则上先打通最小可闭环链路，再做大面积视觉 polish 和能力扩展。

## Current Snapshot (2026-03-18)

- 已完成：Phase 0 / 1 / 2 / 3 / 4 / 6
- 进行中：Phase 7 的基础设施已存在，但核心闭环未完成
- 未开始：Phase 5 / 8 / 9

### Key Facts

- 当前 Tab 4 已经具备 Skill 模板选择、配置预览、部分一键应用。
- 当前 Tab 4 已经具备单条输入、批量 JSONL、保存并回填到 Tab 1 的能力。
- 当前 Tab 4 已经具备 Agent provider 配置和快速对话能力。
- 当前真正缺失的是：将 Agent 的结构化建议解析、预览并完整应用到 Tab 1。
- 当前视觉主题仍然是 teal/green，Tab 2 仍然是基础版对比页。
- 当前模型选择仍以预置下拉为主，尚未形成统一的模型适配层。

## Architecture Split

为避免实现时混淆，固定三条能力链路：

- 任务模型链路
  负责真正被编辑、遗忘、注入或评估的目标模型。
  第一阶段范围：HuggingFace 主流文本模型。

- Assistant / Planner 链路
  负责理解用户目标、结合规则和 Agent 给出配置建议，并输出结构化建议对象。
  第一阶段范围：规则引擎 + GPT / DeepSeek。

- UI Workspace 链路
  负责把配置、对话、数据输入、运行控制和结果对比组织成清晰的工作台。
  第一阶段范围：继续基于 Gradio 改造，不引入新的前端框架。

## Reference Architecture: `text-generation-webui`

本项目后续增强可以参考：

- Repo:
  `https://github.com/oobabooga/text-generation-webui`
- 参考目的：
  不是把当前 WebUI 改造成通用 LLM playground，而是借鉴它如何把 Gradio 从“页面脚本”提升为“可扩展应用框架”。

### What It Gets Right

`text-generation-webui` 最值得借鉴的不是聊天功能本身，而是下面 5 个架构层：

- 壳层与模块层分离
  `server.py` 负责启动和装配，`modules/` 负责具体能力实现。

- 共享状态中心
  通过统一的共享对象保存参数、设置、Gradio 组件引用和运行态，而不是把状态散落在各回调里。

- 页面模块化
  各 Tab 按能力拆成独立 UI 模块，例如 chat、parameters、session、model menu，而不是把所有组件堆在一个文件里。

- Loader / Adapter 体系
  模型加载、后端切换、参数设置不是硬编码到 UI 层，而是通过 adapter / loader / metadata 抽象。

- 扩展机制
  通过 `extensions` 体系把额外能力挂到主应用，而不是每次都侵入式修改核心文件。

### Why It Matters For This Project

当前 `open-unlearning/webui` 也建立在 Gradio 之上，因此以下思路可以直接迁移：

- 把 Tab 1 / Tab 4 的联动从“手工字段更新”升级成“统一工作区状态驱动”
- 把 Agent 从“能聊天”升级成“能输出结构化动作并安全应用到配置页”
- 把后续新能力从“继续往主文件里加回调”升级成“可注册的模块或插件”
- 把主题、工作区设置、最近实验、AI 建议历史纳入统一 session / workspace 层

### What To Borrow

- `workspace_state`
  为整个 WebUI 引入统一工作区状态，覆盖：
  当前 mode、model、trainer、dataset、eval、关键参数、最近 AI 建议、当前草稿状态。

- `ui_registry`
  为跨 Tab 写入建立统一组件注册和更新协议，避免继续扩散手写 `gr.update(...)` 拼装逻辑。

- `assistant action contract`
  让 Agent 输出结构化动作，而不是只输出自然语言解释。
  推荐动作包括：
  `set_mode`
  `set_model`
  `set_trainer`
  `set_dataset`
  `set_eval`
  `set_param`
  `apply_skill`
  `preview_diff`

- `session/workspace panel`
  增加统一的会话与工作区层，用于承载：
  theme、Agent provider、autosave、最近实验、最近建议、当前草稿。

- `lightweight plugins`
  引入轻量插件位，而不是完整复制 `extensions` 系统。
  每个插件至少可声明：
  `id`
  `label`
  `register_ui()`
  `register_actions()`
  `register_settings()`

- `static assets as first-class layer`
  不再只把样式理解成一个 `custom.css`，而是把主题变量、工作台布局、面板组件样式、AI 建议卡片样式当成独立设计层维护。

### What Not To Borrow

以下内容不应成为当前阶段重点：

- 通用本地 LLM playground 的大而全能力
- 角色卡 / 多人格聊天系统
- 图像生成 tab
- 过重的命令行 flags 体系
- 为通用文本生成准备的大量采样参数控制

原因：
当前项目的产品目标不是“通用聊天和推理工作台”，而是“AI 驱动的知识编辑 / 遗忘 / 注入实验控制台”。

### Direct Impact On Current Phases

- 对 Phase 5 的影响
  `text-generation-webui` 的 `loaders + models + model settings` 思路说明：
  模型选择不应停留在普通 Dropdown，而应该逐步演进成 `model_adapter`。

- 对 Phase 7 的影响
  最值得借鉴的是“统一状态 + 模块化会话 + 结构化动作”。
  这正对应当前最重要的目标：
  `对话 -> 建议 -> diff -> 应用 -> 继续追问`

- 对 Phase 8 的影响
  它说明 Gradio 也可以做出更像“产品”的工作台体验。
  因此本项目可以继续坚持 Gradio 路线，但要把工作区层、session 层和静态资源层补齐。

### New Cross-Cutting Tasks

这些任务横跨 Phase 5 / 7 / 8，建议作为独立增强轨道处理：

- [ ] 新增 `utils/workspace_state.py`
  统一保存当前工作区状态和最近 AI 建议。

- [ ] 新增 `utils/ui_registry.py`
  统一登记可被跨 Tab 更新的关键组件。

- [ ] 新增 `utils/assistant_actions.py`
  定义 Agent 输出动作协议和应用逻辑。

- [ ] 新增 `utils/session_store.py`
  管理最近实验、最近建议、当前草稿和 autosave。

- [ ] 评估是否引入 `plugins/` 或 `extensions/` 目录
  用于后续扩展 recommendation、export、charts、provider adapter。

### Priority Decision

结合本项目当前目标，最推荐先借鉴 `text-generation-webui` 的不是“通用聊天”，而是这三项：

1. 统一状态中心
2. Assistant 结构化动作协议
3. Session / Workspace 层

这三项一旦落地，当前 WebUI 就会从“配置页 + 向导”升级为“可对话、可审阅、可应用、可回溯”的 AI 实验工作台。

## Phase 0: Stabilize Existing UI [Done]

### Goal

先修复已知断点，避免在不稳定状态上继续加功能。

### Current State

- `mode -> trainer/experiment/dataset/eval` 联动顺序已修复
- 导入/导出配置已可逆
- 训练模式下的 `eval_suite` 已参与命令预览和执行
- 已打通 `开始运行 -> 输出目录 -> 结果摘要`
- 切换 trainer 时会同步关键默认参数

### Deliverables In Repo

- `utils/ui_state.py`
- 稳定的配置状态同步逻辑
- 可复现的导入/导出配置格式
- 结果摘要回填链路

### Acceptance Snapshot

- 模式切换不再出现 dropdown value warning
- 导出配置重新导入后可还原主要 UI 状态
- 训练页修改评测套件后命令预览会变化
- 运行结束后结果区可以加载 summary

## Phase 1: Shared Foundations [Done]

### Goal

为多语言、智能助手、模型适配和 Agent 接入建立共享基础设施。

### Current State

- 已建立 `i18n` 字典层
- 已抽离统一 UI state schema
- 已有统一应用配置到界面的 helper
- 已有新版 skill schema
- 已明确任务模型适配和 Agent provider 适配的公共接口方向

### Deliverables In Repo

- `utils/i18n.py`
- `utils/ui_state.py`
- `utils/skill_schema.py`

## Phase 2: Bilingual UI [Done]

### Goal

支持 `[中文] | [English]` 全局切换，仅处理界面文案。

### Current State

- 顶栏语言切换已存在
- 4 个 Tab、按钮、状态文案、表头、提示语已接入 i18n
- 不刷新页面即可完成主要界面文案切换

## Phase 3: Skill Knowledge Base Upgrade [Done]

### Goal

把算法核心配置沉淀到 `skills` 模板，作为规则推荐和 Agent 推荐的共同知识源。

### Current State

- 已设计并落地 skill schema
- `unlearn / inject / edit` 模板已补充结构化字段
- 已增加 schema 校验逻辑

### Deliverables In Repo

- `skills/*.json`
- `utils/skill_schema.py`

## Phase 4: JSONL Data Adaptation [Done]

### Goal

支持用户输入单条数据和批量 `JSONL`，并将其串联到现有数据流水线。

### Current State

- 已支持单条输入和批量 JSONL 双入口
- 已支持字段校验、样例填充、预览展示
- 已支持保存 JSONL 并回填到 Tab 1 对应数据集字段
- 已建立统一数据适配层

### Deliverables In Repo

- `components/skill_wizard.py`
- `utils/data_adapter.py`

### Acceptance Snapshot

- 用户上传 JSONL 后可以看到字段解析结果
- 用户数据可映射为 inject / edit / unlearn 配置
- 生成结果可应用到 Tab 1 数据集字段

## Phase 5: Task Model Adaptation [Not Started]

### Goal

支持更多 HuggingFace 主流文本模型，并允许用户输入自己的模型名或路径。

### Current Baseline

- 当前仅有预置模型下拉
- 当前没有独立的模型适配层
- 当前没有模型元信息表
- 当前没有自定义模型校验和降级策略

### Plan Split

#### Phase 5A: Minimum Usable Model Adaptation

- [ ] 允许用户输入自定义 HF 模型名或本地路径
- [ ] 将“预置模型”和“自定义模型”区分显示
- [ ] 增加轻量校验逻辑，至少能判断字段是否为空、路径是否存在、名称是否合法
- [ ] 为后续 Assistant 应用配置预留统一写入口

#### Phase 5B: Full Model Adapter

- [ ] 新增 `utils/model_adapter.py`
- [ ] 维护模型元信息：tokenizer、dtype、chat template、任务支持范围
- [ ] 增加 capability 判断：是否支持 `inject / edit / unlearn / eval`
- [ ] 增加默认值和降级策略

### Deliverables

- 统一的任务模型适配接口
- 预置模型和自定义模型的统一表示
- 可扩展的模型元信息表

### Suggested Files

- `components/config_panel.py`
- `utils/config_loader.py`
- 新增 `utils/model_adapter.py`

### Acceptance

- 用户可选择预置主流模型
- 用户可输入自定义 HF 模型名或本地路径
- 系统可判断该模型是否支持对应任务

### Priority Note

- Phase 5 不是当前最强阻塞项。
- 若目标是优先实现“和编辑后的 AI 对话并操作配置”，可以先做 Phase 7A，再补 Phase 5A。

## Phase 6: Agent Provider Layer [Done]

### Goal

让智能助手可以切换不同大模型 API 作为后端。

### Current State

- 已抽象统一 Agent provider 接口
- 已接入 OpenAI GPT
- 已接入 DeepSeek
- 已统一 provider 配置项
- 已增加错误处理和超时兜底
- 已支持测试连接、保存配置和快速对话

### Implementation Notes

- 当前 provider 实现集中在 `utils/agent_provider.py`
- 当前配置持久化位于 `utils/agent_settings.py`
- 当前 UI 入口位于 `components/skill_wizard.py`

### Minimum Provider Config

- `provider`
- `base_url`
- `api_key`
- `model`
- `timeout`
- `temperature`
- `max_tokens`

### Deliverables In Repo

- 统一 provider 抽象层
- 至少两个可切换 provider 实现
- 基础 Agent 配置 UI
- 快速对话能力

### Acceptance Snapshot

- 可以在配置中切换 GPT / DeepSeek
- 上层智能助手不需要关心具体 provider 差异

### Important Boundary

- 当前 Phase 6 只解决“接上模型并能对话”。
- 它还没有解决“把对话结果安全、结构化地应用到配置页”。

## Phase 7: Intelligent Assistant Closed Loop [In Progress]

### Goal

把当前“模板向导 + 数据输入 + 快速对话”升级成“规则推荐 + Agent 推荐 + 结构化预览 + 一键应用配置”的智能配置页。

### Current Baseline

- 已有 Skill 模板选择和配置预览
- 已有部分一键应用到 Tab 1
- 已有单条输入和批量 JSONL 输入
- 已有 Agent provider 配置和快速对话

### What Is Still Missing

- 缺少用户目标输入区和上下文摘要区
- 缺少规则推荐引擎
- 缺少 Agent 输出的统一解析和校验
- 缺少标准化建议卡片
- 缺少变更 diff 预览
- 缺少对 Tab 1 更完整的一键应用能力
- 缺少“应用后继续追问再微调”的工作流闭环

### Plan Split

#### Phase 7A: Minimum Closed Loop

- [ ] 增加用户目标输入区
- [ ] 新增 `utils/recommendation_engine.py`，先做规则推荐最小版
- [ ] 统一 Assistant 输出合同，产出标准化建议对象
- [ ] 新增 Agent 输出解析与校验逻辑
- [ ] 渲染建议卡片，至少展示 `mode / skill / model / data / eval / core_overrides / reasoning / risk`
- [ ] 增加配置 diff 预览
- [ ] 扩展“一键应用到 Tab 1”，覆盖：
  - `mode`
  - `model`
  - `trainer`
  - `experiment`
  - dataset 相关字段
  - `eval_suite`
  - 核心参数字段
- [ ] 应用后允许继续追问并再次生成建议

#### Phase 7B: Experience Upgrade

- [ ] 重做智能助手页面结构
- [ ] 统一规则模式和 Agent 模式的入口
- [ ] 增加推荐结果的分组视图和风险提醒
- [ ] 增加“仅生成建议 / 生成并预览 / 直接应用”三种操作层级
- [ ] 为后续运行控制预留接口，但当前阶段不直接触发训练

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

可选扩展字段：

- `ui_actions`
- `dataset_overrides`
- `parameter_overrides`
- `warnings`

### Deliverables

- 新版智能助手页面
- 规则推荐引擎
- Agent 需求理解与配置建议引擎
- 建议卡片和 diff 预览
- 从对话到 Tab 1 的一键应用闭环

### Suggested Files

- `components/skill_wizard.py`
- 新增 `utils/recommendation_engine.py`
- 新增 `utils/agent_planner.py`
- 可选新增 `utils/recommendation_contract.py`

### Acceptance

- 用户输入目标和数据后，可以得到结构化配置建议
- 建议可在 UI 中被预览，而不是直接黑盒写入
- 建议可一键应用到 Tab 1
- 应用后用户可以继续通过对话进行修正
- 规则模式和 Agent 模式都能工作

### Immediate Target

- 当前阶段最重要的是完成 Phase 7A。
- 这是“最后可以和编辑后的 AI 进行对话和操作”的真正落地点。

## Phase 8: Tab 2 Redesign and Global Visual Refresh [Not Started]

### Goal

重做结果对比页，并将整体主题从绿色切换为蓝灰科研工作台风格。

### Current Baseline

- 当前主题仍为 teal/green
- 当前 Tab 2 仍是基础选择器 + HTML 输出结构
- 当前 Tab 1 / Tab 4 的工作台层级还不够强

### Tasks

- [ ] 重构全局色板和主题变量
- [ ] 替换 teal/green 主视觉
- [ ] 强化 Tab 1 / Tab 4 的工作台层级
- [ ] 重做 Tab 2 布局
- [ ] 增加筛选区
- [ ] 增加摘要图表区
- [ ] 增加 sticky header、指标说明、最佳值高亮方向控制
- [ ] 整理日志区、状态区、卡片标题和按钮层级

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
- 更清晰的工作台视觉层级
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
- Tab 1 / Tab 4 更接近统一工作台体验

### Priority Note

- Phase 8 很重要，但应排在 Phase 7A 之后。
- 先把智能助手闭环打通，再做大面积视觉升级，收益更高。

## Phase 9: Validation and Documentation [Not Started]

### Goal

在 Phase 7A 和 Phase 8 稳定后，为关键行为补充验证和文档，避免继续扩展时回归。

### Tasks

- [ ] 为 i18n、skill schema、data adapter、recommendation engine 增加单测
- [ ] 为 result parser 和 metric direction 增加单测
- [ ] 为关键 UI 联动增加 smoke test
- [ ] 更新 README，明确已支持和待支持功能
- [ ] 补充 Agent provider 配置说明
- [ ] 补充 Assistant 输出合同和应用行为说明

### Deliverables

- 基础测试集
- 更新后的 README / 使用说明
- Assistant 配置与能力边界文档

### Acceptance

- 关键模块可被最小测试覆盖
- 关键 UI 链路至少有 smoke test
- 文档能准确反映当前能力边界

## Recommended Execution Order

建议按下面顺序推进，而不是机械按 Phase 编号顺序推进：

1. 以 Phase 0 / 1 / 2 / 3 / 4 / 6 作为当前稳定基线
2. 先完成 Phase 7A，打通智能助手最小闭环
3. 再做 Phase 8，完成工作台视觉升级和 Tab 2 重构
4. 然后补 Phase 5A，让模型输入更灵活
5. 再继续完善 Phase 7B 和 Phase 5B
6. 最后做 Phase 9 的验证和文档收口

## Milestone Summary

- Milestone A [Done]
  稳定现有 UI 核心链路并建立共享基础设施

- Milestone B [Done]
  完成双语 UI、skill 知识库和 JSONL 数据适配

- Milestone C [In Progress]
  完成 Agent provider，并打通智能助手的结构化建议和一键应用闭环

- Milestone D [Planned]
  完成工作台视觉升级和 Tab 2 重构

- Milestone E [Planned]
  完成模型适配硬化、测试补齐和文档收口
