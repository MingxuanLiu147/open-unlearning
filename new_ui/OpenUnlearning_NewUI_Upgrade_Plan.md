# `open-unlearning/new_ui` 升级方案

## 1. 方案定位

这是一版面向实现的升级方案，目标不是复制 UltraRAG UI，而是借用它的高价值能力，升级你当前的 `new_ui`：

- 保留你现有的现代前端栈：Vue 3 + TypeScript + Pinia + Element Plus
- 借鉴 UltraRAG 的工作台组织方式和可执行 Copilot 交互
- 让 `new_ui` 从“配置页 + 运行页 + 结果页 + 简单聊天侧边栏”升级成“具备上下文感知和可执行建议能力的实验工作台”

这份方案默认读者是你自己以及后续会参与 `new_ui` 实现的人。

---

## 2. 现状总结

基于当前代码，`new_ui` 已经具备以下基础：

### 前端基础

- 顶层布局已经支持独立 Copilot 侧边栏
- 实验配置状态已经集中在 `experimentStore`
- 运行状态已经集中在 `runnerStore`
- 结果页和技能页已经独立成路由视图
- Copilot 已支持流式输出

### 后端基础

- 已有独立 agent API：`/api/agent/chat`
- 已有独立运行 API：`/api/run/*`
- 已有默认 system prompt 和 OpenAI-compatible 流式适配

### 当前主要缺口

1. Copilot 只能输出文本，不能落地配置。
2. 上下文传递太浅，无法真正理解当前工作区状态。
3. 没有动作层，没有 Apply/Reject 闭环。
4. Copilot 和 Monitor / Results 之间没有能力联动。
5. 运行日志流和 Copilot 流式协议没有统一抽象。

---

## 3. 目标设计

本次升级目标分三层：

### 第一层：Workspace Copilot

面向配置和实验编排。

它应当能：

- 推荐实验模式、模型、trainer、数据集和 eval
- 解释参数和默认值
- 直接生成结构化配置建议
- 让用户一键 Apply 到 `experimentStore`

### 第二层：Runtime Assistant

面向运行态和结果分析。

它应当能：

- 解读当前日志
- 解释失败原因
- 总结一次 run 的关键结果
- 对比两次实验结果
- 推荐下一步实验

### 第三层：Workflow Layer

面向更复杂的实验编排和 skill chaining。

这一层不要求第一阶段就做，但方案需要给它预留空间。

---

## 4. 核心设计原则

### 4.1 保留现代前端实现

不回退到 UltraRAG 那种：

- 巨型 `main.js`
- 大量全局变量
- DOM ID 驱动的状态切换

继续坚持：

- Vue 组件化
- TypeScript 类型约束
- Pinia 状态管理
- API 层与 UI 层分离

### 4.2 让 Copilot 成为“可执行助手”

Copilot 不能停留在“会说建议”，必须变成：

```text
上下文快照
-> LLM 输出结构化建议
-> 后端验证
-> 前端渲染 action cards
-> 用户 Apply / Reject
-> 写入 experimentStore / 其他 store
```

### 4.3 智能能力分层

不要把所有智能能力塞到一个聊天框里。

建议明确划分：

- `Workspace Copilot`
- `Runtime Assistant`
- `Workflow Layer`

### 4.4 流式协议统一

将来所有流式能力尽量统一成事件协议，而不是每条链路各玩一套。

---

## 5. 产品结构改造建议

## 5.1 保持现有导航结构

当前已有：

- `Workshop`
- `Monitor`
- `Results`
- `Skills`

这套结构可以保留，不需要重做。

### 新的语义定位

- `Workshop`: 配置和实验准备中心
- `Monitor`: 运行态中心
- `Results`: 结果分析中心
- `Skills`: 预设方案和工作流模板中心
- `Copilot`: 全局智能入口，但会根据所在页面切换上下文和能力

---

## 5.2 Copilot 从通用聊天升级为上下文感知助手

当前 Copilot 放在全局侧边栏是对的，这点不需要改。

但它需要根据当前路由和 store 状态自动切换 focus：

### 在 `Workshop`

Copilot 重点做：

- 配置推荐
- 参数修改
- Skill 推荐
- 实验模板建议

### 在 `Monitor`

Copilot 重点做：

- 解读日志
- 解释失败
- 总结训练过程
- 提示中断 / 重试 / 参数调整建议

### 在 `Results`

Copilot 重点做：

- 解释对比结果
- 总结关键指标变化
- 推荐下一步实验

### 在 `Skills`

Copilot 重点做：

- 推荐 skill 组合
- 帮助挑选 baseline / ablation / multi-method compare 模板

---

## 6. 前端架构方案

## 6.1 新的前端能力边界

### 保留现有 store

- `experimentStore`
- `runnerStore`
- `resultsStore`
- `skillsStore`
- `appStore`

### 升级 `agentStore`

当前 `agentStore` 过于轻量，只适合最简单的流式聊天。

建议升级后的状态结构：

```ts
type AgentAction =
  | {
      id: string
      type: 'apply_config'
      payload: {
        mode?: string
        model?: string
        trainer?: string
        datasets?: Record<string, string>
        eval?: string
        params?: Record<string, any>
      }
      preview: string
      status: 'pending' | 'applied' | 'rejected'
    }
  | {
      id: string
      type: 'navigate_view'
      payload: { view: 'workshop' | 'monitor' | 'results' | 'skills' }
      preview: string
      status: 'pending' | 'applied' | 'rejected'
    }

type AgentSession = {
  id: string
  title: string
  messages: ChatMessage[]
  actions: AgentAction[]
  updatedAt: number
}
```

推荐新增的 store 字段：

- `sessions`
- `currentSessionId`
- `streaming`
- `controller`
- `lastContextSnapshot`
- `pendingActions`
- `currentChunk`

---

## 6.2 新增统一上下文快照构造

建议把当前组件内联构造 context 的逻辑抽离出来。

### 新增函数

建议位置：

- `frontend/src/stores/agent.ts`
- 或 `frontend/src/services/contextSnapshot.ts`

### 推荐快照结构

```ts
type AgentContextSnapshot = {
  route: 'workshop' | 'monitor' | 'results' | 'skills'
  mode: string
  experiment: {
    model: string
    trainer: string
    datasets: Record<string, string>
    eval: string
    experiment: string
    params: Record<string, any>
    overrides: Record<string, any>
  }
  runner: {
    running: boolean
    exitCode: number | null
    command: string
    recentLogs: string[]
  }
  results: {
    selectedRuns: string[]
    summary?: Record<string, any>
  }
  skills: {
    available: string[]
  }
}
```

### 设计收益

1. Copilot 真正知道“你现在在什么页面做什么”。
2. 不再由 `CopilotPanel.vue` 自己拼装 context。
3. 后续可以把同一个快照用于：
   - Copilot
   - Debug assistant
   - Auto suggestion

---

## 6.3 CopilotPanel UI 改造

建议保留当前侧边栏位置，但升级内容结构。

### 新布局建议

1. 顶部状态栏
   - 当前 focus：`Workshop / Monitor / Results / Skills`
   - 当前实验摘要
   - 清空 / 新会话按钮

2. 中部消息区
   - 消息列表
   - 流式内容
   - Action cards

3. 底部输入区
   - 输入框
   - 发送按钮
   - 可选建议 chips

### 新增组件建议

- `frontend/src/components/copilot/ActionCard.vue`
- `frontend/src/components/copilot/ContextBanner.vue`
- `frontend/src/components/copilot/SessionList.vue`

### ActionCard 交互

每个 action card 至少支持：

- 预览
- Apply
- Reject

`apply_config` 的 Apply 应直接调用：

- `experimentStore.applyConfig(action.payload)`

这是第一阶段最关键的闭环。

---

## 6.4 Monitor / Results 页面与 Copilot 联动

### Monitor

当前 `runnerStore` 只有：

- `running`
- `exitCode`
- `logs`
- `command`

建议保留，但让 Copilot 可读取最近日志切片：

- `recentLogs = logs.slice(-200)`

Copilot 在 `Monitor` 页面时，system prompt 重点引导：

- 解释日志
- 识别错误模式
- 给出下一步操作建议

### Results

当前 `Results.vue` 已支持 run 选择和 compare。

建议让 Copilot 在该页面可读取：

- 当前选中的 labels
- compareData

这样它可以直接回答：

- 哪个方法更好
- 哪个指标最关键
- 为什么结果有 trade-off

---

## 7. 后端方案

## 7.1 升级 `/api/agent/chat`

当前：

- 只返回流式文本
- 由前端纯拼接显示

建议升级为“流式文本 + 最终动作”。

### 推荐返回事件

```text
data: {"type":"message_delta","content":"..."}

data: {"type":"actions","actions":[...]}

data: {"type":"done"}

data: {"type":"error","message":"..."}
```

### 原因

这样做比当前只有：

- `content`
- `done`

更稳定，也更适合后续扩展。

---

## 7.2 动作协议从自由文本升级为 JSON schema

当前 `DEFAULT_SYSTEM_PROMPT` 已经在要求 JSON，这很好。

建议下一步正式化：

### 推荐输出结构

```json
{
  "message": "我建议你切换到 edit 模式并使用某个 trainer。",
  "actions": [
    {
      "type": "apply_config",
      "preview": "切换到 edit 模式，选择 trainer=rome，更新 forget/retain 数据集和学习率。",
      "payload": {
        "mode": "edit",
        "trainer": "rome",
        "datasets": {
          "forget": "xxx",
          "retain": "yyy"
        },
        "params": {
          "learning_rate": 1e-5
        }
      }
    }
  ]
}
```

### 后端职责

在 `backend/services/agent_provider.py` 中：

1. 要求模型只输出 JSON。
2. 尝试解析 JSON。
3. 解析失败时走 fallback：
   - 文本作为 `message`
   - `actions=[]`

### 推荐新增函数

- `parse_agent_response(raw_text) -> { message, actions }`
- `validate_actions(actions) -> actions`

---

## 7.3 system prompt 分层

当前只有一个 `DEFAULT_SYSTEM_PROMPT`。

建议改成按场景构造：

- `workspace_copilot_prompt(context)`
- `runtime_assistant_prompt(context)`

### 路由方式

先不必新增后端路由，可以仍然复用 `/api/agent/chat`。

只需要后端根据 `context.route` 或 `context.focus` 切换 prompt 模板即可。

---

## 7.4 预留 Runtime Assistant 的结果入口

为了后续扩展，建议后端 agent 层能接受：

- `context.runner.recentLogs`
- `context.results.summary`

第一阶段不需要新增数据库或复杂服务，只需要：

- 从现有 store 派生摘要
- 前端带过去

---

## 8. 事件与协议统一方案

你当前有两类流：

1. `runnerApi.connectLog()` 用 `EventSource`
2. `agentApi.streamChat()` 用 `fetch + reader`

建议方向：

- 新功能统一优先使用 `POST + fetch + reader`
- 旧的日志 SSE 可以后续再迁移

### 为什么

因为 Copilot 和后续 Runtime Assistant 都需要复杂 body：

- 当前 context
- 当前页面
- 当前实验配置
- 近期日志
- 当前结果摘要

`EventSource` 不适合这一点。

### 推荐的统一事件格式

```json
{ "type": "message_delta", "content": "..." }
{ "type": "actions", "actions": [...] }
{ "type": "status", "status": "running" }
{ "type": "error", "message": "..." }
{ "type": "done" }
```

---

## 9. 第一阶段 MVP 范围

第一阶段不要做太大，建议只做以下内容：

### 目标

把当前 Copilot 升级为：

- 有上下文快照
- 有结构化动作
- 有 Apply / Reject
- 能直接修改 `experimentStore`

### 明确包含

1. `Workshop` 页面上下文快照
2. `apply_config` 动作
3. `CopilotPanel` 中的 action cards
4. `agentStore` 的 session / actions / controller
5. 后端 JSON 解析和动作校验

### 明确不包含

1. KB 管理
2. 完整 workflow builder
3. 多 agent 协作
4. `Monitor` / `Results` 的深度智能分析

---

## 10. 实施步骤

## Step 1：前端 store 升级

修改：

- `frontend/src/stores/agent.ts`

新增：

- `sessions`
- `currentSessionId`
- `pendingActions`
- `controller`
- `lastContextSnapshot`
- `applyAction(action)`
- `rejectAction(actionId)`
- `buildContextSnapshot()`

## Step 2：Copilot UI 升级

修改：

- `frontend/src/components/copilot/CopilotPanel.vue`

新增组件：

- `frontend/src/components/copilot/ActionCard.vue`

目标：

- 支持渲染流式 message
- 支持渲染最终 actions
- 支持 Apply / Reject

## Step 3：Agent API 升级

修改：

- `backend/api/agent.py`
- `backend/services/agent_provider.py`

目标：

- 统一流式事件 envelope
- 支持 JSON 解析
- 返回 `message + actions`

## Step 4：对接 experimentStore

修改：

- `frontend/src/stores/experiment.ts`

目标：

- 保持现有 `applyConfig(cfg)` 不变
- 只补充必要的健壮性和类型边界

## Step 5：基础验收

验收方式：

1. 在 `Workshop` 打开 Copilot。
2. 输入“帮我配置一个 edit 实验，用某个 trainer，forget/retain 数据集分别为 ...”。
3. Copilot 返回文本说明和 action card。
4. 点击 Apply。
5. `mode`、`trainer`、`datasets`、`params` 被写入 `experimentStore`，页面即时刷新。

---

## 11. 文件级改动建议

### 第一阶段优先修改

- `frontend/src/stores/agent.ts`
- `frontend/src/components/copilot/CopilotPanel.vue`
- `frontend/src/stores/experiment.ts`
- `frontend/src/api/index.ts`
- `backend/api/agent.py`
- `backend/services/agent_provider.py`

### 第二阶段可能新增

- `frontend/src/components/copilot/ActionCard.vue`
- `frontend/src/components/copilot/ContextBanner.vue`
- `frontend/src/services/contextSnapshot.ts`

### 第三阶段可能扩展

- `frontend/src/stores/results.ts`
- `frontend/src/stores/runner.ts`
- `backend/api/results.py`
- `backend/api/runner.py`

---

## 12. 验收标准

## 第一阶段验收标准

### 功能

- Copilot 能读取当前 `Workshop` 上下文
- Copilot 能返回结构化 `apply_config`
- 前端能显示 action cards
- 用户点击 Apply 后，实验配置真实写入 store
- 用户点击 Reject 后，动作标记为 rejected

### 体验

- 流式输出不中断
- Apply 后页面配置同步刷新
- 无需手动刷新页面
- 若模型返回非法 JSON，系统仍能以纯文本回复而不崩溃

### 工程

- 新协议有明确类型
- `CopilotPanel` 不再内联拼复杂 context
- `agentStore` 具备最基本的会话和动作管理能力

---

## 13. 风险与对策

## 风险 1：模型不稳定输出 JSON

### 对策

- 后端实现 `parse_agent_response()`
- 失败时 fallback 到纯文本
- 前端允许 `actions=[]`

## 风险 2：动作过于激进

### 对策

- 第一阶段只支持 `apply_config`
- 不做直接执行命令类动作
- 必须用户显式点击 Apply

## 风险 3：上下文过大

### 对策

- 当前只传摘要，不传全部日志和全部结果
- 例如日志只传最近 N 行
- 结果只传 compare summary，不传完整原始文件

## 风险 4：Copilot 和页面状态不一致

### 对策

- 所有写入都以 store 为唯一真相源
- Apply 只通过 store action 完成，不直接改组件局部状态

---

## 14. 后续路线图

## Phase 2：Runtime Assistant

目标：

- 基于 `Monitor` 和 `Results` 上下文做日志解释和结果分析

重点：

- 增强 `contextSnapshot`
- 增加新的 action 类型
- 让 Copilot 能根据路由切换 prompt

## Phase 3：Skill / Workflow 升级

目标：

- 将 Skills 从静态模板逐步升级为可组合的实验工作流模板

可选方向：

- skill chaining
- run template
- compare template
- result-driven recommendation

## Phase 4：更严格的执行层

目标：

- 若后续需要，可引入真正的工具调用或后端 orchestrator

但这不是第一阶段必须做的。

---

## 15. 最终建议

这版方案的核心主张只有一句：

先把你现在已经有的能力打通，而不是另起炉灶重做一套系统。

也就是说，第一阶段应当聚焦这条链：

```text
CopilotPanel
-> buildContextSnapshot
-> /api/agent/chat
-> parse message + actions
-> ActionCard
-> experimentStore.applyConfig()
```

如果这条链做通，你的 `new_ui` 就已经完成了从“有聊天框”到“有可执行实验助手”的关键跨越。
