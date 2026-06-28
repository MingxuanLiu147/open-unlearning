# UltraRAG UI 全面分析与对 `open-unlearning/new_ui` 的参考建议

## 0. 说明

这份文档基于代码静态阅读完成，主要分析目录如下：

- `UltraRAG/ui/frontend`
- `UltraRAG/ui/backend`
- `UltraRAG/examples`
- `UltraRAG/src/ultrarag/client.py`
- 你的当前项目 `open-unlearning/new_ui` 中与 Copilot / Agent 相关的现有实现

本分析没有实际启动浏览器做视觉走查，因此重点放在：

- 技术栈
- 模块边界
- 前后端交互链路
- 聊天 / 知识库 / Builder / Prompt / AI Assistant 的实现方式
- UltraRAG 中更接近 agent workflow 的部分
- 如何把这些思路迁移到你当前的 `new_ui`

---

## 1. 一句话结论

UltraRAG UI 最值得参考的不是它的前端工程实现，而是它的产品结构和交互协议：

1. 一个统一的工作台，把 Pipeline、参数、Prompt、Chat、知识库、AI Assistant 放在同一个语境里。
2. Builder 和 Chat 并不是完全割裂的，Pipeline 配置、参数配置、Prompt 编辑和对话能力是连通的。
3. AI Assistant 不是纯聊天，而是“上下文感知 + 结构化建议 + 用户 Apply/Reject”的可执行 Copilot。
4. 真正的 agent/workflow 能力主要在 YAML pipeline 里，通过 `loop + branch + router + custom state` 组织。
5. 这套思路适合借鉴到 `open-unlearning/new_ui`，但实现层不建议照搬它的原生单文件 JS 模式。

---

## 2. 目录与总体架构

UltraRAG UI 的目录非常集中：

```text
UltraRAG/
├─ ui/
│  ├─ frontend/
│  │  ├─ index.html
│  │  ├─ main.js
│  │  ├─ style.css
│  │  ├─ i18n/
│  │  └─ vendor/
│  └─ backend/
│     ├─ app.py
│     └─ pipeline_manager.py
├─ examples/
│  ├─ RAG-web.yaml
│  ├─ multiturn_chat.yaml
│  ├─ LightResearch.yaml
│  ├─ AgentCPM-Report-web.yaml
│  └─ parameter/
├─ src/ultrarag/client.py
├─ servers/
└─ prompt/
```

代码量也能说明它的组织方式：

- `ui/frontend/main.js`: 10130 行
- `ui/frontend/style.css`: 8950 行
- `ui/frontend/index.html`: 1358 行
- `ui/backend/app.py`: 1757 行
- `ui/backend/pipeline_manager.py`: 2772 行

这说明它本质上是：

- 一个由 Flask 同时托管静态页面和 API 的单体 Web UI
- 一个很强的工作台产品
- 一个前端实现上已经明显“长成大文件”的系统

整体数据流可以概括为：

```text
Browser
  -> Flask(app.py)
      -> 静态页面(index.html/main.js/style.css)
      -> /api/*
          -> pipeline_manager.py
              -> ultrarag.client
                  -> MCP servers / retriever / generation / prompt / router / custom
```

---

## 3. 技术栈分析

## 3.1 前端技术栈

UltraRAG 前端不是 React / Vue / Next / Vite 这类现代工程，而是原生静态页：

- 没有 `package.json`
- 没有 `vite.config.*`
- 没有 `webpack.config.*`
- 没有 `tsconfig.json`

说明它的前端是：

- `index.html` 直接引入资源
- `main.js` 承担全部状态与交互逻辑
- `style.css` 承担全部样式逻辑

依赖层面：

- UI 基础：Bootstrap
- Markdown 渲染：`marked`
- HTML 清洗：`DOMPurify`
- 代码高亮：`highlight.js`
- 数学公式：`KaTeX + auto-render`
- 字体：本地 `Inter`、`JetBrains Mono`
- 国际化：自定义 `window.I18N_LOCALES`

前端的几个关键技术特征：

1. 路由不是框架路由，而是手写 `history.pushState` + path 判断。
2. 状态管理不是 Pinia/Redux，而是全局对象。
3. UI 切换主要依赖 DOM 显隐和全局状态。
4. 所有复杂功能都塞进 `main.js`。

### 关键文件

- `UltraRAG/ui/frontend/index.html`
- `UltraRAG/ui/frontend/main.js`
- `UltraRAG/ui/frontend/style.css`
- `UltraRAG/ui/frontend/i18n/en.js`
- `UltraRAG/ui/frontend/i18n/zh.js`

---

## 3.2 后端技术栈

后端是 Flask，但 Flask 只是一层 HTTP 门面。真正的执行逻辑在：

- `ui/backend/pipeline_manager.py`
- `src/ultrarag/client.py`

这意味着它不是传统的 CRUD Web 后端，而是“UI API + pipeline runtime bridge”。

后端的职责包括：

- 托管静态前端
- 暴露 Builder / Chat / KB / Prompt / AI API
- 管理 demo session、background session、KB task
- 通过 `ultrarag.client` 执行 pipeline
- 把异步流式事件桥接为 SSE 返回给前端

它同时具备：

- Web server
- Session manager
- Task manager
- Pipeline runtime adapter
- Knowledge base orchestration layer

---

## 3.3 部署模式

UltraRAG UI 不是前后端分离部署，而是同源部署。

启动入口在：

- `UltraRAG/src/ultrarag/client.py`

命令行支持：

- `ultrarag show ui`
- `ultrarag show ui --admin`

这里有两个模式：

1. `chat-only mode`
2. `admin mode`

这点很重要，因为它说明 UI 从产品层面已经考虑到两种使用场景：

- 普通用户只要聊天与使用
- 高级用户需要配置 pipeline / 参数 / prompts

---

## 4. UltraRAG UI 的模块结构

## 4.1 Builder 工作区

Builder 是整个系统的核心工作台，包含三个子模式：

1. `pipeline`
2. `parameters`
3. `prompts`

### Pipeline 模式

特点：

- 左边是可视流程画布
- 右边是 YAML 编辑器
- 中间有可拖拽分隔条
- 支持 branch / loop / tool / custom node
- YAML 和可视结构双向同步

值得注意的实现点：

- 保存时优先保存原始 YAML
- YAML 解析优先走服务端 `/api/pipelines/parse`
- 如果服务端解析失败，前端退回本地简化 parser

这是一个很实用的工程实践：既保证真实解析，又保证前端不中断。

### Parameters 模式

特点：

- 参数按 server 分组渲染
- 有“简化模式”和“完整模式”
- 参数编辑后可以直接持久化

这说明作者已经意识到：

- 模型系统参数很多
- 不能把所有字段平铺给用户
- UI 必须支持面向用户心智的简化展示

### Prompts 模式

特点：

- Prompt 文件列表
- 搜索
- 多 tab
- 编辑 / 保存 / 删除 / 重命名

本质上它把 prompt 视为一等资源，而不是隐藏在后端目录里的模板文件。

---

## 4.2 Chat 模式

Chat 模式并不是一个孤立聊天框，而是依赖当前 pipeline 和知识库。

核心能力：

- 选择 pipeline
- 启动对应 engine session
- 选择知识库 collection
- 发起流式聊天
- 展示 thinking / step / sources / final answer
- 会话列表和本地恢复
- 导出结果
- 后台执行

这其实是“实验工作台中的运行面板”，而不是纯聊天产品。

---

## 4.3 Knowledge Base 模块

KB 模块不只是一个下拉框，而是一个完整知识库控制台：

- 配置 Milvus
- 上传 raw files
- 解析 corpus
- chunk
- 建立 index
- 查看 collection
- 删除 staging files / collection

这套设计说明 UltraRAG 希望用户在同一工作台里完成：

- 数据准备
- 检索配置
- 对话运行

而不是切换到另一个独立后台系统。

---

## 4.4 Background Tasks

Background Tasks 是 UltraRAG 一个很实用的产品点。

它允许用户：

- 把长问题发到后台运行
- 继续留在当前前台工作区
- 后续再把后台结果加载回对话

这适合：

- 长时检索
- 长时生成
- 多步 agent 流程
- 知识库构建

这类能力对 `open-unlearning/new_ui` 也很有参考价值，因为你的系统里同样存在：

- 训练
- 评测
- 结果分析
- 可能的长时推理

---

## 4.5 AI Assistant

UI 右侧的 `AI Assistant` 是一个独立侧边抽屉，它更接近“工作台 Copilot”，不是普通聊天框。

它的特点：

- 感知当前工作区上下文
- 给出结构化修改建议
- 用户可以 Apply / Reject
- 可以直接改 YAML、Prompt、Parameter

这是 UltraRAG UI 最值得借鉴的交互设计。

---

## 5. 前后端交互链路

## 5.1 API 组织方式

UltraRAG 的 API 大致分成几组：

### Builder / Pipeline

- `GET /api/pipelines`
- `POST /api/pipelines`
- `PUT /api/pipelines/<name>/yaml`
- `POST /api/pipelines/parse`
- `GET /api/pipelines/<name>`
- `DELETE /api/pipelines/<name>`
- `POST /api/pipelines/<name>/rename`
- `GET /api/pipelines/<name>/parameters`
- `PUT /api/pipelines/<name>/parameters`
- `POST /api/pipelines/<name>/build`

### Tools / Servers

- `GET /api/servers`
- `GET /api/tools`

### Chat / Engine

- `POST /api/pipelines/<name>/demo/start`
- `POST /api/pipelines/demo/stop`
- `POST /api/pipelines/<name>/chat`
- `POST /api/pipelines/chat/stop`
- `POST /api/pipelines/chat/clear-history`
- `GET /api/pipelines/chat/history`
- `POST /api/chat/export/docx`

### Background Tasks

- `POST /api/pipelines/<name>/chat/background`
- `GET /api/background-tasks`
- `GET /api/background-tasks/<task_id>`
- `DELETE /api/background-tasks/<task_id>`
- `POST /api/background-tasks/clear-completed`

### Knowledge Base

- `GET /api/kb/config`
- `POST /api/kb/config`
- `GET /api/kb/files`
- `GET /api/kb/files/inspect`
- `POST /api/kb/upload`
- `DELETE /api/kb/files/<category>/<filename>`
- `POST /api/kb/staging/clear`
- `POST /api/kb/run`
- `GET /api/kb/status/<task_id>`

### Prompts

- `GET /api/prompts`
- `GET /api/prompts/<path>`
- `POST /api/prompts`
- `PUT /api/prompts/<path>`
- `DELETE /api/prompts/<path>`
- `POST /api/prompts/<path>/rename`

### AI Assistant

- `POST /api/ai/test`
- `POST /api/ai/chat`

---

## 5.2 Chat 数据流

UltraRAG 聊天并不是简单 `POST question -> return answer`，而是一个 session 化的流式系统。

### 前端侧

前端会维护：

- 本地聊天历史
- 当前 pipeline
- 当前 engine session id
- 当前知识库选择
- 当前是否正在流式生成

发问流程大致如下：

```text
用户输入问题
  -> 前端检查是否有可用 engine
  -> 检查是否选择知识库
  -> 将用户消息先写入本地 history
  -> POST /api/pipelines/<name>/chat
      body:
        - question
        - history
        - session_id
        - dynamic_params
  -> fetch(...).body.getReader() 读 SSE 样式流
  -> 根据 event type 更新 UI
```

### 后端侧

后端收到请求后会：

1. 解析前端历史为内部 `role/content`
2. 注入 retriever 的 KB collection 参数
3. 判断是首轮还是多轮
4. 首轮走完整 pipeline
5. 多轮默认走纯多轮生成

这部分很关键：UltraRAG 采用“两阶段聊天”。

---

## 5.3 首轮 RAG vs 后续多轮

这是 UltraRAG 当前聊天架构中最重要的设计事实之一。

### 首轮

首轮走完整 pipeline：

- `RAG-web.yaml` 或其他实际被选中的 pipeline

典型链路：

```text
benchmark.get_data
-> retriever.retriever_init
-> generation.generation_init
-> prompt.qa_boxed / qa_rag_boxed
-> generation.generate
```

### 后续轮次

后续轮次不再默认执行完整 RAG。

而是走：

- `multiturn_chat.yaml`

也就是：

```text
generation.generation_init
-> generation.multiturn_generate
```

这意味着：

- 优点：速度快、成本低、体验流畅
- 缺点：后续轮次不自动重新检索，事实 freshness 下降

这点对你做 `open-unlearning/new_ui` 很重要，因为如果你后续也要做实验助理或结果解释助理，最好明确区分：

1. 每轮都重新检索的 runtime assistant
2. 只基于上下文历史的纯多轮生成 assistant

不要混在一起。

---

## 5.4 SSE 事件协议

UltraRAG 的 Chat 流式协议非常值得借鉴。

常见事件类型：

- `step_start`
- `step_end`
- `sources`
- `token`
- `final`
- `error`

这个协议的优点：

1. 不只是输出 token。
2. 可以显示推理步骤。
3. 可以显示引用来源。
4. 可以在最终结果阶段统一收口。

前端不是用 `EventSource`，而是用 `fetch + ReadableStream reader`。

这样做的好处：

- 可以发送复杂 POST body
- 可以配合 `AbortController`
- 更适合带上下文的大请求

这套实践对 `new_ui` 很有用，因为你当前：

- `runnerApi.connectLog()` 仍然在用 `EventSource`
- `agentApi.streamChat()` 已经在用 `fetch + reader`

也就是说，你已经有迁移基础。

---

## 5.5 Session 设计

UltraRAG 实际上有三层 session / task 语义：

1. 浏览器本地聊天会话
2. 远端 demo engine session
3. 后台任务 session

### 浏览器本地聊天会话

保存在 `localStorage`，用于：

- 保留聊天历史
- 切换会话
- 刷新恢复

### 远端 demo engine session

后端用 `session_id -> DemoSession` 管理，复用运行时上下文。

### 后台任务 session

后台任务不会抢当前前台 session，而是独立创建 background session。

这是一个非常好的架构实践，因为它把：

- UI 会话
- 执行会话
- 后台任务

分离开了。

---

## 5.6 Knowledge Base 数据流

KB 模块的数据流是：

```text
文件上传
  -> raw staging
  -> build_text_corpus
  -> corpus_chunk
  -> milvus_index
  -> collection 可供 chat 动态选择
```

其中值得学习的点有：

1. 显示名和实际文件名分离。
2. 用 `_meta.json` / `_display_names.json` 保存 UI 友好的展示信息。
3. collection 安全名和 display name 分离。
4. KB 构建本身也是后台任务。

这类设计对实验平台很有价值，因为实验输入数据和底层存储标识通常不能完全等价。

---

## 6. 动态工作流和参数系统

UltraRAG 的 Builder 并不是手写死的节点面板，它有动态生成的一面。

关键点在后端：

- `list_servers()`
- `list_server_tools()`
- `_ensure_server_yaml()`
- `_generate_server_stub()`

它会：

1. 扫描 `servers/`
2. 找 `server.yaml`
3. 如果没有，就静态分析 Python AST
4. 自动生成 tools / prompts 元数据

这意味着 UI 可以动态感知：

- 有哪些 server
- 每个 server 有哪些 tool
- tool 需要什么输入
- prompt 有哪些参数

这对可视工作流系统非常关键。

### 这套思路的价值

如果未来你的 `open-unlearning/new_ui` 也要做“可组合实验流”或“技能编排流”，这套元数据驱动方式非常有用：

- 不要把 UI 节点写死
- 用后端暴露 schema / metadata
- 前端按 schema 渲染节点和参数表单

---

## 7. UltraRAG 的“智能助手 / agent”到底是什么

这里必须区分三层能力，否则很容易把 UltraRAG 误读成“一个统一 agent 系统”。

## 7.1 第一层：工作台 Copilot

这是 UI 右侧的 `AI Assistant`。

它的特点：

- 读取当前工作区上下文
- 请求外部 LLM
- 把 LLM 回复解析为结构化 action
- 由用户决定 Apply / Reject

它本质上是：

- 上下文感知型 Copilot
- 配置编辑助手
- Prompt 编辑助手

它不是：

- 真正的工具调用 agent
- server-side planner/executor
- 带长期记忆和内部检索的 agent

---

## 7.2 第二层：RAG Demo Chat

这是聊天页里的主问答能力。

它本质上是：

- 首轮 pipeline 执行
- 后续多轮纯生成

它不是右侧 AI Assistant，也不是完整 agent。

---

## 7.3 第三层：可编排 agent workflow

真正更像 agent 的能力在 `examples/*.yaml`：

- `LightResearch.yaml`
- `AgentCPM-Report-web.yaml`
- `r1_searcher.yaml`

它们通过这些原语组织复杂工作流：

- `loop`
- `branch`
- `router`
- `custom state`
- prompt 生成子任务
- 检索
- citation registry
- 多阶段规划 / 写作 / 扩展

也就是说，UltraRAG 真正的 agentic 能力在 pipeline 层，而不是在 UI 右侧 Copilot 层。

---

## 8. AI Assistant 的实现细节

## 8.1 上下文注入

前端会根据当前模式构造 context snapshot：

- `pipeline` 模式：当前 YAML
- `parameters` 模式：当前参数 JSON
- `prompts` 模式：当前 prompt 文件及内容

然后把这个 context 发给 `/api/ai/chat`。

后端会把 context 拼进 system prompt。

这点很值得借鉴，因为它让 AI 不需要用户反复描述“我现在在改什么”。

---

## 8.2 输出协议

UltraRAG 没用 JSON tool calling，而是用了“文本协议 + 正则解析”：

### 改 pipeline

````text
<fenced block>
```yaml:pipeline
...
```
````

### 改 prompt

````text
<fenced block>
```jinja:prompt:<filename>
...
```
````

### 改参数

```text
Set `a.b.c` to `x`
```

后端会把这些解析成 action：

- `modify_pipeline`
- `modify_prompt`
- `modify_parameter`

前端再渲染 Apply / Reject 按钮。

### 这套方案的优点

- 实现简单
- 易于接入外部兼容 OpenAI 的模型
- 对 demo 和原型很有效

### 缺点

- 协议脆弱
- 正则解析容易误判
- 不适合复杂 diff
- 不适合长生命周期协作系统

如果迁移到 `new_ui`，建议保留“结构化 action”思想，但把协议升级为严格 JSON schema。

---

## 8.3 Apply / Reject 闭环

UltraRAG 的真正亮点不是“AI 会回答”，而是“AI 的建议可以落地”。

前端应用逻辑包括：

- 改 YAML：写入编辑器，重建 canvas，自动保存
- 改 Prompt：切到目标文件，写回内容，自动保存
- 改参数：按 path 写入参数对象，重渲染表单，自动保存

这套闭环才是它最值得参考的部分。

---

## 9. UltraRAG 的优点

## 9.1 产品层优点

1. 工作台形态完整。
2. 配置、Prompt、运行、知识库、AI 助手在一个系统里。
3. Builder 和 Chat 打通。
4. AI Assistant 不是空聊天，而是可执行建议。
5. Background tasks 非常适合长任务。
6. KB 管理做得很产品化。

## 9.2 工程层优点

1. 同源部署，接口简单。
2. `fetch + SSE reader` 支持复杂流式场景。
3. 前端作为多轮历史真相源，后端更聚焦执行。
4. session 分层清晰。
5. pipeline 元数据可动态暴露给前端。
6. YAML 解析有服务端和前端双保险。

---

## 10. UltraRAG 的问题和限制

## 10.1 前端工程债

1. `main.js` 太大，功能高度耦合。
2. 全局状态 + DOM ID 绑定，维护成本高。
3. 存在重复结构和迁移痕迹。
4. 没有 TS，缺少类型约束。
5. 页面结构复杂后，回归测试风险大。

## 10.2 AI 助手限制

1. 动作协议依赖正则。
2. 没有真正 tool calling。
3. 没有 server-side agent orchestration。
4. 会话与设置主要保存在 `localStorage`，更像单机 demo。

## 10.3 Chat 限制

1. 后续轮次默认不重新检索。
2. `/api/pipelines/chat/clear-history` 后端有接口但前端没真正用起来。
3. `chat/history` 中的 `client_ip` 校验实现不完整。
4. Background tasks 的用户隔离只是轻隔离，不是鉴权。

---

## 11. 对 `open-unlearning/new_ui` 的直接参考价值

下面这部分结合了你当前项目的现状。

你当前 `new_ui` 的前端是：

- Vue 3
- Vite
- TypeScript
- Pinia
- Vue Router
- Element Plus

对应文件：

- `open-unlearning/new_ui/frontend/package.json`
- `open-unlearning/new_ui/frontend/src/stores/agent.ts`
- `open-unlearning/new_ui/frontend/src/components/copilot/CopilotPanel.vue`
- `open-unlearning/new_ui/frontend/src/stores/experiment.ts`
- `open-unlearning/new_ui/backend/api/agent.py`
- `open-unlearning/new_ui/backend/services/agent_provider.py`

这意味着你已经比 UltraRAG 更接近一个可维护的现代实现。

结论很明确：

- 你应该借鉴 UltraRAG 的产品设计和交互协议
- 不应该退回 UltraRAG 那种原生单文件 JS 实现

---

## 11.1 你当前 `new_ui` 已经有的基础

### Agent / Copilot 侧

你当前已经有：

1. `CopilotPanel.vue`
2. `agent.ts`
3. `/api/agent/chat`
4. `fetch + reader` 的流式聊天
5. 一个基于实验上下文拼装 system prompt 的后端

这和 UltraRAG 的 AI Assistant 已经有相似基础。

### 更关键的一点

你当前的 `DEFAULT_SYSTEM_PROMPT` 已经要求模型输出：

```json
{
  "action": "apply_config",
  "mode": "...",
  "model": "...",
  "trainer": "...",
  "datasets": {},
  "eval": "...",
  "params": {}
}
```

而 `experimentStore` 里已经有：

- `applyConfig(cfg)`

这意味着你当前项目其实已经“半只脚进入 UltraRAG 式可执行 Copilot”了，只是还没有把闭环真正做完。

换句话说：

- UltraRAG 的亮点不是你没有基础
- 而是它把“建议 -> 预览 -> Apply”真正做完了

---

## 11.2 最值得你直接迁移的能力

## A. 把当前 Copilot 从“只流式文本”升级为“结构化建议 + Apply/Reject”

你当前：

- 模型可以被提示输出 JSON
- 前端只是在流式拼文本
- 没有 action 卡片
- 没有 Apply/Reject

建议直接升级为：

```text
LLM 输出
  -> 后端验证 JSON schema
  -> 返回 message + actions[]
  -> 前端渲染 action cards
  -> 用户 Apply / Reject
  -> 调 experimentStore.applyConfig(...)
```

可直接落点：

- `frontend/src/components/copilot/CopilotPanel.vue`
- `frontend/src/stores/agent.ts`
- `backend/api/agent.py`
- `backend/services/agent_provider.py`
- `frontend/src/stores/experiment.ts`

这是你当前最值得立刻做的一步。

---

## B. 上下文快照机制升级

UltraRAG 的 AI Assistant 会按当前工作区模式注入上下文。

你当前 `CopilotPanel.vue` 已经在传：

- `mode`
- `model`
- `trainer`
- `datasets`
- `skills`

这很好，但还不够。

建议补充：

- `selectedEval`
- `selectedExperiment`
- `params`
- `overrides`
- 当前运行状态
- 最新结果摘要
- 当前视图类型：`Workshop / Skills / Monitor / Results`

推荐把上下文构造从组件里抽出来，做成：

- `frontend/src/stores/agent.ts` 中的 `buildContextSnapshot()`

这样 `CopilotPanel.vue` 不需要自己拼 context。

---

## C. 让 Agent 有会话和动作状态，而不只是消息流

当前 `agent.ts` 只有：

- `messages`
- `streaming`
- `currentChunk`

这更像“简易聊天 store”。

建议升级为：

- `sessions`
- `currentSessionId`
- `messages`
- `pendingActions`
- `lastContextSnapshot`
- `streaming`
- `controller`

这样你就能实现类似 UltraRAG 的：

- 多会话切换
- 记住上一次建议
- 对建议块做 Apply / Reject
- 在刷新后恢复 Copilot 对话

---

## D. 引入“工作台 Copilot”和“运行时 Assistant”的分层

UltraRAG 的一个重要启发是：不要把所有“智能能力”混成一个聊天框。

你可以在 `new_ui` 里做明确分层：

### 1. Workspace Copilot

作用：

- 推荐实验配置
- 修改训练参数
- 解释参数含义
- 生成可执行配置补丁

主要作用域：

- `Workshop.vue`
- `Skills.vue`

### 2. Runtime Assistant

作用：

- 解读运行日志
- 解释失败原因
- 总结结果差异
- 回答“为什么这个方法效果差”

主要作用域：

- `Monitor.vue`
- `Results.vue`

这两个 assistant 可以共用一个后端 provider，但不要共用完全相同的 prompt 和 action schema。

---

## E. 借鉴 UltraRAG 的后台任务与事件协议

你当前已经有：

- `runnerApi.connectLog()` 的日志 SSE

建议进一步统一成更强的事件协议，例如：

- `status`
- `step_start`
- `step_end`
- `log`
- `metric`
- `artifact`
- `result`
- `error`
- `done`

这样后续：

- Monitor 页面
- Copilot 页面
- Results 页面

都可以共享统一事件流，不必每条链路自定义一套协议。

---

## 11.3 推荐保留、推荐改造、不建议照搬

| 类别 | 内容 | 结论 |
|---|---|---|
| 产品结构 | 单工作台，配置与运行打通 | 推荐保留 |
| AI 交互 | 上下文感知 + Apply/Reject | 推荐保留 |
| Workflow | pipeline / DAG / loop / branch 思路 | 推荐保留 |
| 流式协议 | `POST + SSE(fetch reader)` | 推荐保留 |
| KB 控制台 | 若你未来做数据管理，可借鉴 | 视需求保留 |
| 实现方式 | 原生 `main.js` 单文件 | 不建议照搬 |
| 动作协议 | regex 解析 fenced block | 推荐改造成 JSON schema |
| 多轮聊天 | 后续轮次默认不检索 | 不建议直接照搬 |
| 状态存储 | 过度依赖 `localStorage` | 推荐部分保留、核心状态服务端化 |

---

## 12. 面向你当前 `new_ui` 的落地路线

## Phase 1：把 Copilot 真正变成“可执行助手”

### 目标

把你现在的：

- 文本流式聊天

升级成：

- 文本回答 + 结构化 action + Apply / Reject

### 具体改动建议

#### 后端

修改：

- `backend/api/agent.py`
- `backend/services/agent_provider.py`

建议：

1. 用严格 JSON schema 约束输出，不再依赖模型随意写 Markdown。
2. `/api/agent/chat` 最终返回：
   - `content`
   - `actions`
   - `context_snapshot_id`
3. 后端校验 action 后再发给前端。

#### 前端

修改：

- `frontend/src/stores/agent.ts`
- `frontend/src/components/copilot/CopilotPanel.vue`

建议：

1. 在 store 中新增 `pendingActions`
2. 在 UI 中渲染建议卡片
3. 支持 `Apply` / `Reject`
4. 对 `apply_config` 直接调用 `experimentStore.applyConfig()`

---

## Phase 2：做“工作区上下文快照”

### 目标

不只传几项当前选择，而是传结构化工作区上下文。

建议上下文字段：

- 当前视图
- 当前 mode
- 当前 model / trainer / datasets / eval
- 当前 params
- 当前 overrides
- 当前 skill 列表
- 当前运行状态
- 最新日志摘要
- 最新结果摘要

这样 Copilot 才能真正做到类似 UltraRAG 的上下文感知。

---

## Phase 3：做“运行时 Assistant”

### 目标

让 agent 不只会配配置，还会解释运行和结果。

适合接入的位置：

- `Monitor.vue`
- `Results.vue`

能力包括：

- 解释当前失败日志
- 总结一次 run 的关键异常
- 对比两次结果的变化
- 推荐下一步实验

这会比单纯“配置助手”更有价值。

---

## Phase 4：把 Skills / Experiments 逐步 workflow 化

你当前项目里已经有：

- `skills/`
- `skill_engine.py`
- `command_runner.py`

这意味着你未来完全可以往 UltraRAG 那种“工作流层”演化，但建议做得比它更现代：

- 不一定用 YAML
- 也可以用 JSON schema / typed workflow model
- 前端做 DAG / step editor
- 后端做 workflow executor

如果以后要做：

- 多阶段实验编排
- 自动评测
- skill chaining
- result-driven retry

这条路线非常自然。

---

## 13. 最终建议

如果把 UltraRAG 的经验浓缩成三句话，给 `open-unlearning/new_ui` 的建议是：

1. 学它的工作台产品结构，不学它的原生单文件实现。
2. 学它的“AI 建议可执行”闭环，不学它的 regex 文本协议。
3. 学它把配置、运行、知识、助手串起来的思路，同时在你的 Vue + TS + Pinia 架构上做得更严格、更模块化。

对你当前项目，最有价值的不是重写 UI，而是把已经存在的这些能力打通：

- `agent_provider.DEFAULT_SYSTEM_PROMPT`
- `experimentStore.applyConfig()`
- `CopilotPanel.vue`
- `agent.ts`

你已经有基础，只差把这条链真正闭合。

---

## 14. 关键文件清单

### UltraRAG

- `UltraRAG/ui/frontend/index.html`
- `UltraRAG/ui/frontend/main.js`
- `UltraRAG/ui/frontend/style.css`
- `UltraRAG/ui/backend/app.py`
- `UltraRAG/ui/backend/pipeline_manager.py`
- `UltraRAG/examples/RAG-web.yaml`
- `UltraRAG/examples/multiturn_chat.yaml`
- `UltraRAG/examples/LightResearch.yaml`
- `UltraRAG/examples/AgentCPM-Report-web.yaml`
- `UltraRAG/src/ultrarag/client.py`

### open-unlearning/new_ui

- `open-unlearning/new_ui/frontend/package.json`
- `open-unlearning/new_ui/frontend/src/components/copilot/CopilotPanel.vue`
- `open-unlearning/new_ui/frontend/src/stores/agent.ts`
- `open-unlearning/new_ui/frontend/src/stores/experiment.ts`
- `open-unlearning/new_ui/frontend/src/api/index.ts`
- `open-unlearning/new_ui/backend/api/agent.py`
- `open-unlearning/new_ui/backend/services/agent_provider.py`
