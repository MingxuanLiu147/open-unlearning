# Notes: UltraRAG UI 对 open-unlearning/new_ui 的参考要点

## UltraRAG 的核心价值

- 统一工作台：Pipeline、参数、Prompt、Chat、KB、AI Assistant 在一个语境里。
- AI Assistant 不是普通对话，而是“上下文感知 + 结构化建议 + Apply/Reject”。
- 真正的 agent/workflow 在 YAML pipeline 层，不在右侧 Copilot 层。
- Chat 使用 `POST + fetch reader` 消费 SSE 风格流，适合复杂请求体和中断控制。
- 会话分层清晰：本地会话、运行时 session、后台任务 session。

## UltraRAG 不建议照搬的点

- `main.js` 超大单文件，维护性差。
- 动作协议依赖 fenced block + regex。
- 后续轮次默认不重新检索。
- 过度依赖 `localStorage`。

## 当前 new_ui 已有基础

### 前端

- `Vue 3 + Vite + TypeScript + Pinia + Element Plus`
- `MainLayout.vue` 已经有独立 Copilot 侧边栏
- `CopilotPanel.vue` 已经有流式聊天入口
- `agent.ts` 已经有基础消息流状态
- `experiment.ts` 已有 `applyConfig(cfg)`
- `runner.ts` 已有运行日志状态

### 后端

- `backend/api/agent.py` 已有 `/api/agent/chat`
- `backend/services/agent_provider.py` 已有默认 system prompt 和 OpenAI-compatible 流式接口
- `backend/api/runner.py` 已有运行状态和日志流

## 初步结论

- 第一阶段无需重做布局。
- 最值得先做的是把 Copilot 从“文本流式聊天”升级为“结构化 action + Apply/Reject”。
- 上下文快照应从组件内联拼接，升级为 store 或 service 层统一构造。
- 后续应拆出 `Workspace Copilot` 和 `Runtime Assistant` 两条能力线。
