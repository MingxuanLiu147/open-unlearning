# Task Plan: open-unlearning/new_ui 参考 UltraRAG 的 UI/Agent 改造方案

## Goal
产出一份可直接指导 `open-unlearning/new_ui` 下一阶段开发的实施方案，覆盖产品结构、前后端接口、状态设计、分阶段实施和验收标准。

## Phases
- [x] Phase 1: 收集当前 `new_ui` 与 UltraRAG 的关键上下文
- [x] Phase 2: 提炼可迁移能力与不建议照搬项
- [x] Phase 3: 形成实施方案文档
- [ ] Phase 4: 根据方案开始实现第一阶段改造

## Key Questions
1. 当前 `new_ui` 哪些能力已经具备，只差打通闭环？
2. UltraRAG 哪些设计值得直接迁移，哪些只适合作为理念参考？
3. 第一阶段最小可交付应该落在哪些文件上？

## Decisions Made
- 决定不照搬 UltraRAG 的原生前端实现，而是只借鉴其工作台和 Copilot 交互模式。
- 决定优先围绕当前 `CopilotPanel -> /api/agent/chat -> experimentStore.applyConfig()` 做第一阶段方案。
- 决定把方案拆成 `Workspace Copilot`、`Runtime Assistant`、`Workflow Layer` 三层，避免把所有智能能力塞进同一个聊天框。

## Errors Encountered
- 无阻塞错误。

## Status
**Currently in Phase 4** - 调研、笔记与实施方案文档均已完成，下一步是按方案启动第一阶段实现。
