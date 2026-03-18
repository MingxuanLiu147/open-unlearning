# Task Plan: WebUI Implementation Checklist

## Goal
把当前 WebUI 需求整理成一份可直接执行的分阶段实施清单，并固化关键约束。

## Phases
- [x] Phase 1: 汇总已确认需求与边界
- [x] Phase 2: 校验现有 TODO 与数据接入约束
- [x] Phase 3: 产出实施清单文档
- [x] Phase 4: 交付并等待后续实现

## Key Questions
1. 任务模型和 Agent 后端模型的适配边界如何区分？
2. JSONL 如何接入 inject / edit / unlearn 三类现有数据流水线？
3. 新增功能的实现顺序怎样安排最稳？

## Decisions Made
- 任务模型第一阶段只做 HuggingFace 主流文本模型适配
- Agent 后端第一阶段支持 GPT 和 DeepSeek API
- 批量数据第一阶段只支持 JSONL
- 算法核心配置优先沉淀到 `skills/*.json`

## Errors Encountered
- 多次命令执行失败，原因是工作目录路径拼写错误和只读沙箱限制；已改正路径并使用只读提升完成读取

## Status
**Completed** - 已生成实施清单文档，可继续进入逐阶段实现
