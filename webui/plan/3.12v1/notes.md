# Notes: WebUI Feature Expansion

## Confirmed Requirements

- UI 需要新增 `[中文] | [English]` 一键切换
- 双语第一阶段只覆盖界面文案
- 智能助手支持单条数据和批量数据
- 批量数据第一阶段优先支持 `JSONL`
- 用户数据必须和现有算法数据集配置串联
- 任务模型第一阶段只做 HuggingFace 主流文本模型
- Agent 后端需要支持主流 LLM API，第一阶段至少支持 `GPT` 和 `DeepSeek`
- Agent 第一阶段负责"建议并应用配置"，不直接启动训练
- **Agent 需要支持多轮对话交互，而非单轮建议**
- 对不同用户数据要给出对应算法和配置建议
- **模型增强只使用现有算法（inject/edit/unlearn），暂不引入 LoRA 等其他方式**
- **暂不支持多模态模型，第一阶段专注文本模型**
- 算法核心配置优先沉淀到 `skills/*.json`
- Tab 2 需要重设计和美化
- 全局配色改成蓝灰科研风格，不再以绿色为主色

## Data Pipeline Findings

### Injection
- `src/data/inject.py` 已支持本地 `jsonl`
- 已有 `configs/data/datasets/Custom_inject.yaml` 可作为自定义注入数据入口

### Editing
- `src/data/editing.py` 当前本地文件加载偏向 `json`
- 若要统一支持批量 `jsonl`，需要补充编辑任务的数据适配层

### Unlearning
- `src/data/qa.py` 仍主要走现有 QA 数据集配置
- `configs/data/unlearn.yaml` 通过 `forget/retain` 组合进入 unlearn 流水线
- 若接入用户自定义 `jsonl`，需要新增 QA / forget-retain 适配层

## Architecture Direction

- 将"任务模型适配"和"Agent 后端适配"分离
- 将"规则推荐"和"LLM Agent 推荐"分离
- 将"技能知识库"和"UI 展示模板"合并为统一 skill schema
- 将"用户 JSONL 输入"和"Hydra dataset config 生成"之间加一层 adapter
