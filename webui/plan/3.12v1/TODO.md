# WebUI TODO

面向 `webui/` 的实现待办文档。当前先聚焦两类工作：

- 修复现有重构后的状态错配和功能断点
- 为后续新增页面、交互和能力预留清晰入口

## 优先级说明

- `P0` 必须先做，否则现有功能会不稳定或不生效
- `P1` 应尽快补齐，否则 README 和实际能力不一致
- `P2` 后续增强，适合在核心链路稳定后继续扩展

## P0 当前必须处理

- [ ] 修复 `mode -> trainer/experiment/dataset/eval` 的联动顺序
原因：切换到 `inject/edit` 时会出现 dropdown value 不在 choices 里的 warning，说明 UI 状态更新顺序有问题。
方案：
1. 抽一个统一的“应用 UI 配置” helper
2. 第一阶段先切 `mode` 并刷新 choices
3. 第二阶段再写入 `trainer`、`experiment`、dataset、eval 等具体值
涉及文件：`app.py`、`components/config_panel.py`、`components/skill_wizard.py`

- [ ] 让配置导入/导出变成可逆操作
原因：当前导出保存了 `trainer_args`、`method_args`、`overrides`，但导入时没有恢复这些字段。
方案：
1. 定义统一 UI schema
2. 导出时完整写入 schema
3. 导入时完整回填对应组件
涉及文件：`app.py`、`components/params.py`

- [ ] 让训练模式下的“评测套件”真正参与命令生成和执行
原因：训练页面显示的是 `eval_suite`，但预览和运行读取的是 `eval_suite_select`，训练模式下改评测套件不会生效。
方案：
1. 训练模式统一使用 Hydra override `eval=...`
2. 命令预览和实际运行使用同一字段
3. 实验模板存在时明确覆盖策略
涉及文件：`app.py`、`components/config_panel.py`、`components/run_panel.py`

- [ ] 打通“开始运行 -> 输出目录 -> 结果摘要”的闭环
原因：`output_dir` 从未在运行完成后写回，导致“加载结果”按钮基本不可用。
方案：
1. 运行结束后回填本次输出目录
2. 支持自动读取 `*_SUMMARY.json`
3. 结果区支持直接展示摘要
涉及文件：`app.py`、`components/run_panel.py`、`utils/result_parser.py`

- [ ] 切换 trainer 时同步关键默认参数
原因：目前只同步了 `method_args_json`，学习率、epoch、batch size 等仍可能保留旧值。
方案：
1. 从 trainer config 的 `args` 中提取关键参数
2. 同步更新中间参数栏
3. 必要时加“重载默认参数”按钮避免覆盖用户手改值
涉及文件：`components/params.py`、`utils/config_loader.py`

## P1 现有功能补全

- [ ] 补全 Tab 4 智能向导的一键应用能力
原因：当前只应用了部分字段，没有覆盖 dataset、eval、`method_args`、`warmup_ratio` 等。
方案：
1. 扩展 skill 模板 schema
2. 或者收敛为只应用 `experiment=...`
3. 为模板增加校验逻辑
涉及文件：`components/skill_wizard.py`、`skills/*.json`

- [ ] 实现 Tab 3 的实时推理链路
原因：现在只有静态 JSON 演示，还没有 base/after 模型选择、用户输入和实时 before/after 推理。
方案：
1. 新增 `utils/inference.py`
2. 增加模型加载控件和 prompt 输入
3. 支持 before/after 结果对比和懒加载缓存
涉及文件：`components/interactive_demo.py`、`utils/inference.py`、`examples/*.json`

- [ ] 修正 Tab 2 指标高亮逻辑
原因：当前默认所有指标都是“越大越好”，会误导如 `privleak` 这类越低越好的指标。
方案：
1. 为 metric 增加方向元数据
2. 高亮时按 max 或 min 选择
3. 在表格中补充指标含义提示
涉及文件：`utils/result_parser.py`

- [ ] 完善运行结果和对比页的数据来源约定
原因：目前 `saves/` 扫描逻辑、checkpoint 目录结构和结果摘要展示仍偏隐式。
方案：
1. 明确支持的目录结构
2. 对缺失 summary 文件给出更清晰提示
3. 必要时补一个刷新/重扫状态提示
涉及文件：`utils/config_loader.py`、`utils/result_parser.py`、`components/results_compare.py`

## P2 后续增强

- [ ] 为新增功能建立统一的状态模型
目标：避免未来继续把状态分散在多个回调里，导致字段不同步。
方向：
1. 抽取共享配置对象
2. 统一“预览命令”和“实际运行”的输入来源
3. 为向导、导入配置、新增功能共用同一套写入逻辑

- [ ] 为 WebUI 增加基础验证
目标：在继续新增功能前，先锁住关键行为。
方向：
1. 参数解析单测
2. 配置联动单测
3. 结果解析单测
4. 至少补一个 WebUI smoke test

- [ ] 梳理 README 与实际功能的一致性
目标：避免后续继续出现“文档已承诺，但功能未实现”的情况。
方向：
1. 标记已完成/未完成功能
2. 把占位功能写成 roadmap
3. 新增功能上线时同步更新文档

## 新增需求扩展方案（待确认）

### A. 中英文一键切换

- [ ] 增加 `[中文] | [English]` 全局语言切换按钮
目标：让 UI 文案、按钮、提示信息、Tab 名称和表头可以一键切换。
建议方案：
1. 新增 `i18n` 字典层，统一管理所有界面文案
2. 语言状态提升到全局，所有 Tab 共用
3. 第一阶段只覆盖界面文案，不扩展到示例数据和 skill 描述
涉及文件：`app.py`、`components/*.py`、`examples/*.json`、`skills/*.json`

### B. 智能助手升级

- [ ] 支持用户自定义“单条数据”和“批量数据”两种输入方式
目标：让用户不只选模板，而是能输入自己的编辑/遗忘/注入目标。
建议方案：
1. 单条输入支持表单模式
2. 第一阶段批量输入优先支持 `JSONL`
3. 数据输入要和现有算法数据集配置串联，不能只做独立上传
4. 增加输入校验、样例模板和预览区
涉及文件：`components/skill_wizard.py`、新增数据解析工具

- [ ] 建立 `JSONL -> 现有数据集配置` 的适配层
目标：让用户上传的数据可以直接接到现有 `inject / edit / unlearn` 算法流水线。
现状约束：
1. `inject` 现有 `InjectDataset` 已支持 `jsonl`
2. `edit` 当前本地加载仍偏向 `json`
3. `unlearn` 当前主要基于现有 QA 配置，需要一个自定义 QA/forget-retain 适配层
建议方案：
1. 新增统一数据适配层，把用户 JSONL 转换为任务对应的数据描述
2. 自动生成或选择对应的 Hydra dataset config
3. 对 `inject/edit/unlearn` 分别提供字段映射模板
4. 优先支持文本类任务，不先做图像/多模态
涉及文件：`components/skill_wizard.py`、`utils/config_loader.py`、新增 `utils/data_adapter.py`

- [ ] 根据用户目标自动推荐算法与参数
目标：让向导从“模板选择器”升级为“目标驱动配置器”。
建议方案：
1. 先做规则层推荐
2. 再引入 LLM Agent 做需求理解和配置建议
3. 输出建议时同时给出方法、数据、资源、风险说明
4. 对不同类型的用户数据给出对应算法挑选和核心配置建议
涉及文件：`components/skill_wizard.py`、新增推荐规则/agent 适配层

### C. 模型适配层

- [ ] 适配更多主流模型，并允许用户自由指定自己的模型
目标：不要把 UI 绑定在少量预置 model config 上。
建议方案：
1. 建立模型适配层，区分“预置主流模型”和“用户自定义模型路径/名称”
2. 第一阶段只支持 HuggingFace 主流文本模型
3. 预留多模态模型字段和能力标记
4. 模型适配信息包括 tokenizer、dtype、chat_template、是否支持 edit/inject/unlearn
涉及文件：`utils/config_loader.py`、`components/config_panel.py`、新增模型适配工具

### D. LLM Agent 接入

- [ ] 接入一个大模型 Agent，理解用户需求并自动生成配置
目标：用户可以直接描述“我想遗忘什么/编辑什么/注入什么”，由 Agent 帮他落配置。
建议方案：
1. Agent 第一阶段负责“需求理解 + 配置建议 + 应用到 Tab 1”
2. 不直接执行训练，先停在配置层
3. Agent 输出标准化结构：`mode / method / model / data / eval / overrides / explanation`
4. Agent 和规则引擎共存，优先给出可解释建议
涉及文件：`components/skill_wizard.py`、新增 `utils/agent_planner.py` 或同类模块

- [ ] 建立 Agent Provider 适配层，支持主流 LLM API
目标：Agent 后端可以切换不同服务商，而不是绑定单一模型接口。
建议方案：
1. 抽象统一的 agent provider 接口
2. 第一阶段支持 `OpenAI GPT` 和 `DeepSeek`
3. 配置项至少包含 `base_url / api_key / model / timeout`
4. 输出层统一收敛为结构化配置建议
涉及文件：新增 `utils/agent_provider.py`、`utils/agent_planner.py`、UI 中的 agent 设置区域

- [ ] 把算法核心配置沉淀到 `skills` 模板层
目标：让 Agent 和规则引擎都复用同一份算法知识，而不是把推荐逻辑散落在代码里。
建议方案：
1. 每个 skill 模板不只保留 UI 展示字段，还保留算法核心配置
2. skill 内包含 `适用场景 / 数据格式要求 / 推荐模型范围 / 推荐 eval / 关键 overrides / 风险提示`
3. Agent 先从 skill 库中检索候选，再做解释和微调建议
4. 规则引擎和 Agent 共用这套 skill schema
涉及文件：`skills/*.json`、`components/skill_wizard.py`、新增 skill schema 校验逻辑

### E. Tab 2 重设计与配色重构

- [ ] 美化 Tab 2，提升结果对比的科研可读性
目标：让结果对比页从“纯表格输出”升级为“可筛选、可总结、可视化”的分析页。
建议方案：
1. 左侧做筛选区，支持 mode/model/method/run 条件过滤
2. 中间保留对比表格并加 sticky header、最佳值标记、指标说明
3. 右侧增加图形摘要区，优先做 radar / bar / metric cards
涉及文件：`components/results_compare.py`、`utils/result_parser.py`、`assets/custom.css`

- [ ] 把当前偏绿的色调改成更接近参考项目的蓝灰科研风格
目标：保留专业感，但不再以绿色/teal 为主色。
建议方案：
1. 主色改为 cobalt blue
2. 辅色改为 slate / steel / ice blue
3. 警告和强调继续用 amber，不引入绿色作为主视觉
建议色板：
   - Primary: `#355CFF`
   - Primary Dark: `#1D3FDB`
   - Background: `#F5F7FB`
   - Surface: `#FFFFFF`
   - Surface Alt: `#EEF3FB`
   - Text: `#0F172A`
   - Subtle Text: `#475569`
   - Border: `#D7E0F0`
   - Accent: `#7CB7FF`
   - Warning: `#F59E0B`
涉及文件：`assets/custom.css`、`app.py`

## 新功能预留区

后续新增内容可以按下面模板继续追加：

### 功能名称

- 优先级：`P0 / P1 / P2`
- 目标：
- 用户场景：
- 需要改动的文件：
- 风险点：
- 验收标准：

## 建议执行顺序

1. 先完成所有 `P0`
2. 再补 Tab 3 / Tab 4 / Tab 2 的功能闭环
3. 最后再加新功能，避免在不稳定状态上继续堆功能

## 已确认约束

- 语言切换第一阶段只做界面文案双语
- 批量数据第一阶段优先支持 `JSONL`
- 用户数据要和现有算法数据集配置串联
- 任务模型第一阶段范围限定为 HuggingFace 主流文本模型
- LLM Agent 第一阶段负责“建议并应用配置”，不直接执行训练
- Agent 后端 API 需要支持主流服务商，第一阶段至少包含 `GPT` 和 `DeepSeek`
- 不同用户数据需要给出对应算法和配置建议
- 算法核心配置优先沉淀到 `skills` 模板中
