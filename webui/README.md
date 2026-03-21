# Know-Surgery WebUI 使用指南

Know-Surgery WebUI 是一个基于 Gradio 的图形化界面，用于配置和运行大模型知识可控更新任务。

## 快速启动

### 1. 安装依赖

确保已安装项目基础依赖，然后安装 Gradio：

```bash
pip install gradio>=4.0.0
```

如需使用智能助手功能（Agent），还需安装：

```bash
pip install openai
```

### 2. 启动 WebUI

在项目根目录下运行：

```bash
python webui/app.py
```

可选参数：
- `--port`: 指定端口，默认 7860
- `--host`: 指定地址，默认 0.0.0.0
- `--share`: 创建公共链接（用于远程访问）

示例：

```bash
python webui/app.py --port 8080
python webui/app.py --share
```

### 3. 访问界面

启动后，在浏览器中打开：

```
http://localhost:7860
```

## 功能说明（4 个 Tab）

### Tab 1: ⚙️ 配置与运行

三栏布局：

| 左栏（任务配置） | 中栏（参数调节） | 右栏（运行控制） |
|----------------|----------------|----------------|
| 模式/模型/方法/数据集 | 学习率/轮数/批次等训练参数 | 命令预览、GPU 设置 |
| 实验模板、评测套件 | 方法参数（JSON）、Override | 启动/停止、实时日志 |
| 自定义模型输入 | — | 结果展示 |
| 导入/导出配置 | — | — |

**任务模式**：Unlearn（遗忘）· Inject（注入）· Edit（编辑）· Eval（评估）

**自定义模型**：支持 HuggingFace 模型名（`org/model`）或本地路径，勾选「使用自定义模型」启用。

### Tab 2: 📊 结果对比

- **筛选区**：按模式（unlearn/inject/edit）过滤可选 Run
- **多 Run 对比表**：指标横向对比，最佳值高亮
- **Metric Direction**：每个指标标注 ↑（higher is better）或 ↓（lower is better），最佳值按方向正确选取

### Tab 3: 🎬 交互演示

预置 Before/After 示例，直观展示 unlearn/inject/edit 效果，支持按类别筛选。

### Tab 4: 🤖 智能助手

重构为三个逻辑区块：

**A. 智能助手区**
- 输入目标描述 → 选择操作类型 → 点击「获取建议」
- Agent 根据 Skills 知识库生成结构化建议（遵循 Output Contract）
- 建议卡片展示：推荐模板 / 方法 / 模型 / 参数 / 理由 / 风险
- Diff 预览：对比当前配置与建议配置的差异
- 一键应用到 Tab 1，应用后可继续追问
- 三种操作层级：仅生成建议 / 生成并预览 / 直接应用
- 无 API Key 时自动降级为规则匹配模式

**B. Skill 模板区**（Accordion）
- 按目标类型筛选 Skill 模板
- 配置预览 + 一键应用

**C. 数据工作区**（Accordion，与 Assistant 解耦）
- 单条输入 / 批量 JSONL 输入
- 数据校验 + 预览 + 保存 + 自动回填到 Tab 1

**Agent 配置**（Accordion）
- 支持 OpenAI / DeepSeek 等 OpenAI 兼容 API
- API Key 持久化到 `.agent_settings.json`（已 gitignore）
- 连接测试

## Assistant 输出合同

智能助手（Agent / 规则引擎）输出遵循统一的结构化合同：

**必填字段**：
- `mode`: unlearn / inject / edit
- `recommended_skill`: skill 模板 ID
- `recommended_method`: 训练方法名
- `recommended_model`: 推荐模型
- `data_plan`: 数据准备建议
- `eval_plan`: 评测套件
- `core_overrides`: Hydra override 字典
- `reasoning_summary`: 推荐理由
- `risk_notes`: 风险提示

**可选字段**：
- `ui_actions`: UI 动作列表
- `dataset_overrides`: 数据集覆盖
- `parameter_overrides`: 参数覆盖
- `warnings`: 警告信息

建议在应用前必须经过 diff 预览，用户确认后方可写入 Tab 1 配置。

## Agent Provider 配置

### OpenAI

```
Provider: openai
Base URL: https://api.openai.com/v1
Model: gpt-4o (推荐)
```

### DeepSeek

```
Provider: deepseek
Base URL: https://api.deepseek.com/v1
Model: deepseek-chat
```

### 其他 OpenAI 兼容 API

任何兼容 OpenAI Chat Completions API 的服务均可使用，设置对应的 Base URL 和 Model 即可。

## 使用示例

### 示例 1: 智能助手推荐 + 应用

1. 打开 Tab 4「智能助手」
2. 选择目标类型「遗忘」，输入描述
3. 点击「获取建议」
4. 查看建议卡片和 diff 预览
5. 点击「应用建议到配置页」
6. 切换到 Tab 1，确认配置后运行

### 示例 2: 手动配置运行

1. Tab 1 左栏：模式选 `Unlearn`，模型选 `Qwen2.5-7B-Instruct`，方法选 `SimNPO`
2. 中栏：按需调整学习率、训练轮数
3. 右栏：确认命令预览，设置 GPU，点击「开始运行」

### 示例 3: 自定义模型

1. Tab 1 左栏：勾选「使用自定义模型」
2. 输入 HuggingFace 模型名（如 `meta-llama/Llama-3.1-8B`）或本地路径
3. 模型自动进入命令预览和应用链路

### 示例 4: 多 Run 结果对比

1. 打开 Tab 2「结果对比」
2. 按模式筛选 Run，勾选需要对比的实验
3. 点击「生成对比」，↑/↓ 标注指标方向，最佳值绿色高亮

## 常见问题

### Q: 如何指定 GPU？

在「环境配置」中设置 `CUDA_VISIBLE_DEVICES`，例如：
- 单卡: `0`
- 多卡: `0,1`

### Q: 智能助手不工作？

确认已在 Tab 4「Agent 配置」中填写 API Key 并测试连接。无 API Key 时自动降级为规则匹配。

### Q: 配置报错怎么办？

1. 检查「命令预览」中的命令是否正确
2. 复制命令到终端手动运行，查看详细错误信息
3. 检查 Hydra 配置文件是否存在

## 目录结构

```
webui/
├── app.py                         # 主入口（4-Tab 布局 + 语言切换）
├── assets/
│   └── custom.css                 # 蓝灰科研工作台主题
├── components/
│   ├── config_panel.py            # Tab 1 左栏：任务配置 + 自定义模型
│   ├── params.py                  # Tab 1 中栏：参数调节
│   ├── run_panel.py               # Tab 1 右栏：运行控制
│   ├── results_compare.py         # Tab 2：结果对比（筛选 + metric direction）
│   ├── interactive_demo.py        # Tab 3：交互演示
│   └── skill_wizard.py            # Tab 4：智能助手 + Skill 模板 + 数据工作区
├── examples/
│   └── unlearn_examples.json      # Tab 3 预置演示数据
├── skills/
│   ├── unlearn_tofu.json          # Skill 模板：遗忘 TOFU
│   ├── unlearn_muse.json          # Skill 模板：遗忘 MUSE
│   ├── inject_alpaca.json         # Skill 模板：注入 Alpaca
│   └── edit_zsre.json             # Skill 模板：编辑 ZSRE
├── tests/
│   ├── test_skills_context.py     # skills 注入层单测
│   ├── test_agent_planner.py      # Agent 合同 / JSON 解析单测
│   ├── test_result_parser.py      # 结果解析 + metric direction 单测
│   └── test_model_adapter.py      # 模型适配层单测
└── utils/
    ├── config_loader.py           # 配置扫描（含 get_eval_runs）
    ├── runner.py                  # 命令执行
    ├── result_parser.py           # 结果解析 + metric direction
    ├── i18n.py                    # 中英文国际化
    ├── ui_state.py                # UI 状态管理
    ├── data_adapter.py            # JSONL 数据解析 / 校验 / 保存
    ├── skill_schema.py            # Skill 结构定义与校验
    ├── skills_context.py          # Skills 注入与上下文装配层
    ├── agent_planner.py           # Agent 输出合同 + system prompt + JSON 解析
    ├── agent_provider.py          # LLM Provider 抽象层
    ├── agent_settings.py          # Agent 配置持久化
    └── model_adapter.py           # 模型元信息 + capability 判断 + 降级策略
```

## 已支持能力

- [x] Unlearn / Inject / Edit / Eval 四种模式配置与运行
- [x] 智能助手（Agent + 规则降级）结构化建议生成
- [x] 建议 diff 预览 + 一键应用到配置页
- [x] 应用后继续追问工作流
- [x] 自定义 HuggingFace 模型 / 本地模型输入
- [x] 模型元信息与 capability 适配层
- [x] Tab 2 结果对比含 metric direction 高亮
- [x] 数据工作区（单条 + 批量 JSONL）
- [x] 蓝灰科研工作台主题
- [x] 中英文切换
- [x] 配置导入 / 导出（YAML）

## 未来规划

- [ ] Tab 3 升级为大模型实时推理演示
- [ ] 多模态模型 / 数据输入输出支持
- [ ] workspace_state / session_store 状态管理
- [ ] 更多 Skill 模板
