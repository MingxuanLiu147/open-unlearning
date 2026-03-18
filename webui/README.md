# Know-Surgery WebUI 使用指南

Know-Surgery WebUI 是一个基于 Gradio 的图形化界面，用于配置和运行大模型知识可控更新任务。

## 快速启动

### 1. 安装依赖

确保已安装项目基础依赖，然后安装 Gradio：

```bash
pip install gradio>=4.0.0
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
# 指定端口
python webui/app.py --port 8080

# 创建公共链接
python webui/app.py --share
```

env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u NO_PROXY -u no_proxy \
  python3 webui/app.py --port 19191
python3 webui/app.py --port 10001

 
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
| 导入/导出配置 | — | 结果展示 |

**任务模式**：Unlearn（遗忘）· Inject（注入）· Edit（编辑）· Eval（评估）

### Tab 2: 📊 结果对比

勾选多个 `saves/` 下的 checkpoint run，一键生成指标横向对比表格（最优值高亮）。

### Tab 3: 🎬 交互演示

预置 Before/After 示例，直观展示 unlearn/inject/edit 效果，支持按类别筛选。

### Tab 4: 🧙 智能向导

选择目标 → 自动匹配推荐 Skill 模板 → 一键应用配置到 Tab 1。

## 使用示例

### 示例 1: 快速运行 Unlearning（向导方式）

1. 打开 Tab 4「智能向导」
2. 选择目标「🗑️ 我想遗忘某类知识」
3. 点击推荐模板「遗忘 TOFU 基准」
4. 点击「⚡ 一键应用到配置页」
5. 切换到 Tab 1，确认配置后点击「▶️ 开始运行」

### 示例 2: 手动配置运行 Unlearning

1. Tab 1 左栏：模式选 `Unlearn`，模型选 `Qwen2.5-7B-Instruct`，方法选 `SimNPO`
2. 中栏：按需调整学习率、训练轮数
3. 右栏：确认命令预览，设置 GPU，点击「▶️ 开始运行」

### 示例 3: 多 Run 结果对比

1. 打开 Tab 2「结果对比」
2. 勾选多个 `unlearn/my_simnpo_run_v3 @ checkpoint-N`
3. 点击「📊 生成对比」查看指标表格

## 常见问题

### Q: 如何指定 GPU？

在「环境配置」中设置 `CUDA_VISIBLE_DEVICES`，例如：
- 单卡: `0`
- 多卡: `0,1`

### Q: 如何查看完整日志？

运行日志实时显示在「运行日志」区域。如需查看完整日志，可在输出目录中找到 `.hydra/` 子目录。

### Q: 配置报错怎么办？

1. 检查「命令预览」中的命令是否正确
2. 复制命令到终端手动运行，查看详细错误信息
3. 检查 Hydra 配置文件是否存在

## 目录结构

```
webui/
├── app.py                      # 主入口（4-Tab 布局）
├── assets/
│   └── custom.css              # Teal 主题样式
├── components/
│   ├── config_panel.py         # Tab 1 左栏：任务配置
│   ├── params.py               # Tab 1 中栏：参数调节
│   ├── run_panel.py            # Tab 1 右栏：运行控制
│   ├── results_compare.py      # Tab 2：结果对比
│   ├── interactive_demo.py     # Tab 3：交互演示
│   └── skill_wizard.py         # Tab 4：智能向导
├── examples/
│   └── unlearn_examples.json   # Tab 3 预置演示数据
├── skills/
│   ├── unlearn_tofu.json       # Tab 4 模板：遗忘 TOFU
│   ├── unlearn_muse.json       # Tab 4 模板：遗忘 MUSE
│   ├── inject_alpaca.json      # Tab 4 模板：注入 Alpaca
│   └── edit_zsre.json          # Tab 4 模板：编辑 ZSRE
└── utils/
    ├── config_loader.py        # 配置扫描（含 get_eval_runs）
    ├── runner.py               # 命令执行
    └── result_parser.py        # 结果解析（含多 run 对比）
```
