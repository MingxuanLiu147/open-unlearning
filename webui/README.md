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
python3 webui/app.py --port 10001

export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1
### 3. 访问界面

启动后，在浏览器中打开：

```
http://localhost:7860
```

## 功能说明

### 任务模式

WebUI 支持四种任务模式：

| 模式 | 功能 | 入口脚本 |
|------|------|----------|
| **Unlearn** | 知识删除/遗忘 | `train.py` |
| **Inject** | 知识注入/微调 | `train.py` |
| **Edit** | 知识编辑 | `train.py` |
| **Eval** | 模型评估 | `eval.py` |

### 配置面板（左侧）

按步骤选择配置：

1. **任务模式**: 选择 Unlearn/Inject/Edit/Eval
2. **实验模板**: 选择预设配置或自定义
3. **模型选择**: 选择 HuggingFace 模型配置
4. **方法选择**: 选择具体算法（如 SimNPO、LoRA、ROME 等）
5. **数据集选择**: 根据模式选择对应数据集
6. **评测套件**: 选择评估指标集合
7. **运行参数**: 设置任务名称、随机种子等

### 运行面板（右侧）

- **命令预览**: 显示等价的 CLI 命令
- **环境配置**: 设置 CUDA 设备
- **运行控制**: 启动/停止任务
- **实时日志**: 显示运行输出
- **结果展示**: 查看评估指标

### 高级参数

展开「高级参数」可调整：

- 学习率、训练轮数、批次大小等训练参数
- 方法特定参数（JSON 格式）
- 额外 Hydra Overrides

### 导入/导出配置

- **导出**: 将当前配置保存为 YAML 文件
- **导入**: 从 YAML 文件恢复配置

## 使用示例

### 示例 1: 运行 Unlearning

1. 选择模式: `Unlearn`
2. 选择模型: `Qwen2.5-7B-Instruct`
3. 选择方法: `SimNPO`
4. 设置任务名称: `my_unlearn_exp`
5. 点击「开始运行」

### 示例 2: 评估已训练模型

1. 选择模式: `Eval`
2. 选择评测套件: `tofu`
3. 选择已保存模型: `saves/unlearn/my_unlearn_exp`
4. 设置任务名称: `eval_my_model`
5. 点击「开始运行」

### 示例 3: 查看评估结果

1. 在「输出目录」中输入路径: `saves/eval/eval_my_model`
2. 点击「加载结果」
3. 查看指标卡片

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
├── app.py              # 主入口
├── components/         # UI 组件
│   ├── config_panel.py # 配置面板
│   ├── run_panel.py    # 运行面板
│   └── params.py       # 参数面板
├── utils/              # 工具模块
│   ├── config_loader.py # 配置加载
│   ├── runner.py       # 命令执行
│   └── result_parser.py # 结果解析
└── README.md           # 本文档
```
