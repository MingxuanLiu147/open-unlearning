# Know-Surgery New UI

Flask + Vue 3 + Vite 架构的知识手术实验平台，替代原有 Gradio 界面。

## 快速启动

### 1. 安装后端依赖

```bash
pip install flask flask-cors pyyaml openai
```

### 2. 安装前端依赖

```bash
cd new_ui/frontend
npm install
```

### 3. 开发模式（前后端分离）

```bash
# 终端 1: 启动 Flask 后端 (端口 18888)
cd open-unlearning
python -m new_ui.backend.app --port 18888

# 终端 2: 启动 Vite 开发服务器 (端口 3000, 自动代理 /api -> 18888)
cd open-unlearning/new_ui/frontend
npm run dev
```

浏览器访问 `http://localhost:3000`

### 4. 生产部署（单进程）

```bash
cd open-unlearning/new_ui/frontend
npm run build   # 产出到 dist/

cd open-unlearning
python -m new_ui.backend.app --port 18888
```

Flask 自动托管 `dist/` 静态文件，访问 `http://localhost 18888`

## 功能概览

| 页面 | 功能 |
|------|------|
| **Workshop** | 卡片式选择模型/方法/数据集，参数面板，命令预览，一键运行 |
| **Monitor** | SSE 实时日志流，运行状态指示，历史记录 |
| **Results** | 多 run 指标对比表格，ECharts 雷达图 |
| **Skills** | YAML 实验模板管理，可视化编辑，一键应用 |
| **Copilot** | 侧栏常驻 Agent (DeepSeek/GPT)，上下文感知对话，流式回复 |

## 通用特性

- 中英双语一键切换 (vue-i18n)
- 深色/浅色主题切换 (CSS Variables)
- IBM Plex Sans 学术字体 + JetBrains Mono 代码字体
- 模式颜色区分：紫(Unlearn) / 青(Inject) / 橙(Edit) / 靛(Eval)
- 多模态 modality 标签预留

## 目录结构

```
new_ui/
├── backend/
│   ├── app.py              # Flask 入口
│   ├── api/                # REST 路由 (config/runner/results/skills/agent/data)
│   ├── services/           # 业务逻辑 (从 webui/utils/ 适配)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── views/          # Workshop / Monitor / Results / Skills
│   │   ├── components/     # 按功能分组的 Vue 组件
│   │   ├── stores/         # Pinia 状态管理
│   │   ├── api/            # 后端 API 封装
│   │   ├── i18n/           # 中英文翻译
│   │   ├── styles/         # CSS 变量 + 主题 + Element Plus 覆盖
│   │   └── layouts/        # MainLayout (Topbar + Copilot)
│   ├── vite.config.ts
│   └── package.json
└── skills/                 # 预定义 Skill YAML 模板
```

## Copilot Agent 配置

Copilot 支持 OpenAI 兼容 API（GPT / DeepSeek）。配置方式：

```bash
# 环境变量
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.gpt.ge/v1/
export OPENAI_MODEL=gpt-4o

# 或 DeepSeek
export DEEPSEEK_API_KEY=sk-xxx
```

也可在运行后通过 `PUT /api/agent/config` 动态配置。

## 与原有代码的关系

- `src/`、`configs/`、`webui/` 完全不动，原有 Gradio UI 仍可独立使用
- `new_ui/backend/services/` 从 `webui/utils/` 适配而来，共享同一套 configs 和 saves 目录
