# Know-Surgery 代码库收敛任务计划

## 目标
把当前 fork 从“研究原型集合”整理成一个更清晰、可控的大模型编辑库，统一 `unlearn` / `inject` / `edit` / `multimodal` 的配置、入口和文档边界。

本轮工作优先修复会直接失败的问题，再做低风险结构收敛；不删除用户已有数据、checkpoint、下载数据集或历史实验结果。

## 原始计划完整归档

### Summary
目标是把当前 fork 从“研究原型集合”整理成一个更清晰的可控大模型编辑库，统一 unlearn / inject / edit / multimodal 的配置、入口和文档边界。修改分三步推进：先修会直接失败的问题，再清理冗余与文档，再做架构收敛。

### Key Changes

#### 修复当前硬错误
- 给所有 `configs/experiment/edit/**` 和 `configs/experiment/inject/**` 实验配置补 `# @package _global_`，确保 Hydra override 作用到全局。
- 把 inject 实验里的顶层 `method_args` 全部迁到 `trainer.method_args`。
- 将 `configs/mm_edit.yaml` 默认模型从不存在的 `Qwen2-VL` 改为现有 `Qwen2VL-2B`。
- 将 `tests/test_inject_dataset.py` 的旧 `webui.utils.config_loader` 导入迁到 `new_ui.backend.services.config_loader`。
- 更新 `new_ui/README.md` 中“旧 webui 仍可用”的过期说明。

#### 清理仓库冗余和产物边界
- 更新 `.gitignore`，覆盖 `new_ui/frontend/node_modules/`、`*.tsbuildinfo`、LaTeX 编译产物、Hydra/实验输出、`.pytest_cache/`、本地缓存和临时日志。
- 不删除 `data/`、`saves/` 中用户已有资产；只确保后续不进入版本控制。
- 将根目录过程文档和调研文档归类为“archive/working notes”，保留 `README.md`、`CLAUDE.md`、`docs/` 作为权威说明。
- 明确 `.claude/.codex/.cursor/.gemini` 是 agent 工具配置；若保留多套，只在文档中说明用途，否则后续单独做一次精简。

#### 收敛核心架构
- 保持 `src/train.py` 作为文本 `train/unlearn/inject/edit` 统一入口，但把 `mode == "edit"` 特判封装成独立 runner 函数，避免主入口继续膨胀。
- 把 `trainer/data/eval/model` 注册逻辑整理为一致的注册模块，保留现有 handler 名称不变，避免破坏现有配置。
- 暂时保留 `src/mm_train.py` / `src/mm_eval.py` 作为 multimodal sidecar，但把硬编码 `MM_TRAINER_REGISTRY` 抽到可复用注册表，为后续统一入口铺路。
- 给算法配置加元数据字段，例如 `paper_repro: true|false`、`implementation_status: upstream|adapted|smoke_only`，先用于文档和 UI 展示，不改变训练逻辑。

#### 标注文献/参数差异
- 在算法文档或配置注释中标明：unlearning 基本沿用 upstream；ROME 接近官方 GPT-J 默认；MEMIT/MEND/RMU/inject 多数为 adapted/default，不应声明论文复现。
- 为每类算法建立“推荐配置”和“论文复现配置”两层结构：默认配置保证通用可跑，repro 配置单独保存并注明目标模型/数据集。

### Test Plan
- 配置组合测试：
  - 用 Hydra compose 覆盖检查 `unlearn/tofu/default`、`edit/counterfact/default`、`inject/alpaca/default`、`mm_edit.yaml` 均能组合成功。
  - 检查 `trainer.method_args` 中的 LoRA、DoRA、AdaLoRA 参数实际生效。
- 单元测试：
  - 跑 `PYTHONPATH=src .venv/bin/pytest tests/test_inject_dataset.py -q`。
  - 跑现有 edit/inject smoke tests：`tests/test_edit_pipeline.py`、`tests/test_edit_eval.py`、`tests/test_all_editors_smoke.py`、`tests/test_mm_editors_smoke.py`。
- 静态质量：
  - 跑 `make quality`。
  - 对 `src/train.py`、registry、config loader 相关文件单独跑 `ruff check`。
- 行为验收：
  - `python src/train.py --config-name=edit.yaml experiment=edit/counterfact/default task_name=SMOKE max_edits=1` 能进入 edit 流程。
  - `python src/train.py --config-name=inject.yaml experiment=inject/alpaca/default task_name=SMOKE trainer.args.max_steps=1` 能正确读取 LoRA 参数。
  - `python src/mm_train.py --config-name=mm_train.yaml --help` 和 `python src/mm_eval.py --config-name=mm_eval.yaml --help` 不因配置缺失失败。

### Assumptions
- 第一轮只做结构收敛和显性错误修复，不删除用户数据、checkpoint、下载数据集或历史实验结果。
- `new_ui/` 是未来唯一 UI，旧 `webui/` 不再恢复。
- 多模态入口暂不强行并入 `src/train.py`，先通过注册表和配置命名收敛，避免一次改动影响 CUDA/processor 特殊逻辑。
- 算法实现不在本轮重写为论文级复现；本轮只标清状态，防止误用实验结果。

## 本轮执行边界
- 先写中文计划与说明文档，再等待用户确认。
- 不修改 `.claude/`、`.codex/`、`.cursor/`、`.gemini/` 等 agent 工具目录配置。
- 只在文档中说明这些 agent 工具目录的定位和边界。
- 不删除 `data/`、`saves/`、checkpoint、下载数据集、历史日志或用户已有过程文档。
- 当前 worktree 已有大量用户改动；本轮只做明确范围内的增量修改，不回滚非本任务文件。

## 并行子代理审查结论

### 配置审查
- `configs/experiment/edit/**/*.yaml` 缺少 `# @package _global_`，通过 `experiment=...` 组合时会被挂到 `experiment.*` 下，导致 `model@experiment.model` 这类 override 失败。
- `configs/experiment/inject/**/*.yaml` 同样缺少 `# @package _global_`。
- `configs/experiment/inject/alpaca/default.yaml`、`adalora.yaml`、`dora.yaml` 的 `method_args` 在顶层，但运行时读取的是 `cfg.trainer.method_args`。
- `configs/mm_edit.yaml` 默认引用 `model/Qwen2-VL`，仓库没有对应 `configs/model/Qwen2-VL.yaml`，应改为现有 `Qwen2VL-2B`。

### 源码架构审查
- `src/train.py` 中 `mode == "edit"` 分支嵌在通用训练流程里；低风险做法是抽出 `run_edit_mode(...)`，继续复用 `trainer.edit.pipeline`。
- `src/mm_train.py` 内部有独立 `MM_TRAINER_REGISTRY`；低风险做法是把多模态 trainer 注册表移到可复用模块，先不强行并入文本 `load_trainer`。
- `src/mm_eval.py` 有 benchmark 形态的 evaluator 分发；低风险做法是先提供 MM evaluator factory/export，保持现有配置形态不变。
- `MQuAKEDataset` 已实现但未注册到 `src/data/__init__.py`，而配置通过 handler 引用它，这是一个明确导出缺口。
- `get_mm_model` 目前被 sidecar 入口直接导入；可以先从 `src/model/__init__.py` 暴露 wrapper/export，不急于合并进文本 `get_model`。

### 文档、测试和仓库边界审查
- `.gitignore` 缺少 `new_ui/frontend/node_modules/`、`new_ui/frontend/dist/`、`*.tsbuildinfo`、LaTeX 编译产物、`outputs/`、`multirun/`、`.hydra/` 等规则。
- `tests/test_inject_dataset.py` 还引用已删除的 `webui.utils.config_loader`；迁到 `new_ui.backend.services.config_loader` 时，需要适配新 loader 的 project root 语义。
- `new_ui/README.md` 仍说旧 `webui/` 可独立使用，但当前 worktree 中 `webui/` 已删除，应改为 `new_ui` 是活跃 UI。
- `docs/multimodal_editing_guide.md` 和若干算法配置引用论文但没有说明 reproduction 状态；需要加“引用论文仅作方法来源，默认配置是集成预设，不代表论文复现结果”的边界说明。

## 详细修改方案

### 1. Hydra 配置硬错误
- 在所有 `configs/experiment/edit/**/*.yaml` 和 `configs/experiment/inject/**/*.yaml` 第一行加入 `# @package _global_`。
- 将 inject 实验中的顶层 `method_args` 缩进到 `trainer.method_args` 下。
- 修改 `configs/mm_edit.yaml` 默认 model group：`Qwen2-VL` -> `Qwen2VL-2B`。

### 2. UI 和测试迁移
- 把 `tests/test_inject_dataset.py` 的导入改为 `new_ui.backend.services.config_loader.ConfigLoader`。
- 根据新 loader 的 project root 语义调整测试 fixture 路径。
- 如果新 loader 当前无法识别 inject 数据集分类，则补一个最小兼容能力，让测试继续覆盖 `train` / `eval` / `analysis` 分类。
- 将 `new_ui/README.md` 的旧 UI 说明改成：`new_ui` 是当前活跃 UI，相关后端服务已从旧 UI 自包含迁移。

### 3. `.gitignore` 和产物边界
- 追加忽略规则：`new_ui/frontend/node_modules/`、`new_ui/frontend/dist/`、`*.tsbuildinfo`。
- 追加 LaTeX 产物规则：`*.aux`、`*.bbl`、`*.blg`、`*.fls`、`*.fdb_latexmk`、`*.synctex.gz`、`*.toc` 等。
- 追加 Hydra/实验输出规则：`outputs/`、`multirun/`、`.hydra/`。
- 保持 `data/`、`saves/` 不删除、不移动。

### 4. 入口和 registry 收敛
- 在 `src/train.py` 中抽出 `run_edit_mode(cfg, model, tokenizer)` 或相近签名，主函数只保留模式分发。
- 保留现有 `trainer.edit.pipeline`，不改变编辑算法行为。
- 在 trainer 侧新增或整理多模态注册表导出，供 `src/mm_train.py` 使用，先不把 MM trainer 强行接入文本 `load_trainer`。
- 在 data registry 中补上 `MQuAKEDataset` 导出。
- 在 model registry 中导出 `get_mm_model` wrapper，降低 sidecar 对内部模块路径的耦合。

### 5. 文献复现边界和算法元数据
- 在代表性算法配置中加入非行为字段：
  - `paper_repro: false`
  - `implementation_status: upstream|adapted|smoke_only`
- 对 ROME 这类接近官方 GPT-J 默认的配置标明更接近论文默认；对 MEMIT/MEND/RMU/inject/MM 方法标明多为 adapted/default。
- 在 `docs/multimodal_editing_guide.md` 中软化“完整框架 / 前沿论文标准 / 全部可跑”的表述，改为“集成覆盖 / smoke-ready 目标 / 待完整论文级验证”。
- 增加或更新 agent 工具目录说明文档，明确 `.claude/.codex/.cursor/.gemini` 是工具配置，不是训练、评估或算法运行时依赖。

### 6. 验证
- 先跑 Hydra compose 检查：`unlearn/tofu/default`、`edit/counterfact/default`、`inject/alpaca/default`、`mm_edit.yaml`。
- 跑 `PYTHONPATH=src .venv/bin/pytest tests/test_inject_dataset.py -q`。
- 跑 edit/MM smoke tests：`tests/test_edit_pipeline.py`、`tests/test_edit_eval.py`、`tests/test_all_editors_smoke.py`、`tests/test_mm_editors_smoke.py`。
- 跑针对性 lint：`ruff check src/train.py src/trainer src/data/__init__.py src/model/__init__.py src/mm_train.py tests/test_inject_dataset.py`。
- 最后视环境情况跑 `make quality`；如果暴露既有失败，单独记录，不混入本轮修复。

## 待用户确认的问题
1. 是否同意本轮按“硬错误修复 + 低风险架构收敛 + 文档边界说明”完整推进，而不是只做硬错误？
2. 是否确认 `.claude/.codex/.cursor/.gemini` 本轮只写入文档说明，不做配置精简或删除？
3. 是否确认根目录过程文档和调研文档本轮不移动，只先在文档/计划中标注为 working notes，后续单独做归档？

## 状态
**等待用户确认**：确认后再开始修改代码、配置、测试和 README。
