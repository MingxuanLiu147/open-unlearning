# Know-Surgery 审查笔记

## Hubble：配置审查
- `configs/experiment/edit/**/*.yaml` 全部缺少 `# @package _global_`。
- `configs/experiment/inject/**/*.yaml` 全部缺少 `# @package _global_`。
- `configs/experiment/inject/alpaca/default.yaml`、`adalora.yaml`、`dora.yaml` 的 `method_args` 位于顶层，应迁到 `trainer.method_args`。
- `configs/mm_edit.yaml` 引用不存在的 `model/Qwen2-VL`，应改为现有 `Qwen2VL-2B`。
- 子代理用 Hydra compose 只读验证过：edit/inject 会出现 `model@experiment.model` 错误，`mm_edit` 会出现找不到 `model/Qwen2-VL`。

## Hooke：源码架构审查
- `src/train.py` 的 edit 模式可以抽成 `run_edit_mode(...)`，继续复用 `trainer.edit.pipeline`。
- `src/mm_train.py` 的 `MM_TRAINER_REGISTRY` 应迁到可复用注册表，暂不强行接入文本 `load_trainer`。
- `src/mm_eval.py` 的 evaluator 分发可先抽出 MM evaluator factory/export，保持 benchmark 配置稳定。
- `MQuAKEDataset` 已实现但未在 `src/data/__init__.py` 注册。
- `get_mm_model` 可从 `src/model/__init__.py` 导出，减少入口文件对内部模块路径的直接依赖。

## Ampere：文档、测试和产物边界审查
- `.gitignore` 缺少 frontend、LaTeX、Hydra 输出相关规则。
- `tests/test_inject_dataset.py` 仍引用已删除的 `webui.utils.config_loader`。
- `new_ui/README.md` 仍声称旧 `webui/` 可独立使用，与当前 worktree 不一致。
- `docs/multimodal_editing_guide.md` 存在论文复现表述过强的问题，需要加默认配置边界说明。
- 多个算法配置引用论文但没有 reproduction status 注释。
