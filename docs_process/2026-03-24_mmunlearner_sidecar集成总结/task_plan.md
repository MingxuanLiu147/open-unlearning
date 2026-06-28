# Task Plan: MMUnlearner Sidecar 后续实验

## Goal
完成 MMUnlearner sidecar 的 plan 复核，并继续打通其他方法的最小实验链路，将结果持续写入当前文档目录。

## Phases
- [x] Phase 1: 建立计划与记录文件
- [x] Phase 2: 检查代码、配置、环境与已有产物
- [x] Phase 3: 执行最小训练实验并收集日志
- [x] Phase 4: 分析结果，必要时调整配置或代码
- [x] Phase 5: 回写实验记录与阶段结论
- [x] Phase 6: 复核原始 plan 完成度并形成书面结论
- [x] Phase 7: 扩展 runner，打通其他方法的 smoke 实验链路
- [x] Phase 8: 回写其他方法链路实验结果与剩余缺口
- [x] Phase 9: 稳定正式入口并将进度回写集成补完计划
- [ ] Phase 10: 补齐 CLEAR 数据适配与 evaluator，并做 smoke 验证
- [ ] Phase 11: 补齐 MLLMU evaluator 的 test/classification/generation 缺口

## Key Questions
1. 当前 `src/mm_train.py` 最小可运行配置是什么？
2. 本机当前是否有可用 GPU 供 Qwen3-VL-4B + LoRA 训练？
3. 现有代码在最小训练链路上是否仍存在环境或实现阻塞？
4. 本轮实验结束后，文档目录下需要沉淀哪些复现信息？
5. 原 plan 中哪些任务只是“文件已存在”，哪些已经被实验验证？
6. `MMGradDiff / MMKLMin / MMNPO / MMUnlearner` 哪些链路能真正跑通？

## Decisions Made
- 使用当前文档目录作为实验记录主目录，集中放置计划、笔记、实验记录和总结。
- 优先从现有总结中定义的 `MMGradAscent + forget_10` 最小训练闭环开始验证。
- 在重新训练前，先使用现有 `saves/mm_unlearn/test_ga_f10` 做一次评估，判断当前闭环是否已经可用。
- 在当前沙箱内，不再使用 `python src/mm_*.py`、`CUDA_VISIBLE_DEVICES=*`、`python -u`、stdout/stderr 重定向来跑 GPU 实验。
- 后续 MM sidecar 实验统一通过当前文档目录下的 runner 脚本启动。
- 其他方法先做 smoke 级链路验证，避免在未确认入口稳定前直接长跑全量实验。
- `MMUnlearner` 实验前需要先确认方法语义是否仍与当前 LoRA 训练模式兼容，不能只看“命令跑通”。
- 正式入口默认改为 `device=cuda` + `require_cuda=true`，避免入口在 CUDA 不可用时静默退回 CPU。
- CLEAR Phase 2 先按“工程化适配本地 parquet 目录”落地，不再强依赖原仓库 `load_dataset/load_from_disk` 的脚本式入口。
- MLLMU Phase 3 先完成 `classification + generation + test split` 最小闭环，再补 `fill_in_the_blank / real / few-shot`。

## Errors Encountered
- `nvidia-smi` 在当前沙箱内无法访问 NVIDIA driver，需进一步确认是否需要沙箱外执行。
- 已确认使用精简查询参数后，`nvidia-smi` 可正常返回 10 张 24GB GPU 的空闲状态。
- 现有 `mm_eval.py` 初始实现未显式将模型迁移到 CUDA，导致评估实际落在 CPU 上，已修复。
- 使用训练配置直接评估 merged checkpoint 会重复注入 LoRA；本轮实验先通过命令行覆盖 `model.lora.enabled=false` 规避。
- 启动方式本身会影响 CUDA 可见性：
  - `python src/mm_eval.py ...` / `python src/mm_train.py ...` 会让 `torch.cuda.is_available()` 变成 `False`
  - `CUDA_VISIBLE_DEVICES=*` 会让当前进程失去 CUDA
  - `python -u` 会让当前进程失去 CUDA
  - stdout/stderr 重定向到文件会让当前进程失去 CUDA
- Hugging Face cache 在 `/netcache/huggingface` 上存在写权限告警，但未阻断本轮训练与评估。
- `Qwen/Qwen3-VL-2B-Instruct` 当前 cache 只有 `config.json`，权重不完整，无法作为离线最小模型使用。
- `PYTORCH_ALLOC_CONF=expandable_segments:True` 在当前启动方式下会让 CUDA 不可用。
- `MMKLMin` 在 `max_forget_samples=8 + max_retain_samples=16` 的 smoke 配置下仍会 OOM。
- `SFRon` mask 生成初版会在 forget/preserve 两段 Fisher 之间残留梯度显存，已通过 `zero_grad(set_to_none=True) + empty_cache()` 修复。
- `MMUnlearner` 的 `forget_alpha` 初版为死参数，已修复为真正参与 forget loss。
- 直接入口的 CUDA 问题暂时按“导入顺序 + 显式失败”治理：先 `import torch` 再 `import hydra`，并在默认 CUDA 失败时直接报错提示。
- CLEAR 当前的核心风险不再是“没有代码”，而是“新接入 evaluator 还没有做端到端 smoke 验证”。
- MLLMU evaluator 当前的新风险是 few-shot 仍未实现，因此 `shot_num != zero_shot` 目前会显式报错，而不是给出误导性结果。

## Status
**In Phase 11** - 已完成 CLEAR 第一轮接入，并开始补齐 MLLMU evaluator 的正式 split 与指标逻辑。
