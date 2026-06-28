# Notes: MMUnlearner Sidecar 后续实验

## Environment
- 工作目录：`/home/liumingxuan/open-unlearning`
- 文档目录：`/home/liumingxuan/open-unlearning/docs_process/2026-03-24_mmunlearner_sidecar集成总结`

## Initial Findings
- 现有总结文档已给出最小训练目标：`MMGradAscent + MLLMU forget_10`
- 当前文档目录内仅有总结文档，尚无实验计划、实验记录和日志汇总文件
- 沙箱内执行 `nvidia-smi` 失败，报错为无法与 NVIDIA driver 通信
- 使用 `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits` 可正常返回 10 张 24GB GPU，当前几乎空闲
- `src/mm_train.py` 通过 Hydra 驱动，当前最小训练参数可直接覆盖为 `model=Qwen3-VL-4B data.forget_split_ratio=10 data.batch_size=1 trainer=MMGradAscent trainer.args.gradient_accumulation_steps=8 trainer.args.num_epochs=1`
- `saves/mm_unlearn/test_ga_f10` 已存在完整 checkpoint 文件，大小约 8.3G，且 `config.json` 显示为 `Qwen3VLForConditionalGeneration`
- `src/evals/mllmu.py` 当前会在 `forget`/`retain` 两个 split 上执行 classification 和 generation；为先验证闭环，优先做 classification
- `mm_eval.py` 原始实现只加载模型并 `model.eval()`，没有显式 `.to("cuda")`，会导致 `inputs.to(model.device)` 跟着落到 CPU
- 评估 merged checkpoint 时若沿用 `configs/model/Qwen3-VL-4B.yaml`，会把新的 LoRA adapter 再挂到已合并模型上；实验命令需覆盖 `model.lora.enabled=false`
- 在当前沙箱内，以下做法会导致 `torch.cuda.is_available()` 变为 `False`：
  - 直接执行 `python src/mm_train.py ...` 或 `python src/mm_eval.py ...`
  - 设置 `CUDA_VISIBLE_DEVICES=*`
  - 使用 `python -u`
  - 将 stdout/stderr 重定向到文件
- 可工作的方式是：在 repo root 使用普通 `python -c ...`，先导入 `torch`，再导入文档目录下的 runner 模块并调用 `main()`

## Results
- 现有 checkpoint `saves/mm_unlearn/test_ga_f10` 在 `forget_10` classification 上的结果：
  - `accuracy = 0.148`
  - `correct = 37`
  - `total = 250`
- 最小训练复现实验成功完成：
  - 配置：`MMGradAscent + forget_10 + batch_size=1 + grad_accum=8 + num_epochs=1 + lr=1e-4`
  - 训练数据：forget `50` 条，retain `448` 条
  - 训练结果：`Epoch 1/1 - avg loss: -37.8649`
  - 产物：`saves/mm_unlearn/test_ga_f10_bs1_acc8_2026-03-24`
- 新 checkpoint `saves/mm_unlearn/test_ga_f10_bs1_acc8_2026-03-24` 在 `forget_10` classification 上的结果：
  - `accuracy = 0.248`
  - `correct = 62`
  - `total = 250`
- 已有 `MMGradDiff` smoke checkpoint 在 `forget_10` classification 上的结果：
  - `accuracy = 0.276`
  - `correct = 69`
  - `total = 250`
- `MMKLMin` smoke 训练与评估已完成：
  - 较大 smoke 配置 `8/16` 仍 OOM
  - 成功配置为 `max_forget_samples=2 + max_retain_samples=4`
  - `forget_10` classification：`accuracy = 0.260`，`correct = 65`
- `MMNPO` smoke 训练与评估已完成：
  - 成功配置为 `max_forget_samples=2 + max_retain_samples=4`
  - `forget_10` classification：`accuracy = 0.252`，`correct = 63`
- `SFRon` mask 生成已完成：
  - 需要使用本地 snapshot 路径
  - 需要 `disable_lora=true`
  - 成功配置为 `max_forget_samples=1 + max_retain_samples=1`
  - `mask_entries = 713`
  - `sparsity = 48.77%`
- `MMUnlearner` smoke 训练与评估已完成：
  - 已修复 `forget_alpha` 死参数问题
  - 训练入口、mask 加载、checkpoint 保存、forget-only eval 全部跑通
  - `forget_10` classification：`accuracy = 0.264`，`correct = 66`
- 初步判断：
  - 从 forget-only classification 看，新跑出的最小 checkpoint 比现有 `test_ga_f10` 更高准确率，说明它至少没有表现出更强的遗忘效果。
  - 但目前尚未补 retain split 和 generation 结果，因此只能做阶段性判断。
  - 对其他方法也一样：这轮新增结果更适合作为“链路可运行”证据，而不是“方法已经达到上游/论文效果”的证据。
  - `MMUnlearner` 仍有方法语义风险：当前默认 LoRA 训练下，base-weight mask 很可能约束不到真正的 LoRA 可训练参数。

## Open Questions
- 当前 repo 下是否已有 `saves/mm_unlearn/*` 相关产物可复用
- 当前 `.venv`、数据路径、模型路径是否齐备
- 是否需要先做一次 dry-run / import check 再发起长时训练
- 是否需要为 repo 正式入口补一个“不会丢 CUDA”的 root-level launcher，而不是继续依赖文档目录中的实验 runner
- 是否要补做 retain-only / full MLLMU 评估，以判断新 checkpoint 是否在 forget-retain tradeoff 上更合理
- 是否要把 `MMUnlearner` 的 mask 逻辑改成真正面向 LoRA trainable params，而不是只面向 base weights
- 是否要补一个本地 snapshot / offline 优先的多模态模型加载策略，避免 `.locks` 和联网探测问题
