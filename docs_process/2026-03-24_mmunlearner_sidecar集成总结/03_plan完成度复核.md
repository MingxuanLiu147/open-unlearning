# MMUnlearner Sidecar Plan 完成度复核（2026-03-24）

## 1. 总结论

原始 plan **没有完全完成**。

更准确地说：
- sidecar 目录骨架、`MMGradAscent` 主链路、`MMGradDiff / MMKLMin / MMNPO` 的 smoke 训练与 forget-only 评估已经具备。
- `SFRon -> mask.pt -> MMUnlearner trainer` 的 I/O 链路也已被 smoke 验证。
- 但从“plan 是否完整达成”和“代码是否已经达到方法效果”两个角度看，当前都还不能下完全完成的结论。

## 2. 完成度分级

### 2.1 可判定为已完成或基本完成

- `model-adapter`
  - `src/model/multimodal.py` 已支持 `Qwen2VL / Qwen3VL` 适配与 LoRA 注入。
- `multimodal-data`
  - `src/data/multimodal.py` 已适配 MLLMU parquet 数据格式。
- `mm-base-trainer`
  - `src/trainer/unlearn/mm_base.py` 已可稳定驱动 `MMGradAscent` 与其他 sidecar trainer。
- `mm-grad-ascent`
  - 最小训练与 forget-only 评估均已实测跑通。
- `hydra-configs`
  - `configs/mm_train.yaml`、`configs/mm_eval.yaml` 以及对应 trainer/model/eval 配置均已落地。

### 2.2 只能判定为部分完成

- `mm-methods-expand`
  - `MMGradDiff`、`MMKLMin`、`MMNPO` 已有 smoke 级训练和 forget-only eval。
  - 但都只是极小样本实验，不足以证明方法效果达到上游目标。
- `mask-pipeline`
  - `SFRon` Fisher 与 `mask.pt` 生成已跑通。
  - 但需要本地 snapshot、`disable_lora=true`、极小样本和显式显存清理，稳定性仍有限。
- `mm-entry`
  - `src/mm_train.py` / `src/mm_eval.py` 代码已存在。
  - 但在当前环境下，直接入口并不可靠，仍需依赖文档目录下的 CUDA-safe runner。
- `multimodal-eval`
  - `MLLMU` 的 forget/retain 基础评估可用。
  - 但 `CLEAR` 仍是占位实现，`test` split 也未补齐。
- `MMUnlearner`
  - 训练入口和产物链路已 smoke 跑通。
  - 但方法语义仍存在关键风险，暂不能算真正完成。

### 2.3 明确未完成

- `CLEAR` evaluator 完整迁移
  - `src/evals/clear.py` 仍是占位实现。
- 完整 benchmark 闭环
  - 仍缺 `retain/test/generation` 的系统性评估。
- 原始 MVP 目标验证
  - plan 写的是 `Qwen2VL-2B + 3090` 路线。
  - 当前实际跑通的是 `Qwen3-VL-4B` 与文档目录下的 sidecar runner。

## 3. 代码与实验对应判断

### 3.1 `MMGradAscent`

这条链路是目前最扎实的：
- 现有 checkpoint `test_ga_f10` 已可评估。
- 新最小训练 checkpoint `test_ga_f10_bs1_acc8_2026-03-24` 已可训练并保存。
- 这证明 `MMGradAscent` 至少已经做到：
  - 训练可运行
  - 参数可更新
  - 模型行为可改变

但从 forget-only 结果看：
- `test_ga_f10`: `0.148` (`37/250`)
- 新最小 GA checkpoint: `0.248` (`62/250`)

如果把 forget split 上更低准确率理解为“遗忘更充分”，新最小 GA 结果并没有优于已有 checkpoint。

### 3.2 `MMGradDiff / MMKLMin / MMNPO`

这三条链路现在可以说是“代码文件存在”之外，又多了一层 smoke 实证：

- `MMGradDiff smoke checkpoint`: `0.276` (`69/250`)
- `MMKLMin smoke checkpoint`: `0.260` (`65/250`)
- `MMNPO smoke checkpoint`: `0.252` (`63/250`)

但要注意：
- 这些都是极小样本 smoke 训练。
- 结果只能支持“链路可跑”，不能支持“方法效果成立”。

### 3.3 `MMUnlearner`

这条链路本轮补了两个关键进展：

1. `forget_alpha` 已从死参数修成真正参与 forget loss。
2. `SFRon` mask 生成与 `MMUnlearner` smoke 训练都能完成。

对应 smoke 结果：
- `MMUnlearner smoke checkpoint`: `0.264` (`66/250`)

但是，这里有一个比数值更重要的实现问题：

- 当前默认训练路径是 LoRA。
- LoRA 下真正 trainable params 只有约 `0.8993%`。
- 本轮成功生成的 mask 来自 `disable_lora=true` 的 base model。
- 这意味着 mask 的参数名和语义主要对应 base weights，而不是当前实际更新的 LoRA adapter 参数。

因此，当前 `MMUnlearner` 最多只能说：
- `mask.pt` 能生成
- `mask` 能加载
- trainer 能跑
- checkpoint 能保存
- eval 能完成

但还不能说：
- 这个实现已经真正实现了上游 MMUnlearner 的“选择性梯度更新”方法效果。

## 4. 本轮新增修复

本轮为继续实验新增了三处代码修复：

1. `src/trainer/unlearn/mm_grad_diff.py`
   - 将 forget/retain loss 拆成可覆写方法，便于子类定制。

2. `src/trainer/unlearn/mmunlearner.py`
   - 让 `forget_alpha` 真正参与 `compute_forget_loss()`。

3. `src/trainer/unlearn/sfron.py`
   - 在 Fisher 计算结束后显式释放梯度缓存，避免 preserve fisher 阶段的边缘 OOM。

这些修复都已经通过实际实验路径验证到“至少能推进下一步”。

## 5. 环境结论

当前环境对多模态 sidecar 有额外限制：

- 直接运行 `python src/mm_train.py` / `python src/mm_eval.py` 会导致 CUDA 不可用。
- `CUDA_VISIBLE_DEVICES=*` 会导致 CUDA 不可用。
- `python -u` 会导致 CUDA 不可用。
- stdout/stderr 重定向会导致 CUDA 不可用。
- `PYTORCH_ALLOC_CONF=expandable_segments:True` 在本环境下也会破坏当前可用的 CUDA-safe 启动方式。
- `Qwen3-VL-2B-Instruct` 当前 cache 不完整，只有 `config.json`，不能离线用于最小模型实验。

所以本轮实际可用的稳定策略是：
- repo root 启动
- 普通 `.venv/bin/python -c ...`
- 先导入 `torch`
- 再导入文档目录下 runner
- 必要时优先使用本地 snapshot 路径

## 6. 最终判断

如果问题是：

“`/home/liumingxuan/.cursor/plans/mmunlearner集成计划_open-unlearning版_5d0acad7.plan.md` 是否已经完全完成？”

答案是：
- **没有完全完成。**

如果问题是：

“根据这轮最小实验，代码是否已经真正达到 MMUnlearner 及其相关方法的效果？”

答案是：
- **也不能这么说。**

更准确的结论是：
- `MMGradAscent` 主链路已真实可跑。
- `MMGradDiff / MMKLMin / MMNPO` 已达到 smoke 级可运行。
- `SFRon + MMUnlearner` 已达到 I/O 链路级可运行。
- 但完整 benchmark、完整 evaluator、方法语义一致性、以及真正的 forget/retain/generation 效果验证，都还没有完成。
