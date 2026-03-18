# 知识编辑运行链路修改建议

本文档针对 `src/train.py` 当前无法真正执行知识编辑方法的问题，给出一份先讨论、后实施的修改建议。目标是让以下命令具备可预期行为：

```bash
PYTHONPATH=src .venv/bin/python src/train.py --config-name=edit.yaml ...
```

本文档只给建议，不直接实施代码修改。

## 1. 现状结论

当前仓库已经具备知识编辑相关的三类基础能力：

- Hydra 配置入口已经存在：`configs/edit.yaml`
- 编辑方法已经注册：`ROMEEditor`、`MEMITEditor`、`MENDEditor`
- 编辑数据集已经接入：`ZSREDataset`、`CounterFactDataset`

但 `src/train.py` 目前仍然是“训练/评估入口”的思路，而不是“编辑执行入口”的思路，因此 `mode=edit` 时存在链路断点：

1. `train.py` 只会调用 `trainer.train()` 和 `trainer.evaluate()`，不会调用 `trainer.edit(...)`。
2. `mode=edit` 时数据被加载到 `data["edit"]`，但 `train.py` 只把 `data.get("train")` 和 `data.get("eval")` 传给 trainer。
3. `configs/trainer/edit/base_editor.yaml` 默认是 `do_train: false`、`do_eval: true`，因此当前 `edit` 运行看起来像“会评估”，但实际上并没有先执行编辑。
4. `src/eval.py` 能解析 `edit_data_path` / `original_model_path`，而 `train.py` 里的评估路径没有这层运行时参数解析，因此即使走到 edit evaluator，也很容易只得到默认分数。

结论：当前仓库处于“编辑器类已实现，但主入口还没接上”的状态。

## 2. 推荐目标

我建议把目标拆成两个阶段。

### 阶段 A：先让知识编辑真正能跑

目标：

- `src/train.py --config-name=edit.yaml` 可以真正执行编辑
- 支持 `single / batch / sequential` 三种 `edit_type`
- 输出编辑结果摘要并保存编辑后的模型

这一阶段不强行把评估也塞进同一个入口。

### 阶段 B：再决定是否把评估并入 `train.py`

目标：

- 编辑执行完成后可选地直接运行 `eval=edit`
- 与 `src/eval.py` 共享同一套 `edit_data` / `original_model_path` 解析逻辑

这一步不是必须第一时间做。原因是单独的 `src/eval.py` 已经更接近可用状态，先把“编辑能跑”打通，风险最低。

## 3. 推荐修改方案

## 3.1 在 `src/train.py` 增加显式的 `mode=edit` 分支

建议在 trainer 实例化之后，增加一个明确的分支：

- 如果 `mode != "edit"`，保持现有训练逻辑不变
- 如果 `mode == "edit"`，进入新的编辑执行逻辑，而不是沿用 `train()/evaluate()` 的默认路径

推荐思路：

1. 从 `data["edit"]` 读取编辑样本
2. 将样本转换成 `EditRequest`
3. 根据 `edit_type` 分发到不同执行策略
4. 保存编辑结果摘要
5. 保存编辑后的模型
6. 如果显式开启评估，再进入评估阶段

不建议把 `edit` 伪装成普通 `train_dataset` 去走 HuggingFace Trainer 训练循环。这样会让语义更混乱，而且和 ROME/MEMIT 的实现方式不一致。

## 3.2 增加“数据样本 -> EditRequest”转换层

目前 `EditingDataset` 的原始样本已经包含构造编辑请求所需的字段：

- `prompt`
- `subject`
- `target_new`
- `target_old`
- `locality_inputs`
- `portability_inputs`

建议增加一个明确的转换函数，二选一即可：

- 方案 1：在 `src/data/editing.py` 的 `EditingDataset` 上增加 `to_edit_requests()` 方法
- 方案 2：新增一个轻量 helper，例如 `src/trainer/edit/utils.py`

我更推荐方案 1，因为：

- 编辑样本结构定义就在 `EditingDataset`
- 后续扩展 `json/jsonl/HF dataset` 时，转换逻辑集中
- `train.py` 可以保持更薄

建议函数行为：

- 支持 `limit/max_edits`
- 保留 `locality` 与 `portability` 相关字段
- 返回 `List[EditRequest]`

## 3.3 明确 `edit_type` 的执行语义

当前 `configs/edit.yaml` 已经有：

- `single`
- `batch`
- `sequential`

建议把语义明确为：

- `single`
  - 只取第一个请求执行一次编辑
  - 适合 ROME 的最小可验证链路
- `batch`
  - 将请求列表整体传给 `trainer.edit(requests)`
  - 适合 MEMIT 这种需要联合求解的批量编辑方法
- `sequential`
  - 逐条执行 `trainer.edit(request)`，保留每一步后的模型状态
  - 适合测试连续编辑与干扰累积

这里有一个重要细节：

- 对 ROME/MEND 来说，`batch_edit()` 本质上更像分批循环
- 对 MEMIT 来说，真正的“批量”应优先走 `trainer.edit(requests)` 一次性联合编辑

因此不建议统一强行走 `batch_edit()`。推荐按方法语义分发。

## 3.4 调整 `configs/trainer/edit/base_editor.yaml` 的默认行为

当前默认值：

- `do_train: false`
- `do_eval: true`

这对编辑模式不够直观。建议改成：

- `do_train: false`
- `do_eval: false`

理由：

- 编辑不是标准意义上的训练
- 当前评估链路尚未完整接入 `train.py`
- 先默认关闭评估，避免用户误以为“命令跑完即完成编辑并评估”

如果后续完成阶段 B，再考虑把 `do_eval` 打开。

## 3.5 为编辑运行新增最小必要配置项

建议在 `configs/edit.yaml` 中增加以下字段：

```yaml
edit_type: single
max_edits: null
save_edit_summary: true
save_edited_model: true
```

如果希望更细，还可以加：

```yaml
sequential_save_every: false
```

但第一轮实现不建议加太多开关。先把最基本的链路做清楚。

## 3.6 保存结构化产物，而不只是打印日志

建议每次编辑运行后至少保存两个文件到 `output_dir`：

- `edit_results.json`
  - 包含方法名、编辑条数、成功条数、编辑类型、时间戳等
- `edit_requests_preview.json`
  - 保存本次实际执行的请求子集，便于复现实验

这样做的价值：

- CLI 输出更容易复查
- 后续接 WebUI 时更容易展示
- 便于单独运行 `src/eval.py` 进行复评

## 3.7 评估建议先复用 `src/eval.py`

第一阶段我建议不要急着让 `train.py` 直接兼容所有 edit evaluator 的运行时参数。

推荐工作流：

1. 用 `src/train.py --config-name=edit.yaml` 执行编辑并保存模型
2. 再用 `src/eval.py` 对保存后的模型做 `eval=edit`

原因：

- `src/eval.py` 已经支持 `edit_data_path`
- `src/eval.py` 已经支持 `original_model_path`
- 这样能避免在 `train.py` 里复制一套 evaluator runtime 参数解析逻辑

如果后续确认希望“一条命令编辑+评估”，再把 `src/eval.py` 中的运行时参数解析逻辑抽到共享 helper 中，让 `train.py` 复用。

## 4. MEND 的特殊说明

`MEND` 不建议和 `ROME/MEMIT` 按同一优先级直接接成“可用方法”。

原因：

- 代码里 `MENDEditor.edit()` 会在首次使用时初始化一个随机编辑网络
- 同文件中还存在 `train_edit_network(...)`，并明确写了“在使用 MEND 之前，需要先训练编辑网络”
- 但当前主入口没有接入这一步

这意味着：

- 从工程角度看，`MEND` 可以被实例化
- 但从方法有效性角度看，当前并不满足“直接可用”的前提

建议：

- 第一轮先把 `ROME` 与 `MEMIT` 打通并验证
- `MEND` 先标记为 `experimental`
- 等确认需要时，再补一个单独的 MEND 预训练/加载链路

## 5. 受影响文件建议

第一阶段建议只动下面这些文件：

- `src/train.py`
- `src/data/editing.py`
- `configs/edit.yaml`
- `configs/trainer/edit/base_editor.yaml`
- `docs/experiments.md` 或新增一份 edit 运行说明文档

第二阶段如果要把评估并入 `train.py`，再考虑：

- `src/eval.py`
- 新增共享 helper，例如 `src/evals/runtime_utils.py`
- `README.md`
- `EDIT_EVALUATION.md`

## 6. 推荐实施顺序

### 第一步：打通最小可运行链路

- `train.py` 增加 `mode=edit` 分支
- 将 `EditingDataset` 转成 `EditRequest`
- 跑通 `ROME single`
- 保存编辑结果和模型

### 第二步：补齐其他执行模式

- 跑通 `MEMIT batch`
- 跑通 `ROME sequential`
- 校验 `edit_history` 和输出产物

### 第三步：文档和命令示例

- 写清楚三种 `edit_type` 的使用方式
- 写清楚编辑后如何用 `src/eval.py` 评估

### 第四步：决定是否接入 inline eval

- 如果用户确实需要一条命令完成编辑和评估，再合并评估逻辑
- 否则维持“编辑”和“评估”两个入口分离

## 7. 建议的命令形态

下面是建议修改完成后的目标命令形态。

### 7.1 ROME 单条编辑

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --config-name=edit.yaml \
  task_name=edit_zsre_rome \
  experiment=edit/zsre/default \
  edit_type=single \
  max_edits=1
```

### 7.2 MEMIT 批量编辑

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --config-name=edit.yaml \
  task_name=edit_zsre_memit \
  experiment=edit/zsre/memit \
  edit_type=batch \
  max_edits=64
```

### 7.3 顺序编辑

```bash
PYTHONPATH=src .venv/bin/python src/train.py \
  --config-name=edit.yaml \
  task_name=edit_zsre_seq \
  experiment=edit/zsre/default \
  edit_type=sequential \
  max_edits=10
```

### 7.4 编辑后的评估

```bash
PYTHONPATH=src .venv/bin/python src/eval.py \
  mode=eval \
  task_name=edit_eval \
  eval=edit \
  model=Qwen2.5-7B-Instruct \
  model.model_args.pretrained_model_name_or_path=/ABS/PATH/TO/EDITED_MODEL \
  eval.reliability.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.generalization.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.locality.edit_data_path=/ABS/PATH/edit_eval.json \
  eval.locality.original_model_path=/ABS/PATH/TO/BASE_MODEL \
  eval.portability.edit_data_path=/ABS/PATH/edit_eval.json
```

## 8. 不建议当前一起做的事情

第一轮不建议一起做下面这些内容：

- 直接把 MEND 当作稳定可用方法接入
- 在 `train.py` 里复制一套 `src/eval.py` 的运行时参数解析逻辑
- 为 edit 模式引入新的复杂训练循环
- 一次性补所有本地数据格式支持
- 顺手改 WebUI

这些会扩大修改面，降低第一轮落地成功率。

## 9. 我推荐的最终决策

如果你准备让我开始实现，我建议按下面的范围执行：

### 本轮实现范围

- 打通 `src/train.py` 的 `mode=edit`
- 支持 `single / batch / sequential`
- 保存编辑结果与编辑后模型
- 默认关闭 `train.py` 内联评估
- 给出可直接复制的运行命令

### 本轮暂不实现

- `MEND` 的完整预训练链路
- `train.py` 内联 edit evaluator
- WebUI 侧联动

这是最小、最稳、最容易验证的一版。

## 10. 待你确认的两件事

在开始实施前，我建议你确认下面两点：

1. 是否接受“本轮先打通编辑执行，评估继续走 `src/eval.py`”这个分阶段方案。
2. 是否接受“本轮先把 `MEND` 标记为实验性，不承诺结果有效性”。

如果你确认，我下一步会按这份文档开始改代码，而不是扩大范围。

## 11. 本次执行方案

下面这部分是本轮已经采用的执行方案，用于约束实现范围与验证范围。

### 11.1 实现范围

本轮只做阶段 A，对应内容如下：

- 在 `src/train.py` 中新增 `mode=edit` 的显式执行分支
- 在编辑模式下从 `data.edit` 构造 `EditRequest`
- 支持 `single / batch / sequential` 三种执行语义
- 保存 `edit_results.json` 与 `edit_requests_preview.json`
- 保存编辑后的模型到 `output_dir`
- 将 `configs/trainer/edit/base_editor.yaml` 的默认 `do_eval` 改为 `false`

本轮不做：

- `train.py` 内联 edit evaluator
- `MEND` 的完整预训练/加载链路
- WebUI 联动

### 11.2 代码结构方案

本轮采用“轻量 helper + 薄入口”的方式实现：

- `src/train.py`
  - 保留现有 unlearn/train 逻辑
  - 仅在 `mode=edit` 时转入编辑执行逻辑
- `src/data/editing.py`
  - 增加 `to_edit_requests()`，把数据样本转成 `EditRequest`
- 新增 `src/trainer/edit/pipeline.py`
  - 负责构造请求
  - 负责 single/batch/sequential 分发
  - 负责保存编辑产物

这样做的原因是：

- `train.py` 不会变成巨型脚本
- 核心逻辑可以被单元测试直接覆盖
- 后续如果要接 WebUI 或独立 CLI，可以复用同一套 helper

### 11.3 测试方案

本轮采用“单元测试 + 语法检查 + lint + diff 复核”的严格验证方式：

1. 新增针对 `src/trainer/edit/pipeline.py` 的单元测试
2. 运行 Python 语法检查，确保新增与改动文件可编译
3. 运行 `ruff check` 做静态 lint 验证
4. 运行 `pytest` 针对新增测试文件做冒烟与行为校验
5. 查看 `git diff` / `git diff --stat`，确认没有扩散到无关文件

### 11.4 成功标准

本轮实现完成后，应满足以下标准：

- `mode=edit` 时不再走空的 `train/eval` 路径
- `ROME/MEMIT` 至少具备“命令能执行到编辑逻辑、能保存产物”的工程可用性
- 新增 helper 具备单元测试覆盖
- 默认配置不会再误导用户以为 `train.py` 已支持完整 edit 评估

### 11.5 风险处理

本轮对风险的处理原则如下：

- 对 `MEND` 不扩大承诺，只保留入口兼容
- 对评估能力不做半成品接入，避免返回误导性分数
- 对已有脏工作区文件只做最小必要修改，避免覆盖现有改动
