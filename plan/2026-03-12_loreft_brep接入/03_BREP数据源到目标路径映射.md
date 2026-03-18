# BREP 数据源到目标路径映射

## 目标
把每个 BREP 相关数据集的推荐来源、license、原始格式、目标落盘路径、以及规范化转换规则补齐。当前只做映射记录，不执行下载。

## 训练数据

| 数据集 | 推荐来源 | License | 原始格式 | 目标路径 | 规范化规则 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `prm800k` | OpenAI 官方仓库 `openai/prm800k` | MIT | `phase2_train.jsonl` 等 JSONL | `/home/liumingxuan/open-unlearning/data/inject/brep/train/prm800k/train.json` | 参考 `code/BREP/Prefix/deal_data.py`：`question.problem -> instruction`，`question.ground_truth_solution -> output`，`question.ground_truth_answer -> answer` | 可直接下载并转换 |
| `math10k` | 参考 `LLM-Adapters` / `LoRA-RITE` 的 `math_10k.json` 配方 | 继承自上游混合数据集，需逐项核对 | 通常是聚合后的 JSON | `/home/liumingxuan/open-unlearning/data/inject/brep/train/math10k/train.json` | 按 Alpaca 风格保留 `instruction/input/output`，建议额外保留 `answer` 和 `source_dataset` | 需人工重建 |
| `commonsense` | 参考 `LLM-Adapters` 的 commonsense recipe | 继承自上游混合数据集，需逐项核对 | 聚合后的 JSON/JSONL | `/home/liumingxuan/open-unlearning/data/inject/brep/train/commonsense/train.json` | 统一转为 `instruction/input/output/source_dataset`，其中 `input` 可为空串 | 需人工重建 |
| `ultrafeedback` | Hugging Face `openbmb/UltraFeedback` | MIT | HF dataset / JSON | `/home/liumingxuan/open-unlearning/data/inject/brep/train/ultrafeedback/train.json` | 从 `instruction` 抽题目，后续需单独定义如何从 `completions` 中选 `output` | 可直接下载，转换规则待定 |

## 评测数据

| 数据集 | 推荐来源 | License | 原始格式 | 目标路径 | 规范化规则 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | Hugging Face `openai/gsm8k` | MIT | HF dataset / parquet | `/home/liumingxuan/open-unlearning/data/inject/brep/eval/gsm8k/test.json` | `question -> instruction`，`answer -> output`，并从 `####` 后提取 `answer` 标量 | 可直接下载并转换 |
| `hellaswag` | Hugging Face `Rowan/hellaswag` | MIT | HF dataset / parquet | `/home/liumingxuan/open-unlearning/data/inject/brep/eval/hellaswag/test.json` | `ctx -> instruction`，正确 `ending` 作为 `output` 和 `answer` 候选，保留 `label` 与 `endings` 便于复查 | 可直接下载并转换 |
| `svamp` | Hugging Face `ChilleD/SVAMP` | MIT | HF dataset / parquet | `/home/liumingxuan/open-unlearning/data/inject/brep/eval/svamp/test.json` | `question_concat` 或 `Body + Question -> instruction`，`Answer -> answer`，`Equation` 可留作附加字段 | 可直接下载并转换 |
| `mathqa` | Hugging Face `allenai/math_qa` | Apache-2.0 | HF dataset script format | `/home/liumingxuan/open-unlearning/data/inject/brep/eval/mathqa/test.json` | `Problem -> instruction`，由 `options + correct` 解出正确选项文本作为 `answer`，`Rationale` 可作为 `output` | 可直接下载并转换 |
| `math500` | Hugging Face `HuggingFaceH4/MATH-500`，其卡片明确来源于 OpenAI `prm800k` 的 `math_splits` | 数据卡未显式列出；可推断上游 OpenAI PRM 仓库为 MIT | JSON | `/home/liumingxuan/open-unlearning/data/inject/brep/eval/math500/test.json` | `problem -> instruction`，`solution -> output`，`answer -> answer` | 可直接下载，license 需备注为上游继承/待确认 |
| `amc23` | Hugging Face `ScaleFrontierData/amc23` 或 `baohao/amc23` | 数据卡未见明确 license，需人工确认 | HF dataset / parquet | `/home/liumingxuan/open-unlearning/data/inject/brep/eval/amc23/test.json` | `question/problem -> instruction`，`answer/gt -> answer`，可把 `answer` 同步为 `output`，保留原始 URL 元数据 | 可直接下载，license 待确认 |

## 辅助分析数据

| 数据集 | 推荐来源 | License | 原始格式 | 目标路径 | 规范化规则 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `TruthfulQA` | 官方仓库 `sylinrl/TruthfulQA` 或 HF `domenicrosati/TruthfulQA` | Apache-2.0 | CSV / HF dataset | `/home/liumingxuan/open-unlearning/data/inject/brep/analysis/TruthfulQA/truthful_qa.jsonl` | 至少保留 `question`、`correct_answers`、`incorrect_answers` | 可直接下载并转换 |
| `Faithful` | BREP 本地脚本 `Truthful/make_faithful_data.py` 从 `math10k` 调用 API 派生 | 无单独上游 license；取决于 `math10k` 与生成流程 | JSON | `/home/liumingxuan/open-unlearning/data/inject/brep/analysis/Faithful/faithful.jsonl` | 不下载，按脚本从 `math10k` 生成 | 派生数据 |
| `cal_raw` | BREP 本地分析目录引用 | 未找到明确公开源 | JSONL | `/home/liumingxuan/open-unlearning/data/inject/brep/analysis/cal_raw/raw.jsonl` | 保持原格式，待进一步追源 | 来源待确认 |
| `cal_4` | BREP 本地分析目录引用 | 未找到明确公开源 | JSONL | `/home/liumingxuan/open-unlearning/data/inject/brep/analysis/cal_4/4.jsonl` | 保持原格式，待进一步追源 | 来源待确认 |

## 组合数据集说明

### `math10k`
- 公开线索来自 `LLM-Adapters` / `LoRA-RITE` 项目说明。
- 其公开描述是：`math_10k.json` 由 `GSM8K + MAWPS + AQuA(1000 examples)` 组合而成。
- 这意味着它不是一个在 BREP 仓库内可直接下载的单文件数据源，而是一个需要按配方重建的派生训练集。

### `commonsense`
- BREP 仓库只引用 `dataset/commonsense/train.json`，没有提供构造脚本。
- 从 `LLM-Adapters` 的 commonsense 设定可推断，常见组成包括：
  - `BoolQ`
  - `PIQA`
  - `SIQA`
  - `HellaSwag`
  - `WinoGrande`
  - `ARC-e`
  - `ARC-c`
  - `OBQA`
- 因此该数据应在执行下载前被视为“组合训练集 recipe”，而不是可直接获取的单一公开文件。

## 推荐执行顺序
1. 先下载可直接获取且 license 明确的单源数据：`prm800k`、`ultrafeedback`、`gsm8k`、`hellaswag`、`svamp`、`mathqa`、`truthfulqa`。
2. 再处理 `math500`、`amc23` 这类可获取但仍需补 license 备注的数据。
3. 最后单独设计 `math10k` 和 `commonsense` 的重建脚本与字段映射。

## 参考来源
- OpenAI PRM800K 官方仓库: `https://github.com/openai/prm800k`
- UltraFeedback HF 数据卡: `https://huggingface.co/datasets/openbmb/UltraFeedback`
- GSM8K HF 数据卡: `https://huggingface.co/datasets/openai/gsm8k`
- HellaSwag HF 数据卡: `https://huggingface.co/datasets/Rowan/hellaswag`
- SVAMP HF 数据卡: `https://huggingface.co/datasets/ChilleD/SVAMP`
- MathQA HF 数据卡: `https://huggingface.co/datasets/allenai/math_qa`
- MATH-500 HF 数据卡: `https://huggingface.co/datasets/HuggingFaceH4/MATH-500`
- AMC23 HF 数据页: `https://huggingface.co/datasets/ScaleFrontierData/amc23`
- TruthfulQA 官方仓库: `https://github.com/sylinrl/TruthfulQA`
- LLM-Adapters 仓库: `https://github.com/AGI-Edgerunners/LLM-Adapters`
- LoRA-RITE 项目页: `https://gkevinyen5418.github.io/LoRA-RITE/LLM-Adapters/`
