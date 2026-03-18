# BREP 数据集分类与下载格式

## 目标
在正式下载前，先把 BREP 参考仓库涉及的数据集按用途分类，并确定统一落盘到 `/home/liumingxuan/open-unlearning/data/inject/brep/` 后的目录和文件格式。

## 训练集
- `prm800k`
- `math10k`
- `commonsense`
- `ultrafeedback`

目标目录：
```text
/home/liumingxuan/open-unlearning/data/inject/brep/train/
  prm800k/train.json
  math10k/train.json
  commonsense/train.json
  ultrafeedback/train.json
```

统一最小 JSON 结构：
```json
[
  {
    "instruction": "string",
    "input": "string",
    "output": "string",
    "source_dataset": "prm800k|math10k|commonsense|ultrafeedback",
    "index": 0
  }
]
```

说明：
- BREP 原训练脚本最核心依赖的是 `instruction` 和 `output`。
- 为了兼容 `open-unlearning` 的 `InjectDataset`，统一保留 Alpaca 风格字段最稳。
- `input` 可以为空字符串。
- 原始数据若带 `answer`、`weight` 等字段，可作为附加字段保留，不作为首批规范化必填项。

## 评测集
- `gsm8k`
- `hellaswag`
- `svamp`
- `mathqa`
- `math500`
- `amc23`

目标目录：
```text
/home/liumingxuan/open-unlearning/data/inject/brep/eval/
  gsm8k/test.json
  hellaswag/test.json
  svamp/test.json
  mathqa/test.json
  math500/test.json
  amc23/test.json
```

统一最小 JSON 结构：
```json
[
  {
    "instruction": "string",
    "output": "string",
    "answer": "string",
    "index": 0
  }
]
```

说明：
- 这是按 `code/BREP/Prefix/make_answer_json.py` 的读取逻辑整理的。
- 若原始 benchmark 文件没有 `index`，后处理阶段可以自动补。

## 辅助分析数据
- `TruthfulQA`
- `Faithful`
- `cal_raw`
- `cal_4`

目标目录：
```text
/home/liumingxuan/open-unlearning/data/inject/brep/analysis/
  TruthfulQA/truthful_qa.jsonl
  Faithful/faithful.jsonl
  cal_raw/raw.jsonl
  cal_4/4.jsonl
```

格式要求：
- `TruthfulQA` / `Faithful`：JSONL
- `cal_raw` / `cal_4`：JSONL

## 本阶段结论
- 现在只完成了分类和格式归档，尚未实施下载。
- 真正下载前还要补一张“数据源到目标路径”的映射表，明确每个数据集从哪里获取、是否需要 license、是否需要字段转换。
