# 多模态知识编辑 (Multimodal Knowledge Editing) 使用指南

## 一、总览

本仓库已集成完整的多模态知识编辑框架，覆盖 **7 种编辑算法**、**4 个评测基准**、**4 项评估指标**，支持 **5 种 VL 模型**。所有组件通过 Hydra 配置驱动，可自由组合。

### 架构关系图

```
EditingSample (含 image/image_rephrase/multimodal_locality_inputs 字段)
    │
    ▼
EditRequest  ──────────────► MM Editor.edit(request)
    │                            │
    │  ┌─────────────────────────┤
    │  │                         │
    │  │  image=None?            │  image≠None?
    │  │  ▼                      │  ▼
    │  │  tokenizer path         │  MMEditMixin.mm_tokenize()
    │  │  (text-only fallback)   │  processor(text+image)
    │  │                         │
    │  └────────► model(**inputs) ◄───────┘
    │                    │
    ▼                    ▼
MM Evaluator         weight update / ICL store
```

---

## 二、评测基准 (Benchmarks)

| 基准 | 来源 | 数据集类 | 配置文件 | 规模 | 特色 |
|------|------|---------|---------|------|------|
| **MMEdit E-VQA** | EMNLP 2023 | `MMEditVQADataset` | `MMEdit_VQA.yaml` | ~5k | Visual QA 编辑，含 image_rephrase + multimodal locality (OK-VQA) |
| **MMEdit E-IC** | EMNLP 2023 | `MMEditCaptionDataset` | `MMEdit_Caption.yaml` | ~5k | Image Captioning 编辑，与 E-VQA 字段格式相同 |
| **MMKE-Bench** | ICLR 2025 | `MMKEBenchDataset` | `MMKE_Bench_*.yaml` (3 个子集) | 2,940 条知识 + 8,363 图 | 三种编辑任务：visual_entity / visual_semantic / user_specific |
| **VLKEB** | NeurIPS 2024 | `VLKEBDataset` | `VLKEB_eval.yaml`, `VLKEB_multihop.yaml` | 8,174 edits + 18,434 图 | 知识图谱三元组 + 图像，含 **1-4 hop 多跳 portability** |

### 数据字段映射

所有多模态数据集统一映射到 `EditingSample`：

```
原始字段              →  EditingSample 字段
─────────────────────────────────────────────
src / prompt          →  prompt
alt / target_new      →  target_new
pred                  →  target_old
image                 →  image (PIL.Image)
rephrase              →  rephrase_prompts
image_rephrase        →  image_rephrase (PIL.Image)
loc / loc_ans         →  locality_inputs
m_loc / m_loc_q / a   →  multimodal_locality_inputs
port_new (VLKEB)      →  portability_inputs["multihop_portability"]
```

---

## 三、编辑算法 (Methods)

### 3.1 多模态编辑器（7 种）

| 算法 | 类名 | 配置 | 论文 | 改权重? | 核心机制 |
|------|------|------|------|---------|---------|
| **MM-IKE** | `MMIKEEditor` | `MM_IKE.yaml` | [IKE](https://arxiv.org/abs/2305.12740) + [MMEdit](https://arxiv.org/abs/2310.08475) | 否 | ICL 检索，sentence-transformer 编码 |
| **MM-GRACE** | `MMGRACEEditor` | `MM_GRACE.yaml` | [GRACE](https://arxiv.org/abs/2211.11031) + MMEdit | 是 (adapter) | 离散 key-value 码本 |
| **MM-WISE** | `MMWISEEditor` | `MM_WISE.yaml` | [WISE](https://arxiv.org/abs/2405.14768) + MMEdit | 是 (adapter+merge) | 知识分片 + slerp/ties/linear 合并 |
| **MM-MEND** | `MMMENDEditor` | `MM_MEND.yaml` | [MEND](https://arxiv.org/abs/2110.11309) + MMEdit | 是 (元学习) | 梯度分解 + 编辑网络 |
| **MM-SERAC** | `MMSERACEditor` | `MM_SERAC.yaml` | [SERAC](https://arxiv.org/abs/2206.06520) + MMEdit | 否 (外部模型) | 范围分类器 + 反事实模型 |
| **UniKE** | `UniKEEditor` | `UniKE.yaml` | [UniKE](https://arxiv.org/abs/2409.19872) NeurIPS 2024 Spotlight | 是 | 统一 IKE + ROME：同化(外部记忆) + 顺应(秩一更新) |
| **MM-UniKE** | `MMUniKEEditor` | `MM_UniKE.yaml` | UniKE + 多模态扩展 | 是 | 多模态 forward 的统一编辑框架 |

### 3.2 继承关系

```
EditTrainer (base.py)
├── IKEEditor ─────── MMIKEEditor (+ MMEditMixin)
├── GRACEEditor ───── MMGRACEEditor (+ MMEditMixin)
├── WISEEditor ────── MMWISEEditor (+ MMEditMixin)
├── MENDEditor ────── MMMENDEditor (+ MMEditMixin)
├── SERACEditor ───── MMSERACEditor (+ MMEditMixin)
├── ROMEEditor
└── UniKEEditor ───── MMUniKEEditor (+ MMEditMixin)
```

所有 MM 编辑器都继承对应的文本版 + `MMEditMixin`，支持双路径：
- `request.image ≠ None` → 多模态路径 (`mm_tokenize` + `mm_forward`)
- `request.image = None` → 文本回退路径（与文本版完全相同）

---

## 四、评估指标 (Metrics)

| 指标 | 评估器类 | 配置 | 含义 |
|------|---------|------|------|
| **MM Reliability** | `MMEditReliabilityEvaluator` | `mm_reliability.yaml` | 编辑后对原始 prompt+image 的 rewrite accuracy |
| **MM Generalization** | `MMEditGeneralizationEvaluator` | `mm_generalization.yaml` | text_rephrase_acc + image_rephrase_acc |
| **MM Locality** | `MMEditLocalityEvaluator` | `mm_locality.yaml` | text_locality_acc + multimodal_locality_acc (OK-VQA) |
| **MM Portability** | `MMEditPortabilityEvaluator` | `mm_portability.yaml` | 多跳推理迁移准确率 (VLKEB 1-4 hop) |

评估流程：使用 `processor` 将 prompt+image 编码，通过多模态 forward 计算 teacher-forcing token 准确率。

---

## 五、支持的模型

| 模型 | 配置文件 | 参数量 | 用途 |
|------|---------|-------|------|
| **Qwen2-VL-2B-Instruct** | `Qwen2VL-2B.yaml` | 2B | 快速实验 / 开发调试 |
| **Qwen2.5-VL-7B-Instruct** | `Qwen2.5VL-7B.yaml` | 7B | 主力评测（前沿论文标准） |
| **Qwen3-VL-4B** | `Qwen3-VL-4B.yaml` | 4B | 中等规模实验 |
| **InternVL2.5-2B** | `InternVL2_5-2B.yaml` | 2B | 跨架构验证 |
| **InternVL2.5-8B** | `InternVL2_5-8B.yaml` | 8B | 大规模评测 |

所有模型通过 `AutoModelForImageTextToText` + `AutoProcessor` 加载（HF Processor 路径），不依赖旧版 BLIP2/MiniGPT4 接口。

---

## 六、使用方法

### 6.1 基本用法（Hydra 命令行）

```bash
# MM-GRACE 在 MMEdit E-VQA 上编辑 Qwen2-VL-2B
python src/train.py --config-name=mm_edit \
    model=Qwen2VL-2B \
    trainer=edit/MM_GRACE \
    data.edit.MMEdit_VQA.args.data_path=data/edit/mmedit/vqa.json \
    task_name=mm_grace_vqa

# MM-UniKE 在 VLKEB 上编辑 Qwen2.5-VL-7B
python src/train.py --config-name=mm_edit \
    model=Qwen2.5VL-7B \
    trainer=edit/MM_UniKE \
    data=mm_edit \
    data.edit=VLKEB_eval \
    task_name=mm_unike_vlkeb

# 切换评测基准：MMKE-Bench visual_entity
python src/train.py --config-name=mm_edit \
    model=Qwen2VL-2B \
    trainer=edit/MM_WISE \
    data.edit=MMKE_Bench_visual_entity \
    task_name=mm_wise_mmke
```

### 6.2 Python API 用法

```python
import sys; sys.path.insert(0, "src")
from transformers import AutoModelForImageTextToText, AutoProcessor
from trainer.edit.base import EditRequest
from trainer.edit.mm_grace import MMGRACEEditor
from PIL import Image

# 加载模型
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="bfloat16", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# 构建编辑请求（含图像）
request = EditRequest(
    prompt="What color is this car?",
    subject="car",
    target_new="red",
    target_old="blue",
    image=Image.open("car.jpg"),
)

# 初始化 MM 编辑器（绕过 HF Trainer 的完整初始化）
editor = object.__new__(MMGRACEEditor)
editor.model = model
editor.tokenizer = processor.tokenizer
editor.processor = processor  # 关键：多模态 processor
editor.layers = [5]
editor.inner_params = "model.layers.5.mlp.down_proj"
editor.target_layer = editor.inner_params
editor.edit_lr = 0.1
editor.n_iter = 40
editor.eps = 5e-4
# ... 其他 GRACE 参数 ...

# 执行编辑
result = editor.edit(request)
print(result)  # {"success": True, "edited_count": 1, ...}
```

### 6.3 纯文本回退

所有 MM 编辑器对 `image=None` 的请求自动走文本路径：

```python
text_request = EditRequest(
    prompt="The capital of France is",
    subject="France",
    target_new="Berlin",
)
result = editor.edit(text_request)  # 自动走 tokenizer 路径
```

### 6.4 评估

```python
from evals.mm_edit import MMEditReliabilityEvaluator

evaluator = MMEditReliabilityEvaluator(eval_cfg)
result = evaluator.evaluate(
    model=model,
    processor=processor,
    edit_data=[{
        "prompt": "What color?",
        "target_new": "red",
        "image": image,
    }],
)
# {"mm_reliability": 0.95, "total_samples": 1}
```

---

## 七、文件清单

### 核心代码（10 个文件）

| 文件 | 内容 |
|------|------|
| `src/trainer/edit/mm_mixin.py` | `MMEditMixin` — 多模态 tokenize/forward 工具 |
| `src/trainer/edit/mm_ike.py` | MM-IKE 编辑器 |
| `src/trainer/edit/mm_grace.py` | MM-GRACE 编辑器 |
| `src/trainer/edit/mm_wise.py` | MM-WISE 编辑器 |
| `src/trainer/edit/mm_mend.py` | MM-MEND 编辑器 |
| `src/trainer/edit/mm_serac.py` | MM-SERAC 编辑器 |
| `src/trainer/edit/unike.py` | UniKE 编辑器（文本版，IKE+ROME 统一） |
| `src/trainer/edit/mm_unike.py` | MM-UniKE 编辑器（多模态版） |
| `src/data/mm_editing.py` | 4 个多模态数据集类 |
| `src/evals/mm_edit.py` | 4 个多模态评估器 |

### 配置文件（18 个）

| 类型 | 文件 |
|------|------|
| 入口 | `configs/mm_edit.yaml` |
| 算法 | `configs/trainer/edit/MM_IKE.yaml`, `MM_GRACE.yaml`, `MM_WISE.yaml`, `MM_MEND.yaml`, `MM_SERAC.yaml`, `UniKE.yaml`, `MM_UniKE.yaml` |
| 数据 | `configs/data/mm_edit.yaml`, `datasets/MMEdit_VQA.yaml`, `MMEdit_Caption.yaml`, `MMKE_Bench_*.yaml` (3), `VLKEB_eval.yaml`, `VLKEB_multihop.yaml` |
| 评估 | `configs/eval/mm_edit.yaml`, `mm_edit_metrics/mm_reliability.yaml`, `mm_generalization.yaml`, `mm_locality.yaml`, `mm_portability.yaml` |
| 模型 | `configs/model/Qwen2VL-2B.yaml`, `Qwen2.5VL-7B.yaml` |

### 修改的已有文件（6 个）

| 文件 | 改动 |
|------|------|
| `src/data/editing.py` | `EditingSample` +3 多模态字段 |
| `src/trainer/edit/base.py` | `EditRequest` +3 多模态字段 + `_input_device()` |
| `src/trainer/edit/grace.py` | `_safe_layer_forward()` dtype 兼容 |
| `src/trainer/edit/wise.py` | `_orig_forward()` dtype 兼容 + `_dtype` 记录 |
| `src/trainer/edit/pipeline.py` | PIL.Image 序列化 |
| `src/trainer/edit/__init__.py` | 导出所有 MM 编辑器 |

---

## 八、算法 × 基准 × 指标 兼容矩阵

| | Reliability | Generalization | Locality | Portability |
|------|:-:|:-:|:-:|:-:|
| **MMEdit E-VQA** | ✅ | ✅ text + image_rephrase | ✅ text + mm (OK-VQA) | — |
| **MMEdit E-IC** | ✅ | ✅ text + image_rephrase | ✅ text + mm (OK-VQA) | — |
| **MMKE-Bench** | ✅ | ✅ text | ✅ text | ✅ standard |
| **VLKEB** | ✅ | ✅ text + image_rephrase | ✅ text + mm | ✅ **1-4 hop** |

所有 7 种 MM 编辑算法均可在以上 4 个基准上运行。
