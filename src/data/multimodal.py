"""多模态数据集和 collate 函数（sidecar，不修改现有 data/__init__.py）。

复用上游 MMUnlearner 的 MLLMU 数据格式（Parquet + image bytes + metadata JSON）。
collate 使用 transformers 原生 processor.apply_chat_template，不依赖 qwen_vl_utils。
"""

import json
import logging
import os
from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class MLLMUDataset(Dataset):
    """MLLMU-Bench 数据集，从 Parquet 读取图像+QA 对并展平为单条样本。"""

    def __init__(self, parquet_path: str):
        super().__init__()
        df = pd.read_parquet(parquet_path)
        self.samples = self._flatten(df)
        logger.info("MLLMUDataset loaded %d samples from %s", len(self.samples), parquet_path)

    def _flatten(self, df: pd.DataFrame) -> List[Dict]:
        data = []
        has_metadata = "metadata" in df.columns
        for idx, row in df.iterrows():
            image_bytes = row["image"].get("bytes")
            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                logger.warning("Skipping row %d: image decode error: %s", idx, e)
                continue

            if has_metadata:
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError as e:
                    logger.warning("Skipping row %d: metadata JSON error: %s", idx, e)
                    continue
                for qa in metadata:
                    q, a = qa.get("Question", ""), qa.get("Answer", "")
                    if q and a:
                        data.append({"image": image, "question": q, "answer": a})
            else:
                q = row.get("question", "")
                a = row.get("answer", "")
                if q and a:
                    data.append({"image": image, "question": q, "answer": a})
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def mm_collate_fn(examples, processor, ans_only=False):
    """将 MLLMUDataset 样本列表编码为模型可用的 batch。

    使用 processor.apply_chat_template 构建 messages，
    再调用 processor() 编码图像+文本，不依赖 qwen_vl_utils。

    Args:
        examples: 样本列表，每个含 image / question / answer。
        processor: AutoProcessor 实例。
        ans_only: 是否只在 answer token 上计算 loss。
    """
    images = []
    texts = []
    answer_tokens_list = []

    for ex in examples:
        image = ex["image"]
        question = ex["question"]
        answer = ex["answer"]

        user_content = [{"type": "image"}, {"type": "text", "text": question}]
        if image is not None:
            images.append(image)
        else:
            user_content = [{"type": "text", "text": question}]

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())

        if ans_only:
            ans_tok = processor.tokenizer(answer, return_tensors="pt")["input_ids"][0][1:]
            answer_tokens_list.append(ans_tok)

    batch = processor(
        text=texts,
        images=images if images else None,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()

    if ans_only and answer_tokens_list:
        for i, ans_ids in enumerate(answer_tokens_list):
            matched = False
            for start in range(len(labels[i]) - len(ans_ids) + 1):
                if torch.equal(labels[i][start : start + len(ans_ids)], ans_ids):
                    labels[i][:start] = -100
                    labels[i][start + len(ans_ids) :] = -100
                    matched = True
                    break
            if not matched:
                labels[i][labels[i] == processor.tokenizer.pad_token_id] = -100
    else:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return dict(batch)


def get_mm_data(data_cfg: DictConfig, processor):
    """根据 Hydra 数据配置构建 forget / retain DataLoader。

    Returns:
        (forget_loader, retain_loader)，retain_loader 可能为 None。
    """
    data_dir = data_cfg.data_dir
    forget_ratio = data_cfg.get("forget_split_ratio", 5)
    batch_size = data_cfg.get("batch_size", 4)
    ans_only = data_cfg.get("ans_only", False)

    forget_parquet = os.path.join(
        data_dir, f"forget_{forget_ratio}", "train-00000-of-00001.parquet"
    )
    forget_dataset = MLLMUDataset(forget_parquet)
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: mm_collate_fn(x, processor, ans_only=ans_only),
    )

    retain_loader = None
    retain_parquet_path = os.path.join(
        data_dir, f"retain_{100 - forget_ratio}", "train-00000-of-00001.parquet"
    )
    if os.path.exists(retain_parquet_path):
        retain_dataset = MLLMUDataset(retain_parquet_path)
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: mm_collate_fn(x, processor, ans_only=ans_only),
        )

    full_parquet = data_cfg.get("full_set_parquet", None)
    if full_parquet and not retain_loader and os.path.exists(full_parquet):
        retain_dataset = MLLMUDataset(full_parquet)
        retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: mm_collate_fn(x, processor, ans_only=ans_only),
        )

    return forget_loader, retain_loader
