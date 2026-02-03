"""
Knowledge Editing 数据集
=======================

支持知识编辑任务的数据集类，适配 ZSRE、CounterFact 等主流基准。

数据格式：
- prompt: 触发编辑的输入
- subject: 编辑的主体实体
- target_new: 新的目标输出
- target_old: 原始输出（可选）
- rephrase_prompts: 改写的提示（用于泛化性测试）
- locality_inputs: 局部性测试输入
- portability_inputs: 可移植性测试输入
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class EditingSample:
    """编辑样本数据结构"""

    prompt: str
    subject: str
    target_new: str
    target_old: Optional[str] = None
    rephrase_prompts: Optional[List[str]] = None
    locality_inputs: Optional[List[Dict[str, str]]] = None
    portability_inputs: Optional[List[Dict[str, str]]] = None


class EditingDataset(Dataset):
    """知识编辑数据集基类

    支持加载 ZSRE、CounterFact 等知识编辑基准数据集。

    Attributes:
        data: 编辑样本列表
        tokenizer: 分词器
        max_length: 最大序列长度
    """

    def __init__(
        self,
        hf_args: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        prompt_key: str = "prompt",
        subject_key: str = "subject",
        target_new_key: str = "target_new",
        target_old_key: str = "target_true",
        tokenizer=None,
        max_length: int = 512,
        template_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """初始化编辑数据集

        Args:
            hf_args: HuggingFace 数据集加载参数
            data_path: 本地数据路径（优先级高于 hf_args）
            prompt_key: 提示字段名
            subject_key: 主体字段名
            target_new_key: 新目标字段名
            target_old_key: 原目标字段名
            tokenizer: 分词器
            max_length: 最大长度
            template_args: 模板参数
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args or {}

        self.prompt_key = prompt_key
        self.subject_key = subject_key
        self.target_new_key = target_new_key
        self.target_old_key = target_old_key

        # 加载数据
        self.data = self._load_data(hf_args, data_path)

        logger.info(f"EditingDataset loaded with {len(self.data)} samples")

    def _load_data(
        self, hf_args: Optional[Dict[str, Any]], data_path: Optional[str]
    ) -> List[EditingSample]:
        """加载数据

        Args:
            hf_args: HuggingFace 数据集参数
            data_path: 本地数据路径

        Returns:
            编辑样本列表
        """
        samples = []

        if data_path:
            # 从本地文件加载
            import json

            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif hf_args:
            # 从 HuggingFace 加载
            dataset = load_dataset(**hf_args)
            raw_data = list(dataset)
        else:
            logger.warning("No data source specified")
            return samples

        # 转换为 EditingSample
        for item in raw_data:
            sample = EditingSample(
                prompt=item.get(self.prompt_key, ""),
                subject=item.get(self.subject_key, ""),
                target_new=item.get(self.target_new_key, ""),
                target_old=item.get(self.target_old_key, None),
                rephrase_prompts=item.get("rephrase_prompts", None),
                locality_inputs=item.get("locality", None),
                portability_inputs=item.get("portability", None),
            )
            samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本

        Returns:
            包含 tokenized 输入的字典
        """
        sample = self.data[idx]

        # 构造输入文本
        input_text = f"{sample.prompt}"
        target_text = f"{sample.prompt} {sample.target_new}"

        # Tokenize
        if self.tokenizer:
            inputs = self.tokenizer(
                input_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": targets["input_ids"].squeeze(0),
                "subject": sample.subject,
                "target_new": sample.target_new,
                "target_old": sample.target_old,
            }
        else:
            return {
                "prompt": sample.prompt,
                "subject": sample.subject,
                "target_new": sample.target_new,
                "target_old": sample.target_old,
            }


class ZSREDataset(EditingDataset):
    """ZSRE 数据集

    Zero-Shot Relation Extraction 知识编辑基准。
    """

    def __init__(self, tokenizer=None, split: str = "train", **kwargs):
        hf_args = {
            "path": "zjunlp/KnowEdit",
            "name": "zsre",
            "split": split,
        }
        super().__init__(
            hf_args=hf_args,
            tokenizer=tokenizer,
            prompt_key="prompt",
            subject_key="subject",
            target_new_key="target_new",
            target_old_key="target_true",
            **kwargs,
        )


class CounterFactDataset(EditingDataset):
    """CounterFact 数据集

    反事实知识编辑基准，用于测试模型编辑能力。
    """

    def __init__(self, tokenizer=None, split: str = "train", **kwargs):
        hf_args = {
            "path": "zjunlp/KnowEdit",
            "name": "counterfact",
            "split": split,
        }
        super().__init__(
            hf_args=hf_args,
            tokenizer=tokenizer,
            prompt_key="prompt",
            subject_key="subject",
            target_new_key="target_new",
            target_old_key="target_true",
            **kwargs,
        )
