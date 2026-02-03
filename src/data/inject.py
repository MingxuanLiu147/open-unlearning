"""
Knowledge Injection 数据集
=========================

支持知识注入（微调）任务的数据集类。
支持 Alpaca、ShareGPT 等主流数据格式。

数据格式支持：
- Alpaca: instruction, input, output
- ShareGPT: conversations (多轮对话)
- 自定义: 可配置的字段映射
"""

import logging
from typing import Dict, Any, Optional, List, Union

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


class InjectDataset(Dataset):
    """知识注入数据集基类

    支持多种数据格式，用于参数高效微调。

    Attributes:
        data: 样本列表
        tokenizer: 分词器
        max_length: 最大序列长度
        format_type: 数据格式类型 (alpaca/sharegpt/custom)
    """

    def __init__(
        self,
        hf_args: Optional[Dict[str, Any]] = None,
        data_path: Optional[str] = None,
        format_type: str = "alpaca",
        instruction_key: str = "instruction",
        input_key: str = "input",
        output_key: str = "output",
        conversations_key: str = "conversations",
        tokenizer=None,
        max_length: int = 2048,
        template_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """初始化注入数据集

        Args:
            hf_args: HuggingFace 数据集参数
            data_path: 本地数据路径
            format_type: 数据格式 (alpaca/sharegpt/custom)
            instruction_key: 指令字段名
            input_key: 输入字段名
            output_key: 输出字段名
            conversations_key: 对话字段名（ShareGPT 格式）
            tokenizer: 分词器
            max_length: 最大长度
            template_args: 模板参数
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_args = template_args or {}
        self.format_type = format_type

        self.instruction_key = instruction_key
        self.input_key = input_key
        self.output_key = output_key
        self.conversations_key = conversations_key

        # 加载数据
        self.data = self._load_data(hf_args, data_path)

        logger.info(
            f"InjectDataset loaded with {len(self.data)} samples, format={format_type}"
        )

    def _load_data(
        self, hf_args: Optional[Dict[str, Any]], data_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """加载数据"""
        if data_path:
            import json

            with open(data_path, "r", encoding="utf-8") as f:
                if data_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            return data
        elif hf_args:
            dataset = load_dataset(**hf_args)
            return list(dataset)
        else:
            logger.warning("No data source specified")
            return []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        item = self.data[idx]

        if self.format_type == "alpaca":
            return self._process_alpaca(item)
        elif self.format_type == "sharegpt":
            return self._process_sharegpt(item)
        else:
            return self._process_custom(item)

    def _process_alpaca(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理 Alpaca 格式数据

        Alpaca 格式：
        {
            "instruction": "任务指令",
            "input": "可选的输入内容",
            "output": "期望的输出"
        }
        """
        instruction = item.get(self.instruction_key, "")
        input_text = item.get(self.input_key, "")
        output = item.get(self.output_key, "")

        # 构造输入
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + output

        return self._tokenize(prompt, full_text)

    def _process_sharegpt(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理 ShareGPT 格式数据

        ShareGPT 格式：
        {
            "conversations": [
                {"from": "human", "value": "用户输入"},
                {"from": "gpt", "value": "助手回复"},
                ...
            ]
        }
        """
        conversations = item.get(self.conversations_key, [])

        # 构造对话文本
        prompt_parts = []
        response_parts = []

        for i, turn in enumerate(conversations):
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if role in ["human", "user"]:
                prompt_parts.append(f"User: {content}")
            elif role in ["gpt", "assistant"]:
                if i == len(conversations) - 1:
                    response_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(f"Assistant: {content}")

        prompt = "\n".join(prompt_parts) + "\nAssistant: "
        full_text = prompt + (
            response_parts[0].replace("Assistant: ", "") if response_parts else ""
        )

        return self._tokenize(prompt, full_text)

    def _process_custom(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """处理自定义格式数据"""
        # 尝试多种常见字段名
        text = item.get("text", "")
        if not text:
            text = item.get("content", "")
        if not text:
            # 拼接所有字符串字段
            text = " ".join(str(v) for v in item.values() if isinstance(v, str))

        return self._tokenize(text, text)

    def _tokenize(self, prompt: str, full_text: str) -> Dict[str, Any]:
        """Tokenize 文本

        Args:
            prompt: 输入提示（不计入损失）
            full_text: 完整文本（包含输出）

        Returns:
            tokenized 字典
        """
        if self.tokenizer is None:
            return {"text": full_text, "prompt": prompt}

        # Tokenize 完整文本
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 创建 labels（prompt 部分设为 -100）
        labels = encodings["input_ids"].clone()

        # 计算 prompt 长度
        prompt_encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encodings["input_ids"].shape[1]

        # 将 prompt 部分的 labels 设为 -100（不计入损失）
        labels[0, :prompt_len] = -100

        # Padding token 也设为 -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


class AlpacaDataset(InjectDataset):
    """Alpaca 格式数据集"""

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(format_type="alpaca", tokenizer=tokenizer, **kwargs)


class ShareGPTDataset(InjectDataset):
    """ShareGPT 格式数据集"""

    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(format_type="sharegpt", tokenizer=tokenizer, **kwargs)
