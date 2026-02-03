"""
Knowledge Editing (知识编辑) 训练器基类
======================================

本模块实现知识编辑的基础训练器。
支持单次编辑和批量编辑两种模式。

知识编辑的核心目标：
1. Reliability（可靠性）：编辑后的知识能够正确输出
2. Generalization（泛化性）：对改写表达也能正确输出
3. Locality（局部性）：不影响无关知识
4. Portability（可移植性）：编辑后的知识能迁移到相关推理
"""

import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import torch
from torch import nn
from trainer.base import FinetuneTrainer

logger = logging.getLogger(__name__)


@dataclass
class EditRequest:
    """编辑请求数据结构

    Attributes:
        prompt: 触发编辑的提示语
        subject: 编辑的主体（实体）
        target_new: 新的目标输出
        target_old: 原始目标输出（可选）
        locality_inputs: 局部性测试输入（可选）
        portability_inputs: 可移植性测试输入（可选）
    """

    prompt: str
    subject: str
    target_new: str
    target_old: Optional[str] = None
    locality_inputs: Optional[List[Dict[str, str]]] = None
    portability_inputs: Optional[List[Dict[str, str]]] = None


class EditTrainer(FinetuneTrainer):
    """知识编辑训练器基类

    所有知识编辑方法（ROME、MEMIT、MEND 等）都应继承此类。
    提供统一的编辑接口和评估方法。

    Attributes:
        edit_requests: 待编辑的请求列表
        layers: 编辑的目标层
        preserve_memory: 是否保留编辑历史
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        preserve_memory: bool = True,
        *args,
        **kwargs,
    ):
        """初始化知识编辑训练器

        Args:
            layers: 编辑的目标层索引列表
            preserve_memory: 是否保留编辑历史（用于连续编辑）
        """
        self.layers = layers or [5, 6, 7, 8]
        self.preserve_memory = preserve_memory
        self.edit_history: List[EditRequest] = []

        super().__init__(*args, **kwargs)

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行知识编辑

        子类需要实现此方法的具体编辑逻辑。

        Args:
            requests: 单个或多个编辑请求
            **kwargs: 额外参数

        Returns:
            包含编辑结果和指标的字典
        """
        raise NotImplementedError("Subclass must implement edit() method")

    def batch_edit(
        self, requests: List[EditRequest], batch_size: int = 1, **kwargs
    ) -> List[Dict[str, Any]]:
        """批量执行知识编辑

        Args:
            requests: 编辑请求列表
            batch_size: 批大小
            **kwargs: 额外参数

        Returns:
            编辑结果列表
        """
        results = []
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            result = self.edit(batch, **kwargs)
            results.append(result)

            if self.preserve_memory:
                self.edit_history.extend(batch)

        return results

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算编辑损失

        知识编辑通常不使用标准的训练损失，
        而是通过直接修改模型权重实现。
        此方法作为兼容接口保留。
        """
        # 对于需要训练的编辑方法（如 MEND），在子类中重写
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """根据名称获取模型子模块

        Args:
            model: 模型实例
            name: 模块名称（如 "model.layers.5.mlp.down_proj"）

        Returns:
            目标子模块
        """
        parts = name.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _set_module_by_name(self, model: nn.Module, name: str, new_module: nn.Module):
        """根据名称设置模型子模块

        Args:
            model: 模型实例
            name: 模块名称
            new_module: 新的子模块
        """
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)

        last_part = parts[-1]
        if last_part.isdigit():
            parent[int(last_part)] = new_module
        else:
            setattr(parent, last_part, new_module)
