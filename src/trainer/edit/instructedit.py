"""
InstructEdit 知识编辑器
======================

实现 Instruction-based Knowledge Editing (InstructEdit) 方法。
InstructEdit 在 MEND 的基础上引入指令模板，为编辑网络提供任务上下文，
使同一编辑网络能根据不同任务类型（事实修改、关系更新、反事实编辑等）
自适应调整编辑策略。

参考论文: InstructEdit: Instruction-based Knowledge Editing for Large Language Models
https://arxiv.org/abs/2402.16123

核心思想：
1. 继承 MEND 的梯度分解 + 编辑网络架构
2. 为不同编辑任务定义指令模板（task_type -> description）
3. 在调用 MEND edit() 之前，用模板包装原始 prompt
4. 模板为编辑网络提供额外的任务语义信息
"""

import logging
import random
from dataclasses import replace
from typing import Optional, Dict, Any, List, Union

from trainer.edit.base import EditRequest
from trainer.edit.mend import MENDEditor

logger = logging.getLogger(__name__)

DEFAULT_TASK_DESCRIPTIONS: Dict[str, List[str]] = {
    "fact_update": [
        "Update a factual association in the model's knowledge.",
        "Modify the model to reflect a corrected fact.",
        "Change an outdated factual statement to the current truth.",
    ],
    "relation_edit": [
        "Edit the relational knowledge between entities.",
        "Modify the relationship information stored in the model.",
        "Update the model's understanding of entity relationships.",
    ],
    "counterfactual": [
        "Apply a counterfactual edit to the model's knowledge.",
        "Introduce hypothetical knowledge that differs from reality.",
        "Modify the model to reason about an alternative scenario.",
    ],
    "knowledge_erase": [
        "Remove specific knowledge from the model.",
        "Erase the model's ability to recall this information.",
        "Unlearn a specific piece of stored knowledge.",
    ],
    "default": [
        "Edit the model's knowledge to produce the desired output.",
        "Apply a targeted knowledge modification.",
        "Modify the model's internal knowledge representation.",
    ],
}

INSTRUCTION_TEMPLATE = "Task: {task_type}\nDescription: {description}\nInput: {prompt}"


class InstructEditEditor(MENDEditor):
    """InstructEdit 指令引导知识编辑器

    继承 MEND 的全部能力，增加指令模板包装机制。
    编辑网络接收到的梯度来自包含任务描述的增强 prompt，
    从而在梯度空间中编码了任务类型信息。

    InstructEdit vs MEND:
    - MEND: 直接对原始 prompt 计算梯度
    - InstructEdit: 用 "Task / Description / Input" 模板包装后再计算

    Attributes:
        task_descriptions: task_type -> 描述文本列表的映射
        default_task_type: 未指定 task_type 时的默认任务
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        edit_lr: float = 1e-4,
        n_hidden: int = 128,
        rank: int = 1920,
        task_descriptions: Optional[Dict[str, List[str]]] = None,
        default_task_type: str = "default",
        *args,
        **kwargs,
    ):
        """初始化 InstructEdit 编辑器

        Args:
            layers: 编辑目标层
            edit_lr: 编辑网络学习率
            n_hidden: 编辑网络隐藏层大小
            rank: 梯度分解秩
            task_descriptions: 自定义任务描述映射，为 None 时使用内置默认值
            default_task_type: 默认任务类型
        """
        super().__init__(
            layers=layers,
            edit_lr=edit_lr,
            n_hidden=n_hidden,
            rank=rank,
            *args,
            **kwargs,
        )

        self.task_descriptions = task_descriptions or DEFAULT_TASK_DESCRIPTIONS
        self.default_task_type = default_task_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        task_type: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """执行 InstructEdit 知识编辑

        对每条请求：
        1. 确定 task_type（优先用参数，否则推断或用默认值）
        2. 从 task_descriptions 中采样一条描述
        3. 用指令模板包装原始 prompt
        4. 调用 MEND 的 edit() 完成实际编辑

        Args:
            requests: 编辑请求（单条或列表）
            task_type: 全局任务类型，若为 None 则逐条推断
            **kwargs: 传递给父类 edit() 的额外参数

        Returns:
            编辑结果字典
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        wrapped_requests = []
        for request in requests:
            resolved_type = task_type or self._infer_task_type(request)
            wrapped = self._wrap_with_instruction(request, resolved_type)
            wrapped_requests.append(wrapped)

        logger.info(
            "InstructEdit: wrapping %d requests with instruction templates",
            len(wrapped_requests),
        )

        return super().edit(wrapped_requests, **kwargs)

    # ------------------------------------------------------------------
    # Instruction wrapping
    # ------------------------------------------------------------------

    def _wrap_with_instruction(
        self, request: EditRequest, task_type: str
    ) -> EditRequest:
        """用指令模板包装单条编辑请求。

        Args:
            request: 原始编辑请求
            task_type: 任务类型标识

        Returns:
            prompt 被替换为模板包装后的新 EditRequest
        """
        description = self._select_description(task_type)

        wrapped_prompt = INSTRUCTION_TEMPLATE.format(
            task_type=task_type,
            description=description,
            prompt=request.prompt,
        )

        return replace(request, prompt=wrapped_prompt)

    def _select_description(self, task_type: str) -> str:
        """从 task_descriptions 中为指定任务采样一条描述。

        若 task_type 不在映射中，回退到 "default"。
        """
        descriptions = self.task_descriptions.get(task_type)
        if not descriptions:
            descriptions = self.task_descriptions.get(
                self.default_task_type, ["Edit the model's knowledge."]
            )
        return random.choice(descriptions)

    def _infer_task_type(self, request: EditRequest) -> str:
        """根据请求内容启发式推断任务类型。

        简单规则：
        - 有 target_old -> fact_update（修正事实）
        - prompt 中含 "if" / "counterfactual" -> counterfactual
        - 有 locality_inputs -> relation_edit（涉及关系保持）
        - 否则 -> default
        """
        if request.target_old:
            return "fact_update"

        prompt_lower = request.prompt.lower()
        if "if " in prompt_lower or "counterfactual" in prompt_lower:
            return "counterfactual"

        if request.locality_inputs:
            return "relation_edit"

        return self.default_task_type

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def register_task_type(self, task_type: str, descriptions: List[str]):
        """动态注册新的任务类型及其描述。

        Args:
            task_type: 任务类型标识
            descriptions: 该任务的描述文本列表
        """
        self.task_descriptions[task_type] = descriptions
        logger.info(
            "InstructEdit: registered task type '%s' with %d descriptions",
            task_type,
            len(descriptions),
        )
