"""
AdaLoRA 训练器
==============

实现 Adaptive Low-Rank Adaptation (AdaLoRA) 微调方法。
AdaLoRA 通过自适应地调整不同层的秩来优化参数效率。

参考论文: AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
https://arxiv.org/abs/2303.10512

核心思想：
1. 使用 SVD 分解来表示增量更新
2. 根据重要性得分动态调整每层的秩
3. 在训练过程中逐步剪枝不重要的奇异值
"""

import logging
from typing import Optional, List

from trainer.inject.base import InjectTrainer

logger = logging.getLogger(__name__)


class AdaLoRATrainer(InjectTrainer):
    """AdaLoRA 自适应微调训练器

    通过自适应秩分配实现更高效的参数微调。

    AdaLoRA vs LoRA：
    - LoRA：固定秩，所有层使用相同的 r
    - AdaLoRA：自适应秩，根据重要性动态调整

    Attributes:
        init_r: 初始秩（训练开始时的秩）
        target_r: 目标秩（训练结束时的秩）
        tinit: 秩调整的初始训练步数
        tfinal: 秩调整的结束训练步数
        deltaT: 秩更新的间隔步数
    """

    def __init__(
        self,
        init_r: int = 12,
        target_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        tinit: int = 0,
        tfinal: int = 0,
        deltaT: int = 1,
        beta1: float = 0.85,
        beta2: float = 0.85,
        orth_reg_weight: float = 0.5,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        *args,
        **kwargs,
    ):
        """初始化 AdaLoRA 训练器

        Args:
            init_r: 初始秩（开始训练时）
            target_r: 目标秩（训练结束时，通过剪枝达到）
            lora_alpha: 缩放因子
            lora_dropout: Dropout 比例
            target_modules: 应用 AdaLoRA 的模块
            tinit: 开始调整秩的训练步数
            tfinal: 停止调整秩的训练步数
            deltaT: 秩更新间隔
            beta1: 重要性得分的 EMA 系数（奇异值）
            beta2: 重要性得分的 EMA 系数（敏感度）
            orth_reg_weight: 正交正则化权重
            bias: 是否训练 bias
            task_type: 任务类型
        """
        self.init_r = init_r
        self.target_r = target_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        self.tinit = tinit
        self.tfinal = tfinal
        self.deltaT = deltaT
        self.beta1 = beta1
        self.beta2 = beta2
        self.orth_reg_weight = orth_reg_weight
        self.bias = bias
        self.task_type = task_type

        super().__init__(*args, **kwargs)

        # 应用 AdaLoRA 配置
        self._apply_adalora_config()

    def _apply_adalora_config(self):
        """应用 AdaLoRA 配置到模型"""
        try:
            from peft import AdaLoraConfig, get_peft_model, TaskType

            task_type_map = {
                "CAUSAL_LM": TaskType.CAUSAL_LM,
                "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
                "SEQ_CLS": TaskType.SEQ_CLS,
            }

            # 创建 AdaLoRA 配置
            adalora_config = AdaLoraConfig(
                init_r=self.init_r,
                target_r=self.target_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                tinit=self.tinit,
                tfinal=self.tfinal,
                deltaT=self.deltaT,
                beta1=self.beta1,
                beta2=self.beta2,
                orth_reg_weight=self.orth_reg_weight,
                bias=self.bias,
                task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            )

            # 应用到模型
            self.model = get_peft_model(self.model, adalora_config)

            self.model.print_trainable_parameters()
            logger.info(
                f"AdaLoRA config applied: init_r={self.init_r}, "
                f"target_r={self.target_r}, alpha={self.lora_alpha}"
            )

        except ImportError:
            logger.error("peft library not installed. Please run: pip install peft")
            raise

    def compute_loss(self, model, inputs, return_outputs=False):
        """计算 AdaLoRA 微调损失

        包含正交正则化以保持 SVD 分解的稳定性。

        Args:
            model: PEFT 包装后的模型
            inputs: 输入 batch
            return_outputs: 是否返回输出

        Returns:
            loss 或 (loss, outputs)
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss

        # 添加正交正则化（AdaLoRA 特性）
        if self.orth_reg_weight > 0 and hasattr(model, "base_model"):
            orth_loss = self._compute_orth_regularization(model)
            loss = loss + self.orth_reg_weight * orth_loss

        return (loss, outputs) if return_outputs else loss

    def _compute_orth_regularization(self, model) -> float:
        """计算正交正则化损失

        确保 AdaLoRA 的 SVD 分解保持正交性。

        Args:
            model: PEFT 模型

        Returns:
            正交正则化损失
        """
        import torch

        orth_loss = 0.0
        count = 0

        # 遍历 AdaLoRA 层
        for name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                # AdaLoRA 使用 P, Lambda, Q 分解
                # 这里简化为检查 A 和 B 的正交性
                if hasattr(module, "lora_E"):  # AdaLoRA 特有
                    A = module.lora_A["default"].weight
                    B = module.lora_B["default"].weight

                    # 计算 A^T A - I 的 Frobenius 范数
                    if A.shape[0] < A.shape[1]:
                        AAT = A @ A.T
                        I = torch.eye(AAT.shape[0], device=AAT.device)
                        orth_loss += torch.norm(AAT - I, p="fro")
                        count += 1

        return orth_loss / max(count, 1)
