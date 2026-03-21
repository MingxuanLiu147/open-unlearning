"""多模态 NPO (Negative Preference Optimization) 遗忘方法。

使用 oracle（未遗忘）模型作参考，在 forget 数据上优化 NPO 目标。
oracle 模型需要单独指定路径，frozen 状态参与前向计算。
"""

import logging

import torch
import torch.nn.functional as F

from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)


class MMNPO(MMUnlearnBase):
    """NPO: -logsigmoid(beta * (current_loss - oracle_loss)).mean() * 2 / beta。"""

    def __init__(self, *args, oracle_model=None, beta: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if oracle_model is None:
            raise ValueError(
                "MMNPO requires oracle_model. "
                "Pass a pre-loaded frozen model or specify oracle_model_path in config."
            )
        self.oracle_model = oracle_model
        self.oracle_model.eval()
        for p in self.oracle_model.parameters():
            p.requires_grad = False
        self.oracle_model = self.accelerator.prepare(self.oracle_model)
        logger.info("NPO: oracle model loaded (frozen), beta=%.2f", self.beta)

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        current_loss = outputs.loss

        with torch.no_grad():
            oracle_outputs = self.oracle_model(**batch)
            oracle_loss = oracle_outputs.loss

        neg_log_ratios = current_loss - oracle_loss
        loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
        return loss
