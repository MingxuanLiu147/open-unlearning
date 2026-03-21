"""MMUnlearner 核心方法：GradDiff + SFRon 权重显著性掩码。

在 mm_grad_diff 的基础上，反向传播后对梯度应用 saliency mask，
只更新"对 forget 数据敏感但对 retain 数据不敏感"的参数子集。
"""

import logging

import torch

from trainer.unlearn.mm_grad_diff import MMGradDiff

logger = logging.getLogger(__name__)


class MMUnlearner(MMGradDiff):
    """MMUnlearner = GradDiff + grad_mask selective update。

    grad_mask 通过 scripts/generate_mm_mask.py 预生成，
    训练时通过 grad_mask_path 加载。
    """

    def __init__(self, *args, grad_mask_path: str = None, forget_alpha: float = 1.0, **kwargs):
        self.forget_alpha = forget_alpha

        loaded_mask = None
        if grad_mask_path:
            data = torch.load(grad_mask_path, map_location="cpu")
            loaded_mask = data.get("weight", data)
            logger.info(
                "Loaded grad mask from %s (%d parameters)", grad_mask_path, len(loaded_mask)
            )

        kwargs["grad_mask"] = loaded_mask
        super().__init__(*args, **kwargs)
