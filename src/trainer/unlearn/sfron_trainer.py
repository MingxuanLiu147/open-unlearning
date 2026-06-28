"""SFRON 文本 Unlearning 训练器：GradDiff + Fisher 权重显著性掩码。

这是多模态 ``MMUnlearner``（``trainer/unlearn/mmunlearner.py``）在单模态文本
上的对应实现：在 GradDiff 的 forget/retain 梯度差分基础上，对反向传播得到的
梯度施加一个**权重显著性掩码**，只更新"对 forget 数据敏感、但对 retain/preserve
数据不敏感"的参数子集。

掩码由 ``trainer/unlearn/sfron.py`` 的 :func:`generate_saliency_mask` 预先生成
（基于 Fisher 信息：``mask = (forget_fisher / preserve_fisher) >= threshold``），
训练时通过 ``grad_mask_path`` 加载。掩码方向与 MM 版保持一致——
``grad *= mask``，即"保留 forget-显著位置的梯度，置零其余"。

掩码通过 ``param.register_hook`` 在反向传播时逐参数施加，因此与梯度累积
（gradient accumulation）天然兼容：每个 micro-batch 的梯度在累加进 ``.grad``
之前就已被掩码，无需手动在 optimizer.step 前介入。

用法::

    # 1) 预生成掩码（见 SFRON.generate_mask 或 sfron.generate_saliency_mask）
    # 2) 训练时指定 grad_mask_path
    python src/train.py --config-name=unlearn.yaml trainer=SFRON \
        trainer.method_args.grad_mask_path=saves/mask/mask.pt ...

若未提供 ``grad_mask_path``（或文件缺失），训练器会发出告警并**退化为普通
GradDiff**，保证流程端到端可跑通。
"""

import logging
import os
from typing import Dict, List, Optional

import torch

from trainer.unlearn.grad_diff import GradDiff

logger = logging.getLogger(__name__)


class SFRON(GradDiff):
    """GradDiff + SFRon Fisher 显著性掩码的选择性梯度更新。

    Attributes:
        grad_mask: ``{参数名: bool 掩码}``；None 时退化为标准 GradDiff。
    """

    def __init__(
        self,
        *args,
        grad_mask_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.grad_mask: Optional[Dict[str, torch.Tensor]] = None
        if grad_mask_path:
            if os.path.exists(grad_mask_path):
                data = torch.load(grad_mask_path, map_location="cpu")
                # 兼容两种存储格式：{"weight": {...}} 或直接 {...}
                self.grad_mask = data.get("weight", data) if isinstance(data, dict) else data
                logger.info(
                    "SFRON loaded saliency mask from %s (%d parameters)",
                    grad_mask_path, len(self.grad_mask),
                )
            else:
                logger.warning(
                    "SFRON grad_mask_path '%s' not found; falling back to "
                    "plain GradDiff (no selective masking).",
                    grad_mask_path,
                )

        self._mask_hooks: List = []
        if self.grad_mask is not None:
            self._register_mask_hooks()

    def _register_mask_hooks(self):
        """为每个被掩码的参数注册反向钩子，在梯度生成时即施加掩码。

        逐参数注册（而非在 optimizer.step 前统一处理）可保证梯度累积下的
        正确性：每个 micro-batch 的梯度在累加前就被掩码。
        """
        registered, skipped = 0, 0
        for name, param in self.model.named_parameters():
            if name not in self.grad_mask:
                continue
            mask = self.grad_mask[name]
            # 跳过形状不匹配的占位掩码（generate_saliency_mask 的异常回退会写入 zeros(1)）
            if tuple(mask.shape) != tuple(param.shape):
                skipped += 1
                continue

            def make_hook(m: torch.Tensor):
                def hook(grad):
                    return grad * m.to(dtype=grad.dtype, device=grad.device)
                return hook

            self._mask_hooks.append(param.register_hook(make_hook(mask)))
            registered += 1

        logger.info(
            "SFRON registered %d gradient masks (%d skipped due to shape mismatch)",
            registered, skipped,
        )

    @staticmethod
    def generate_mask(
        model,
        forget_loader,
        preserve_loader,
        modules: Optional[List[str]] = None,
        threshold: float = 1.0,
        save_path: str = "saves/sfron_mask",
        max_batches: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """生成并保存 Fisher 权重显著性掩码（薄封装 sfron.generate_saliency_mask）。

        Args:
            model: 待计算 Fisher 的模型。
            forget_loader / preserve_loader: forget 与 retain DataLoader。
            modules: 参与计算的参数名关键词；文本 LLM 默认 ``["model"]``。
            threshold: 显著性阈值。
            save_path: 掩码保存目录（写出 ``mask.pt``）。
            max_batches: 每侧 Fisher 估计的最大 batch 数（小实验加速用）。

        Returns:
            ``{参数名: bool 掩码}``。
        """
        from trainer.unlearn.sfron import generate_saliency_mask

        return generate_saliency_mask(
            model,
            forget_loader,
            preserve_loader,
            modules=modules or ["model"],
            threshold=threshold,
            save_path=save_path,
            max_batches=max_batches,
        )
