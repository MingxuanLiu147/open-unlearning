"""多模态 KL_Min 遗忘方法。

forget 数据梯度上升 + retain 数据梯度下降并最小化与 vanilla 模型的 KL 散度。
vanilla 模型通过 deepcopy 创建，frozen 状态用于计算参考 logits。
"""

import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
from tqdm import tqdm

from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)


class MMKLMin(MMUnlearnBase):
    """forget 梯度上升 + retain CE + KL(current, vanilla) 最小化。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.vanilla_model = deepcopy(unwrapped)
        self.vanilla_model.eval()
        for p in self.vanilla_model.parameters():
            p.requires_grad = False
        self.vanilla_model = self.accelerator.prepare(self.vanilla_model)
        logger.info("KL_Min: vanilla model created via deepcopy (frozen)")

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        return -outputs.loss

    def train(self):
        if self.retain_loader is None:
            raise ValueError("MMKLMin requires retain_loader")

        for epoch in range(self.num_epochs):
            self.model.train()
            total_forget_loss = 0.0
            total_retain_loss = 0.0

            forget_iter = iter(self.forget_loader)
            n_iters = len(self.retain_loader)
            forget_freq = max(1, n_iters // len(self.forget_loader))

            progress = tqdm(
                enumerate(self.retain_loader),
                total=n_iters,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            for step, retain_batch in progress:
                if step % forget_freq == 0:
                    try:
                        forget_batch = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(self.forget_loader)
                        forget_batch = next(forget_iter)

                    with self.accelerator.accumulate(self.model):
                        forget_out = self.model(**forget_batch)
                        loss_forget = -forget_out.loss
                        self.accelerator.backward(loss_forget)
                        self._apply_grad_mask()
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    total_forget_loss += loss_forget.detach().item()

                with self.accelerator.accumulate(self.model):
                    retain_out_current = self.model(**retain_batch)
                    with torch.no_grad():
                        retain_out_vanilla = self.vanilla_model(**retain_batch)

                    kl_div = F.kl_div(
                        F.log_softmax(retain_out_current.logits, dim=-1),
                        F.softmax(retain_out_vanilla.logits, dim=-1),
                        reduction="batchmean",
                    )
                    loss_retain = retain_out_current.loss + kl_div
                    self.accelerator.backward(loss_retain)
                    self._apply_grad_mask()
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                total_retain_loss += loss_retain.detach().item()

                progress.set_postfix(
                    f_loss=total_forget_loss / max(1, (step // forget_freq) + 1),
                    r_loss=total_retain_loss / (step + 1),
                )

            logger.info(
                "Epoch %d - forget_loss: %.4f, retain_loss: %.4f",
                epoch + 1,
                total_forget_loss / max(1, n_iters // forget_freq),
                total_retain_loss / n_iters,
            )
