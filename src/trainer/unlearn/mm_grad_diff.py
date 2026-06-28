"""多模态 GradDiff 遗忘方法。

交替训练：forget 数据梯度上升 + retain 数据梯度下降。
retain_loader 作为主循环，每 forget_freq 步采一次 forget batch。
"""

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)


class MMGradDiff(MMUnlearnBase):
    """forget 梯度上升 + retain 梯度下降，交替训练。"""

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        return -outputs.loss

    def compute_forget_loss(self, model, batch) -> torch.Tensor:
        """Compute the forget-side objective for alternating training."""
        outputs = model(**batch)
        return -outputs.loss

    def compute_retain_loss(self, model, batch) -> torch.Tensor:
        """Compute the retain-side objective for alternating training."""
        outputs = model(**batch)
        return outputs.loss

    def train(self):
        if self.retain_loader is None:
            raise ValueError("MMGradDiff requires retain_loader")

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
                        loss_forget = self.compute_forget_loss(self.model, forget_batch)
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
                    loss_retain = self.compute_retain_loss(self.model, retain_batch)
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
