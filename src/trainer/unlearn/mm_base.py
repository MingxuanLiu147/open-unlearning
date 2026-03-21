"""多模态 unlearning 训练基类（sidecar，不修改现有 trainer 体系）。

提供 Accelerator 驱动的训练循环，子类只需覆写 compute_loss()。
复用上游 MMUnlearner 的训练模式：AdamW + linear scheduler + 梯度裁剪。
"""

import json
import logging
import os
from abc import ABC, abstractmethod

import torch
from accelerate import Accelerator
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_scheduler

logger = logging.getLogger(__name__)


class MMUnlearnBase(ABC):
    """多模态 unlearning 训练基类。

    子类需要实现 compute_loss(model, batch) -> loss。
    基类负责：Accelerator 初始化、训练循环、梯度裁剪、模型保存。
    """

    def __init__(
        self,
        model,
        processor,
        forget_loader: DataLoader,
        retain_loader: DataLoader = None,
        *,
        lr: float = 5e-4,
        num_epochs: int = 5,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 0,
        save_dir: str = "./saves/mm_unlearn",
        grad_mask: dict = None,
    ):
        self.processor = processor
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        self.grad_mask = grad_mask

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps
        )

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        self.optimizer = AdamW(model.parameters(), lr=lr)
        total_steps = len(forget_loader) * num_epochs
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        prepare_args = [model, self.optimizer, forget_loader, self.lr_scheduler]
        if retain_loader is not None:
            prepare_args.append(retain_loader)
            (
                self.model,
                self.optimizer,
                self.forget_loader,
                self.lr_scheduler,
                self.retain_loader,
            ) = self.accelerator.prepare(*prepare_args)
        else:
            (
                self.model,
                self.optimizer,
                self.forget_loader,
                self.lr_scheduler,
            ) = self.accelerator.prepare(*prepare_args)

    @abstractmethod
    def compute_loss(self, model, batch) -> torch.Tensor:
        """子类实现具体的 unlearning 损失计算。"""
        ...

    def _apply_grad_mask(self):
        """在 backward 之后、optimizer.step 之前，对梯度应用掩码（用于 MMUnlearner）。"""
        if self.grad_mask is None:
            return
        for name, p in self.model.named_parameters():
            if p.grad is not None and name in self.grad_mask:
                p.grad *= self.grad_mask[name].to(p.grad.device)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            progress = tqdm(
                self.forget_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )
            for step, batch in enumerate(progress):
                with self.accelerator.accumulate(self.model):
                    loss = self.compute_loss(self.model, batch)
                    self.accelerator.backward(loss)
                    self._apply_grad_mask()
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                total_loss += loss.detach().item()
                progress.set_postfix(loss=total_loss / (step + 1))

            avg_loss = total_loss / len(self.forget_loader)
            logger.info("Epoch %d/%d - avg loss: %.4f", epoch + 1, self.num_epochs, avg_loss)

    def save_model(self, output_dir: str = None):
        save_path = output_dir or self.save_dir
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        if isinstance(unwrapped, PeftModel):
            unwrapped = unwrapped.merge_and_unload()
        if self.accelerator.is_main_process:
            os.makedirs(save_path, exist_ok=True)
            unwrapped.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            logger.info("Model saved to %s", save_path)
