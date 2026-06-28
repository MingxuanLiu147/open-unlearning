"""
MMVKD (Visual Knowledge Distillation) for multimodal unlearning.

Reference: arXiv:2512.11325 — "Robust MLLM Unlearning via Visual
           Knowledge Distillation"

Core idea:
    Disentangle visual and textual knowledge within MLLMs.  Use the
    pre-unlearning (oracle) model's intermediate visual representations
    as supervision to selectively erase target visual knowledge while
    preserving textual knowledge.

    1. Freeze the LLM backbone — only fine-tune visual encoder and
       connector/projector layers.
    2. On forget data: gradient ascent on the standard CE loss (erase
       visual association).
    3. On retain data: distillation loss that aligns current model's
       intermediate visual features with the oracle model's features
       (preserve visual understanding).

    Total loss = alpha * forget_loss + beta * distill_loss

This trainer follows the MMNPO pattern for oracle model loading.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from trainer.unlearn.mm_base import MMUnlearnBase

logger = logging.getLogger(__name__)


def _get_visual_features(model, batch):
    """Extract intermediate visual representations from the vision encoder.

    Works with Qwen2-VL, LLaVA, InternVL — looks for common attribute
    names for the vision tower / visual encoder.
    """
    pixel_values = batch.get("pixel_values")
    if pixel_values is None:
        pixel_values = batch.get("pixel_values_videos")
    if pixel_values is None:
        return None

    vision_tower = None
    for attr in ("vision_tower", "visual", "vision_model", "img_encoder"):
        vision_tower = getattr(model, attr, None)
        if vision_tower is None:
            inner = getattr(model, "model", None)
            if inner is not None:
                vision_tower = getattr(inner, attr, None)
        if vision_tower is not None:
            break

    if vision_tower is None:
        return None

    with torch.set_grad_enabled(model.training):
        features = vision_tower(pixel_values)
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
    return features


class MMVKD(MMUnlearnBase):
    """Visual Knowledge Distillation for multimodal unlearning.

    Requires an oracle (pre-unlearning) model passed via
    ``oracle_model`` kwarg, same as MMNPO.
    """

    def __init__(
        self,
        *args,
        oracle_model=None,
        alpha: float = 1.0,
        beta: float = 1.0,
        freeze_llm: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta

        if oracle_model is None:
            raise ValueError("MMVKD requires oracle_model for visual distillation.")
        self.oracle_model = oracle_model
        self.oracle_model.eval()
        for p in self.oracle_model.parameters():
            p.requires_grad = False
        self.oracle_model = self.accelerator.prepare(self.oracle_model)

        if freeze_llm:
            self._freeze_llm_params()

        logger.info(
            "MMVKD: oracle loaded (frozen), alpha=%.2f, beta=%.2f, freeze_llm=%s",
            self.alpha, self.beta, freeze_llm,
        )

    def _freeze_llm_params(self):
        """Freeze LLM backbone parameters, keep vision encoder trainable."""
        model = self.accelerator.unwrap_model(self.model)
        frozen, trainable = 0, 0
        for name, param in model.named_parameters():
            is_visual = any(
                kw in name.lower()
                for kw in ("vision", "visual", "img_", "image_", "projector", "connector", "mm_projector")
            )
            if is_visual:
                param.requires_grad = True
                trainable += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        logger.info(
            "VKD freeze: %d params frozen (LLM), %d trainable (vision)",
            frozen, trainable,
        )

    def compute_loss(self, model, batch) -> torch.Tensor:
        outputs = model(**batch)
        forget_loss = -outputs.loss * self.alpha

        oracle_features = _get_visual_features(self.oracle_model, batch)
        current_features = _get_visual_features(model, batch)

        distill_loss = torch.tensor(0.0, device=forget_loss.device)
        if oracle_features is not None and current_features is not None:
            min_len = min(oracle_features.size(1), current_features.size(1))
            oracle_feat = oracle_features[:, :min_len]
            current_feat = current_features[:, :min_len]
            distill_loss = F.mse_loss(current_feat, oracle_feat.detach())

        loss = forget_loss + self.beta * distill_loss
        return loss
