"""
MM-GRACE: Multimodal GRACE Knowledge Editing
=============================================

Extends GRACEEditor with multimodal tokenization so the discrete key-value
codebook receives visual information during editing.

Reference:
  - GRACE: https://arxiv.org/abs/2211.11031
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch

from trainer.edit.base import EditRequest
from trainer.edit.grace import GRACEAdapter, GRACEEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMGRACEEditor(GRACEEditor, MMEditMixin):
    """Multimodal GRACE editor.

    Overrides the edit loop to tokenize inputs through the multimodal
    processor so that the model forward pass includes visual tokens.
    """

    def __init__(self, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)

    def _apply_grace_edit(self, request: EditRequest):
        model = self.model
        device = next(model.parameters()).device
        adapter = self._get_adapter()

        if request.image is not None and self.processor is not None:
            tokens = self.mm_tokenize(
                request.prompt, request.image, request.target_new,
                device=device,
            )
        else:
            full_text = request.prompt + " " + request.target_new
            tokenizer = self.tokenizer
            prompt_ids = tokenizer(
                request.prompt, return_tensors="pt",
            )["input_ids"].to(device)
            full_ids = tokenizer(
                full_text, return_tensors="pt",
            )["input_ids"].to(device)
            labels = full_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100
            tokens = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
                "labels": labels,
            }

        key_id = (tokens["labels"][0] == -100).sum().item() - 1
        adapter.key_id = key_id
        adapter._training_edit = True
        adapter.edit_label = tokens["labels"]
        adapter.edit_id = self._edit_count

        losses: List[float] = []
        for i in range(self.n_iter):
            adapter.iter = i

            if request.image is not None and self.processor is not None:
                outputs = self.mm_forward(model, tokens)
            else:
                outputs = model(**tokens)

            if i == 0:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=self.edit_lr,
                )

            loss = outputs.loss

            if self.val_reg > 0 and adapter.has_keys:
                loss = loss + self.val_reg * adapter.values.norm()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        adapter._training_edit = False
        self._edit_count += 1

        logger.info(
            "MM-GRACE edit: '%s' -> '%s' (loss=%.4f, codebook=%d)",
            request.prompt,
            request.target_new,
            losses[-1] if losses else float("inf"),
            adapter.codebook_size,
        )
