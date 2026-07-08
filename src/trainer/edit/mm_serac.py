"""
MM-SERAC: Multimodal SERAC Knowledge Editing
=============================================

Extends SERACEditor with multimodal capabilities.  The scope classifier
and counterfactual model can receive image inputs through the multimodal
processor.

Reference:
  - SERAC: https://arxiv.org/abs/2206.06520
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.optim import Adam

from trainer.edit.base import EditRequest
from trainer.edit.serac import SERACEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMSERACEditor(SERACEditor, MMEditMixin):
    """Multimodal SERAC editor.

    Overrides the edit and predict flows to handle image inputs through the
    multimodal processor.  The scope classifier receives the last hidden
    state of the multimodal forward pass.
    """

    def __init__(self, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)

    def edit(
        self,
        requests: Union[EditRequest, List[EditRequest]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Store multimodal edits and fine-tune the counterfactual model."""
        if isinstance(requests, EditRequest):
            requests = [requests]

        self._ensure_initialized()

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
        }

        for request in requests:
            try:
                self._mm_store_and_finetune(request)
                results["edited_count"] += 1
            except Exception as exc:
                logger.error("MM-SERAC edit failed: %s", exc)
                results["success"] = False

        return results

    def _mm_store_and_finetune(self, request: EditRequest):
        """Store the edit and fine-tune the counterfactual model."""
        entry = {
            "prompt": request.prompt,
            "target_new": request.target_new,
            "subject": request.subject,
        }
        if request.image is not None:
            entry["has_image"] = True
        self.edit_memory.append(entry)

        model = self.model
        device = next(model.parameters()).device

        if request.image is not None and self.processor is not None:
            tokens = self.mm_tokenize(
                request.prompt, request.image, request.target_new, device=device,
            )
        else:
            tokenizer = self.tokenizer
            full_text = request.prompt + " " + request.target_new
            prompt_ids = tokenizer(
                request.prompt, return_tensors="pt",
            )["input_ids"].to(device)
            full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
            labels = full_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100
            tokens = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
                "labels": labels,
            }

        cf_model = self.counterfactual_model or model
        optimizer = Adam(cf_model.parameters(), lr=self.edit_lr)

        for _ in range(self.num_edit_steps):
            if request.image is not None and self.processor is not None:
                outputs = self.mm_forward(cf_model, tokens)
            else:
                outputs = cf_model(**tokens)

            loss = outputs.loss * self.cedit
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info(
            "MM-SERAC edit stored and fine-tuned: '%s' -> '%s' (memory=%d)",
            request.prompt,
            request.target_new,
            len(self.edit_memory),
        )

    def _get_hidden_state(self, model, request: EditRequest):
        """Get the last hidden state for classifier input."""
        device = next(model.parameters()).device

        if request.image is not None and self.processor is not None:
            tokens = self.mm_tokenize(
                request.prompt, request.image, "", device=device,
            )
            with torch.no_grad():
                outputs = self.mm_forward(
                    model, tokens, output_hidden_states=True,
                )
        else:
            tokenizer = self.tokenizer
            inputs = tokenizer(
                request.prompt, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]
        return hidden[:, -1, :]
