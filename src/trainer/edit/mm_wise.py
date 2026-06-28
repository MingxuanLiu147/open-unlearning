"""
MM-WISE: Multimodal WISE Knowledge Editing
===========================================

Extends WISEEditor with multimodal tokenization so the adapter receives
visual information during the fine-tuning loop.

Reference:
  - WISE: https://arxiv.org/abs/2405.14768
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from trainer.edit.base import EditRequest
from trainer.edit.wise import WISEEditor, _activation_distance
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMWISEEditor(WISEEditor, MMEditMixin):
    """Multimodal WISE editor.

    Overrides _apply_wise_edit to handle image inputs through the
    multimodal processor while reusing the WISEAdapter machinery.
    """

    def __init__(self, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)

    def _apply_wise_edit(self, request: EditRequest):
        model = self.model
        device = next(model.parameters()).device
        adapter = self._get_adapter()

        use_mm = request.image is not None and self.processor is not None

        if use_mm:
            tokens = self.mm_tokenize(
                request.prompt, request.image, request.target_new,
                device=device,
            )
        else:
            tokenizer = self.tokenizer
            full_text = request.prompt + " " + request.target_new
            prompt_ids = tokenizer(
                request.prompt, return_tensors="pt",
            )["input_ids"].to(device)
            full_ids = tokenizer(
                full_text, return_tensors="pt",
            )["input_ids"].to(device)
            labels = full_ids.clone()
            labels[:, : prompt_ids.shape[1]] = -100

            if request.locality_inputs:
                loc_text = self._first_locality(request)
                if loc_text:
                    loc_ids = tokenizer(
                        loc_text,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=full_ids.shape[1],
                        truncation=True,
                    )["input_ids"].to(device)
                    full_ids = torch.cat([full_ids, loc_ids], dim=0)
                    labels = torch.cat(
                        [labels, torch.full_like(loc_ids, -100)], dim=0,
                    )

            tokens = {
                "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids),
                "labels": labels,
            }

        self._edit_history.append(
            {
                k: v.detach().cpu()
                for k, v in tokens.items()
                if isinstance(v, torch.Tensor)
            }
        )

        adapter.editing = True
        adapter.set_parameter_tunable()

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.generate_activation_mask()

        last_prompt_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        optimizer = torch.optim.SGD(
            [adapter.new_weight], lr=self.edit_lr, weight_decay=1e-5,
        )

        model_dtype = next(model.parameters()).dtype
        use_amp = model_dtype in (torch.float16, torch.bfloat16)

        for i in range(self.n_iter):
            with torch.amp.autocast("cuda", dtype=model_dtype, enabled=use_amp):
                if use_mm:
                    outputs = self.mm_forward(model, tokens)
                else:
                    outputs = model(**tokens)

            logits = outputs.logits
            shift_logits = logits[:1, :-1, :].contiguous()
            shift_labels = tokens["labels"][:1, 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="none")
            per_tok = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(1, -1)

            lm = torch.zeros_like(per_tok, dtype=torch.bool)
            for b, col in enumerate(last_prompt_loc[:1]):
                lm[b, max(0, col - 1):] = True
            ft_loss = (per_tok * lm).sum() / lm.sum().clamp(min=1)

            act_loss = torch.tensor(0.0, device=device)
            orig_out = adapter.original_layer_output
            new_out = adapter.new_weight_layer_output
            if orig_out is not None and new_out is not None:
                in_s = _activation_distance(orig_out[:1], new_out[:1])
                if orig_out.shape[0] > 1:
                    out_s = _activation_distance(
                        orig_out[1:], new_out[1:],
                    )
                else:
                    out_s = torch.tensor(0.0, device=device)
                act_loss = (
                    F.relu(out_s - in_s + self.gamma)
                    + F.relu(out_s - self.alpha)
                    + F.relu(self.beta - in_s)
                )

            loss = ft_loss + act_loss

            if i == self.n_iter - 1:
                adapter.save_editing_activation()

            optimizer.zero_grad()
            loss.backward()
            adapter.mask_new_weight_gradient()
            optimizer.step()

            if self.norm_constraint is not None:
                with torch.no_grad():
                    orig_w = adapter._original_weight_cpu.to(device)
                    adapter.new_weight.clamp_(
                        orig_w - self.norm_constraint,
                        orig_w + self.norm_constraint,
                    )

        adapter.editing = False
        adapter.editing_total_cnt += 1

        if adapter.editing_total_cnt % self.merge_freq == 0:
            adapter.merge_weight()

        logger.info(
            "MM-WISE edit: '%s' -> '%s' (ft_loss=%.4f)",
            request.prompt,
            request.target_new,
            ft_loss.item(),
        )

    @staticmethod
    def _first_locality(request: EditRequest) -> Optional[str]:
        loc = request.locality_inputs
        if isinstance(loc, dict):
            for entries in loc.values():
                if entries and isinstance(entries, list):
                    return entries[0].get("prompt")
        elif isinstance(loc, list) and loc:
            return loc[0].get("prompt")
        return None
