"""
MM-UniKE: Multimodal Unified Knowledge Editing
===============================================

Extends UniKEEditor with multimodal capabilities via MMEditMixin.
When image is present, the accommodation phase uses multimodal
forward passes to compute subject keys and optimise target values
with visual context.

Reference:
  - UniKE: https://arxiv.org/abs/2409.19872
  - MMEdit: https://arxiv.org/abs/2310.08475
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from trainer.edit.base import EditRequest
from trainer.edit.unike import UniKEEditor
from trainer.edit.mm_mixin import MMEditMixin

logger = logging.getLogger(__name__)


class MMUniKEEditor(UniKEEditor, MMEditMixin):
    """Multimodal UniKE editor.

    Overrides the accommodation phase to use multimodal forward passes
    when images are present, while keeping the assimilation (ICL store)
    phase text-based.
    """

    def __init__(self, processor=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if processor is not None:
            self.init_mm(processor)

    def _assimilate(self, request: EditRequest) -> torch.Tensor:
        """Assimilation: store fact with image context prefix if available."""
        if request.image is not None:
            enriched_fact = f"[Visual] {request.prompt} {request.target_new}"
        else:
            enriched_fact = f"{request.prompt} {request.target_new}"

        entry = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
            "fact": enriched_fact,
            "has_image": request.image is not None,
        }
        self.knowledge_store.append(entry)

        if self.use_icl_examples:
            try:
                from sentence_transformers import util as st_util
                device = next(self.model.parameters()).device
                emb = self.sentence_model.encode(
                    enriched_fact, show_progress_bar=False, convert_to_tensor=True,
                )
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                emb = st_util.normalize_embeddings(emb.to(device))
                if self.knowledge_embeddings is None:
                    self.knowledge_embeddings = emb
                else:
                    self.knowledge_embeddings = torch.cat(
                        [self.knowledge_embeddings, emb], dim=0,
                    )
                return emb.squeeze(0)
            except ImportError:
                pass

        return torch.zeros(1)

    def _compute_subject_key(
        self, request: EditRequest, layer_idx: int, device: torch.device,
    ) -> torch.Tensor:
        """Compute subject key using multimodal forward when image is present."""
        if request.image is None or self.processor is None:
            return super()._compute_subject_key(request, layer_idx, device)

        model = self.model
        tokens = self.mm_tokenize(
            request.prompt, request.image, request.target_new, device=device,
        )

        layers_container = self._get_layers_container(model)
        hook_output = {}

        def hook_fn(module, inp, out):
            if isinstance(out, tuple):
                hook_output["hidden"] = out[0]
            else:
                hook_output["hidden"] = out

        handle = layers_container[layer_idx].register_forward_hook(hook_fn)
        with torch.no_grad():
            self.mm_forward(model, tokens)
        handle.remove()

        hidden = hook_output["hidden"]
        labels = tokens["labels"]
        target_start = (labels[0] != -100).nonzero()
        if target_start.numel() > 0:
            pos = max(0, target_start[0].item() - 1)
        else:
            pos = hidden.shape[1] - 1

        return hidden[0, pos].detach().float()

    def _optimise_target_value(
        self, request: EditRequest, layer_idx: int,
        key: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """Optimise target value using multimodal forward when image is present."""
        if request.image is None or self.processor is None:
            return super()._optimise_target_value(request, layer_idx, key, device)

        model = self.model
        tokens = self.mm_tokenize(
            request.prompt, request.image, request.target_new, device=device,
        )

        layers_container = self._get_layers_container(model)
        target_layer = layers_container[layer_idx]
        target_weight = self._get_target_weight(layer_idx)
        current_value = F.linear(key, target_weight.float())
        delta = torch.zeros_like(current_value, requires_grad=True)

        optimizer = torch.optim.Adam([delta], lr=self.v_lr)
        model_dtype = next(model.parameters()).dtype

        for step in range(self.v_num_grad_steps):
            def edit_hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                else:
                    h = out.clone()
                h[0, -1] = h[0, -1] + delta.to(h.dtype)
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = target_layer.register_forward_hook(edit_hook)
            with torch.amp.autocast("cuda", dtype=model_dtype, enabled=model_dtype != torch.float32):
                outputs = self.mm_forward(model, tokens)
            handle.remove()

            loss = outputs.loss + self.kl_factor * delta.norm()

            optimizer.zero_grad()
            loss.backward()

            if self.clamp_norm_factor > 0 and delta.grad is not None:
                max_norm = self.clamp_norm_factor * current_value.norm()
                torch.nn.utils.clip_grad_norm_([delta], max_norm.item())

            optimizer.step()

        return (current_value + delta).detach().float()
