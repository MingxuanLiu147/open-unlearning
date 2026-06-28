"""
NMKE: Neuron-level Model Knowledge Editing
===========================================

NeurIPS 2025 — neuron-level lifelong editing via attribution + dynamic sparse
masking.  Instead of rank-one updates to entire weight matrices, NMKE:

1. **Neuron Attribution**: identifies the most relevant neurons for a given
   fact by computing integrated-gradient-style attribution scores over the
   MLP output projection weight.
2. **Dynamic Sparse Mask**: constructs a binary mask selecting the top-k%
   attributed neurons, so each edit only touches a small, disjoint subset
   of parameters.
3. **Targeted Value Update**: optimises only the masked weights to encode
   the new fact, leaving others frozen — enabling many sequential edits
   with minimal interference.

Paper: https://arxiv.org/abs/2410.xxxxx (NeurIPS 2025)

Core advantage over ROME/MEMIT: edits are localised to specific neurons
rather than adding a global rank-one update, dramatically reducing
cross-edit interference for lifelong editing scenarios.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)


class NMKEEditor(EditTrainer):
    """Neuron-level Model Knowledge Editing.

    Attributes:
        layers: target layer indices for editing
        attribution_steps: number of integration steps for neuron attribution
        sparsity_ratio: fraction of neurons to mask (top-k selection)
        v_lr: learning rate for value optimisation
        v_num_grad_steps: optimisation steps
        clamp_norm_factor: gradient clipping factor
        attribution_batch_size: batch size for attribution computation
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        attribution_steps: int = 20,
        sparsity_ratio: float = 0.01,
        v_lr: float = 0.1,
        v_num_grad_steps: int = 25,
        clamp_norm_factor: float = 5.0,
        attribution_batch_size: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(layers=layers, *args, **kwargs)
        self.attribution_steps = attribution_steps
        self.sparsity_ratio = sparsity_ratio
        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.clamp_norm_factor = clamp_norm_factor
        self.attribution_batch_size = attribution_batch_size
        self._neuron_usage: Dict[int, torch.Tensor] = {}

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "per_edit": [],
        }

        for request in requests:
            try:
                info = self._nmke_edit(request)
                results["edited_count"] += 1
                results["per_edit"].append(info)
            except Exception as exc:
                logger.error("NMKE edit failed for '%s': %s", request.prompt, exc)
                results["success"] = False
                results["per_edit"].append({"error": str(exc)})

        return results

    def _nmke_edit(self, request: EditRequest) -> Dict[str, Any]:
        model = self.model
        tokenizer = self.tokenizer
        layer_indices = self._resolve_layer_indices(self.layers)
        device = self._input_device(model)
        info: Dict[str, Any] = {"layers": layer_indices}

        for layer_idx in layer_indices:
            attribution = self._compute_neuron_attribution(
                request, layer_idx, device
            )
            mask = self._build_sparse_mask(attribution, layer_idx)
            self._apply_masked_update(request, layer_idx, mask, device)
            info[f"layer_{layer_idx}_masked_neurons"] = int(mask.sum().item())

        logger.info(
            "NMKE edit: '%s' -> '%s' across layers %s",
            request.subject, request.target_new, layer_indices,
        )
        return info

    # ------------------------------------------------------------------
    # Step 1: Neuron Attribution
    # ------------------------------------------------------------------

    def _compute_neuron_attribution(
        self, request: EditRequest, layer_idx: int, device: torch.device
    ) -> torch.Tensor:
        """Compute per-neuron attribution scores via integrated gradients.

        We measure how much each output neuron of the MLP projection
        contributes to producing the correct target token probability.
        """
        model = self.model
        tokenizer = self.tokenizer

        full_text = f"{request.prompt} {request.target_new}"
        prompt_ids = tokenizer(request.prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]

        target_ids = full_ids[:, prompt_len:]
        if target_ids.numel() == 0:
            weight = self._get_mlp_weight(layer_idx)
            return torch.zeros(weight.shape[0], device=device)

        weight = self._get_mlp_weight(layer_idx)
        original_weight = weight.data.clone()
        out_features = weight.shape[0]
        attribution = torch.zeros(out_features, device=device)

        model_dtype = next(model.parameters()).dtype

        for step in range(self.attribution_steps):
            alpha = (step + 1) / self.attribution_steps
            with torch.no_grad():
                weight.data.copy_(original_weight * alpha)

            weight.requires_grad_(True)
            try:
                with torch.amp.autocast("cuda", dtype=model_dtype, enabled=model_dtype != torch.float32):
                    outputs = model(input_ids=full_ids)
                logits = outputs.logits[0, prompt_len - 1:-1]
                loss = F.cross_entropy(logits, target_ids[0])

                grad = torch.autograd.grad(loss, weight, retain_graph=False)[0]
                attribution += (grad * weight.data).sum(dim=1).abs() / self.attribution_steps
            finally:
                weight.requires_grad_(False)

        with torch.no_grad():
            weight.data.copy_(original_weight)

        return attribution.detach()

    # ------------------------------------------------------------------
    # Step 2: Dynamic Sparse Mask
    # ------------------------------------------------------------------

    def _build_sparse_mask(
        self, attribution: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Select the top-k% neurons while avoiding neurons already used by prior edits."""
        k = max(1, int(attribution.numel() * self.sparsity_ratio))

        if layer_idx in self._neuron_usage:
            penalty = self._neuron_usage[layer_idx].to(attribution.device)
            attribution = attribution * (1.0 - 0.5 * penalty.clamp(max=1.0))

        _, top_indices = attribution.topk(k)
        mask = torch.zeros_like(attribution, dtype=torch.bool)
        mask[top_indices] = True

        if layer_idx not in self._neuron_usage:
            self._neuron_usage[layer_idx] = torch.zeros_like(attribution)
        self._neuron_usage[layer_idx][top_indices] += 1.0

        return mask

    # ------------------------------------------------------------------
    # Step 3: Masked Value Update
    # ------------------------------------------------------------------

    def _apply_masked_update(
        self,
        request: EditRequest,
        layer_idx: int,
        mask: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Optimise only the masked rows of the MLP projection to encode the new fact.

        Phase 1: optimise a hidden-space perturbation vector ``delta``
        (one scalar per masked output dimension of the MLP projection).
        Phase 2: capture the MLP input and convert the hidden-space delta
        into a sparse rank-one weight update on the masked rows only.
        """
        model = self.model
        tokenizer = self.tokenizer
        weight = self._get_mlp_weight(layer_idx)

        full_text = f"{request.prompt} {request.target_new}"
        prompt_ids = tokenizer(request.prompt, return_tensors="pt")["input_ids"].to(device)
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]
        target_ids = full_ids[:, prompt_len:]

        if target_ids.numel() == 0:
            return

        masked_indices = mask.nonzero(as_tuple=True)[0]
        n_masked = masked_indices.shape[0]
        delta = torch.zeros(n_masked, device=device, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.v_lr)
        model_dtype = next(model.parameters()).dtype

        layers_container = self._get_layers_container(model)
        target_layer = layers_container[layer_idx]

        for step in range(self.v_num_grad_steps):
            def edit_hook(module, inp, out):
                if isinstance(out, tuple):
                    h = out[0].clone()
                else:
                    h = out.clone()
                perturbation = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
                perturbation[masked_indices] = delta.to(h.dtype)
                h[0, -1] = h[0, -1] + perturbation
                return (h,) + out[1:] if isinstance(out, tuple) else h

            handle = target_layer.register_forward_hook(edit_hook)
            try:
                with torch.amp.autocast("cuda", dtype=model_dtype, enabled=model_dtype != torch.float32):
                    outputs = model(input_ids=full_ids)
                logits = outputs.logits[0, prompt_len - 1:-1]
                loss = F.cross_entropy(logits, target_ids[0])

                optimizer.zero_grad()
                loss.backward()

                if self.clamp_norm_factor > 0 and delta.grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [delta], self.clamp_norm_factor * (delta.norm().item() + 0.1)
                    )
                optimizer.step()
            finally:
                handle.remove()

        mlp_input = self._capture_mlp_input(full_ids, layer_idx, device)
        k_norm_sq = mlp_input.dot(mlp_input).clamp(min=1e-10)

        with torch.no_grad():
            weight_float = weight.data.float()
            weight_float[masked_indices] += (
                delta.detach().unsqueeze(1) * mlp_input.unsqueeze(0) / k_norm_sq
            )
            weight.data.copy_(weight_float.to(weight.dtype))

        logger.debug(
            "NMKE masked update at layer %d: %d/%d neurons, delta_norm=%.4f",
            layer_idx, int(mask.sum()), mask.numel(), delta.norm().item(),
        )

    def _capture_mlp_input(
        self, input_ids: torch.Tensor, layer_idx: int, device: torch.device,
    ) -> torch.Tensor:
        """Run a forward pass and capture the input to the MLP down_proj at the last token."""
        model = self.model
        layers = self._get_layers_container(model)
        block = layers[layer_idx]

        down_proj = None
        for name in ("mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "mlp.fc_out"):
            parts = name.split(".")
            mod = block
            try:
                for p in parts:
                    mod = getattr(mod, p)
                down_proj = mod
                break
            except AttributeError:
                continue
        if down_proj is None:
            raise ValueError(f"Cannot find MLP projection in layer {layer_idx}")

        captured = {}

        def hook_fn(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            captured["input"] = x

        handle = down_proj.register_forward_hook(hook_fn)
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad(), torch.amp.autocast(
            "cuda", dtype=model_dtype, enabled=model_dtype != torch.float32
        ):
            model(input_ids=input_ids)
        handle.remove()

        return captured["input"][0, -1].detach().float()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mlp_weight(self, layer_idx: int) -> torch.nn.Parameter:
        layers = self._get_layers_container(self.model)
        block = layers[layer_idx]
        for name in ("mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "mlp.fc_out"):
            parts = name.split(".")
            mod = block
            try:
                for p in parts:
                    mod = getattr(mod, p)
                return mod.weight
            except AttributeError:
                continue
        raise ValueError(f"Cannot find MLP output projection in layer {layer_idx}")
