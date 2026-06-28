"""
UNKE knowledge editor
======================

Implements Unstructured Knowledge Editing (UNKE).
Unlike ROME/MEMIT which apply closed-form weight updates, UNKE fine-tunes
the target transformer layers to match modified input-output mappings while
using weight regularisation to preserve existing knowledge.

Paper: UnKE: Unstructured Knowledge Editing in Large Language Models
https://arxiv.org/abs/2405.15349

Algorithm:
1. Save original weights of target layers
2. Compute target representations z at the last edit layer (compute_z)
3. For each layer:
   a. Capture layer input/output via Trace
   b. Recompute current z, derive residual, modify target output at subject
      positions
   c. Fine-tune the entire layer (AdamW + CosineAnnealingLR) with:
      - Edit loss:  MSE(layer(cached_input), modified_output)
      - Weight reg:  L2 distance to original weights
4. Restore weights that deviated beyond a relative threshold
"""

import inspect
import logging
from typing import Dict, Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.utils.nethook import (
    Trace,
    get_parameter,
    get_module,
    set_requires_grad,
)
from trainer.edit.utils.compute_z import compute_z, compute_ks

logger = logging.getLogger(__name__)


class UNKEEditor(EditTrainer):
    """UNKE knowledge editor via layer-level fine-tuning.

    Compared with ROME/MEMIT:
    - ROME/MEMIT: closed-form weight update on MLP projection only
    - UNKE: gradient-based fine-tuning of the *entire* transformer layer,
      regularised by L2 distance to original weights and optional
      post-hoc restoration of overly-deviated parameters
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        v_lr: float = 5e-1,
        v_num_grad_steps: int = 20,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 4.0,
        v_loss_layer: int = 31,
        ft_lr: float = 1e-5,
        ft_epochs: int = 25,
        weight_decay_factor: float = 0.01,
        max_weight_deviation: float = 5.0,
        rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj",
        layer_module_tmp: str = "model.layers.{}",
        lm_head_module: str = "lm_head",
        ln_f_module: str = "model.norm",
        *args,
        **kwargs,
    ):
        """
        Args:
            layers: Target transformer layers to edit.
            v_lr: Learning rate for compute_z value optimisation.
            v_num_grad_steps: Steps for compute_z.
            v_weight_decay: Weight decay for compute_z.
            clamp_norm_factor: Gradient clamp factor for compute_z.
            v_loss_layer: Layer index to evaluate NLL in compute_z.
            ft_lr: Learning rate for layer fine-tuning.
            ft_epochs: Number of fine-tuning steps per layer.
            weight_decay_factor: L2 regularisation strength toward
                original weights during fine-tuning.
            max_weight_deviation: Relative deviation threshold; parameters
                exceeding this ratio are restored to originals.
            rewrite_module_tmp: Template for the MLP projection module.
            layer_module_tmp: Template for the full transformer layer.
            lm_head_module: Name of the LM head module.
            ln_f_module: Name of the final layer-norm module.
        """
        layers = layers or [4, 5, 6, 7, 8]
        super().__init__(layers=layers, *args, **kwargs)

        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_weight_decay = v_weight_decay
        self.clamp_norm_factor = clamp_norm_factor
        self.v_loss_layer = v_loss_layer
        self.ft_lr = ft_lr
        self.ft_epochs = ft_epochs
        self.weight_decay_factor = weight_decay_factor
        self.max_weight_deviation = max_weight_deviation
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.lm_head_module = lm_head_module
        self.ln_f_module = ln_f_module

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """Execute UNKE knowledge editing.

        Args:
            requests: Single or list of edit requests.

        Returns:
            Dict with ``success``, ``edited_count``, ``metrics``,
            ``layers_edited``.
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "layers_edited": [],
        }

        try:
            layer_indices = self._resolve_layer_indices(self.layers)
            results["layers_edited"] = layer_indices
            self._apply_unke(requests, layer_indices)
            results["edited_count"] = len(requests)
        except Exception as e:
            logger.error("UNKE edit failed: %s", e, exc_info=True)
            results["success"] = False

        return results

    # ------------------------------------------------------------------
    # Core editing
    # ------------------------------------------------------------------

    def _apply_unke(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
    ):
        """Apply UNKE layer-level fine-tuning across target layers."""
        model = self.model
        tok = self.tokenizer
        device = next(model.parameters()).device

        # 1. Snapshot all parameters in target layers
        layer_params_orig: Dict[str, torch.Tensor] = {}
        for layer_idx in layer_indices:
            layer_name = self.layer_module_tmp.format(layer_idx)
            layer_mod = get_module(model, layer_name)
            for pname, param in layer_mod.named_parameters():
                full_name = f"{layer_name}.{pname}"
                layer_params_orig[full_name] = param.detach().clone()

        # 2. Compute target z at the last edit layer
        z_layer = layer_indices[-1]
        req_dicts = [
            {"prompt": r.prompt, "target_new": r.target_new}
            for r in requests
        ]

        z_list = []
        for rd in req_dicts:
            z = compute_z(
                model,
                tok,
                rd,
                z_layer,
                layer_module_tmp=self.layer_module_tmp,
                lm_head_module=self.lm_head_module,
                ln_f_module=self.ln_f_module,
                v_lr=self.v_lr,
                v_num_grad_steps=self.v_num_grad_steps,
                v_weight_decay=self.v_weight_decay,
                v_loss_layer=self.v_loss_layer,
                clamp_norm_factor=self.clamp_norm_factor,
            )
            z_list.append(z)
        zs = torch.stack(z_list, dim=0)  # (batch, hidden)

        prompts = [r.prompt for r in requests]

        # 3. Layer-by-layer fine-tuning
        for i, layer_idx in enumerate(layer_indices):
            layer_name = self.layer_module_tmp.format(layer_idx)
            layer_mod = get_module(model, layer_name)

            # 3a. Capture layer input/output on edit prompts
            ctx_tok = tok(prompts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                with Trace(
                    module=model,
                    layer=layer_name,
                    retain_input=True,
                    retain_output=True,
                    detach=True,
                    clone=True,
                ) as tr:
                    model(**ctx_tok)
                    layer_in = tr.input
                    layer_out = tr.output

            layer_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            # 3b. Current z at z_layer -> residual
            cur_zs, idxs = compute_ks(
                model, tok, prompts, z_layer,
                layer_module_tmp=self.layer_module_tmp,
            )
            if zs.device != cur_zs.device:
                cur_zs = cur_zs.to(zs.device)
            targets = zs - cur_zs

            logger.debug(
                "Layer %d z-error: %.4f",
                layer_idx,
                torch.linalg.norm(targets, dim=0).mean().item(),
            )

            # Distribute residual across remaining layers
            resid = targets.to(device) / max(len(layer_indices) - i, 1)

            # Modify target output at subject last-token positions
            target_out = layer_out.clone()
            for j in range(len(idxs)):
                target_out[j, idxs[j]] += resid[j]

            # 3c. Fine-tune this layer
            self._finetune_layer(
                layer_mod, layer_name, layer_in, target_out,
                ctx_tok["attention_mask"], layer_params_orig,
            )

            # Freeze layer again
            set_requires_grad(False, layer_mod)

            del layer_in, layer_out, target_out, cur_zs, targets, resid
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 4. Restore overly-deviated weights
        restored = self._restore_deviated_weights(layer_indices, layer_params_orig)
        if restored:
            logger.info("Restored %d parameters that deviated too far", restored)

        logger.info(
            "UNKE applied across layers %s: %d requests",
            layer_indices,
            len(requests),
        )

    # ------------------------------------------------------------------
    # Layer fine-tuning
    # ------------------------------------------------------------------

    def _finetune_layer(
        self,
        layer_mod: nn.Module,
        layer_name: str,
        layer_in: torch.Tensor,
        target_out: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        orig_params: Dict[str, torch.Tensor],
    ):
        """Fine-tune a single transformer layer to match target output.

        Loss = MSE(layer(input), target) + weight_decay_factor * L2_reg
        """
        # Enable gradients for this layer
        for param in layer_mod.parameters():
            param.requires_grad = True

        # Build causal attention mask & position ids for direct layer call
        causal_mask, position_ids, cache_position = _build_causal_inputs(
            layer_in, attention_mask_2d,
        )
        fwd_kwargs = _layer_forward_kwargs(
            layer_mod, causal_mask, position_ids, cache_position,
        )

        # Optimiser & scheduler
        optimizer = optim.AdamW(
            layer_mod.parameters(), lr=self.ft_lr, weight_decay=0.0,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(self.ft_epochs, 1))
        criterion = nn.MSELoss()

        # Snapshot current (pre-finetune) weights for regularisation
        local_orig: Dict[str, torch.Tensor] = {}
        for pname, param in layer_mod.named_parameters():
            full_name = f"{layer_name}.{pname}"
            if full_name in orig_params:
                local_orig[pname] = orig_params[full_name].to(param.device)
            else:
                local_orig[pname] = param.detach().clone()

        # Ensure layer_in is a plain tensor
        fwd_input = layer_in[0] if isinstance(layer_in, tuple) else layer_in

        for step in range(self.ft_epochs):
            optimizer.zero_grad()

            # Forward through the single layer
            out = layer_mod(fwd_input, **fwd_kwargs)
            out = out[0] if isinstance(out, tuple) else out

            # Edit loss
            edit_loss = criterion(out, target_out)

            # Weight regularisation: L2 distance to originals
            reg_loss = torch.tensor(0.0, device=out.device)
            for pname, param in layer_mod.named_parameters():
                if pname in local_orig:
                    reg_loss = reg_loss + (param - local_orig[pname]).pow(2).sum()

            loss = edit_loss + self.weight_decay_factor * reg_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 5 == 0:
                logger.debug(
                    "UNKE finetune step %d/%d: loss=%.6f (edit=%.6f, reg=%.6f)",
                    step + 1,
                    self.ft_epochs,
                    loss.item(),
                    edit_loss.item(),
                    reg_loss.item(),
                )

    # ------------------------------------------------------------------
    # Post-hoc weight restoration
    # ------------------------------------------------------------------

    def _restore_deviated_weights(
        self,
        layer_indices: List[int],
        orig_params: Dict[str, torch.Tensor],
    ) -> int:
        """Restore parameters whose relative deviation exceeds threshold."""
        model = self.model
        count = 0

        with torch.no_grad():
            for layer_idx in layer_indices:
                layer_name = self.layer_module_tmp.format(layer_idx)
                layer_mod = get_module(model, layer_name)
                for pname, param in layer_mod.named_parameters():
                    full_name = f"{layer_name}.{pname}"
                    if full_name not in orig_params:
                        continue
                    orig = orig_params[full_name].to(param.device)
                    orig_norm = orig.norm().clamp(min=1e-10)
                    deviation = (param - orig).norm() / orig_norm
                    if deviation > self.max_weight_deviation:
                        param.copy_(orig)
                        count += 1
                        logger.debug(
                            "Restored %s (deviation %.2f > %.2f)",
                            full_name, deviation.item(), self.max_weight_deviation,
                        )
        return count


# ======================================================================
# Helpers (module-level)
# ======================================================================

def _build_causal_inputs(
    hidden_states: torch.Tensor,
    attention_mask_2d: torch.Tensor,
):
    """Build 4-D causal attention mask, position_ids and cache_position
    for calling a single transformer decoder layer directly.

    Returns:
        (causal_mask, position_ids, cache_position)
    """
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    batch_size, seq_len = hidden_states.shape[:2]
    device = hidden_states.device
    dtype = hidden_states.dtype

    position_ids = (
        torch.arange(seq_len, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    cache_position = torch.arange(seq_len, device=device)

    min_val = torch.finfo(dtype).min
    causal_mask = torch.zeros(
        batch_size, 1, seq_len, seq_len, device=device, dtype=dtype,
    )
    triu_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    causal_mask[:, :, triu_mask] = min_val

    if attention_mask_2d is not None:
        padding = attention_mask_2d[:, None, None, :].eq(0)
        causal_mask = causal_mask.masked_fill(padding, min_val)

    return causal_mask, position_ids, cache_position


def _layer_forward_kwargs(
    layer_mod: nn.Module,
    causal_mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache_position: torch.Tensor,
) -> dict:
    """Inspect the layer's ``forward`` signature and build a kwargs dict
    containing only the arguments it accepts, avoiding TypeError on
    architecture-specific differences (e.g. ``cache_position`` in Llama 3
    but not Qwen 2).
    """
    sig = inspect.signature(layer_mod.forward)
    params = sig.parameters
    kwargs: dict = {}

    if "attention_mask" in params:
        kwargs["attention_mask"] = causal_mask
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position

    return kwargs
