"""
Compute the target value vector *z* for locate-then-edit methods
(ROME / MEMIT / AlphaEdit / UNKE).

Ported from:
- https://github.com/TrustedLLM/UnKE  (UnKE / AnyEdit)
- https://github.com/zjunlp/EasyEdit   (AlphaEdit)

The function optimises a residual *delta* so that injecting
``target_init + delta`` at the critical token position makes the model
produce the desired target text.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from trainer.edit.utils import nethook

logger = logging.getLogger(__name__)


def compute_z(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    request: Dict[str, Any],
    layer: int,
    *,
    layer_module_tmp: str = "model.layers.{}",
    lm_head_module: str = "lm_head",
    ln_f_module: str = "model.norm",
    v_lr: float = 5e-1,
    v_num_grad_steps: int = 20,
    v_weight_decay: float = 0.5,
    v_loss_layer: int = 31,
    clamp_norm_factor: float = 4.0,
    prompt_key: str = "prompt",
    target_key: str = "target_new",
) -> torch.Tensor:
    """Optimise the value vector for a single edit request.

    Returns the optimised target representation (``target_init + delta``).
    """
    device = next(model.parameters()).device

    lm_w = nethook.get_parameter(model, f"{lm_head_module}.weight").T
    ln_f = nethook.get_module(model, ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{lm_head_module}.bias")
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    target_ids = tok(request[target_key], return_tensors="pt").to(device)["input_ids"][0]
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    input_tok = tok([request[prompt_key]], return_tensors="pt", padding=True).to(device)
    input_ids = torch.cat(
        [input_tok["input_ids"], target_ids[:-1].unsqueeze(0)], dim=1
    )

    rewriting_targets = torch.full_like(input_ids, -100)
    ex_len = input_ids.shape[1]
    rewriting_targets[0, ex_len - len(target_ids) : ex_len] = target_ids

    lookup_idxs = [ex_len - len(target_ids)]
    loss_layer = max(v_loss_layer, layer)

    hidden_size = getattr(model.config, "hidden_size", None) or model.config.n_embd
    delta = torch.zeros(hidden_size, requires_grad=True, device=device)
    target_init: Optional[torch.Tensor] = None

    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == layer_module_tmp.format(layer):
            if target_init is None:
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta
        return cur_out

    opt = torch.optim.Adam([delta], lr=v_lr)
    nethook.set_requires_grad(False, model)

    for it in range(v_num_grad_steps):
        opt.zero_grad()
        with nethook.TraceDict(
            module=model,
            layers=[
                layer_module_tmp.format(loss_layer),
                layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            model(input_ids)

        output = tr[layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = output.transpose(0, 1)
        full_repr = output

        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device),
            dim=2,
        )
        loss_tokens = torch.where(rewriting_targets != -100, rewriting_targets, 0)
        loss_gather = torch.gather(
            log_probs, 2, loss_tokens.unsqueeze(2).to(log_probs.device)
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        nll_loss_each = -(loss_gather * mask.to(loss_gather.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        weight_decay = v_weight_decay * (torch.norm(delta) / torch.norm(target_init) ** 2)
        total_loss = nll_loss + weight_decay.to(nll_loss.device)

        if it % 5 == 0:
            logger.debug(
                "compute_z step %d: loss=%.4f (nll=%.4f, wd=%.4f) avg_prob=%.4f",
                it, total_loss.item(), nll_loss.item(), weight_decay.item(),
                torch.exp(-nll_loss_each).mean().item(),
            )

        if it == v_num_grad_steps - 1:
            break

        total_loss.backward()
        opt.step()

        max_norm = clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    logger.info(
        "compute_z done: init_norm=%.4f delta_norm=%.4f target_norm=%.4f",
        target_init.norm().item(), delta.norm().item(), target.norm().item(),
    )
    return target


def compute_ks(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompts: list[str],
    layer: int,
    *,
    layer_module_tmp: str = "model.layers.{}",
) -> Tuple[torch.Tensor, list[int]]:
    """Collect the hidden-state keys at the last non-padding token for each
    prompt at the given layer.

    Returns ``(keys, idxs)`` where *keys* has shape ``(batch, hidden)``
    and *idxs* is a list of per-sample last-token indices.
    """
    input_tok = tok(prompts, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    idxs = [int(m.sum()) - 1 for m in input_tok["attention_mask"]]

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
        ) as tr:
            model(**input_tok)
            zs_out = tr.output

    zs_out = zs_out[0] if isinstance(zs_out, tuple) else zs_out
    keys = torch.stack([zs_out[i, idxs[i]] for i in range(len(zs_out))], dim=0)
    return keys, idxs
