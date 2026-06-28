"""
WISE 知识编辑器
===============

实现 WISE (Knowledge Memory for Lifelong Model Editing) 方法。
WISE 在 GRACE 基础上增加知识分片与合并机制，支持大规模持续编辑。

参考论文: WISE: Rethinking the Knowledge Memory for Lifelong Model Editing
of Large Language Models
https://arxiv.org/abs/2405.14768

核心思想：
1. 使用 WISEAdapter 包装目标层，维护可训练的新权重矩阵
2. 编辑时训练新权重，利用激活距离损失保持局部性
3. 累积编辑后，通过合并策略 (slerp / ties / linear) 整合权重
4. 推理时根据激活距离判断使用原始权重还是编辑后权重
"""

import copy
import logging
from typing import Optional, Dict, Any, List, Union, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.utils.nethook import get_module, replace_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------

def _slerp(
    t: float,
    v0: torch.Tensor,
    v1: torch.Tensor,
    dot_threshold: float = 0.9995,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spherical linear interpolation between two weight tensors."""
    v0_np = v0.detach().cpu().float().numpy()
    v1_np = v1.detach().cpu().float().numpy()
    v0_copy, v1_copy = np.copy(v0_np), np.copy(v1_np)

    norm0, norm1 = np.linalg.norm(v0_np), np.linalg.norm(v1_np)
    if norm0 > eps:
        v0_np = v0_np / norm0
    if norm1 > eps:
        v1_np = v1_np / norm1

    dot = np.sum(v0_np * v1_np)
    if np.abs(dot) > dot_threshold:
        return torch.from_numpy((1 - t) * v0_copy + t * v1_copy).to(v0.device)

    theta_0 = np.arccos(np.clip(dot, -1, 1))
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = np.sin(theta_t) / sin_theta_0
    return torch.from_numpy(s0 * v0_copy + s1 * v1_copy).to(v0.device)


def _ties_merge(
    weights: List[float],
    base: torch.Tensor,
    tensors: List[torch.Tensor],
    density: float = 0.5,
) -> torch.Tensor:
    """TIES merge: magnitude pruning + sign consensus + weighted sum."""
    deltas = [t - base for t in tensors]

    pruned = []
    for d in deltas:
        flat = d.abs().view(-1)
        threshold = torch.quantile(flat.float(), 1.0 - density)
        pruned.append(d * (d.abs() >= threshold))

    stacked = torch.stack(pruned)
    weight_t = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
    while weight_t.dim() < stacked.dim():
        weight_t = weight_t.unsqueeze(-1)

    weighted = stacked * weight_t
    sign_sum = stacked.sign().sum(dim=0)
    majority = (sign_sum >= 0).to(stacked.dtype) * 2 - 1
    mask = (stacked.sign() == majority).to(stacked.dtype)

    mixed = (weighted * mask).sum(dim=0)
    divisor = (weight_t * mask).sum(dim=0)
    divisor[divisor == 0] = 1
    return (base + mixed / divisor).to(base.dtype)


def _linear_merge(t: float, base: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    return (1.0 - t) * base + t * new


def _activation_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.norm(a - b, p=2, dim=-1).mean()


# ---------------------------------------------------------------------------
# WISEAdapter
# ---------------------------------------------------------------------------

class WISEAdapter(nn.Module):
    """Knowledge-sharding adapter that wraps a single linear layer.

    Maintains a trainable ``new_weight`` matrix that is optimised during editing.
    Accumulated edits are periodically merged into the base layer using a
    configurable strategy (slerp / ties / linear).
    """

    def __init__(
        self,
        layer: nn.Module,
        merge_strategy: str = "slerp",
        merge_freq: int = 5,
        save_freq: int = 1,
        mask_ratio: float = 0.5,
        act_ratio: float = 1.0,
        merge_weight: float = 0.5,
        density: float = 0.5,
        transpose: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.weight = layer.weight
        self.device = layer.weight.device
        self.merge_strategy = merge_strategy
        self.merge_freq = merge_freq
        self.save_freq = save_freq
        self.mask_ratio = mask_ratio
        self.act_ratio = act_ratio
        self.merge_weight_factor = merge_weight
        self.density = density

        self._dtype = self.weight.dtype
        self._original_weight_cpu = self.weight.detach().cpu().clone()
        self.original_layer = layer
        torch.cuda.empty_cache()
        self.new_weight = torch.nn.Parameter(self.weight.detach().clone())
        self.memory_weight: List[torch.Tensor] = []

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]

        self.editing: bool = False
        self.editing_total_cnt: int = 0
        self.weight_mask: Optional[torch.Tensor] = None
        self._min_act: float = float("inf")

        self.original_layer_output: Optional[torch.Tensor] = None
        self.new_weight_layer_output: Optional[torch.Tensor] = None

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def generate_activation_mask(self):
        flat_size = self.new_weight.reshape(-1).size()[0]
        mask_np = np.random.choice(
            [1, 0], size=flat_size, p=[self.mask_ratio, 1 - self.mask_ratio],
        )
        self.weight_mask = torch.from_numpy(mask_np).to(self.new_weight.device)

    def mask_new_weight_gradient(self):
        if self.new_weight.grad is None or self.weight_mask is None:
            return
        p_size = self.new_weight.grad.size()
        self.new_weight.grad = (
            (self.new_weight.grad.reshape(-1) * self.weight_mask)
            .view(p_size)
            .to(self.new_weight.grad.dtype)
        )

    def _new_weight_forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.new_weight.to(x.dtype))

    def save_weight(self):
        self.memory_weight.append(self.new_weight.detach().clone())
        self.new_weight = torch.nn.Parameter(
            self._original_weight_cpu.clone().to(device=self.device, dtype=self._dtype)
        )

    def save_editing_activation(self):
        if self.original_layer_output is None or self.new_weight_layer_output is None:
            return
        dist = _activation_distance(self.original_layer_output, self.new_weight_layer_output)
        self._min_act = min(self._min_act, dist.item())

    def merge_weight(self):
        """Merge accumulated memory weights into the base layer."""
        if not self.memory_weight:
            merged = self._apply_merge(
                self.merge_weight_factor,
                self.layer.weight.data,
                [self.new_weight.data],
            )
            self.layer.weight = nn.Parameter(merged.to(dtype=self._dtype), requires_grad=False)
            self.new_weight = torch.nn.Parameter(
                self._original_weight_cpu.clone().to(device=self.device, dtype=self._dtype)
            )
            return

        weights = [
            self.merge_weight_factor / len(self.memory_weight)
            for _ in self.memory_weight
        ]
        merged = self._apply_merge(
            weights, self.original_layer.weight.data, self.memory_weight,
        )
        self.layer.weight = nn.Parameter(merged.to(device=self.device, dtype=self._dtype), requires_grad=False)
        self.new_weight = torch.nn.Parameter(
            self._original_weight_cpu.clone().to(device=self.device, dtype=self._dtype)
        )
        self.memory_weight.clear()

    def _apply_merge(self, weights, base, tensors):
        if self.merge_strategy == "slerp":
            t = weights if isinstance(weights, float) else weights[0]
            v1 = tensors[0] if len(tensors) == 1 else tensors[-1]
            return _slerp(t, base, v1)
        elif self.merge_strategy == "ties":
            w = [weights] * len(tensors) if isinstance(weights, float) else weights
            return _ties_merge(w, base, tensors, density=self.density)
        elif self.merge_strategy == "linear":
            t = weights if isinstance(weights, float) else weights[0]
            v1 = tensors[0] if len(tensors) == 1 else tensors[-1]
            return _linear_merge(t, base, v1)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def _orig_forward(self, *args):
        """Forward through original layer with dtype matching."""
        x = args[0]
        w = self.original_layer.weight
        b = getattr(self.original_layer, "bias", None)
        return F.linear(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)

    def forward(self, *args):
        if self.editing:
            layer_out = self._new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self._orig_forward(*args)
        else:
            original_out = self._orig_forward(*args)
            new_out = self._new_weight_forward(*args)
            dist = _activation_distance(original_out, new_out)

            threshold = (
                self._min_act * self.act_ratio
                if self._min_act < float("inf")
                else 0.0
            )
            layer_out = new_out if dist.item() >= threshold else original_out

            for mem_w in self.memory_weight:
                mem_out = F.linear(args[0], mem_w.to(args[0].dtype))
                mem_dist = _activation_distance(original_out, mem_out)
                if mem_dist.item() > dist.item():
                    layer_out = mem_out
                    dist = mem_dist

        return layer_out


# ---------------------------------------------------------------------------
# WISEEditor
# ---------------------------------------------------------------------------

class WISEEditor(EditTrainer):
    """WISE 知识编辑器

    基于知识分片与合并的持续知识编辑方法。
    支持 slerp / ties / linear 三种权重合并策略。

    Paper: https://arxiv.org/abs/2405.14768
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        inner_params: str = "model.layers.9.mlp.down_proj",
        edit_lr: float = 1e-4,
        n_iter: int = 70,
        merge_strategy: Literal["slerp", "ties", "linear"] = "slerp",
        merge_freq: int = 5,
        save_freq: int = 1,
        mask_ratio: float = 0.5,
        norm_constraint: Optional[float] = None,
        merge_weight: float = 0.5,
        density: float = 0.5,
        act_ratio: float = 1.0,
        gamma: float = 5.0,
        alpha: float = 20.0,
        beta: float = 5.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            layers: 编辑的目标层索引列表（兼容基类接口）
            inner_params: 目标层路径，如 ``"model.layers.9.mlp.down_proj"``
            edit_lr: 新权重优化学习率
            n_iter: 每次编辑的训练迭代次数
            merge_strategy: 权重合并策略 (``"slerp"`` | ``"ties"`` | ``"linear"``)
            merge_freq: 每隔多少次编辑执行一次合并
            save_freq: 每隔多少次编辑保存权重快照
            mask_ratio: 梯度掩码中被激活的比例
            norm_constraint: 权重更新的范数约束（None 表示不约束）
            merge_weight: 合并时新权重的插值系数
            density: TIES 合并的密度参数
            act_ratio: 激活距离阈值缩放因子
            gamma: margin loss 间隔参数
            alpha: out-of-scope 距离上界
            beta: in-scope 距离下界
        """
        super().__init__(layers=layers, *args, **kwargs)

        self.inner_params = inner_params
        self.edit_lr = edit_lr
        self.n_iter = n_iter
        self.merge_strategy = merge_strategy
        self.merge_freq = merge_freq
        self.save_freq = save_freq
        self.mask_ratio = mask_ratio
        self.norm_constraint = norm_constraint
        self.merge_weight = merge_weight
        self.density = density
        self.act_ratio = act_ratio
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

        suffixes = (".weight", ".bias")
        self.target_layer = (
            inner_params.rsplit(".", 1)[0]
            if inner_params.endswith(suffixes)
            else inner_params
        )

        self._adapter_installed = False
        self._original_layer = None
        self._edit_history: List[Dict[str, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Adapter lifecycle
    # ------------------------------------------------------------------

    def _ensure_adapter(self):
        if self._adapter_installed:
            return

        model = self.model
        try:
            target = get_module(model, self.target_layer)
        except LookupError:
            raise ValueError(f"Target layer '{self.target_layer}' not found")

        if not isinstance(target, WISEAdapter):
            import transformers
            transpose = not isinstance(
                model,
                transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
            )
            adapter = WISEAdapter(
                layer=target,
                merge_strategy=self.merge_strategy,
                merge_freq=self.merge_freq,
                save_freq=self.save_freq,
                mask_ratio=self.mask_ratio,
                act_ratio=self.act_ratio,
                merge_weight=self.merge_weight,
                density=self.density,
                transpose=transpose,
            ).to(device=target.weight.device, dtype=target.weight.dtype)

            self._original_layer_state = {
                k: v.detach().cpu().clone() for k, v in target.state_dict().items()
            }
            self._original_layer_cls = type(target)
            self._original_layer_config = {
                "in_features": getattr(target, "in_features", None),
                "out_features": getattr(target, "out_features", None),
                "bias": target.bias is not None if hasattr(target, "bias") else False,
            }
            replace_module(model, self.target_layer, adapter)

        self._adapter_installed = True

    def _get_adapter(self) -> WISEAdapter:
        adapter = get_module(self.model, self.target_layer)
        assert isinstance(adapter, WISEAdapter)
        return adapter

    def reset_layer(self):
        """Restore the original layer, removing the WISE adapter."""
        if hasattr(self, "_original_layer_state") and self._original_layer_state is not None:
            cfg = self._original_layer_config
            if cfg["in_features"] is not None:
                restored = self._original_layer_cls(
                    cfg["in_features"], cfg["out_features"], bias=cfg["bias"],
                )
            else:
                restored = self._original_layer_cls.__new__(self._original_layer_cls)
            restored.load_state_dict(self._original_layer_state)
            device = next(self.model.parameters()).device
            replace_module(self.model, self.target_layer, restored.to(device))
            self._adapter_installed = False
            self._original_layer_state = None
            self._edit_history.clear()

    # ------------------------------------------------------------------
    # Edit interface
    # ------------------------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        self._ensure_adapter()

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
        }

        for request in requests:
            try:
                self._apply_wise_edit(request)
                results["edited_count"] += 1
            except Exception as e:
                logger.error("WISE edit failed for '%s': %s", request.prompt, e, exc_info=True)
                results["success"] = False

        return results

    def _apply_wise_edit(self, request: EditRequest):
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device
        adapter = self._get_adapter()

        full_text = request.prompt + " " + request.target_new
        prompt_ids = tokenizer(
            request.prompt, return_tensors="pt",
        )["input_ids"].to(device)
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)

        labels = full_ids.clone()
        labels[:, : prompt_ids.shape[1]] = -100

        if request.locality_inputs:
            loc_text = None
            if isinstance(request.locality_inputs, dict):
                for group_entries in request.locality_inputs.values():
                    if group_entries and isinstance(group_entries, list):
                        loc_text = group_entries[0].get("prompt")
                        break
            elif isinstance(request.locality_inputs, list) and request.locality_inputs:
                loc_text = request.locality_inputs[0].get("prompt")
            if loc_text:
                loc_ids = tokenizer(
                    loc_text, return_tensors="pt",
                    padding="max_length", max_length=full_ids.shape[1], truncation=True,
                )["input_ids"].to(device)
                full_ids = torch.cat([full_ids, loc_ids], dim=0)
                labels = torch.cat([labels, torch.full_like(loc_ids, -100)], dim=0)

        tokens = {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
        }

        self._edit_history.append({k: v.detach().cpu() for k, v in tokens.items()})

        adapter.editing = True
        adapter.set_parameter_tunable()

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.generate_activation_mask()

        last_prompt_loc = (labels == -100).sum(dim=-1) - 1

        optimizer = torch.optim.SGD(
            [adapter.new_weight], lr=self.edit_lr, weight_decay=1e-5,
        )

        for i in range(self.n_iter):
            outputs = model(**tokens)

            logits = outputs.logits
            shift_logits = logits[:1, :-1, :].contiguous()
            shift_labels = labels[:1, 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(1, -1)

            label_mask = torch.zeros_like(per_token_loss, dtype=torch.bool)
            for b, col in enumerate(last_prompt_loc[:1]):
                label_mask[b, max(0, col - 1):] = True
            ft_loss = (per_token_loss * label_mask).sum() / label_mask.sum().clamp(min=1)

            act_loss = torch.tensor(0.0, device=device)
            orig_out = adapter.original_layer_output
            new_out = adapter.new_weight_layer_output
            if orig_out is not None and new_out is not None:
                in_scope = _activation_distance(orig_out[:1], new_out[:1])
                if orig_out.shape[0] > 1:
                    out_scope = _activation_distance(orig_out[1:], new_out[1:])
                else:
                    out_scope = torch.tensor(0.0, device=device)
                act_loss = (
                    F.relu(out_scope - in_scope + self.gamma)
                    + F.relu(out_scope - self.alpha)
                    + F.relu(self.beta - in_scope)
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

        if adapter.editing_total_cnt % self.save_freq == 0:
            adapter.save_weight()
            logger.info("WISE: saved weight snapshot to memory")

        if adapter.editing_total_cnt % self.merge_freq == 0:
            adapter.merge_weight()
            logger.info(
                "WISE: merged weights using '%s' strategy", self.merge_strategy,
            )

        logger.info(
            "WISE edit applied: '%s' -> '%s' (loss=%.4f, total_edits=%d)",
            request.prompt,
            request.target_new,
            loss.item(),
            adapter.editing_total_cnt,
        )
