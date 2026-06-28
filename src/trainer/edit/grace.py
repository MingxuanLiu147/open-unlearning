"""
GRACE 知识编辑器
================

实现 GRACE (Lifelong Model Editing with Discrete Key-Value Adaptors) 方法。
GRACE 通过在目标层插入可学习的 key-value 适配器实现持续知识编辑。

参考论文: T. Hartvigsen et al., "Aging with GRACE: Lifelong Model Editing
with Discrete Key-Value Adaptors", NeurIPS 2023
https://arxiv.org/abs/2211.11031

核心思想：
1. 在目标层插入 GRACEAdapter，维护 (key, value, epsilon) 码本
2. 前向传播时，计算输入与码本中所有 key 的欧氏距离
3. 若距离小于 epsilon 阈值，替换输出为对应 value
4. 编辑时训练新的 key-value 对并加入码本
"""


import logging
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.utils.nethook import get_module, replace_module

logger = logging.getLogger(__name__)


def _euclidean_distance(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance between *key* rows and *query* rows."""
    if key.dim() < 2:
        key = key.unsqueeze(0)
    orig_dtype = key.dtype
    return torch.cdist(key.float(), query.float(), p=2).to(orig_dtype)


def _perturb_values(
    value: torch.Tensor, num_pert: int, device: torch.device,
) -> torch.Tensor:
    """Create noisy copies of a value vector for adversarial training.

    The first copy retains the original value (zero noise).
    """
    noise = torch.normal(0, 1, (num_pert,) + value.shape[1:], device=device)
    noise[0] = 0
    noise.requires_grad = True
    return value.expand(num_pert, -1) + noise


# ---------------------------------------------------------------------------
# GRACEAdapter
# ---------------------------------------------------------------------------

class GRACEAdapter(nn.Module):
    """Discrete key-value codebook adapter wrapping a single linear layer.

    Maintains a growing codebook of (key, value, epsilon) entries.  During
    forward pass, if the input activation at the designated token position
    falls within epsilon distance of a stored key, the layer output at that
    position is replaced with the corresponding learned value.
    """

    def __init__(
        self,
        layer: nn.Module,
        init_epsilon: float = 0.1,
        num_pert: int = 1,
        replacement: str = "replace_last",
        val_init: str = "warm",
        val_train: str = "sgd",
        transpose: bool = True,
    ):
        super().__init__()
        self.layer = layer
        self.weight = layer.weight
        self.init_epsilon = init_epsilon
        self.num_pert = num_pert
        self.replacement = replacement
        self.val_init = val_init
        self.val_train = val_train
        self.device = layer.weight.device
        self._dtype = layer.weight.dtype

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]

        self.key_id: int = -1
        self._training_edit: bool = False
        self.edit_label: Optional[torch.Tensor] = None
        self.edit_id: int = 0
        self.iter: int = 0
        self.chosen_key: Optional[torch.Tensor] = None

        # Codebook (populated lazily on first edit)
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[nn.Parameter] = None
        self.epsilons: Optional[torch.Tensor] = None
        self.key_labels: List = []
        self.edit_ids: List[int] = []

    @property
    def has_keys(self) -> bool:
        return self.keys is not None and self.keys.nelement() > 0

    @property
    def codebook_size(self) -> int:
        return self.keys.shape[0] if self.has_keys else 0

    # -- codebook management ------------------------------------------------

    def _init_key_value(self, query: torch.Tensor, value: nn.Parameter):
        """Initialise the codebook with the first key-value pair."""
        self.keys = query.detach()
        self.values = value
        self.epsilons = torch.tensor(
            [self.init_epsilon], device=self.device, requires_grad=False,
        )
        self.key_labels = [self.edit_label]
        self.edit_ids = [self.edit_id]

    def _add_key(self, new_key: torch.Tensor, new_value: nn.Parameter):
        """Append a new key-value-epsilon entry to the codebook."""
        self.keys = torch.vstack([self.keys, new_key.detach()])
        self.values = nn.Parameter(
            torch.vstack([self.values.data, new_value.data]), requires_grad=True,
        )
        new_eps = torch.tensor([self.init_epsilon], device=self.device)
        self.epsilons = torch.cat([self.epsilons, new_eps])
        self.key_labels = [self.edit_label] + self.key_labels
        self.edit_ids = self.edit_ids + [self.edit_id]

    def _label_match(self, label_a, label_b) -> bool:
        if label_a is None or label_b is None:
            return False
        return label_a.float().mean().item() == label_b.float().mean().item()

    def delete_key(self, edit_id: int):
        """Remove a codebook entry by its *edit_id*."""
        if not self.has_keys or edit_id not in self.edit_ids:
            return
        i = self.edit_ids.index(edit_id)
        self.keys = torch.cat([self.keys[:i], self.keys[i + 1:]])
        self.values = nn.Parameter(
            torch.cat([self.values.data[:i], self.values.data[i + 1:]]),
            requires_grad=True,
        )
        self.epsilons = torch.cat([self.epsilons[:i], self.epsilons[i + 1:]])
        self.key_labels = self.key_labels[:i] + self.key_labels[i + 1:]
        self.edit_ids = self.edit_ids[:i] + self.edit_ids[i + 1:]

    # -- forward ------------------------------------------------------------

    def _safe_layer_forward(self, *args):
        x = args[0]
        w = self.layer.weight
        b = getattr(self.layer, "bias", None)
        return F.linear(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)

    def forward(self, *args):
        layer_out = self._safe_layer_forward(*args)

        if not self._training_edit and not self.has_keys:
            return layer_out

        seq_len = args[0].shape[1]
        token_to_edit = (
            (seq_len - 1) if self.key_id == -1
            else min(self.key_id, seq_len - 1)
        )
        query = args[0][:, token_to_edit, :]

        if self.val_init == "cold":
            new_value = nn.Parameter(
                torch.rand(1, self.value_shape, device=self.device, dtype=self._dtype),
                requires_grad=True,
            )
        else:  # warm
            new_value = nn.Parameter(
                layer_out[:, token_to_edit, :].detach().to(self._dtype), requires_grad=True,
            )

        # -- codebook update (first iteration of a new edit only) --
        if not self.has_keys:
            self._init_key_value(query, new_value)
        elif self._training_edit and self.iter == 0:
            dists = _euclidean_distance(query, self.keys).view(-1, query.shape[0])
            smallest_dist, nearest_key = dists.min(0)

            if smallest_dist > (self.init_epsilon + self.epsilons[nearest_key]):
                self._add_key(query, new_value)
            elif not self._label_match(
                self.edit_label, self.key_labels[nearest_key],
            ):
                self._add_key(query, new_value)
                self.epsilons[nearest_key] = (smallest_dist / 2) - 1e-5
                self.epsilons[-1] = smallest_dist / 2
            else:
                if smallest_dist > self.epsilons[nearest_key]:
                    self.epsilons[nearest_key] = smallest_dist

        # -- distance-based output replacement --
        dists = _euclidean_distance(query, self.keys).view(-1, query.shape[0])
        if dists.nelement() == 0:
            return layer_out

        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        if self.val_train == "adv" and self._training_edit:
            chosen_value = _perturb_values(chosen_value, self.num_pert, self.device)

        if self.replacement == "replace_all":
            layer_out = torch.where(
                (smallest_dist <= eps).view(-1, 1, 1),
                chosen_value.unsqueeze(1).expand_as(layer_out),
                layer_out,
            )
        elif self.replacement == "replace_last":
            layer_out[:, token_to_edit] = torch.where(
                smallest_dist <= eps,
                chosen_value,
                layer_out[:, token_to_edit],
            )
        elif self.replacement == "replace_prompt":
            cond = (smallest_dist <= eps).view(-1, 1, 1)
            target_slice = layer_out[:, :token_to_edit]
            layer_out[:, :token_to_edit] = torch.where(
                cond.expand_as(target_slice),
                chosen_value.unsqueeze(1).expand_as(target_slice),
                target_slice,
            )

        return layer_out


# ---------------------------------------------------------------------------
# GRACEEditor
# ---------------------------------------------------------------------------

class GRACEEditor(EditTrainer):
    """GRACE 知识编辑器

    通过离散 key-value 适配器实现持续知识编辑。
    每次编辑仅向码本添加一个 key-value 对，支持无限次连续编辑。

    Paper: https://arxiv.org/abs/2211.11031
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        inner_params: str = "model.layers.9.mlp.down_proj",
        edit_lr: float = 1e-1,
        n_iter: int = 40,
        eps: float = 0.1,
        num_pert: int = 1,
        val_init: str = "warm",
        val_train: str = "sgd",
        val_reg: float = 0.0,
        dropout: float = 0.0,
        replacement: str = "replace_last",
        *args,
        **kwargs,
    ):
        """
        Args:
            layers: 编辑的目标层索引列表（兼容基类接口）
            inner_params: 目标层路径，如 ``"model.layers.9.mlp.down_proj"``
            edit_lr: 码本 value 优化学习率
            n_iter: 每次编辑的训练迭代次数
            eps: 欧氏距离阈值 (epsilon)
            num_pert: 对抗训练扰动数量
            val_init: value 初始化方式 (``"warm"`` | ``"cold"``)
            val_train: value 训练方式 (``"sgd"`` | ``"adv"``)
            val_reg: value 正则化系数
            dropout: dropout 比率（预留扩展）
            replacement: 输出替换策略
                (``"replace_last"`` | ``"replace_all"`` | ``"replace_prompt"``)
        """
        super().__init__(layers=layers, *args, **kwargs)

        self.inner_params = inner_params
        self.edit_lr = edit_lr
        self.n_iter = n_iter
        self.eps = eps
        self.num_pert = num_pert
        self.val_init = val_init
        self.val_train = val_train
        self.val_reg = val_reg
        self.dropout = dropout
        self.replacement = replacement

        suffixes = (".weight", ".bias")
        self.target_layer = (
            inner_params.rsplit(".", 1)[0]
            if inner_params.endswith(suffixes)
            else inner_params
        )

        self._adapter_installed: bool = False
        self._original_layer: Optional[nn.Module] = None
        self._edit_count: int = 0

    # -- adapter lifecycle --------------------------------------------------

    def _ensure_adapter(self):
        if self._adapter_installed:
            return

        model = self.model
        try:
            target = get_module(model, self.target_layer)
        except LookupError:
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in model"
            )

        if not isinstance(target, GRACEAdapter):
            import transformers

            transpose = not isinstance(
                model,
                transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
            )
            adapter = GRACEAdapter(
                layer=target,
                init_epsilon=self.eps,
                num_pert=self.num_pert,
                replacement=self.replacement,
                val_init=self.val_init,
                val_train=self.val_train,
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

    def _get_adapter(self) -> GRACEAdapter:
        adapter = get_module(self.model, self.target_layer)
        assert isinstance(adapter, GRACEAdapter)
        return adapter

    def reset_layer(self):
        """Restore the original layer, removing the GRACE adapter."""
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
            self._edit_count = 0

    # -- edit interface -----------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs,
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
                self._apply_grace_edit(request)
                results["edited_count"] += 1
            except Exception as e:
                logger.error("GRACE edit failed for '%s': %s", request.prompt, e)
                results["success"] = False

        return results

    def _apply_grace_edit(self, request: EditRequest):
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

        tokens = {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
        }

        key_id = (labels[0] == -100).sum().item() - 1
        adapter.key_id = key_id
        adapter._training_edit = True
        adapter.edit_label = labels
        adapter.edit_id = self._edit_count

        losses: List[float] = []
        for i in range(self.n_iter):
            adapter.iter = i
            outputs = model(**tokens)

            if i == 0:
                optimizer = torch.optim.Adam(model.parameters(), lr=self.edit_lr)

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
            "GRACE edit applied: '%s' -> '%s' (loss=%.4f, codebook_size=%d)",
            request.prompt,
            request.target_new,
            losses[-1] if losses else float("inf"),
            adapter.codebook_size,
        )
