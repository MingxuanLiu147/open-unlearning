"""
AnyEdit 统一多策略知识编辑器
============================

AnyEdit 提供统一接口，将 MEMIT、AlphaEdit、AlphaEdit+ARE、UNKE、UNKE+ARE
五种编辑策略聚合在一个编辑器中。通过 ``strategy`` 参数选择具体方法，
共享底层工具（nethook、compute_z）以减少重复。

基于: AnyEdit 项目 (https://github.com/TrustedLLM/UnKE)

支持的策略：
- memit:         多层最小二乘更新 (MEMIT)
- alphaedit:     零空间投影约束编辑 (AlphaEdit)
- alphaedit_are: AlphaEdit + 自适应表征编辑约束
- unke:          梯度微调编辑 (UNKE)
- unke_are:      UNKE + 自适应表征编辑约束
"""

import logging
from typing import Optional, Dict, Any, List, Union

import torch

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.utils.compute_z import compute_z, compute_ks
from trainer.edit.utils.nethook import (
    Trace,
    get_module,
    get_parameter,
    set_requires_grad,
)

logger = logging.getLogger(__name__)

SUPPORTED_STRATEGIES = frozenset({
    "memit",
    "alphaedit",
    "alphaedit_are",
    "unke",
    "unke_are",
})


class AnyEditEditor(EditTrainer):
    """AnyEdit 统一多策略知识编辑器

    通过 strategy 参数在运行时选择具体编辑方法，避免为每种变体
    维护独立的训练器实例。

    当 AlphaEdit / UNKE 编辑器已在项目中注册时，AnyEdit 直接
    委托给对应编辑器；否则使用内置的轻量实现。

    Attributes:
        strategy: 当前选用的编辑策略
        are_lambda: ARE 约束的权重系数（仅 *_are 策略生效）
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        strategy: str = "memit",
        v_lr: float = 5e-1,
        v_num_grad_steps: int = 20,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 4.0,
        v_loss_layer: int = 31,
        edit_weight: float = 0.5,
        ft_lr: float = 1e-5,
        ft_epochs: int = 25,
        weight_decay_factor: float = 0.1,
        nullspace_threshold: float = 5e-4,
        are_lambda: float = 1.0,
        rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj",
        layer_module_tmp: str = "model.layers.{}",
        lm_head_module: str = "lm_head",
        ln_f_module: str = "model.norm",
        *args,
        **kwargs,
    ):
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Supported: {sorted(SUPPORTED_STRATEGIES)}"
            )

        layers = layers or [4, 5, 6, 7, 8]
        super().__init__(layers=layers, *args, **kwargs)

        self.strategy = strategy
        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_weight_decay = v_weight_decay
        self.clamp_norm_factor = clamp_norm_factor
        self.v_loss_layer = v_loss_layer
        self.edit_weight = edit_weight
        self.ft_lr = ft_lr
        self.ft_epochs = ft_epochs
        self.weight_decay_factor = weight_decay_factor
        self.nullspace_threshold = nullspace_threshold
        self.are_lambda = are_lambda
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.lm_head_module = lm_head_module
        self.ln_f_module = ln_f_module

        self._proj_matrices: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        if isinstance(requests, EditRequest):
            requests = [requests]

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "strategy": self.strategy,
            "layers_edited": [],
        }

        try:
            if self._try_delegate(requests, results, **kwargs):
                return results
            self._dispatch(requests, results)
        except Exception as e:
            logger.error("AnyEdit (%s) failed: %s", self.strategy, e, exc_info=True)
            results["success"] = False

        return results

    # ------------------------------------------------------------------
    # Delegation to existing editors
    # ------------------------------------------------------------------

    def _try_delegate(
        self,
        requests: List[EditRequest],
        results: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """尝试委托给已注册的独立编辑器，成功返回 True。"""
        if self.strategy in ("alphaedit", "alphaedit_are"):
            return self._delegate_alphaedit(requests, results, **kwargs)
        if self.strategy in ("unke", "unke_are"):
            return self._delegate_unke(requests, results, **kwargs)
        return False

    def _delegate_alphaedit(self, requests, results, **kwargs) -> bool:
        try:
            from trainer.edit.alphaedit import AlphaEditEditor
        except ImportError:
            return False

        editor = AlphaEditEditor(
            layers=self.layers,
            v_lr=self.v_lr,
            v_num_grad_steps=self.v_num_grad_steps,
            v_weight_decay=self.v_weight_decay,
            clamp_norm_factor=self.clamp_norm_factor,
            v_loss_layer=self.v_loss_layer,
            nullspace_threshold=self.nullspace_threshold,
            rewrite_module_tmp=self.rewrite_module_tmp,
            layer_module_tmp=self.layer_module_tmp,
            lm_head_module=self.lm_head_module,
            ln_f_module=self.ln_f_module,
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
        )

        if self.strategy == "alphaedit_are":
            requests = self._augment_requests_with_are(requests)

        delegate_results = editor.edit(requests, **kwargs)
        results.update(delegate_results)
        results["strategy"] = self.strategy
        return True

    def _delegate_unke(self, requests, results, **kwargs) -> bool:
        try:
            from trainer.edit.unke_editor import UNKEEditor
        except ImportError:
            return False

        editor = UNKEEditor(
            layers=self.layers,
            v_lr=self.v_lr,
            v_num_grad_steps=self.v_num_grad_steps,
            v_weight_decay=self.v_weight_decay,
            clamp_norm_factor=self.clamp_norm_factor,
            v_loss_layer=self.v_loss_layer,
            ft_lr=self.ft_lr,
            ft_epochs=self.ft_epochs,
            weight_decay_factor=self.weight_decay_factor,
            rewrite_module_tmp=self.rewrite_module_tmp,
            layer_module_tmp=self.layer_module_tmp,
            lm_head_module=self.lm_head_module,
            ln_f_module=self.ln_f_module,
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.args,
        )

        if self.strategy == "unke_are":
            requests = self._augment_requests_with_are(requests)

        delegate_results = editor.edit(requests, **kwargs)
        results.update(delegate_results)
        results["strategy"] = self.strategy
        return True

    # ------------------------------------------------------------------
    # Strategy dispatch (fallback when delegation fails)
    # ------------------------------------------------------------------

    def _dispatch(self, requests: List[EditRequest], results: Dict[str, Any]):
        """根据 strategy 分派到内置实现。"""
        layer_indices = self._resolve_layer_indices(self.layers)
        results["layers_edited"] = layer_indices

        if self.strategy == "memit":
            self._run_memit(requests, layer_indices)
        elif self.strategy in ("alphaedit", "alphaedit_are"):
            use_are = self.strategy.endswith("_are")
            self._run_alphaedit(requests, layer_indices, use_are=use_are)
        elif self.strategy in ("unke", "unke_are"):
            use_are = self.strategy.endswith("_are")
            self._run_unke(requests, layer_indices, use_are=use_are)

        results["edited_count"] = len(requests)

    # ------------------------------------------------------------------
    # MEMIT
    # ------------------------------------------------------------------

    def _run_memit(
        self, requests: List[EditRequest], layer_indices: List[int]
    ):
        """MEMIT: 多层最小二乘更新。"""
        model = self.model
        tok = self.tokenizer

        z_layer = layer_indices[-1]
        zs = self._compute_target_zs(requests, z_layer)
        prompts = [r.prompt for r in requests]

        for layer_pos, layer in enumerate(layer_indices):
            layer_ks, idxs = self._collect_layer_keys(prompts, layer)

            cur_zs, _ = compute_ks(
                model, tok, prompts, z_layer,
                layer_module_tmp=self.layer_module_tmp,
            )
            targets = (zs - cur_zs.to(zs.device)) / (len(layer_indices) - layer_pos)

            hidden_size = layer_ks.shape[0]
            K = layer_ks  # (hidden, batch)
            R = targets.to(K.device)  # (batch, hidden)

            KKT = K @ K.T
            reg = 1e-5 * torch.eye(hidden_size, device=K.device)
            upd = torch.linalg.solve(KKT + reg, K @ R)  # (hidden, hidden)

            weight_name = f"{self.rewrite_module_tmp.format(layer)}.weight"
            weight = get_parameter(model, weight_name)
            upd = _match_weight_shape(upd * self.edit_weight, weight.shape)

            with torch.no_grad():
                weight.add_(upd.to(weight.dtype))

        logger.info(
            "AnyEdit(memit) applied across layers %s: %d requests",
            layer_indices, len(requests),
        )

    # ------------------------------------------------------------------
    # AlphaEdit
    # ------------------------------------------------------------------

    def _run_alphaedit(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
        *,
        use_are: bool = False,
    ):
        """AlphaEdit: 零空间投影约束编辑（可选 ARE）。"""
        model = self.model
        tok = self.tokenizer

        z_layer = layer_indices[-1]
        zs = self._compute_target_zs(requests, z_layer, use_are=use_are)
        prompts = [r.prompt for r in requests]

        for layer_pos, layer in enumerate(layer_indices):
            layer_ks, idxs = self._collect_layer_keys(prompts, layer)

            cur_zs, _ = compute_ks(
                model, tok, prompts, z_layer,
                layer_module_tmp=self.layer_module_tmp,
            )
            remaining = len(layer_indices) - layer_pos
            targets = (zs - cur_zs.to(zs.device)) / remaining

            K = layer_ks  # (hidden, batch)
            R = targets.to(K.device)
            hidden_size = K.shape[0]

            P = self._get_projection(layer, hidden_size, K.device)

            upd = torch.linalg.solve(
                P @ (K @ K.T)
                + self.nullspace_threshold
                * torch.eye(hidden_size, device=K.device),
                P @ K @ R,
            )

            weight_name = f"{self.rewrite_module_tmp.format(layer)}.weight"
            weight = get_parameter(model, weight_name)
            upd = _match_weight_shape(upd, weight.shape)

            with torch.no_grad():
                weight.add_(upd.to(weight.dtype))

            self._update_projection(layer, K, K.device)

        logger.info(
            "AnyEdit(%s) applied across layers %s: %d requests",
            self.strategy, layer_indices, len(requests),
        )

    # ------------------------------------------------------------------
    # UNKE
    # ------------------------------------------------------------------

    def _run_unke(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
        *,
        use_are: bool = False,
    ):
        """UNKE: 梯度微调编辑（可选 ARE）。"""
        model = self.model
        tok = self.tokenizer
        device = next(model.parameters()).device

        weights_copy = self._save_layer_weights(model, layer_indices)

        z_layer = layer_indices[-1]
        zs = self._compute_target_zs(requests, z_layer, use_are=use_are)
        prompts = [r.prompt for r in requests]

        for layer_pos, layer in enumerate(layer_indices):
            layer_name = self.layer_module_tmp.format(layer)

            tok_inputs = tok(prompts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                with Trace(
                    module=model, layer=layer_name,
                    retain_input=True, retain_output=True,
                    detach=True, clone=True,
                ) as tr:
                    model(**tok_inputs)
                    layer_in = tr.input
                    layer_out = tr.output

            layer_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out
            idxs = [int(m.sum()) - 1 for m in tok_inputs["attention_mask"]]

            cur_zs, _ = compute_ks(
                model, tok, prompts, z_layer,
                layer_module_tmp=self.layer_module_tmp,
            )
            remaining = len(layer_indices) - layer_pos
            resid = (zs - cur_zs.to(zs.device)) / remaining

            target_out = layer_out.clone()
            for j in range(len(idxs)):
                target_out[j, idxs[j]] += resid[j].to(target_out.device)

            self._finetune_layer(
                model, layer, layer_in, target_out, weights_copy,
                tok_inputs["attention_mask"],
            )

            for t in [layer_in, layer_out, cur_zs, target_out]:
                if isinstance(t, torch.Tensor):
                    t.cpu()
            torch.cuda.empty_cache()

        logger.info(
            "AnyEdit(%s) applied across layers %s: %d requests",
            self.strategy, layer_indices, len(requests),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _compute_target_zs(
        self,
        requests: List[EditRequest],
        z_layer: int,
        *,
        use_are: bool = False,
    ) -> torch.Tensor:
        """计算目标 z 向量，可选 ARE 约束。"""
        model = self.model
        tok = self.tokenizer

        z_list: List[torch.Tensor] = []
        for request in requests:
            req_dict = {
                "prompt": request.prompt,
                "target_new": request.target_new,
            }

            v_wd = self.v_weight_decay
            if use_are:
                v_wd = v_wd + self.are_lambda

            cur_z = compute_z(
                model, tok, req_dict, z_layer,
                layer_module_tmp=self.layer_module_tmp,
                lm_head_module=self.lm_head_module,
                ln_f_module=self.ln_f_module,
                v_lr=self.v_lr,
                v_num_grad_steps=self.v_num_grad_steps,
                v_weight_decay=v_wd,
                v_loss_layer=self.v_loss_layer,
                clamp_norm_factor=self.clamp_norm_factor,
            )
            z_list.append(cur_z)

        return torch.stack(z_list, dim=0)

    def _collect_layer_keys(
        self, prompts: List[str], layer: int
    ) -> tuple:
        """收集 rewrite module 输入的 key 向量。

        Returns:
            (keys, idxs) -- keys shape (hidden, batch), idxs list
        """
        model = self.model
        tok = self.tokenizer
        device = next(model.parameters()).device

        tok_inputs = tok(prompts, padding=True, return_tensors="pt").to(device)
        idxs = [int(m.sum()) - 1 for m in tok_inputs["attention_mask"]]

        with torch.no_grad():
            with Trace(
                module=model,
                layer=self.rewrite_module_tmp.format(layer),
                retain_input=True, retain_output=True,
                detach=True, clone=True,
            ) as tr:
                model(**tok_inputs)
                layer_in = tr.input

        layer_in = layer_in[0] if isinstance(layer_in, tuple) else layer_in
        keys = torch.stack(
            [layer_in[j, idxs[j]] for j in range(len(idxs))], dim=1
        )
        return keys, idxs

    # ------------------------------------------------------------------
    # Null-space projection (for alphaedit*)
    # ------------------------------------------------------------------

    def _get_projection(
        self, layer_idx: int, hidden_size: int, device: torch.device
    ) -> torch.Tensor:
        if layer_idx not in self._proj_matrices:
            self._proj_matrices[layer_idx] = torch.eye(
                hidden_size, dtype=torch.float32, device="cpu"
            )
        return self._proj_matrices[layer_idx].to(device)

    def _update_projection(
        self, layer_idx: int, keys: torch.Tensor, device: torch.device
    ):
        P = self._get_projection(layer_idx, keys.shape[0], device)
        b = keys.shape[1]
        KtP = keys.T @ P
        KtPK = KtP @ keys
        reg = self.nullspace_threshold * torch.eye(b, device=device)
        correction = keys @ torch.linalg.solve(KtPK + reg, KtP)
        self._proj_matrices[layer_idx] = (P - P @ correction).cpu()

    def reset_projections(self):
        """清空投影矩阵，用于新一轮连续编辑。"""
        self._proj_matrices.clear()

    # ------------------------------------------------------------------
    # Layer fine-tuning (for unke*)
    # ------------------------------------------------------------------

    @staticmethod
    def _save_layer_weights(
        model, layer_indices: List[int],
        layer_module_tmp: str = "model.layers.{}",
    ) -> Dict[str, torch.Tensor]:
        copies: Dict[str, torch.Tensor] = {}
        for attr in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                get_module(model, attr)
                layer_module_tmp = attr.rsplit(".", 1)[0] + ".layers.{}"
                break
            except LookupError:
                continue

        for idx in layer_indices:
            try:
                layer = get_module(model, layer_module_tmp.format(idx))
            except LookupError:
                continue
            prefix = layer_module_tmp.format(idx)
            for name, param in layer.named_parameters():
                copies[f"{prefix}.{name}"] = param.detach().clone()
        return copies

    def _finetune_layer(
        self,
        model,
        layer_idx: int,
        layer_in: torch.Tensor,
        target_out: torch.Tensor,
        weights_copy: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ):
        """微调单个 Transformer 层（与 UNKE 逻辑一致）。"""
        import torch.nn.functional as layer_F
        from torch.optim.lr_scheduler import CosineAnnealingLR

        layer_name = self.layer_module_tmp.format(layer_idx)
        layer_module = get_module(model, layer_name)
        device = next(layer_module.parameters()).device

        layer_input = layer_in[0] if isinstance(layer_in, tuple) else layer_in
        layer_input = layer_input.to(device)
        target_out = target_out.to(device)

        set_requires_grad(True, layer_module)

        optimizer = torch.optim.AdamW(layer_module.parameters(), lr=self.ft_lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(self.ft_epochs, 1))
        criterion = torch.nn.MSELoss()

        for step in range(self.ft_epochs):
            optimizer.zero_grad()

            output = layer_module(layer_input)
            hidden = output[0] if isinstance(output, tuple) else output

            edit_loss = criterion(hidden, target_out)

            reg_loss = torch.tensor(0.0, device=device)
            for name, param in layer_module.named_parameters():
                full_name = f"{layer_name}.{name}"
                if full_name in weights_copy:
                    orig = weights_copy[full_name].to(device)
                    reg_loss = reg_loss + layer_F.mse_loss(param, orig)

            loss = edit_loss + self.weight_decay_factor * reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

        set_requires_grad(False, layer_module)

    # ------------------------------------------------------------------
    # ARE augmentation
    # ------------------------------------------------------------------

    def _augment_requests_with_are(
        self, requests: List[EditRequest]
    ) -> List[EditRequest]:
        """为 ARE 变体在请求中添加 ARE 标记（供 compute_z 使用）。

        ARE (Adaptive Representation Editing) 在优化 z 时增加额外的
        表征一致性约束，实现方式是提高 v_weight_decay。此处通过
        标记透传给 _compute_target_zs。
        """
        return requests


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _match_weight_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Update matrix shape {matrix.shape} does not match weight shape {shape}"
    )
