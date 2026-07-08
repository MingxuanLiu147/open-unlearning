"""
AlphaEdit knowledge editor
===========================

Implements AlphaEdit: null-space constrained knowledge editing.
Extends MEMIT with null-space projection to restrict weight updates to the
orthogonal complement of existing knowledge representations, improving
locality in sequential editing scenarios.

Paper: AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models
https://arxiv.org/abs/2410.02355

Algorithm:
1. Estimate covariance C of rewrite-module inputs via layer_stats
2. Build null-space projection P = I - C(C^2 + lambda*I)^{-1} C
3. Optimise target representations z at the last edit layer (compute_z)
4. Per-layer: collect rewrite-module input keys, compute residual,
   solve constrained update  (P K K^T + lambda I) dW = P K resid
"""

import logging
from typing import Dict, Any, List, Optional, Union

import torch

from trainer.edit.base import EditTrainer, EditRequest
from trainer.edit.utils.nethook import Trace, get_parameter, set_requires_grad
from trainer.edit.utils.compute_z import compute_z, compute_ks
from trainer.edit.utils.layer_stats import layer_stats

logger = logging.getLogger(__name__)


class AlphaEditEditor(EditTrainer):
    """AlphaEdit editor with null-space projection constraints.

    Key difference from MEMIT:
    - MEMIT:     dW = (V - W K) K^T (K K^T)^{-1}
    - AlphaEdit: (P K K^T + lambda I) dW = P K resid
    where P projects onto the null space of cached key covariance,
    preventing edits from overwriting existing factual associations.
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        v_lr: float = 5e-1,
        v_num_grad_steps: int = 20,
        v_weight_decay: float = 0.5,
        clamp_norm_factor: float = 4.0,
        v_loss_layer: int = 31,
        kl_factor: float = 0.0625,
        mom2_update_weight: float = 4000.0,
        nullspace_threshold: float = 1e-5,
        rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj",
        layer_module_tmp: str = "model.layers.{}",
        lm_head_module: str = "lm_head",
        ln_f_module: str = "model.norm",
        mom2_dataset: str = "wikipedia",
        mom2_n_samples: int = 100000,
        mom2_dtype: str = "float32",
        stats_dir: str = "data/stats",
        *args,
        **kwargs,
    ):
        layers = layers or [4, 5, 6, 7, 8]
        super().__init__(layers=layers, *args, **kwargs)

        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.v_weight_decay = v_weight_decay
        self.clamp_norm_factor = clamp_norm_factor
        self.v_loss_layer = v_loss_layer
        self.kl_factor = kl_factor
        self.mom2_update_weight = mom2_update_weight
        self.nullspace_threshold = nullspace_threshold
        self.rewrite_module_tmp = rewrite_module_tmp
        self.layer_module_tmp = layer_module_tmp
        self.lm_head_module = lm_head_module
        self.ln_f_module = ln_f_module
        self.mom2_dataset = mom2_dataset
        self.mom2_n_samples = mom2_n_samples
        self.mom2_dtype = mom2_dtype
        self.stats_dir = stats_dir

        self._cov_cache: Dict[str, torch.Tensor] = {}

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """Execute AlphaEdit knowledge editing.

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

            P = self._compute_nullspace_projections(layer_indices)
            self._apply_alphaedit(requests, layer_indices, P)
            results["edited_count"] = len(requests)
        except Exception as e:
            logger.error("AlphaEdit edit failed: %s", e, exc_info=True)
            results["success"] = False

        return results

    # ------------------------------------------------------------------
    # Null-space projection
    # ------------------------------------------------------------------

    def _get_covariance(self, layer_name: str) -> torch.Tensor:
        """Return (cached) second-moment matrix of rewrite-module inputs."""
        model_tag = self.model.config._name_or_path.replace("/", "_")
        cache_key = f"{model_tag}_{layer_name}"

        if cache_key not in self._cov_cache:
            stats = layer_stats(
                self.model,
                self.tokenizer,
                layer_name,
                self.stats_dir,
                ds_name=self.mom2_dataset,
                to_collect=["mom2"],
                sample_size=self.mom2_n_samples,
                precision=self.mom2_dtype,
            )
            self._cov_cache[cache_key] = stats["mom2"].float().cpu()

        return self._cov_cache[cache_key]

    def _compute_nullspace_projections(
        self, layer_indices: List[int]
    ) -> torch.Tensor:
        """Build null-space projection matrices for each edit layer.

        P = I - C (C^2 + lambda I)^{-1} C

        where C is the second-moment (covariance) of rewrite-module input
        features.  P suppresses directions already occupied by existing
        knowledge so that new edits land in the orthogonal complement.

        Returns:
            Tensor of shape ``(num_layers, input_dim, input_dim)``.
        """
        device = next(self.model.parameters()).device
        projections = []

        for layer_idx in layer_indices:
            layer_name = self.rewrite_module_tmp.format(layer_idx)
            cov = self._get_covariance(layer_name).to(device)
            d = cov.shape[0]

            lam = self.nullspace_threshold
            # P = I - C @ solve(C^2 + lam*I, C)
            P = torch.eye(d, device=device) - cov @ torch.linalg.solve(
                cov @ cov + lam * torch.eye(d, device=device), cov
            )
            projections.append(P)
            logger.debug(
                "Null-space projection for layer %d: effective rank ~ %d / %d",
                layer_idx,
                int((torch.linalg.eigvalsh(P) > 0.5).sum().item()),
                d,
            )

        return torch.stack(projections, dim=0)

    # ------------------------------------------------------------------
    # Core editing loop
    # ------------------------------------------------------------------

    def _apply_alphaedit(
        self,
        requests: List[EditRequest],
        layer_indices: List[int],
        P: torch.Tensor,
    ):
        """Apply AlphaEdit across multiple layers.

        1. compute_z at the last edit layer for each request
        2. For each layer (first to last):
           a. Trace rewrite-module input keys at subject last-token
           b. Recompute current z -> residual = target_z - current_z
           c. Distribute residual among remaining layers
           d. Solve projected normal equations for weight update
        """
        model = self.model
        tok = self.tokenizer
        device = next(model.parameters()).device

        # Snapshot rewrite-module weights
        weights = {
            f"{self.rewrite_module_tmp.format(layer)}.weight": get_parameter(
                model, f"{self.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in layer_indices
        }
        weights_copy = {k: v.detach().clone() for k, v in weights.items()}

        z_layer = layer_indices[-1]
        req_dicts = [
            {"prompt": r.prompt, "target_new": r.target_new}
            for r in requests
        ]

        # Step 1: optimise target z at the last edit layer
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

        # Step 2: layer-by-layer projected update
        for i, layer_idx in enumerate(layer_indices):
            rewrite_name = self.rewrite_module_tmp.format(layer_idx)

            # 2a. Collect rewrite-module input keys
            ctx_tok = tok(prompts, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                with Trace(
                    module=model,
                    layer=rewrite_name,
                    retain_input=True,
                    retain_output=False,
                    detach=True,
                    clone=True,
                ) as tr:
                    model(**ctx_tok)
                    layer_in = tr.input
            if isinstance(layer_in, tuple):
                layer_in = layer_in[0]

            # 2b. Recompute current z at z_layer -> residual
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

            # 2c. Extract keys at subject last-token: (input_dim, batch)
            ks_list = [layer_in[j, idxs[j]] for j in range(len(idxs))]
            layer_ks = torch.stack(ks_list, dim=1)

            dev = layer_ks.device
            targets = targets.to(dev)

            # Distribute residual across remaining layers
            resid = targets / max(len(layer_indices) - i, 1)  # (batch, hidden)

            # 2d. Solve: (P K K^T + lam I) dW = P K resid
            p_slice = P[i].to(dev)
            lam = self.nullspace_threshold
            upd_matrix = torch.linalg.solve(
                p_slice @ (layer_ks @ layer_ks.T)
                + lam * torch.eye(layer_ks.shape[0], device=dev),
                p_slice @ layer_ks @ resid,
            )

            weight_name = f"{rewrite_name}.weight"
            upd_matrix = _match_weight_shape(upd_matrix, weights[weight_name].shape)

            logger.debug(
                "Layer %d update: orig_norm=%.4f upd_norm=%.4f",
                layer_idx,
                torch.linalg.norm(weights[weight_name]).item(),
                torch.linalg.norm(upd_matrix).item(),
            )

            with torch.no_grad():
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()

            del layer_ks, cur_zs, targets, layer_in, p_slice, upd_matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(
            "AlphaEdit applied across layers %s: %d requests",
            layer_indices,
            len(requests),
        )


def _match_weight_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Handle GPT-2/GPT-J transposed weight representations."""
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        f"Update matrix shape {matrix.shape} does not match weight shape {shape}"
    )
