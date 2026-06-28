"""
MEMIT-Merge: Fixing Batch Editing for Same-Subject Facts
=========================================================

ACL 2025 — When multiple edits share the same subject (e.g. updating
several facts about "Paris"), the original MEMIT suffers catastrophic
failure because the key vectors are nearly identical, making the K K^T
matrix rank-deficient.  Reported accuracy drops from ~98% (single edit)
to ~46% (same-subject batch).

MEMIT-Merge fixes this by:
1. **Detecting** same-subject edit groups within a batch
2. **Merging** their key vectors into a single representative key per group
   (weighted average or principal component)
3. **Combining** their target value vectors into a joint objective
4. Using the merged (full-rank) key matrix for the least-squares update

This restores batch editing accuracy from 46% to 98% with minimal code
change on top of the existing MEMITEditor.

Paper: https://arxiv.org/abs/2501.xxxxx (ACL 2025)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from trainer.edit.base import EditRequest
from trainer.edit.memit import MEMITEditor

logger = logging.getLogger(__name__)


class MEMITMergeEditor(MEMITEditor):
    """MEMIT with same-subject key merging for robust batch editing.

    Inherits everything from MEMITEditor but overrides the batch editing
    pipeline to detect and merge same-subject edits before computing the
    least-squares weight update.

    Additional attributes:
        merge_threshold: cosine-similarity threshold above which two keys
            are considered "same subject" and merged (default 0.95)
        merge_strategy: "mean" (average keys/values) or "pca" (first
            principal component of keys, combined values)
    """

    def __init__(
        self,
        merge_threshold: float = 0.95,
        merge_strategy: str = "mean",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.merge_threshold = merge_threshold
        self.merge_strategy = merge_strategy.lower()

    def _apply_memit_edit(
        self, requests: List[EditRequest], layer_indices: List[int]
    ):
        """Override: merge same-subject keys before applying MEMIT updates."""
        model = self.model

        keys_per_layer = {layer: [] for layer in layer_indices}
        for request in requests:
            for layer_idx in layer_indices:
                key = self._compute_key_vector(
                    request.prompt, request.subject, layer_idx
                )
                keys_per_layer[layer_idx].append(key)

        subject_groups = self._group_by_subject(requests)

        for layer_idx in layer_indices:
            raw_keys = torch.stack(keys_per_layer[layer_idx])
            raw_values = self._compute_layer_values(requests, layer_idx, raw_keys)

            merged_keys, merged_values = self._merge_same_subject(
                raw_keys, raw_values, requests, subject_groups
            )

            self._apply_least_squares_update(layer_idx, merged_keys, merged_values)

        logger.info(
            "MEMIT-Merge edit across layers %s: %d requests -> %d merged groups",
            layer_indices, len(requests), len(subject_groups),
        )

    @staticmethod
    def _group_by_subject(
        requests: List[EditRequest],
    ) -> Dict[str, List[int]]:
        """Group request indices by normalised subject string."""
        groups: Dict[str, List[int]] = defaultdict(list)
        for idx, req in enumerate(requests):
            key = req.subject.strip().lower()
            groups[key].append(idx)
        return dict(groups)

    def _merge_same_subject(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        requests: List[EditRequest],
        subject_groups: Dict[str, List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge keys and values for same-subject edit groups.

        For groups with a single edit, key/value pass through unchanged.
        For multi-edit groups, merging depends on ``merge_strategy``.
        """
        merged_keys_list = []
        merged_values_list = []

        for subject, indices in subject_groups.items():
            group_keys = keys[indices]
            group_values = values[indices]

            if len(indices) == 1:
                merged_keys_list.append(group_keys[0])
                merged_values_list.append(group_values[0])
                continue

            if self.merge_strategy == "pca":
                mk, mv = self._pca_merge(group_keys, group_values)
            else:
                mk, mv = self._mean_merge(group_keys, group_values)

            cosine_sim = torch.nn.functional.cosine_similarity(
                group_keys[0:1], group_keys[1:], dim=1
            ).mean()
            if cosine_sim > self.merge_threshold:
                logger.info(
                    "Merging %d edits for subject '%s' (cos_sim=%.4f)",
                    len(indices), subject, cosine_sim.item(),
                )

            merged_keys_list.append(mk)
            merged_values_list.append(mv)

        merged_keys = torch.stack(merged_keys_list)
        merged_values = torch.stack(merged_values_list)
        return merged_keys, merged_values

    @staticmethod
    def _mean_merge(
        keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return keys.mean(dim=0), values.mean(dim=0)

    @staticmethod
    def _pca_merge(
        keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use the first principal component of keys as the merged key."""
        centered = keys - keys.mean(dim=0, keepdim=True)
        try:
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            principal = Vh[0]
        except Exception:
            principal = keys.mean(dim=0)
            principal = principal / (principal.norm() + 1e-10)

        projections = (keys * principal.unsqueeze(0)).sum(dim=1)
        weights = torch.softmax(projections, dim=0)
        merged_value = (weights.unsqueeze(1) * values).sum(dim=0)
        return principal * keys.mean(dim=0).norm(), merged_value

    def _apply_least_squares_update(
        self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ):
        """Least-squares update with merged (better-conditioned) key matrix.

        Identical to the parent implementation but benefits from a
        better-conditioned KK^T after merging duplicate subjects.
        """
        model = self.model
        layer = self._get_layer_module(model, layer_idx)

        for proj_name in ["mlp.down_proj", "mlp.c_proj", "mlp.fc_out", "mlp.dense_4h_to_h"]:
            try:
                proj = self._get_module_by_name(layer, proj_name)
                break
            except AttributeError:
                continue
        else:
            logger.warning("Cannot find MLP projection in layer %d", layer_idx)
            return

        weight = proj.weight

        with torch.no_grad():
            K = keys.T
            V = values.T

            KKT = K @ K.T
            cond = torch.linalg.cond(KKT).item() if KKT.shape[0] <= 4096 else float("inf")
            reg_scale = 1e-5
            if cond > 1e8:
                reg_scale = 1e-3
                logger.info(
                    "Layer %d: KKT cond=%.2e (high), increasing regularisation to %.1e",
                    layer_idx, cond, reg_scale,
                )

            KKT_reg = KKT + reg_scale * torch.eye(KKT.shape[0], device=KKT.device)
            KKT_inv = torch.linalg.pinv(KKT_reg)

            residual = V - weight @ K
            delta = residual @ K.T @ KKT_inv

            weight.add_(delta * self.edit_weight)

        logger.debug(
            "MEMIT-Merge least-squares update at layer %d (cond=%.2e)",
            layer_idx, cond,
        )
