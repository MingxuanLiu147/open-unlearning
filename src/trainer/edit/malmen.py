"""
MALMEN 知识编辑器
================

实现 Fast Model Editing at Scale (MALMEN) 方法。
MALMEN 是基于元学习的知识编辑方法，与 MEND 类似但面向批量编辑优化，
通过辅助网络（auxiliary network）和正规方程求解最优批量更新。

参考论文: Fast Model Editing at Scale
https://arxiv.org/abs/2304.00740

核心思想：
1. 预训练一个辅助网络（hypernetwork），将梯度映射为权重更新
2. 对一批编辑请求，收集各自梯度并通过辅助网络得到初步权重增量
3. 使用正规方程在批维度上求解最优组合权重，减少编辑间冲突
4. 将最终更新写回目标层权重
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)


class _AuxiliaryNetwork(nn.Module):
    """MALMEN 辅助网络

    接收展平后的梯度向量，输出与目标权重形状匹配的增量向量。
    使用两层 MLP + 残差连接以提升大批量时的稳定性。
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.proj_in(x))
        h = h + self.act(self.hidden(h))
        return self.proj_out(h)


class MALMENEditor(EditTrainer):
    """MALMEN 批量元学习知识编辑器

    与 MEND 相比，MALMEN 使用正规方程在批维度上全局求解，
    减少多条编辑同时应用时的相互干扰。

    MALMEN vs MEND:
    - MEND: 逐条编辑，每条独立通过编辑网络
    - MALMEN: 收集整批梯度 -> 辅助网络 -> 正规方程求解最优组合

    Attributes:
        archive: 预训练辅助网络权重路径
        edit_lr: 编辑学习率（控制最终更新幅度）
        n_hidden: 辅助网络隐藏层维度
        n_edits: 单批最大编辑数
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        archive: Optional[str] = None,
        edit_lr: float = 1e-4,
        n_hidden: int = 256,
        rank: int = 1920,
        batch_size: int = 10,
        n_edits: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(layers=layers, *args, **kwargs)

        self.archive = archive
        self.edit_lr = edit_lr
        self.n_hidden = n_hidden
        self.rank = rank
        self.batch_size = batch_size
        self.n_edits = n_edits

        self.aux_net: Optional[_AuxiliaryNetwork] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_model(self):
        """初始化辅助网络并加载预训练权重（若有）。"""
        if self._initialized:
            return

        model = self.model
        device = next(model.parameters()).device
        hidden_size = model.config.hidden_size

        input_dim = self.rank * 2
        output_dim = hidden_size

        self.aux_net = _AuxiliaryNetwork(input_dim, self.n_hidden, output_dim).to(
            device
        )

        if self.archive and Path(self.archive).exists():
            state = torch.load(
                self.archive, map_location=device, weights_only=True
            )
            if "aux_net" in state:
                self.aux_net.load_state_dict(state["aux_net"])
            else:
                self.aux_net.load_state_dict(state)
            logger.info("MALMEN auxiliary network loaded from %s", self.archive)
        else:
            logger.warning(
                "No MALMEN archive at '%s'; auxiliary network uses random init",
                self.archive,
            )

        self._initialized = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 MALMEN 批量知识编辑

        将请求按 batch_size 分组，每组：
        1. 收集梯度 -> 辅助网络得到初步增量
        2. 正规方程全局求解最优组合
        3. 应用更新到目标层
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        self._init_model()

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
        }

        for batch_start in range(0, len(requests), self.batch_size):
            batch = requests[batch_start : batch_start + self.batch_size]
            try:
                self._apply_malmen_batch(batch)
                results["edited_count"] += len(batch)
            except Exception as e:
                logger.error("MALMEN batch edit failed: %s", e, exc_info=True)
                results["success"] = False

        return results

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def _apply_malmen_batch(self, requests: List[EditRequest]):
        """对一个 batch 的请求执行 MALMEN 编辑。

        Steps:
          1. 对每条请求计算编辑梯度并分解为低秩表示
          2. 辅助网络将每条梯度映射为初步权重增量 delta_i
          3. 正规方程求解最优线性组合系数 alpha
          4. 最终增量 = sum(alpha_i * delta_i)，写回目标层
        """
        model = self.model
        device = next(model.parameters()).device
        layer_indices = self._resolve_layer_indices(self.layers)

        deltas: List[torch.Tensor] = []
        target_losses: List[torch.Tensor] = []

        for request in requests:
            grad = self._compute_edit_gradient(request)
            u, v = self._decompose_gradient(grad, device)
            delta = self.aux_net(torch.cat([u, v]).unsqueeze(0)).squeeze(0)
            deltas.append(delta)

            target_loss = self._compute_target_loss(request)
            target_losses.append(target_loss)

        if not deltas:
            return

        delta_matrix = torch.stack(deltas, dim=0)  # (batch, hidden)
        loss_vec = torch.stack(target_losses, dim=0)  # (batch,)

        # Normal equation: alpha = (D^T D + lambda I)^{-1} D^T l
        DtD = delta_matrix @ delta_matrix.T  # (batch, batch)
        reg = 1e-5 * torch.eye(DtD.shape[0], device=device)
        alpha = torch.linalg.solve(DtD + reg, loss_vec)  # (batch,)

        combined_delta = (alpha.unsqueeze(1) * delta_matrix).sum(dim=0)

        for layer_idx in layer_indices:
            self._apply_delta_to_layer(layer_idx, combined_delta)

        logger.info(
            "MALMEN batch edit applied: %d requests across layers %s",
            len(requests),
            layer_indices,
        )

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def _compute_edit_gradient(self, request: EditRequest) -> torch.Tensor:
        """计算编辑目标的梯度向量。"""
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device

        prompt = f"{request.prompt} {request.target_new}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        outputs.loss.backward()

        grads = []
        layer_idx = self.layers[0] if self.layers else 5
        layer = self._get_layer_module(model, layer_idx)
        if layer is not None:
            for _, param in layer.named_parameters():
                if param.grad is not None:
                    grads.append(param.grad.flatten())

        model.zero_grad()
        model.eval()

        if grads:
            return torch.cat(grads)
        return torch.zeros(model.config.hidden_size, device=device)

    def _decompose_gradient(
        self, grad: torch.Tensor, device: torch.device
    ) -> tuple:
        """低秩分解梯度为 (u, v) 向量对。"""
        grad_2d = grad.reshape(1, -1).float()
        min_dim = min(grad_2d.shape)
        rank = min(self.rank, min_dim)

        if rank > 0 and grad.numel() > 1:
            try:
                U, S, V = torch.svd_lowrank(grad_2d, q=rank)
                sqrt_s = S[:rank].sqrt()
                u = (U[:, :rank] @ torch.diag(sqrt_s)).flatten()
                v = (V[:, :rank] @ torch.diag(sqrt_s)).flatten()
            except Exception:
                u = v = grad.flatten().float()
        else:
            u = v = grad.flatten().float()

        u = F.pad(u, (0, max(0, self.rank - u.numel())))[: self.rank]
        v = F.pad(v, (0, max(0, self.rank - v.numel())))[: self.rank]
        return u.to(device), v.to(device)

    def _compute_target_loss(self, request: EditRequest) -> torch.Tensor:
        """计算单条请求的编辑目标损失（正规方程右侧）。"""
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device

        prompt = f"{request.prompt} {request.target_new}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        return outputs.loss.detach()

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def _apply_delta_to_layer(self, layer_idx: int, delta: torch.Tensor):
        """将组合增量应用到指定层的 MLP 投影权重。"""
        layer = self._get_layer_module(self.model, layer_idx)
        if layer is None:
            return

        for proj_name in ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"]:
            try:
                proj = self._get_module_by_name(layer, proj_name)
                with torch.no_grad():
                    weight = proj.weight
                    n = min(delta.numel(), weight.numel())
                    flat = weight.flatten()
                    flat[:n] += delta[:n].to(flat.dtype) * self.edit_lr
                break
            except AttributeError:
                continue

    def _get_layer_module(self, model: nn.Module, layer_idx: int):
        """获取指定层模块。"""
        for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._get_module_by_name(model, attr_name)
                return layers[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_auxiliary_network(
        self,
        train_requests: List[EditRequest],
        num_epochs: int = 10,
        lr: float = 1e-3,
        save_path: Optional[str] = None,
    ):
        """训练 MALMEN 辅助网络

        Args:
            train_requests: 训练用编辑请求
            num_epochs: 训练轮数
            lr: 辅助网络学习率
            save_path: 训练完成后保存路径
        """
        self._init_model()

        optimizer = torch.optim.Adam(self.aux_net.parameters(), lr=lr)
        device = next(self.model.parameters()).device

        for epoch in range(num_epochs):
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_requests), self.batch_size):
                batch = train_requests[i : i + self.batch_size]
                optimizer.zero_grad()

                batch_loss = torch.tensor(
                    0.0, device=device, requires_grad=True
                )
                for request in batch:
                    grad = self._compute_edit_gradient(request)
                    u, v = self._decompose_gradient(grad, device)
                    delta = self.aux_net(
                        torch.cat([u, v]).unsqueeze(0)
                    ).squeeze(0)
                    batch_loss = batch_loss + delta.norm()

                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                n_batches += 1

            logger.info(
                "MALMEN training epoch %d/%d, avg_loss: %.4f",
                epoch + 1,
                num_epochs,
                total_loss / max(n_batches, 1),
            )

        if save_path:
            torch.save({"aux_net": self.aux_net.state_dict()}, save_path)
            logger.info("MALMEN auxiliary network saved to %s", save_path)
