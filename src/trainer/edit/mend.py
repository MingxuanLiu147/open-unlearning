"""
MEND 知识编辑器
==============

实现 Model Editor Networks with Gradient Decomposition (MEND) 方法。
MEND 是基于元学习的知识编辑方法，通过学习一个编辑网络来高效编辑模型。

参考论文: Fast Model Editing at Scale
https://arxiv.org/abs/2110.11309

核心思想：
1. 训练一个小型编辑网络（hypernetwork）
2. 编辑网络接收梯度作为输入，输出权重更新
3. 通过梯度分解降低计算复杂度
"""

import logging
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)


class MENDEditor(EditTrainer):
    """MEND 元学习知识编辑器

    通过训练编辑网络实现快速知识编辑。

    MEND 的优势：
    - 编辑速度快（只需一次前向传播）
    - 支持连续编辑
    - 编辑网络可复用

    Attributes:
        edit_lr: 编辑网络的学习率
        n_hidden: 编辑网络的隐藏层维度
        rank: 梯度分解的秩
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        edit_lr: float = 1e-4,
        n_hidden: int = 128,
        rank: int = 1920,
        *args,
        **kwargs,
    ):
        """初始化 MEND 编辑器

        Args:
            layers: 编辑的目标层
            edit_lr: 编辑网络学习率
            n_hidden: 编辑网络隐藏层大小
            rank: 梯度分解秩（控制计算量和精度的平衡）
        """
        super().__init__(layers=layers, *args, **kwargs)

        self.edit_lr = edit_lr
        self.n_hidden = n_hidden
        self.rank = rank

        # 编辑网络将在首次使用时初始化
        self.edit_network = None
        self._initialized = False

    def _init_edit_network(self):
        """初始化编辑网络

        编辑网络结构：
        - 输入：梯度向量（经过分解）
        - 隐藏层：MLP
        - 输出：权重更新向量
        """
        if self._initialized:
            return

        model = self.model
        device = next(model.parameters()).device

        # 获取模型隐藏层大小
        hidden_size = model.config.hidden_size

        # 创建编辑网络（简化版）
        self.edit_network = nn.Sequential(
            nn.Linear(self.rank * 2, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, hidden_size * 2),
        ).to(device)

        self._initialized = True
        logger.info(
            f"MEND edit network initialized with rank={self.rank}, n_hidden={self.n_hidden}"
        )

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 MEND 知识编辑

        Args:
            requests: 编辑请求

        Returns:
            编辑结果
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        # 初始化编辑网络
        self._init_edit_network()

        results = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
        }

        for request in requests:
            try:
                self._apply_mend_edit(request)
                results["edited_count"] += 1
            except Exception as e:
                logger.error(f"MEND edit failed: {e}")
                results["success"] = False

        return results

    def _apply_mend_edit(self, request: EditRequest):
        """应用单个 MEND 编辑

        Args:
            request: 编辑请求
        """
        model = self.model
        tokenizer = self.tokenizer

        # 1. 计算编辑损失的梯度
        grad = self._compute_edit_gradient(request)

        # 2. 梯度分解
        u, v = self._decompose_gradient(grad)

        # 3. 通过编辑网络计算权重更新
        delta = self._compute_update_from_gradient(u, v)

        # 4. 应用更新
        self._apply_delta_update(delta)

        logger.info(f"MEND edit applied: {request.subject} -> {request.target_new}")

    def _compute_edit_gradient(self, request: EditRequest) -> torch.Tensor:
        """计算编辑目标的梯度

        Args:
            request: 编辑请求

        Returns:
            梯度向量
        """
        model = self.model
        tokenizer = self.tokenizer

        # 构造编辑输入
        prompt = f"{request.prompt} {request.target_new}"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 前向传播
        model.train()
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # 计算梯度
        loss.backward()

        # 收集目标层的梯度
        grads = []
        layer_idx = self.layers[0] if self.layers else 5
        layer = self._get_layer_module(model, layer_idx)

        for name, param in layer.named_parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten())

        # 清除梯度
        model.zero_grad()
        model.eval()

        if grads:
            return torch.cat(grads)
        else:
            # 返回零梯度作为后备
            hidden_size = model.config.hidden_size
            return torch.zeros(hidden_size, device=model.device)

    def _decompose_gradient(self, grad: torch.Tensor):
        """梯度分解

        将梯度分解为低秩形式以减少计算量。

        Args:
            grad: 原始梯度

        Returns:
            (u, v) 分解后的向量对
        """
        # 简化的分解：使用 SVD 的前 rank 个分量
        grad_2d = grad.reshape(1, -1) if grad.dim() == 1 else grad

        # 确保维度足够
        min_dim = min(grad_2d.shape)
        rank = min(self.rank, min_dim)

        if rank > 0:
            # 使用截断 SVD
            U, S, V = torch.svd_lowrank(grad_2d.float(), q=rank)
            u = U[:, :rank] @ torch.diag(S[:rank].sqrt())
            v = V[:, :rank] @ torch.diag(S[:rank].sqrt())
        else:
            # 后备：直接使用梯度的部分
            u = (
                grad[: self.rank].unsqueeze(0)
                if len(grad) >= self.rank
                else grad.unsqueeze(0)
            )
            v = u.clone()

        return u.flatten(), v.flatten()

    def _compute_update_from_gradient(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """通过编辑网络计算权重更新

        Args:
            u, v: 分解后的梯度向量

        Returns:
            权重更新向量
        """
        # 拼接输入
        # 确保维度匹配
        target_size = self.rank
        u_padded = F.pad(u, (0, max(0, target_size - len(u))))[:target_size]
        v_padded = F.pad(v, (0, max(0, target_size - len(v))))[:target_size]

        edit_input = torch.cat([u_padded, v_padded]).unsqueeze(0)

        # 通过编辑网络
        delta = self.edit_network(edit_input)

        return delta.squeeze(0)

    def _apply_delta_update(self, delta: torch.Tensor):
        """应用权重更新

        Args:
            delta: 权重更新向量
        """
        model = self.model
        layer_idx = self.layers[0] if self.layers else 5
        layer = self._get_layer_module(model, layer_idx)

        # 获取 MLP 层并应用更新
        for proj_name in ["mlp.down_proj", "mlp.c_proj"]:
            try:
                proj = self._get_module_by_name(layer, proj_name)

                with torch.no_grad():
                    # 将 delta reshape 并应用
                    weight = proj.weight
                    delta_reshaped = delta[: weight.numel()].reshape(weight.shape)
                    weight.add_(delta_reshaped * self.edit_lr)

                break
            except AttributeError:
                continue

        logger.debug(f"Delta update applied to layer {layer_idx}")

    def _get_layer_module(self, model: nn.Module, layer_idx: int):
        """获取指定层模块"""
        for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._get_module_by_name(model, attr_name)
                return layers[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None

    def train_edit_network(
        self,
        train_requests: List[EditRequest],
        num_epochs: int = 10,
        batch_size: int = 32,
    ):
        """训练编辑网络

        在使用 MEND 之前，需要先训练编辑网络。

        Args:
            train_requests: 训练数据（编辑请求列表）
            num_epochs: 训练轮数
            batch_size: 批大小
        """
        self._init_edit_network()

        optimizer = torch.optim.Adam(self.edit_network.parameters(), lr=self.edit_lr)

        for epoch in range(num_epochs):
            total_loss = 0

            for i in range(0, len(train_requests), batch_size):
                batch = train_requests[i : i + batch_size]

                optimizer.zero_grad()

                batch_loss = 0
                for request in batch:
                    # 计算编辑损失
                    grad = self._compute_edit_gradient(request)
                    u, v = self._decompose_gradient(grad)
                    delta = self._compute_update_from_gradient(u, v)

                    # 简化的训练目标：delta 应该减小编辑损失
                    batch_loss += delta.norm()

                batch_loss /= len(batch)
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()

            logger.info(
                f"MEND training epoch {epoch + 1}/{num_epochs}, loss: {total_loss:.4f}"
            )
