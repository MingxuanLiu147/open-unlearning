"""
MEMIT 知识编辑器
===============

实现 Mass Editing Memory in Transformer (MEMIT) 方法。
MEMIT 是 ROME 的扩展，支持同时编辑多条知识。

参考论文: Mass-Editing Memory in a Transformer
https://arxiv.org/abs/2210.07229

核心思想：
1. 在多个层同时应用编辑（分散编辑负担）
2. 使用最小二乘法同时优化多个编辑请求
3. 保持更好的局部性和稳定性
"""

import logging
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn.functional as F
from torch import nn

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)


class MEMITEditor(EditTrainer):
    """MEMIT 批量知识编辑器

    支持同时编辑多条知识，通过在多层分散编辑实现更好的稳定性。

    MEMIT vs ROME：
    - ROME：单层编辑，适合单条知识
    - MEMIT：多层编辑，适合批量知识，稳定性更好

    Attributes:
        layers: 编辑的目标层列表（MEMIT 通常使用多层）
        v_lr: value 优化学习率
        v_num_grad_steps: 优化步数
        clamp_norm_factor: 梯度裁剪
        edit_weight: 编辑强度权重
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        v_lr: float = 0.5,
        v_num_grad_steps: int = 20,
        clamp_norm_factor: float = 4.0,
        edit_weight: float = 0.5,
        *args,
        **kwargs,
    ):
        """初始化 MEMIT 编辑器

        Args:
            layers: 编辑层列表，默认使用多个中间层
            v_lr: value 优化学习率
            v_num_grad_steps: 优化步数
            clamp_norm_factor: 梯度裁剪因子
            edit_weight: 每层编辑的权重分配
        """
        # MEMIT 默认使用更多层
        layers = layers or [4, 5, 6, 7, 8]
        super().__init__(layers=layers, *args, **kwargs)

        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.clamp_norm_factor = clamp_norm_factor
        self.edit_weight = edit_weight

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 MEMIT 批量知识编辑

        Args:
            requests: 编辑请求列表

        Returns:
            编辑结果字典
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
            "layers_edited": self.layers,
        }

        try:
            # MEMIT 同时处理所有请求
            self._apply_memit_edit(requests)
            results["edited_count"] = len(requests)
        except Exception as e:
            logger.error(f"MEMIT edit failed: {e}")
            results["success"] = False

        return results

    def _apply_memit_edit(self, requests: List[EditRequest]):
        """应用 MEMIT 批量编辑

        Args:
            requests: 编辑请求列表
        """
        model = self.model
        tokenizer = self.tokenizer

        # 1. 为每个请求收集 key 向量
        keys_per_layer = {layer: [] for layer in self.layers}
        targets = []

        for request in requests:
            for layer_idx in self.layers:
                key = self._compute_key_vector(
                    request.prompt, request.subject, layer_idx
                )
                keys_per_layer[layer_idx].append(key)
            targets.append(request.target_new)

        # 2. 计算每层的编辑更新
        for layer_idx in self.layers:
            keys = torch.stack(keys_per_layer[layer_idx])

            # 计算该层的 value 向量
            values = self._compute_layer_values(requests, layer_idx, keys)

            # 应用编辑（使用伪逆求解最小二乘）
            self._apply_least_squares_update(layer_idx, keys, values)

        logger.info(
            f"MEMIT edit applied across layers {self.layers}: {len(requests)} requests"
        )

    def _compute_key_vector(
        self, prompt: str, subject: str, layer_idx: int
    ) -> torch.Tensor:
        """计算 key 向量（与 ROME 类似）"""
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        subject_tokens = tokenizer(subject, add_special_tokens=False)["input_ids"]

        # 定位 subject
        input_ids = inputs["input_ids"][0].tolist()
        subject_end = len(input_ids) - 1

        for i in range(len(input_ids) - len(subject_tokens) + 1):
            if input_ids[i : i + len(subject_tokens)] == subject_tokens:
                subject_end = i + len(subject_tokens) - 1
                break

        # 获取隐藏状态
        hidden_states = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states.append(output[0])
            else:
                hidden_states.append(output)

        layer = self._get_layer_module(model, layer_idx)
        handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        key = hidden_states[0][0, subject_end, :].clone()
        return key

    def _get_layer_module(self, model: nn.Module, layer_idx: int):
        """获取指定层模块"""
        for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._get_module_by_name(model, attr_name)
                return layers[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None

    def _compute_layer_values(
        self, requests: List[EditRequest], layer_idx: int, keys: torch.Tensor
    ) -> torch.Tensor:
        """计算某层所有请求的 value 向量

        通过优化找到使模型输出目标的 value 向量。
        MEMIT 将编辑分散到多层，每层只承担部分编辑责任。

        Args:
            requests: 请求列表
            layer_idx: 层索引
            keys: key 向量矩阵 [num_requests, hidden_size]

        Returns:
            value 向量矩阵 [num_requests, hidden_size]
        """
        model = self.model
        tokenizer = self.tokenizer
        device = keys.device
        num_requests = len(requests)
        hidden_size = keys.shape[1]

        # 获取 MLP 投影层
        layer = self._get_layer_module(model, layer_idx)
        mlp_proj = self._get_mlp_projection(layer)
        if mlp_proj is None:
            # 如果找不到 MLP，返回基于 key 的简化 value
            logger.warning(f"Cannot find MLP projection in layer {layer_idx}, using fallback")
            return keys.clone()

        # 计算当前输出作为基准
        with torch.no_grad():
            weight = mlp_proj.weight
            current_outputs = weight @ keys.T  # [hidden_size, num_requests]

        # 初始化 delta（要添加的扰动）
        deltas = torch.zeros(num_requests, hidden_size, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([deltas], lr=self.v_lr)

        # 优化 delta
        for step in range(self.v_num_grad_steps):
            optimizer.zero_grad()
            total_loss = 0

            for i, request in enumerate(requests):
                # 准备输入
                prompt = request.prompt
                target_new = request.target_new

                prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
                target_ids = tokenizer(
                    target_new, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(device)

                # 注入 delta 的 hook
                def create_hook(delta_vec, target_pos):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hidden = output[0].clone()
                        else:
                            hidden = output.clone()
                        # 按编辑权重分配 delta（MEMIT 特性：分散编辑）
                        hidden[:, target_pos, :] = hidden[:, target_pos, :] + delta_vec * self.edit_weight
                        if isinstance(output, tuple):
                            return (hidden,) + output[1:]
                        return hidden
                    return hook_fn

                target_pos = prompt_ids.shape[1] - 1
                handle = layer.register_forward_hook(create_hook(deltas[i], target_pos))

                try:
                    outputs = model(prompt_ids)
                    logits = outputs.logits[0, -1, :]
                    
                    # 目标损失
                    target_token_id = target_ids[0, 0].item()
                    target_loss = -F.log_softmax(logits, dim=-1)[target_token_id]
                    total_loss = total_loss + target_loss
                finally:
                    handle.remove()

            # 反向传播
            total_loss = total_loss / num_requests
            total_loss.backward()

            # 梯度裁剪
            if self.clamp_norm_factor > 0 and deltas.grad is not None:
                grad_norm = deltas.grad.norm()
                max_norm = self.clamp_norm_factor * deltas.norm().clamp(min=0.1)
                if grad_norm > max_norm:
                    deltas.grad.data = deltas.grad.data / grad_norm * max_norm

            optimizer.step()

        # 计算最终的 value: value = current_output + delta
        with torch.no_grad():
            values = current_outputs.T + deltas  # [num_requests, hidden_size]

        return values.detach()

    def _get_mlp_projection(self, layer: nn.Module):
        """获取 MLP 投影层"""
        for proj_name in ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"]:
            try:
                return self._get_module_by_name(layer, proj_name)
            except AttributeError:
                continue
        return None

    def _apply_least_squares_update(
        self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ):
        """使用最小二乘法应用更新

        解决：min ||W'K - V||^2
        解：W' = W + (V - WK) K^T (K K^T)^{-1}

        Args:
            layer_idx: 层索引
            keys: key 矩阵 [num_requests, hidden_size]
            values: value 矩阵 [num_requests, hidden_size]
        """
        model = self.model
        layer = self._get_layer_module(model, layer_idx)

        # 获取 MLP 投影层
        for proj_name in ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"]:
            try:
                proj = self._get_module_by_name(layer, proj_name)
                break
            except AttributeError:
                continue
        else:
            logger.warning(f"Cannot find MLP projection in layer {layer_idx}")
            return

        weight = proj.weight

        with torch.no_grad():
            # K: [num_requests, hidden_size] -> [hidden_size, num_requests]
            K = keys.T
            V = values.T  # [hidden_size, num_requests]

            # 计算 K K^T 的伪逆
            KKT = K @ K.T
            KKT_inv = torch.linalg.pinv(
                KKT + 1e-5 * torch.eye(KKT.shape[0], device=KKT.device)
            )

            # 计算更新：delta = (V - W @ K) @ K^T @ (K K^T)^{-1}
            residual = V - weight @ K
            delta = residual @ K.T @ KKT_inv

            # 应用带权重的更新
            weight.add_(delta * self.edit_weight)

        logger.debug(f"Least squares update applied to layer {layer_idx}")
