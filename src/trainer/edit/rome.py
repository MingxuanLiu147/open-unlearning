"""
ROME 知识编辑器
==============

实现 Rank-One Model Editing (ROME) 方法。
ROME 通过对 MLP 层进行秩一更新来实现精确的知识编辑。

参考论文: Locating and Editing Factual Associations in GPT
https://arxiv.org/abs/2202.05262

核心思想：
1. 定位知识存储的位置（通常在中间层的 MLP）
2. 计算使模型输出目标值的秩一更新
3. 直接修改 MLP 权重实现编辑
"""

import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)


class ROMEEditor(EditTrainer):
    """ROME 知识编辑器

    通过秩一更新实现精确的知识编辑。

    ROME 算法步骤：
    1. 选择编辑层（通常是中间层的 MLP down_proj）
    2. 收集 subject 的隐藏状态作为 key
    3. 计算使模型输出 target_new 的 value
    4. 应用秩一更新：W' = W + (v - Wk) k^T / ||k||^2

    Attributes:
        v_lr: value 向量优化的学习率
        v_num_grad_steps: value 优化的梯度步数
        clamp_norm_factor: 梯度裁剪因子
        kl_factor: KL 散度正则化因子
    """

    def __init__(
        self,
        layers: Optional[List[int]] = None,
        v_lr: float = 0.5,
        v_num_grad_steps: int = 20,
        clamp_norm_factor: float = 4.0,
        kl_factor: float = 0.0625,
        mom2_update_weight: float = 4000.0,
        *args,
        **kwargs,
    ):
        """初始化 ROME 编辑器

        Args:
            layers: 编辑的目标层
            v_lr: value 优化学习率
            v_num_grad_steps: value 优化步数
            clamp_norm_factor: 梯度裁剪因子
            kl_factor: KL 正则化权重
            mom2_update_weight: 二阶矩估计更新权重
        """
        super().__init__(layers=layers, *args, **kwargs)

        self.v_lr = v_lr
        self.v_num_grad_steps = v_num_grad_steps
        self.clamp_norm_factor = clamp_norm_factor
        self.kl_factor = kl_factor
        self.mom2_update_weight = mom2_update_weight

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 ROME 知识编辑

        Args:
            requests: 编辑请求（单个或列表）

        Returns:
            包含编辑结果的字典：
            - success: 编辑是否成功
            - metrics: 编辑相关指标
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        results = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
        }

        for request in requests:
            try:
                self._apply_rome_edit(request)
                results["edited_count"] += 1
            except Exception as e:
                logger.error(
                    f"ROME edit failed for request: {request.prompt}, error: {e}"
                )
                results["success"] = False

        return results

    def _apply_rome_edit(self, request: EditRequest):
        """应用单个 ROME 编辑

        Args:
            request: 编辑请求
        """
        model = self.model
        tokenizer = self.tokenizer

        # 编辑第一个指定的层
        layer_idx = self.layers[0] if self.layers else 5

        # 获取模型架构信息
        layer_module = self._get_layer_module(model, layer_idx)
        if layer_module is None:
            raise ValueError(f"Cannot find layer {layer_idx} in model")

        # 1. 获取 subject 的 key 向量
        key = self._compute_key_vector(request.prompt, request.subject, layer_idx)

        # 2. 计算目标 value 向量
        value = self._compute_value_vector(request, layer_idx, key)

        # 3. 应用秩一更新
        self._apply_rank_one_update(layer_idx, key, value)

        logger.info(
            f"ROME edit applied at layer {layer_idx}: {request.subject} -> {request.target_new}"
        )

    def _get_layer_module(
        self, model: nn.Module, layer_idx: int
    ) -> Optional[nn.Module]:
        """获取指定层的模块"""
        # 尝试不同的模型架构
        for attr_name in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            try:
                layers = self._get_module_by_name(model, attr_name)
                return layers[layer_idx]
            except (AttributeError, IndexError, KeyError):
                continue
        return None

    def _compute_key_vector(
        self, prompt: str, subject: str, layer_idx: int
    ) -> torch.Tensor:
        """计算 subject 位置的 key 向量

        Args:
            prompt: 完整提示
            subject: 主体实体
            layer_idx: 层索引

        Returns:
            key 向量
        """
        tokenizer = self.tokenizer
        model = self.model

        # Tokenize 并定位 subject
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        subject_tokens = tokenizer(subject, add_special_tokens=False)["input_ids"]

        # 找到 subject 在 prompt 中的位置
        input_ids = inputs["input_ids"][0].tolist()
        subject_start = None
        for i in range(len(input_ids) - len(subject_tokens) + 1):
            if input_ids[i : i + len(subject_tokens)] == subject_tokens:
                subject_start = i
                break

        if subject_start is None:
            # 如果找不到精确匹配，使用最后一个 token
            subject_end = len(input_ids) - 1
        else:
            subject_end = subject_start + len(subject_tokens) - 1

        # 获取隐藏状态
        hidden_states = []

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states.append(output[0])
            else:
                hidden_states.append(output)

        # 注册 hook 获取指定层的输出
        layer = self._get_layer_module(model, layer_idx)
        handle = layer.register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**inputs)

        handle.remove()

        # 提取 subject 最后一个 token 位置的隐藏状态
        key = hidden_states[0][0, subject_end, :].clone()

        return key

    def _compute_value_vector(
        self, request: EditRequest, layer_idx: int, key: torch.Tensor
    ) -> torch.Tensor:
        """计算目标 value 向量

        通过优化找到使模型输出 target_new 的 value 向量。
        ROME 的核心：在 MLP 输出位置注入 delta，使模型输出新目标。

        优化目标：
        1. 使编辑后的输出概率最大化目标 token
        2. 保持与原始输出的 KL 散度最小（局部性）

        Args:
            request: 编辑请求
            layer_idx: 层索引
            key: key 向量

        Returns:
            优化后的 value 向量
        """
        model = self.model
        tokenizer = self.tokenizer
        device = next(model.parameters()).device

        # 准备输入和目标
        prompt = request.prompt
        target_new = request.target_new

        # Tokenize 目标文本
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        target_ids = tokenizer(
            target_new, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(device)

        # 获取 MLP 输出维度
        layer = self._get_layer_module(model, layer_idx)
        mlp_proj = self._get_mlp_projection(layer)
        if mlp_proj is None:
            raise ValueError(f"Cannot find MLP projection in layer {layer_idx}")

        hidden_size = mlp_proj.weight.shape[0]

        # 初始化 delta（要添加到 MLP 输出的扰动）
        delta = torch.zeros(hidden_size, device=device, requires_grad=True)

        # 获取原始模型的输出分布（用于 KL 约束）
        with torch.no_grad():
            original_outputs = model(prompt_ids)
            original_logits = original_outputs.logits[0, -1, :]
            original_probs = F.softmax(original_logits, dim=-1)

        # 优化 delta
        optimizer = torch.optim.Adam([delta], lr=self.v_lr)

        for step in range(self.v_num_grad_steps):
            optimizer.zero_grad()

            # 使用 hook 注入 delta 到 MLP 输出
            def create_hook(delta_vec, target_pos):
                def hook_fn(module, input, output):
                    # output shape: (batch, seq_len, hidden_size)
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    # 在最后一个位置添加 delta
                    hidden[:, target_pos, :] = hidden[:, target_pos, :] + delta_vec
                    if isinstance(output, tuple):
                        return (hidden,) + output[1:]
                    return hidden
                return hook_fn

            # 目标位置是 prompt 的最后一个 token
            target_pos = prompt_ids.shape[1] - 1
            handle = layer.register_forward_hook(create_hook(delta, target_pos))

            try:
                # 前向传播
                outputs = model(prompt_ids)
                logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits

                # 目标损失：最大化目标 token 的概率
                # 使用第一个目标 token
                target_token_id = target_ids[0, 0].item()
                target_loss = -F.log_softmax(logits, dim=-1)[target_token_id]

                # KL 散度损失：保持与原始分布接近
                new_probs = F.softmax(logits, dim=-1)
                kl_loss = F.kl_div(
                    torch.log(new_probs + 1e-10),
                    original_probs,
                    reduction="sum"
                )

                # 总损失
                loss = target_loss + self.kl_factor * kl_loss

                # 梯度裁剪
                loss.backward()
                if self.clamp_norm_factor > 0:
                    max_norm = self.clamp_norm_factor * delta.norm().item()
                    if delta.grad is not None and delta.grad.norm() > max_norm:
                        delta.grad.data = (
                            delta.grad.data / delta.grad.norm() * max_norm
                        )

                optimizer.step()

            finally:
                handle.remove()

            if step % 5 == 0:
                logger.debug(
                    f"ROME opt step {step}: loss={loss.item():.4f}, "
                    f"target_loss={target_loss.item():.4f}, kl_loss={kl_loss.item():.4f}"
                )

        # 计算最终的 value 向量
        # value = current_output + delta，其中 current_output = W @ key
        with torch.no_grad():
            weight = mlp_proj.weight
            current_output = weight @ key
            value = current_output + delta

        return value.detach()

    def _get_mlp_projection(self, layer: nn.Module) -> Optional[nn.Module]:
        """获取 MLP 的投影层（down_proj）"""
        for proj_name in ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h"]:
            try:
                return self._get_module_by_name(layer, proj_name)
            except AttributeError:
                continue
        return None

    def _apply_rank_one_update(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ):
        """应用秩一更新到模型权重

        W' = W + (v - Wk) k^T / ||k||^2

        ROME 的核心更新公式：通过秩一矩阵更新 MLP 权重，
        使得在 key 向量输入时，输出变为 value 向量。

        Args:
            layer_idx: 层索引
            key: key 向量（subject 位置的隐藏状态）
            value: value 向量（优化得到的目标输出）
        """
        model = self.model
        layer = self._get_layer_module(model, layer_idx)

        # 获取 MLP down_proj 权重
        proj = self._get_mlp_projection(layer)
        if proj is None:
            logger.warning(f"Cannot find MLP projection in layer {layer_idx}")
            return

        weight = proj.weight

        # 计算秩一更新
        # 公式：ΔW = (v - Wk) ⊗ k / ||k||^2
        # 这确保 W'k = Wk + ΔWk = Wk + (v - Wk) = v
        with torch.no_grad():
            key_norm_sq = (key @ key).clamp(min=1e-10)
            current_output = weight @ key
            residual = value - current_output

            # 秩一更新矩阵
            delta = torch.outer(residual, key) / key_norm_sq

            # 可选：使用 mom2 权重进行缩放（提高稳定性）
            if self.mom2_update_weight > 0:
                # 简化版本：直接使用权重缩放
                delta = delta * (self.mom2_update_weight / (self.mom2_update_weight + 1))

            # 应用更新
            weight.add_(delta)

        logger.info(
            f"Rank-one update applied to layer {layer_idx}, "
            f"residual norm: {residual.norm().item():.4f}"
        )
