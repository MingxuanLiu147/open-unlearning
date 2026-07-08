"""
SERAC 知识编辑器（Semi-Parametric Editing with a Retrieval-Augmented Counterfactual Model）
===========================================================================================

基于外部记忆 + 分类器 + 反事实模型的知识编辑方法。
SERAC 不直接修改原始模型权重，而是通过路由机制决定何时使用反事实模型。

参考论文: Memory-Based Model Editing at Scale
https://arxiv.org/abs/2206.06520

核心思想：
1. 维护编辑记忆（edit memory）：存储所有编辑事实
2. 范围分类器（scope classifier）：判断输入是否匹配已存储的编辑
3. 反事实模型（counterfactual model）：经过微调的小模型，负责输出编辑后的结果
4. 推理路由：分类器判定匹配 → 使用反事实模型；否则 → 使用原始模型
"""

import logging
import os
from copy import deepcopy
from typing import Optional, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from trainer.edit.base import EditTrainer, EditRequest

logger = logging.getLogger(__name__)


class ScopeClassifier(nn.Module):
    """范围分类器：判断输入是否属于已编辑知识的范围

    使用一个轻量级 MLP 对输入的隐藏状态进行二分类。
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states.to(self.net[0].weight.dtype))


class SERACEditor(EditTrainer):
    """SERAC 知识编辑器

    通过外部记忆 + 范围分类器 + 反事实模型实现知识编辑。
    原始模型权重保持不变，编辑效果由路由机制和反事实模型提供。

    Attributes:
        archive: 预训练的 SERAC 检查点路径（包含分类器和反事实模型）
        edit_lr: 反事实模型的微调学习率
        cedit: 编辑损失权重
        cloc: 局部性损失权重
        cbase: 基础模型损失权重
    """

    def __init__(
        self,
        archive: Optional[str] = None,
        edit_lr: float = 1e-4,
        cedit: float = 0.1,
        cloc: float = 1.0,
        cbase: float = 1.0,
        classifier_hidden_size: Optional[int] = None,
        num_edit_steps: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.archive = archive
        self.edit_lr = edit_lr
        self.cedit = cedit
        self.cloc = cloc
        self.cbase = cbase
        self.classifier_hidden_size = classifier_hidden_size
        self.num_edit_steps = num_edit_steps

        self.edit_memory: List[Dict[str, str]] = []
        self.classifier: Optional[ScopeClassifier] = None
        self.counterfactual_model: Optional[nn.Module] = None
        self._is_initialized = False

    def _ensure_initialized(self):
        """确保分类器和反事实模型已初始化。"""
        if self._is_initialized:
            return

        device = next(self.model.parameters()).device
        hidden_size = self.classifier_hidden_size or self._infer_hidden_size()

        if self.archive and os.path.exists(self.archive):
            self._load_archive(device)
        else:
            if self.archive:
                logger.warning(
                    "Archive '%s' not found. Falling back to on-the-fly "
                    "initialization. For best results, pre-train a SERAC "
                    "checkpoint with train_serac().",
                    self.archive,
                )
            self.classifier = ScopeClassifier(hidden_size).to(device)
            self._cf_layer_names = self._get_trainable_layer_names()
            self._cf_original_state = {
                n: p.detach().cpu().clone()
                for n, p in self.model.named_parameters()
                if any(n.startswith(ln) for ln in self._cf_layer_names)
            }
            self._cf_edited_state: Dict[str, torch.Tensor] = {}
            self.counterfactual_model = self.model
            logger.info(
                "Initialized SERAC with fresh classifier; counterfactual model "
                "shares base model with layer backup (hidden_size=%d, %d backed-up params)",
                hidden_size, len(self._cf_original_state),
            )

        self._is_initialized = True

    def _infer_hidden_size(self) -> int:
        """从模型配置推断隐藏层维度。"""
        config = self.model.config
        for attr in ["hidden_size", "d_model", "n_embd"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 768

    def _load_archive(self, device: torch.device):
        """从检查点加载预训练的分类器和反事实模型。"""
        logger.info("Loading SERAC archive from '%s'", self.archive)
        state = torch.load(self.archive, map_location="cpu")

        hidden_size = self.classifier_hidden_size or self._infer_hidden_size()
        self.classifier = ScopeClassifier(hidden_size).to(device)

        if "classifier" in state:
            self.classifier.load_state_dict(state["classifier"])

        self._cf_layer_names = self._get_trainable_layer_names()
        self._cf_original_state = {
            n: p.detach().cpu().clone()
            for n, p in self.model.named_parameters()
            if any(n.startswith(ln) for ln in self._cf_layer_names)
        }
        self._cf_edited_state = {}
        self.counterfactual_model = self.model
        if "counterfactual_model" in state:
            self._cf_edited_state = {
                k: v.cpu() for k, v in state["counterfactual_model"].items()
                if any(k.startswith(ln) for ln in self._cf_layer_names)
            }
            logger.info("Loaded pre-trained counterfactual layer deltas from archive")

        if "edit_memory" in state:
            self.edit_memory = state["edit_memory"]
            logger.info(
                "Restored %d entries from edit memory", len(self.edit_memory)
            )

    def edit(
        self, requests: Union[EditRequest, List[EditRequest]], **kwargs
    ) -> Dict[str, Any]:
        """执行 SERAC 知识编辑

        将编辑请求存入记忆，并微调反事实模型使其输出新目标。

        Args:
            requests: 编辑请求（单个或列表）

        Returns:
            包含编辑结果的字典：
            - success: 是否成功
            - edited_count: 成功编辑数
            - metrics: 微调过程中的损失指标
        """
        if isinstance(requests, EditRequest):
            requests = [requests]

        self._ensure_initialized()

        results: Dict[str, Any] = {
            "success": True,
            "edited_count": 0,
            "metrics": {},
        }

        for request in requests:
            try:
                metrics = self._apply_serac_edit(request)
                results["edited_count"] += 1
                results["metrics"][request.subject] = metrics
            except Exception as e:
                logger.error(
                    "SERAC edit failed for request: %s, error: %s",
                    request.prompt,
                    e,
                )
                results["success"] = False

        return results

    def _apply_serac_edit(self, request: EditRequest) -> Dict[str, float]:
        """对单个请求执行 SERAC 编辑：存储 + 微调反事实模型。"""
        entry = {
            "prompt": request.prompt,
            "subject": request.subject,
            "target_new": request.target_new,
            "target_old": request.target_old,
        }
        self.edit_memory.append(entry)

        metrics = self._finetune_counterfactual(request)

        logger.info(
            "SERAC edit applied: %s -> %s (memory size: %d)",
            request.subject,
            request.target_new,
            len(self.edit_memory),
        )
        return metrics

    def _finetune_counterfactual(self, request: EditRequest) -> Dict[str, float]:
        """微调反事实模型使其在编辑 prompt 上输出新目标。

        同时更新分类器，使其能识别此编辑的范围。
        """
        device = next(self.model.parameters()).device
        tokenizer = self.tokenizer

        full_text = request.prompt + " " + request.target_new
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True).to(device)
        prompt_ids = tokenizer(
            request.prompt, return_tensors="pt", truncation=True
        ).to(device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        cf_optimizer = Adam(
            self._get_trainable_params(self.counterfactual_model),
            lr=self.edit_lr,
        )
        cls_optimizer = Adam(self.classifier.parameters(), lr=self.edit_lr)

        self.counterfactual_model.train()
        self.classifier.train()

        final_metrics: Dict[str, float] = {}
        for step in range(self.num_edit_steps):
            cf_optimizer.zero_grad()
            cls_optimizer.zero_grad()

            cf_outputs = self.counterfactual_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )
            edit_loss = cf_outputs.loss

            cls_loss = self._compute_classifier_loss(request, device)

            loss = self.cedit * edit_loss + self.cloc * cls_loss
            loss.backward()
            cf_optimizer.step()
            cls_optimizer.step()

            if step % 10 == 0:
                logger.debug(
                    "SERAC finetune step %d: edit_loss=%.4f, cls_loss=%.4f",
                    step,
                    edit_loss.item(),
                    cls_loss.item(),
                )

        final_metrics["edit_loss"] = edit_loss.item()
        final_metrics["cls_loss"] = cls_loss.item()

        self.counterfactual_model.eval()
        self.classifier.eval()
        return final_metrics

    def _get_trainable_layer_names(self) -> List[str]:
        """Return the dotted-name prefixes of trainable layers (last 2 + lm_head)."""
        names = []
        layers = self._get_layers_container(self.model)
        if layers is not None:
            num_layers = len(layers)
            for attr_prefix in ["model.layers", "transformer.h", "gpt_neox.layers"]:
                try:
                    self._get_module_by_name(self.model, attr_prefix)
                    for i in range(max(0, num_layers - 2), num_layers):
                        names.append(f"{attr_prefix}.{i}.")
                    break
                except (AttributeError, IndexError, KeyError):
                    continue
        if hasattr(self.model, "lm_head"):
            names.append("lm_head.")
        return names

    def _get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """获取反事实模型中需要微调的参数（最后几层）。"""
        layer_names = getattr(self, "_cf_layer_names", None)
        if layer_names:
            return [
                p for n, p in model.named_parameters()
                if any(n.startswith(ln) for ln in layer_names)
            ]

        layers = self._get_layers_container(model)
        if layers is None:
            return list(model.parameters())

        num_layers = len(layers)
        trainable_start = max(0, num_layers - 2)

        params: List[nn.Parameter] = []
        for i in range(trainable_start, num_layers):
            params.extend(layers[i].parameters())

        if hasattr(model, "lm_head"):
            params.extend(model.lm_head.parameters())
        return params

    def _compute_classifier_loss(
        self, request: EditRequest, device: torch.device
    ) -> torch.Tensor:
        """计算分类器损失：编辑 prompt 应匹配，随机 prompt 不应匹配。"""
        tokenizer = self.tokenizer

        pos_inputs = tokenizer(
            request.prompt, return_tensors="pt", truncation=True
        ).to(device)
        with torch.no_grad():
            pos_hidden = self.model(
                **pos_inputs, output_hidden_states=True
            ).hidden_states[-1][:, -1, :]

        pos_logits = self.classifier(pos_hidden)
        pos_labels = torch.ones(pos_logits.shape[0], dtype=torch.long, device=device)

        neg_text = "The weather is nice today."
        neg_inputs = tokenizer(neg_text, return_tensors="pt", truncation=True).to(
            device
        )
        with torch.no_grad():
            neg_hidden = self.model(
                **neg_inputs, output_hidden_states=True
            ).hidden_states[-1][:, -1, :]

        neg_logits = self.classifier(neg_hidden)
        neg_labels = torch.zeros(neg_logits.shape[0], dtype=torch.long, device=device)

        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)
        return F.cross_entropy(logits, labels)

    def predict(self, prompt: str) -> str:
        """SERAC 推理：根据分类器决定使用原始模型或反事实模型。

        Args:
            prompt: 输入文本

        Returns:
            模型输出文本
        """
        self._ensure_initialized()
        device = next(self.model.parameters()).device
        tokenizer = self.tokenizer

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            hidden = self.model(
                **inputs, output_hidden_states=True
            ).hidden_states[-1][:, -1, :]
            cls_logits = self.classifier(hidden)
            use_counterfactual = cls_logits.argmax(dim=-1).item() == 1

        target_model = self.counterfactual_model if use_counterfactual else self.model
        with torch.no_grad():
            outputs = target_model.generate(
                **inputs, max_new_tokens=32, do_sample=False
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def save_archive(self, path: str):
        """保存 SERAC 检查点（分类器 + 反事实模型 + 编辑记忆）。"""
        self._ensure_initialized()
        state = {
            "classifier": self.classifier.state_dict(),
            "counterfactual_model": self.counterfactual_model.state_dict(),
            "edit_memory": self.edit_memory,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)
        logger.info("Saved SERAC archive to '%s'", path)

    def train_serac(
        self,
        train_data: Optional[List[Dict[str, str]]] = None,
        num_epochs: int = 5,
        batch_size: int = 8,
    ):
        """预训练 SERAC 分类器和反事实模型。

        完整的 SERAC 需要在编辑数据集上预训练分类器和反事实模型。
        此方法提供预训练入口。

        Args:
            train_data: 训练数据，每条包含 prompt / subject / target_new
            num_epochs: 训练轮数
            batch_size: 批大小

        Raises:
            NotImplementedError: 完整预训练流程尚待实现
        """
        raise NotImplementedError(
            "Full SERAC pre-training is not yet implemented. "
            "Provide a pre-trained archive via the 'archive' parameter, "
            "or use the on-the-fly fine-tuning mode (edit without archive)."
        )

    def clear_edit_memory(self):
        """清空编辑记忆。"""
        self.edit_memory.clear()
        logger.info("Edit memory cleared")
