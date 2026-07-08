"""Minimal BREP integration for the inject pipeline."""

import json
import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn

from data.utils import IGNORE_INDEX
from trainer.inject.base import InjectTrainer

logger = logging.getLogger(__name__)


class ActivationLayer(nn.Module):
    """Wrap a projection layer with trainable activation scaling and bias."""

    def __init__(
        self,
        update_layer: nn.Module,
        hidden_size: int,
        layer_type: str = "all",
        prefix: int = -1,
    ):
        super().__init__()
        self.update_layer = update_layer
        self.layer_type = layer_type
        self.prefix = prefix
        self.modify_count = 0

        if layer_type == "all":
            self.activation_scaling = nn.Parameter(torch.ones(1, hidden_size))
            self.activation_bias = nn.Parameter(torch.zeros(1, hidden_size))
        elif layer_type == "scaling":
            self.activation_scaling = nn.Parameter(torch.ones(1, hidden_size))
            self.activation_bias = None
        elif layer_type == "bias":
            self.activation_scaling = None
            self.activation_bias = nn.Parameter(torch.zeros(1, hidden_size))
        elif layer_type == "ln":
            self.activation_scaling = nn.Parameter(torch.ones(1, hidden_size))
            self.activation_bias = nn.Parameter(torch.zeros(1, hidden_size))
            self.activation_ln = nn.LayerNorm(hidden_size)
        else:
            raise ValueError(f"Unsupported BREP layer_type: {layer_type}")

    def _apply_delta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.activation_scaling is not None:
            hidden_states = hidden_states * self.activation_scaling.to(
                hidden_states.device, dtype=hidden_states.dtype
            )
        if self.activation_bias is not None:
            hidden_states = hidden_states + self.activation_bias.to(
                hidden_states.device, dtype=hidden_states.dtype
            )
        if hasattr(self, "activation_ln"):
            hidden_states = self.activation_ln(hidden_states)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.update_layer(hidden_states)
        if self.prefix == 0:
            return hidden_states
        if hidden_states.ndim < 3:
            return self._apply_delta(hidden_states)
        if hidden_states.shape[1] > 1 or self.prefix == -1:
            self.modify_count = 0
            return self._apply_delta(hidden_states)
        if self.modify_count < self.prefix:
            self.modify_count += 1
            return self._apply_delta(hidden_states)
        return hidden_states


class BREPInterventionModel(nn.Module):
    """Inject trainable activation modifiers into decoder projection layers."""

    TARGET_SUFFIX = {
        "ffn_down": "mlp.down_proj",
        "ffn_up": "mlp.up_proj",
        "attn_q": "self_attn.q_proj",
        "attn_k": "self_attn.k_proj",
        "attn_v": "self_attn.v_proj",
        "attn_o": "self_attn.o_proj",
    }

    def __init__(
        self,
        base_model: nn.Module,
        op_position: str = "ffn_down",
        layer_type: str = "all",
        exclude_layers: Optional[List[int]] = None,
        prefix: int = -1,
    ):
        super().__init__()
        if op_position not in self.TARGET_SUFFIX:
            raise ValueError(f"Unsupported BREP op_position: {op_position}")

        self.base_model = base_model
        self.config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.op_position = op_position
        self.layer_type = layer_type
        self.exclude_layers = set(exclude_layers or [])
        self.prefix = prefix

        self._freeze_base_model()
        self._replace_target_layers()

        if hasattr(self.base_model.config, "use_cache"):
            self.base_model.config.use_cache = False

    def _freeze_base_model(self):
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def _target_keys(self) -> List[str]:
        layers = getattr(getattr(self.base_model, "model", None), "layers", None)
        if layers is None:
            raise ValueError(
                "BREP currently expects a decoder-only model with `model.layers`."
            )

        suffix = self.TARGET_SUFFIX[self.op_position]
        keys = []
        for layer_idx in range(len(layers)):
            if layer_idx in self.exclude_layers:
                continue
            keys.append(f"model.layers.{layer_idx}.{suffix}")
        return keys

    def _replace_target_layers(self):
        for key in self._target_keys():
            parent_key = ".".join(key.split(".")[:-1])
            leaf_name = key.split(".")[-1]
            parent_module = self.base_model.get_submodule(parent_key)
            replaced_module = self.base_model.get_submodule(key)
            hidden_size = getattr(replaced_module, "out_features", None)
            if hidden_size is None:
                raise ValueError(f"BREP target module `{key}` is not a linear layer.")
            setattr(
                parent_module,
                leaf_name,
                ActivationLayer(
                    update_layer=replaced_module,
                    hidden_size=hidden_size,
                    layer_type=self.layer_type,
                    prefix=self.prefix,
                ),
            )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            return self.base_model.gradient_checkpointing_enable(**kwargs)
        return None

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)

    def get_save_dict(self):
        state_dict = self.base_model.state_dict()
        return {
            key: value.detach().cpu()
            for key, value in state_dict.items()
            if "activation_" in key
        }

    def save_model(self, save_path: str):
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(self.get_save_dict(), save_path)

    def print_trainable_parameters(self):
        total_parameters = 0
        trainable_parameters = 0
        for parameter in self.parameters():
            total_parameters += parameter.numel()
            if parameter.requires_grad:
                trainable_parameters += parameter.numel()
        logger.info(
            "BREP trainable parameters: %s / %s (%.4f%%)",
            trainable_parameters,
            total_parameters,
            100 * trainable_parameters / max(total_parameters, 1),
        )


class BREPTrainer(InjectTrainer):
    """Train BREP activation interventions on top of a frozen base model."""

    def __init__(
        self,
        op_position: str = "ffn_down",
        layer_type: str = "all",
        exclude_layers: Optional[List[int]] = None,
        prefix: int = -1,
        use_weighted_loss: bool = True,
        *args,
        **kwargs,
    ):
        self.op_position = op_position
        self.layer_type = layer_type
        self.exclude_layers = list(exclude_layers) if exclude_layers else []
        self.prefix = prefix
        self.use_weighted_loss = use_weighted_loss
        super().__init__(*args, **kwargs)
        self._apply_brep_config()

    def _apply_brep_config(self):
        self.model = BREPInterventionModel(
            base_model=self.model,
            op_position=self.op_position,
            layer_type=self.layer_type,
            exclude_layers=self.exclude_layers,
            prefix=self.prefix,
        )
        self.model.print_trainable_parameters()
        logger.info(
            "BREP config applied: op_position=%s layer_type=%s prefix=%s",
            self.op_position,
            self.layer_type,
            self.prefix,
        )

    def _compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        valid_mask = shift_labels.ne(IGNORE_INDEX)
        sample_losses = (token_losses * valid_mask).sum(dim=-1) / valid_mask.sum(
            dim=-1
        ).clamp(min=1)

        if sample_weight is None or not self.use_weighted_loss:
            return sample_losses.mean()

        sample_weight = sample_weight.to(sample_losses.device)
        return (sample_losses * sample_weight).sum() / sample_weight.sum().clamp(
            min=1e-8
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        loss = self._compute_weighted_loss(
            outputs.logits,
            inputs["labels"],
            inputs.get("sample_weight"),
        )
        return (loss, outputs) if return_outputs else loss

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_model(os.path.join(output_dir, "delta_vector.pth"))

        with open(
            os.path.join(output_dir, "brep_config.json"), "w", encoding="utf-8"
        ) as fout:
            json.dump(
                {
                    "method": "brep",
                    "op_position": self.op_position,
                    "layer_type": self.layer_type,
                    "exclude_layers": self.exclude_layers,
                    "prefix": self.prefix,
                    "use_weighted_loss": self.use_weighted_loss,
                },
                fout,
                ensure_ascii=False,
                indent=2,
            )

        with open(
            os.path.join(output_dir, "inject_artifact.json"), "w", encoding="utf-8"
        ) as fout:
            json.dump(
                {
                    "method": "brep",
                    "path": output_dir,
                },
                fout,
                ensure_ascii=False,
                indent=2,
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
