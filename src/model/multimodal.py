"""多模态模型加载模块（sidecar，不修改现有 model/__init__.py）。

提供 get_mm_model()，返回 (model, processor)。
支持 Qwen2VL / Qwen2.5VL / Qwen3VL 系列，不依赖 qwen_vl_utils。
"""

import logging
from typing import List, Set

import torch
from omegaconf import DictConfig, open_dict
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def find_all_linear_names(model) -> List[str]:
    """自动发现所有适合 LoRA 的 Linear 模块名，排除 embedding / lm_head / patch_embed。"""
    cls = torch.nn.Linear
    lora_module_names: Set[str] = set()
    exclude_keywords = {"embeddings", "embed_tokens", "patch_embed"}
    for name, module in model.named_modules():
        if any(kw in name for kw in exclude_keywords):
            continue
        if isinstance(module, cls):
            parts = name.split(".")
            lora_module_names.add(
                parts[0] if len(parts) == 1 else ".".join(parts[-2:])
            )
    lora_module_names.discard("lm_head")
    return list(lora_module_names)


def _get_torch_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)


def get_mm_model(model_cfg: DictConfig):
    """加载多模态模型和 processor。

    Args:
        model_cfg: Hydra 模型配置，包含 model_path / processor_path / lora / dtype 等字段。

    Returns:
        (model, processor) 二元组。
    """
    model_path = model_cfg.model_path
    processor_path = model_cfg.get("processor_path", model_path)
    torch_dtype = _get_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))
    attn_impl = model_cfg.get("attn_implementation", None)

    model_kwargs = dict(
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForImageTextToText.from_pretrained(
        model_path, **model_kwargs
    )
    model_family = model_cfg.get("model_family", type(model).__name__)

    logger.info("Loaded %s from %s (dtype=%s)", model_family, model_path, torch_dtype)

    lora_cfg = model_cfg.get("lora", None)
    if lora_cfg and lora_cfg.get("enabled", False):
        target_modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(processor_path)
    processor.tokenizer.padding_side = "right"

    return model, processor
