# -*- coding: utf-8 -*-
"""
模型适配层
==========

维护模型元信息表（tokenizer、dtype、chat template、任务支持、modality 标签），
提供 capability 判断、默认值选择和降级策略。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set


@dataclass
class ModelMeta:
    """模型元信息"""
    name: str
    family: str = ""
    modality: str = "text"  # text / multimodal (预留)
    dtype: str = "bfloat16"
    has_chat_template: bool = True
    supported_tasks: Set[str] = field(default_factory=lambda: {"unlearn", "inject", "edit", "eval"})
    default_max_length: int = 512
    default_batch_size: int = 4
    recommended_lr: str = "1e-5"
    notes: str = ""


# 预置模型元信息表
MODEL_REGISTRY: Dict[str, ModelMeta] = {
    "Qwen2.5-7B-Instruct": ModelMeta(
        name="Qwen2.5-7B-Instruct",
        family="qwen2.5",
        dtype="bfloat16",
        has_chat_template=True,
        default_max_length=512,
        default_batch_size=4,
        recommended_lr="1e-5",
    ),
    "Llama-3.1-8B-Instruct": ModelMeta(
        name="Llama-3.1-8B-Instruct",
        family="llama3",
        dtype="bfloat16",
        has_chat_template=True,
        default_max_length=512,
        default_batch_size=4,
        recommended_lr="2e-5",
    ),
    "Llama-3.2-1B-Instruct": ModelMeta(
        name="Llama-3.2-1B-Instruct",
        family="llama3",
        dtype="bfloat16",
        has_chat_template=True,
        default_max_length=512,
        default_batch_size=8,
        recommended_lr="5e-5",
        notes="Smaller model, faster training",
    ),
}

# 未知模型的降级默认值
_FALLBACK_META = ModelMeta(
    name="unknown",
    family="unknown",
    dtype="float16",
    has_chat_template=False,
    supported_tasks={"unlearn", "inject", "edit", "eval"},
    default_max_length=512,
    default_batch_size=4,
    recommended_lr="1e-5",
    notes="Unknown model; using conservative defaults",
)


def get_model_meta(model_name: str) -> ModelMeta:
    """获取模型元信息，未注册时返回降级默认值。"""
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    if short_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[short_name]
    for key, meta in MODEL_REGISTRY.items():
        if key.lower() in model_name.lower():
            return meta
    fallback = ModelMeta(
        name=model_name,
        family="custom",
        dtype=_FALLBACK_META.dtype,
        has_chat_template=_FALLBACK_META.has_chat_template,
        supported_tasks=_FALLBACK_META.supported_tasks.copy(),
        default_max_length=_FALLBACK_META.default_max_length,
        default_batch_size=_FALLBACK_META.default_batch_size,
        recommended_lr=_FALLBACK_META.recommended_lr,
        notes=_FALLBACK_META.notes,
    )
    return fallback


def supports_task(model_name: str, task: str) -> bool:
    """判断模型是否支持指定任务。"""
    meta = get_model_meta(model_name)
    return task in meta.supported_tasks


def get_defaults_for_model(model_name: str) -> Dict[str, Any]:
    """返回模型推荐的默认参数。"""
    meta = get_model_meta(model_name)
    return {
        "max_length": meta.default_max_length,
        "batch_size": meta.default_batch_size,
        "learning_rate": meta.recommended_lr,
        "dtype": meta.dtype,
    }


def validate_model_name(model_name: str) -> tuple[bool, str]:
    """轻量校验模型名称。

    Returns:
        (is_valid, message)
    """
    if not model_name or not model_name.strip():
        return False, "模型名不能为空"

    name = model_name.strip()

    if name.startswith("/") or name.startswith("./"):
        import os
        if os.path.isdir(name):
            return True, f"本地模型路径有效: {name}"
        return False, f"本地路径不存在: {name}"

    if "/" in name:
        parts = name.split("/")
        if len(parts) == 2 and parts[0] and parts[1]:
            return True, f"HuggingFace 模型: {name}"
        return False, f"格式不合法，期望 org/model: {name}"

    if name in MODEL_REGISTRY:
        return True, f"预置模型: {name}"

    return True, f"将作为 HuggingFace 模型名使用: {name}"


def get_registered_models() -> List[str]:
    """返回所有预注册模型名列表。"""
    return list(MODEL_REGISTRY.keys())
