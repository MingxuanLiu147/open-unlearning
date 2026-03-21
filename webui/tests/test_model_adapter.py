# -*- coding: utf-8 -*-
"""model_adapter 单测"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_adapter import (
    get_model_meta,
    supports_task,
    get_defaults_for_model,
    validate_model_name,
    get_registered_models,
    MODEL_REGISTRY,
)


def test_get_registered_models():
    models = get_registered_models()
    assert len(models) >= 3
    assert "Qwen2.5-7B-Instruct" in models


def test_get_model_meta_known():
    meta = get_model_meta("Qwen2.5-7B-Instruct")
    assert meta.family == "qwen2.5"
    assert meta.has_chat_template is True


def test_get_model_meta_with_org():
    meta = get_model_meta("Qwen/Qwen2.5-7B-Instruct")
    assert meta.family == "qwen2.5"


def test_get_model_meta_unknown():
    meta = get_model_meta("some-random-model-xyz")
    assert meta.family == "custom"
    assert meta.notes  # fallback should have notes


def test_supports_task():
    assert supports_task("Qwen2.5-7B-Instruct", "unlearn") is True
    assert supports_task("Qwen2.5-7B-Instruct", "eval") is True


def test_get_defaults_for_model():
    defaults = get_defaults_for_model("Llama-3.2-1B-Instruct")
    assert defaults["batch_size"] == 8
    assert defaults["learning_rate"] == "5e-5"


def test_validate_model_name_empty():
    ok, msg = validate_model_name("")
    assert ok is False


def test_validate_model_name_hf():
    ok, msg = validate_model_name("meta-llama/Llama-3.1-8B")
    assert ok is True
    assert "HuggingFace" in msg


def test_validate_model_name_preset():
    ok, msg = validate_model_name("Qwen2.5-7B-Instruct")
    assert ok is True
    assert "预置" in msg


def test_validate_model_name_local_nonexist():
    ok, msg = validate_model_name("/nonexistent/path/model")
    assert ok is False
    assert "不存在" in msg


if __name__ == "__main__":
    test_get_registered_models()
    test_get_model_meta_known()
    test_get_model_meta_with_org()
    test_get_model_meta_unknown()
    test_supports_task()
    test_get_defaults_for_model()
    test_validate_model_name_empty()
    test_validate_model_name_hf()
    test_validate_model_name_preset()
    test_validate_model_name_local_nonexist()
    print("All model_adapter tests passed!")
