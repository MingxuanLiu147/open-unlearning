# -*- coding: utf-8 -*-
"""
Agent 配置持久化
================

将用户在界面设置的 Agent provider 配置保存到本地 JSON 文件，
下次启动时自动恢复，避免重复填写 API Key。

文件位置：webui/.agent_settings.json（已在 .gitignore 中忽略，避免泄漏密钥）
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

SETTINGS_FILE = Path(__file__).parent.parent / ".agent_settings.json"

# 每个 provider 的默认值
PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 60,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 2048,
        "timeout": 60,
    },
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "provider": "openai",
    "api_key": "",
    **PROVIDER_DEFAULTS["openai"],
}


def load_settings() -> Dict[str, Any]:
    """从文件加载 Agent 配置，文件不存在时返回默认值。"""
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()
    try:
        raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        # 以默认值为底，覆盖已保存的字段
        settings = DEFAULT_SETTINGS.copy()
        settings.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
        return settings
    except Exception:
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> Optional[str]:
    """保存配置到文件。返回 None 表示成功，否则返回错误信息。"""
    try:
        # 只保存白名单字段，防止写入多余数据
        allowed = set(DEFAULT_SETTINGS.keys())
        filtered = {k: v for k, v in settings.items() if k in allowed}
        SETTINGS_FILE.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return None
    except Exception as e:
        return str(e)


def get_provider_default(provider: str) -> Dict[str, Any]:
    """返回指定 provider 的默认 base_url 和 model。"""
    return PROVIDER_DEFAULTS.get(provider, PROVIDER_DEFAULTS["openai"]).copy()


def test_connection(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 10,
) -> tuple[bool, str]:
    """发送一条最小 ping 请求，验证 API Key 和端点是否可达。

    Returns:
        (success, message)
    """
    if not api_key or not api_key.strip():
        return False, "API Key 不能为空"
    if not model or not model.strip():
        return False, "模型名称不能为空"

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key.strip(),
            base_url=base_url.strip() or None,
            timeout=timeout,
        )
        # 发送最短 prompt 验证连通性
        resp = client.chat.completions.create(
            model=model.strip(),
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )
        model_id = resp.model or model
        return True, f"连接成功！模型：{model_id}"
    except ImportError:
        return False, "请先安装 openai 库：pip install openai"
    except Exception as e:
        err = str(e)
        # 简化常见错误信息
        if "401" in err or "Unauthorized" in err or "invalid api key" in err.lower():
            return False, "API Key 无效或已过期"
        if "404" in err or "not found" in err.lower():
            return False, f"模型 `{model}` 不存在，请检查模型名称"
        if "Connection" in err or "connect" in err.lower():
            return False, f"无法连接到 {base_url}，请检查网络或 Base URL"
        return False, f"连接失败：{err[:120]}"
