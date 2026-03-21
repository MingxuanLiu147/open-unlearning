"""
Agent LLM 抽象层 - 适配自 webui/utils/agent_provider.py，支持 OpenAI / DeepSeek 流式输出。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional


SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".agent_settings.json"

DEFAULT_SYSTEM_PROMPT = """你是 Know-Surgery 智能助手，专门帮助用户配置大模型知识更新实验。

你可以帮助用户：
1. 理解 Unlearn（知识遗忘）、Inject（知识注入）、Edit（知识编辑）三种操作的区别
2. 根据用户的需求推荐合适的算法和配置
3. 解释各个参数的含义和建议值
4. 推荐合适的 Skill 模板

当你推荐配置时，请以 JSON 格式输出，包含以下字段：
```json
{
  "action": "apply_config",
  "mode": "unlearn/inject/edit",
  "model": "模型名称",
  "trainer": "算法名称",
  "datasets": {"forget": "...", "retain": "..."},
  "eval": "评测套件",
  "params": {}
}
```

请根据用户的语言偏好回复。"""


@dataclass
class AgentConfig:
    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


def load_settings() -> AgentConfig:
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text("utf-8"))
            return AgentConfig(
                **{k: v for k, v in data.items() if hasattr(AgentConfig, k)}
            )
        except Exception:
            pass
    return AgentConfig(
        provider="openai",
        base_url=os.getenv("OPENAI_API_BASE", "https://api.gpt.ge/v1/"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-5.2-high"),
    )


def save_settings(cfg: AgentConfig):
    SETTINGS_PATH.write_text(
        json.dumps(cfg.__dict__, ensure_ascii=False, indent=2), "utf-8"
    )


def _get_client(cfg: AgentConfig):
    from openai import OpenAI

    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=60)


def test_connection(cfg: AgentConfig) -> Dict:
    try:
        client = _get_client(cfg)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return {"ok": True, "model": cfg.model}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stream_chat(
    messages: List[Dict[str, str]],
    cfg: AgentConfig = None,
    system_prompt: str = None,
) -> Generator[str, None, None]:
    if cfg is None:
        cfg = load_settings()
    client = _get_client(cfg)

    full_messages = []
    full_messages.append(
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
    )
    full_messages.extend(messages)

    try:
        stream = client.chat.completions.create(
            model=cfg.model,
            messages=full_messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"\n[ERROR] {e}"
