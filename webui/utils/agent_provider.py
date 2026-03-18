# -*- coding: utf-8 -*-
"""
Agent Provider 抽象层
======================

支持多种 LLM API 作为智能助手后端。
第一阶段支持 OpenAI GPT 和 DeepSeek。
支持多轮对话。
"""

import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Generator
from pathlib import Path


@dataclass
class Message:
    """对话消息"""
    role: str           # system / user / assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class AgentConfig:
    """Agent 配置"""
    provider: str = "openai"           # openai / deepseek
    base_url: str = ""                 # API 端点
    api_key: str = ""                  # API 密钥
    model: str = ""                    # 模型名称
    timeout: int = 60                  # 超时时间（秒）
    temperature: float = 0.7           # 温度
    max_tokens: int = 2048             # 最大 token 数
    
    @classmethod
    def from_env(cls, provider: str = "openai") -> "AgentConfig":
        """从环境变量创建配置"""
        if provider == "openai":
            return cls(
                provider="openai",
                base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
            )
        elif provider == "deepseek":
            return cls(
                provider="deepseek",
                base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                timeout=int(os.getenv("DEEPSEEK_TIMEOUT", "60")),
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")


@dataclass
class AgentResponse:
    """Agent 响应"""
    content: str                       # 响应内容
    finish_reason: str = "stop"        # 结束原因
    usage: Dict[str, int] = field(default_factory=dict)  # token 使用量
    error: Optional[str] = None        # 错误信息


class BaseProvider(ABC):
    """Provider 基类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._client = None
    
    @abstractmethod
    def chat(self, messages: List[Message]) -> AgentResponse:
        """单轮对话"""
        pass
    
    @abstractmethod
    def stream_chat(self, messages: List[Message]) -> Generator[str, None, None]:
        """流式对话"""
        pass
    
    def validate_config(self) -> List[str]:
        """校验配置，返回错误列表"""
        errors = []
        if not self.config.api_key:
            errors.append("API key is required")
        if not self.config.model:
            errors.append("Model name is required")
        return errors


class OpenAIProvider(BaseProvider):
    """OpenAI / OpenAI-compatible API Provider"""
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout,
                )
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
        return self._client
    
    def chat(self, messages: List[Message]) -> AgentResponse:
        """单轮对话"""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[m.to_dict() for m in messages],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            choice = response.choices[0]
            return AgentResponse(
                content=choice.message.content or "",
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
        except Exception as e:
            return AgentResponse(content="", error=str(e))
    
    def stream_chat(self, messages: List[Message]) -> Generator[str, None, None]:
        """流式对话"""
        try:
            client = self._get_client()
            stream = client.chat.completions.create(
                model=self.config.model,
                messages=[m.to_dict() for m in messages],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[ERROR] {str(e)}"


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek API Provider（兼容 OpenAI 格式）"""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # DeepSeek 使用 OpenAI 兼容格式
        if not self.config.base_url:
            self.config.base_url = "https://api.deepseek.com/v1"


class AgentSession:
    """Agent 会话管理（支持多轮对话）"""
    
    def __init__(self, provider: BaseProvider, system_prompt: str = None):
        self.provider = provider
        self.messages: List[Message] = []
        
        # 设置系统提示词
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        else:
            self.messages.append(Message(
                role="system",
                content=self._default_system_prompt()
            ))
    
    def _default_system_prompt(self) -> str:
        """默认系统提示词"""
        return """你是 Know-Surgery 智能助手，专门帮助用户配置大模型知识更新实验。

你可以帮助用户：
1. 理解 Unlearn（知识遗忘）、Inject（知识注入）、Edit（知识编辑）三种操作的区别
2. 根据用户的需求推荐合适的算法和配置
3. 解释各个参数的含义和建议值
4. 帮助用户准备数据集和配置文件

当你推荐配置时，请以 JSON 格式输出，包含以下字段：
```json
{
  "mode": "unlearn/inject/edit",
  "recommended_skill": "skill 模板名称",
  "recommended_method": "算法名称",
  "recommended_model": "模型名称",
  "data_plan": "数据准备建议",
  "eval_plan": "评测建议",
  "core_overrides": {"参数名": "值"},
  "reasoning_summary": "推荐理由",
  "risk_notes": "风险提示"
}
```

请用中文回复用户的问题。"""
    
    def chat(self, user_message: str) -> AgentResponse:
        """发送消息并获取回复
        
        Args:
            user_message: 用户消息
            
        Returns:
            Agent 响应
        """
        # 添加用户消息
        self.messages.append(Message(role="user", content=user_message))
        
        # 获取回复
        response = self.provider.chat(self.messages)
        
        # 添加助手回复到历史
        if not response.error:
            self.messages.append(Message(role="assistant", content=response.content))
        
        return response
    
    def stream_chat(self, user_message: str) -> Generator[str, None, None]:
        """流式发送消息
        
        Args:
            user_message: 用户消息
            
        Yields:
            响应文本片段
        """
        # 添加用户消息
        self.messages.append(Message(role="user", content=user_message))
        
        # 收集完整回复
        full_response = ""
        
        for chunk in self.provider.stream_chat(self.messages):
            full_response += chunk
            yield chunk
        
        # 添加助手回复到历史
        self.messages.append(Message(role="assistant", content=full_response))
    
    def clear_history(self):
        """清空对话历史（保留系统提示词）"""
        system_msg = self.messages[0] if self.messages else None
        self.messages.clear()
        if system_msg and system_msg.role == "system":
            self.messages.append(system_msg)
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return [m.to_dict() for m in self.messages]


def create_provider(config: AgentConfig) -> BaseProvider:
    """创建 Provider 实例
    
    Args:
        config: Agent 配置
        
    Returns:
        Provider 实例
    """
    providers = {
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
    }
    
    provider_cls = providers.get(config.provider)
    if not provider_cls:
        raise ValueError(f"Unknown provider: {config.provider}, supported: {list(providers.keys())}")
    
    return provider_cls(config)


def create_session(
    provider_name: str = "openai",
    system_prompt: str = None,
    **config_kwargs
) -> AgentSession:
    """创建 Agent 会话的便捷函数
    
    Args:
        provider_name: Provider 名称
        system_prompt: 系统提示词
        **config_kwargs: 配置参数
        
    Returns:
        AgentSession 实例
    """
    # 从环境变量获取基础配置
    config = AgentConfig.from_env(provider_name)
    
    # 覆盖配置参数
    for key, value in config_kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    provider = create_provider(config)
    return AgentSession(provider, system_prompt)
