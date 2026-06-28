from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

class QuotaExceededError(Exception):
    pass

def is_quota_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in ("quota", "rate limit", "rate_limit", "resource_exhausted", "credit balance", "insufficient_quota"))

class ToolCall(TypedDict):
    id: str
    name: str
    arguments: dict

class ToolTurn(TypedDict):
    content: Optional[str]
    tool_calls: list[ToolCall]

class LLMProvider(ABC):
    @abstractmethod
    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[ToolTurn]:
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...
