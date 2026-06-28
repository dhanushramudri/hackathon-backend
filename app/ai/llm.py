import logging
import os
from typing import Optional

from dotenv import load_dotenv

from .providers.base import LLMProvider

load_dotenv()

logger = logging.getLogger("resourceiq.ai.llm")

_cached_providers: Optional[list[LLMProvider]] = None
_provider_checked = False

def _build_providers() -> list[LLMProvider]:
    providers: list[LLMProvider] = []

    if os.environ.get("GEMINI_API_KEY", ""):
        from .providers.gemini import GeminiProvider
        providers.append(GeminiProvider())

    if os.environ.get("ANTHROPIC_API_KEY", ""):
        from .providers.claude import ClaudeProvider
        providers.append(ClaudeProvider())

    if providers:
        logger.info("AI providers, in priority order: %s", [p.provider_name for p in providers])
    else:
        logger.warning("No AI provider configured -- copilot falls back to the deterministic router")
    return providers

def get_providers() -> list[LLMProvider]:
    global _cached_providers, _provider_checked
    if not _provider_checked:
        _cached_providers = _build_providers()
        _provider_checked = True
    return _cached_providers or []

def get_provider() -> Optional[LLMProvider]:
    providers = get_providers()
    return providers[0] if providers else None

def provider_name() -> str:
    p = get_provider()
    return p.provider_name if p else "none"

def reset() -> None:
    global _cached_providers, _provider_checked
    _cached_providers = None
    _provider_checked = False
