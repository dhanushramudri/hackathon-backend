import logging
import os
from typing import Any, Optional

from .base import LLMProvider, QuotaExceededError, ToolTurn, is_quota_error

logger = logging.getLogger("resourceiq.ai.claude")

_MODEL = "claude-sonnet-4-6"

def _to_claude_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[str]]:
    system_parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    claude_messages: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tool_results():
        if pending_tool_results:
            claude_messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for m in messages:
        role = m["role"]
        if role == "system":
            continue
        if role == "tool":
            pending_tool_results.append({"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or ""})
            continue

        flush_tool_results()
        if role == "user":
            claude_messages.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
                claude_messages.append({"role": "assistant", "content": content})
            else:
                claude_messages.append({"role": "assistant", "content": m.get("content") or ""})
    flush_tool_results()
    return claude_messages, ("\n\n".join(system_parts) or None)

class ClaudeProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "claude"

    def _client(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None
        import anthropic
        return anthropic.Anthropic(api_key=api_key)

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> Optional[ToolTurn]:
        client = self._client()
        if client is None:
            return None
        try:
            claude_messages, system = _to_claude_messages(messages)
            claude_tools = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]

            response = client.messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=claude_messages,
                **({"tools": claude_tools} if claude_tools else {}),
            )

            tool_calls = []
            text_parts = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "arguments": dict(block.input or {})})
                elif block.type == "text":
                    text_parts.append(block.text)

            return {"content": ("\n".join(text_parts).strip() or None) if not tool_calls else None, "tool_calls": tool_calls}
        except Exception as e:
            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e
            logger.warning("Claude generate_with_tools failed: %s", e)
            return None
