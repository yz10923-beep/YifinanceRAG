"""An OpenAI-compatible Chat Completions provider (DeepSeek, Qwen, ...), called over HTTP
with httpx2: no vendor SDK, so the wire format stays visible.

The rest of the code keeps conversation history in the Anthropic Messages format
(`llm/types.py`); this module translates both ways:
- system prompt -> a `system` message; text blocks -> `content`;
- assistant `tool_use` blocks -> `tool_calls` (arguments as a JSON string);
- user `tool_result` blocks -> one `tool` message each, before any user text;
- tool specs `{name, description, input_schema}` -> `{"type": "function", ...}`.
Usage maps to our `Usage`: `input_tokens` counts uncached prompt tokens only, as in the
Anthropic API, with cache hits in `cache_read_input_tokens` (DeepSeek reports
`prompt_cache_hit_tokens`; others `prompt_tokens_details.cached_tokens`).

Non-streaming: one POST, one JSON reply. `on_text` gets the whole text once.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx2

from yifinance_rag.llm.anthropic_provider import retry_after, status_kind
from yifinance_rag.llm.types import (
    RETRYABLE_KINDS,
    LLMError,
    LLMResponse,
    Message,
    StopReason,
    TextCallback,
    ToolSpec,
    Usage,
)

DEEPSEEK_URL = "https://api.deepseek.com"
_FINISH: dict[str, StopReason] = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def _text(blocks: object) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    return "".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    )


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        content = m.get("content")
        if m["role"] == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": _text(content) or None}
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                    },
                }
                for b in (content if isinstance(content, list) else [])
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        blocks = content if isinstance(content, list) else []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                result = b.get("content")
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": b["tool_use_id"],
                        "content": result if isinstance(result, str) else _text(result),
                    }
                )
        text = _text(content)
        if text or not blocks:
            out.append({"role": "user", "content": text})
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def to_usage(raw: Mapping[str, Any] | None) -> Usage:
    raw = raw or {}
    prompt = int(raw.get("prompt_tokens") or 0)
    details = raw.get("prompt_tokens_details") or {}
    hit = int(raw.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0)
    return Usage(
        input_tokens=max(0, prompt - hit),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_input_tokens=hit,
    )


def to_llm_response(body: Mapping[str, Any], model: str) -> LLMResponse:
    choices = body.get("choices") or []
    if not choices:
        raise LLMError("reply has no choices", retryable=False, kind="unknown")
    choice = choices[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed_arguments": fn.get("arguments")}
        content.append(
            {
                "type": "tool_use",
                "id": call["id"],
                "name": fn.get("name", ""),
                "input": args if isinstance(args, dict) else {"_arguments": args},
            }
        )
    return LLMResponse(
        stop_reason=_FINISH.get(str(choice.get("finish_reason")), "other"),
        content=content,
        usage=to_usage(body.get("usage")),
        model=str(body.get("model") or model),
    )


def error_from(response: httpx2.Response) -> LLMError:
    try:
        err = response.json().get("error") or {}
        detail = str(err.get("message") or err)
    except (ValueError, AttributeError):
        detail = response.text[:300]
    kind = status_kind(response.status_code, None)
    if response.status_code == 400 and "context" in detail.lower() and "length" in detail.lower():
        kind = "context_overflow"
    return LLMError(
        f"HTTP {response.status_code}: {detail}",
        retryable=kind in RETRYABLE_KINDS,
        kind=kind,
        status=response.status_code,
        retry_after_s=retry_after(response.headers),
    )


class OpenAICompatProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEEPSEEK_URL,
        max_tokens: int = 4096,
        extra: Mapping[str, Any] | None = None,
        timeout_s: float = 120.0,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        # Vendor options sent as is, e.g. DeepSeek's {"thinking": {"type": "disabled"}}.
        self._extra = dict(extra or {})
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx2.AsyncClient(timeout=timeout_s)

    @property
    def model(self) -> str:
        return self._model

    def build_request(
        self, *, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(system, messages),
            "max_tokens": self._max_tokens,
            "stream": False,
            **self._extra,
        }
        if tools:
            body["tools"] = to_openai_tools(tools)
        return body

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        body = self.build_request(system=system, messages=messages, tools=tools)
        try:
            response = await self._client.post(self._url, json=body, headers=self._headers)
        except httpx2.TransportError as e:
            raise LLMError(f"connection error: {e}", retryable=True, kind="connection") from e
        if response.status_code != 200:
            raise error_from(response)
        try:
            payload = response.json()
        except ValueError as e:
            raise LLMError("reply is not JSON", retryable=True, kind="server") from e
        result = to_llm_response(payload, self._model)
        if on_text is not None and result.text:
            await on_text(result.text)
        return result

    async def aclose(self) -> None:
        await self._client.aclose()
