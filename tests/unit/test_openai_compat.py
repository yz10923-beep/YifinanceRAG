from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx2
import pytest

from yifinance_rag.llm.openai_compat import OpenAICompatProvider, to_openai_messages, to_usage
from yifinance_rag.llm.types import LLMError, LLMResponse, Message


def _provider(handler: Any, **kw: Any) -> OpenAICompatProvider:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAICompatProvider(model="deepseek-v4-flash", api_key="sk-test", client=client, **kw)


def _ok(message: dict[str, Any], finish: str = "stop", usage: dict[str, Any] | None = None) -> Any:
    def handler(request: httpx2.Request) -> httpx2.Response:
        handler.seen.append(request)  # type: ignore[attr-defined]
        body = {
            "model": "deepseek-v4-flash",
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 100, "completion_tokens": 20},
        }
        return httpx2.Response(200, json=body)

    handler.seen = []  # type: ignore[attr-defined]
    return handler


def _run(p: OpenAICompatProvider, **kw: Any) -> LLMResponse:
    async def go() -> LLMResponse:
        try:
            return await p.complete(
                system=kw.get("system", "sys"), messages=kw["messages"], tools=kw.get("tools", [])
            )
        finally:
            await p.aclose()

    return asyncio.run(go())


def test_request_carries_system_options_and_auth() -> None:
    handler = _ok({"content": "好的"})
    r = _run(
        _provider(handler, extra={"thinking": {"type": "disabled"}}),
        messages=[{"role": "user", "content": "你好"}],
    )
    (req,) = handler.seen
    body = json.loads(req.content)
    assert str(req.url) == "https://api.deepseek.com/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "你好"},
    ]
    assert (
        body["thinking"] == {"type": "disabled"}
        and body["max_tokens"] == 4096
        and "tools" not in body
    )
    assert r.text == "好的" and r.stop_reason == "end_turn" and r.model == "deepseek-v4-flash"


def test_tool_calls_round_trip_in_our_message_format() -> None:
    handler = _ok(
        {
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "query_payments", "arguments": '{"payment_id": "pay_1"}'},
                }
            ],
        },
        finish="tool_calls",
    )
    tools = [
        {"name": "query_payments", "description": "查支付", "input_schema": {"type": "object"}}
    ]
    r = _run(_provider(handler), messages=[{"role": "user", "content": "查 pay_1"}], tools=tools)
    assert r.stop_reason == "tool_use" and r.tool_calls[0].input == {"payment_id": "pay_1"}
    sent = json.loads(handler.seen[0].content)
    assert sent["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "query_payments",
                "description": "查支付",
                "parameters": {"type": "object"},
            },
        }
    ]
    # the next turn: our history with the tool_use and its tool_result, translated back
    history: list[Message] = [
        {"role": "user", "content": "查 pay_1"},
        {"role": "assistant", "content": r.content},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "succeeded"}],
        },
    ]
    msgs = to_openai_messages("sys", history)
    assert msgs[2]["tool_calls"][0]["function"] == {
        "name": "query_payments",
        "arguments": '{"payment_id": "pay_1"}',
    }
    assert msgs[2]["content"] is None
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "succeeded"}
    assert len(msgs) == 4  # no empty user message after a tool result


def test_usage_counts_cache_hits_apart_like_the_anthropic_api() -> None:
    deepseek = to_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_cache_hit_tokens": 800,
            "prompt_cache_miss_tokens": 200,
        }
    )
    assert (deepseek.input_tokens, deepseek.cache_read_input_tokens, deepseek.output_tokens) == (
        200,
        800,
        50,
    )
    openai = to_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tokens_details": {"cached_tokens": 600},
        }
    )
    assert (openai.input_tokens, openai.cache_read_input_tokens) == (400, 600)


@pytest.mark.parametrize(
    ("status", "body", "kind", "retryable"),
    [
        (429, {"error": {"message": "rate limited"}}, "rate_limit", True),
        (503, {"error": {"message": "busy"}}, "server", True),
        (402, {"error": {"message": "Insufficient Balance"}}, "billing", False),
        (401, {"error": {"message": "bad key"}}, "auth", False),
        (
            400,
            {"error": {"message": "maximum context length is 65536 tokens"}},
            "context_overflow",
            False,
        ),
    ],
)
def test_errors_are_classified(
    status: int, body: dict[str, Any], kind: str, retryable: bool
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json=body, headers={"retry-after": "2"})

    with pytest.raises(LLMError) as e:
        _run(_provider(handler), messages=[{"role": "user", "content": "x"}])
    assert e.value.kind == kind and e.value.retryable is retryable and e.value.status == status
    assert e.value.retry_after_s == 2.0


def test_a_dropped_connection_is_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused")

    with pytest.raises(LLMError) as e:
        _run(_provider(handler), messages=[{"role": "user", "content": "x"}])
    assert e.value.kind == "connection" and e.value.retryable
