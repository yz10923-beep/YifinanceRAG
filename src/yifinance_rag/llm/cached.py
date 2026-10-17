"""One-shot LLM calls with a reply cache, trials, retries and cost accounting: what the
reranker and the judge share.

- Replies are cached on disk keyed by (model, effort, system, prompt, trial): a rerun or a
  crash never pays twice, and `trial` > 0 asks the same prompts again to measure
  run-to-run variance (the model has no fixed temperature to pin).
- API errors that survive the retry policy propagate: the run stops (an infra failure,
  never a score).
- One event loop and one API client for the whole run, closed by `close()`.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.llm.pricing import cost_usd
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMError, LLMProvider, LLMResponse


@dataclass
class CallStats:
    calls: int = 0  # requests actually sent (cache misses)
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None once any price is unknown, never $0


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int  # uncached input of the call that produced it
    output_tokens: int
    cached: bool
    cost_usd: float | None = None  # what this call cost now; None if cached or unknown


class CachedCaller:
    def __init__(
        self,
        provider_factory: Callable[[], LLMProvider],
        *,
        model: str,
        effort: str,
        system: str,
        cache_path: Path,
        stats: CallStats,
        trial: int = 0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.provider_factory = provider_factory
        self.model, self.effort, self.system, self.trial = model, effort, system, trial
        self.stats = stats
        self.retry = retry or RetryPolicy()
        self._loop = asyncio.new_event_loop()
        self._provider: LLMProvider | None = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path, timeout=60)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, text TEXT NOT NULL, "
            "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)"
        )

    def ask(self, prompt: str) -> str:
        """The model's reply text ("" for a refusal), from the cache when it has one."""
        return self.ask_full(prompt).text

    def ask_full(self, prompt: str) -> Reply:
        """The reply with this call's usage; a cached reply carries the usage it cost."""
        key = sha256_text(json.dumps([self.model, self.effort, self.system, prompt, self.trial]))
        row = self.db.execute(
            "SELECT text, input_tokens, output_tokens FROM replies WHERE key = ?", (key,)
        ).fetchone()
        if row is not None:
            self.stats.cached += 1
            return Reply(str(row[0]), int(row[1]), int(row[2]), cached=True)
        response = self._loop.run_until_complete(self._call(prompt))
        self.stats.calls += 1
        usage = response.usage
        total_in = (
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        )
        self.stats.input_tokens += total_in
        self.stats.output_tokens += usage.output_tokens
        cost = cost_usd(response.model or self.model, usage.model_dump())
        if cost is None or self.stats.cost_usd is None:
            self.stats.cost_usd = None
        else:
            self.stats.cost_usd += cost
        text = response.text if response.stop_reason != "refusal" else ""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?)",
                (key, text, usage.input_tokens, usage.output_tokens),
            )
        return Reply(text, usage.input_tokens, usage.output_tokens, cached=False, cost_usd=cost)

    async def _call(self, prompt: str) -> LLMResponse:
        if self._provider is None:
            self._provider = self.provider_factory()
        provider = self._provider
        messages = [{"role": "user", "content": prompt}]
        retry, waited = 0, 0.0
        while True:
            try:
                return await provider.complete(system=self.system, messages=messages, tools=[])
            except LLMError as e:
                retry += 1
                delay = self.retry.delay(e, retry)
                if not self.retry.should_retry(e, retry, waited, delay):
                    raise
                waited += delay
                await asyncio.sleep(delay)

    def close(self) -> None:
        """Close the API client on the loop it ran on, then the loop and the cache."""
        if self._loop.is_closed():
            return
        aclose = getattr(self._provider, "aclose", None)
        if aclose is not None:
            self._loop.run_until_complete(aclose())
        self._provider = None
        self._loop.close()
        self.db.close()
