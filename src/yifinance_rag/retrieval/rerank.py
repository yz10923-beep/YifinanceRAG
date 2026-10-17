"""A listwise LLM reranker over a first-stage retriever's top N.

One call per question: the question and the top N passages, numbered; the model replies
with a JSON array of all N numbers, most useful first. It can only reorder what the first
stage found, so recall@N is unchanged by design; what it can move is MRR and recall@k<N.

- Answers are cached on disk keyed by (model, effort, prompt, trial): a rerun or a crash
  never pays twice, and `trial` > 0 asks the same questions again to measure run-to-run
  variance (the model has no fixed temperature to pin).
- A reply that is not a usable ranking keeps the first-stage order for the numbers it
  leaves out, and the run counts such fallbacks. API errors that survive the retry
  policy stop the run (an infra failure, never a score).
- One event loop and one API client for the whole run, closed by `close()`: a loop per
  call left each client's connection pool to be closed after its loop was gone.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.llm.pricing import cost_usd
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMError, LLMProvider, LLMResponse
from yifinance_rag.retrieval.retrievers import Hit, Retriever

SYSTEM = (
    "You rank passages from FlowPay's product documentation by how useful each one is "
    "for answering a merchant's support question. Judge only from the passage text. "
    "Reply with a JSON array of the passage numbers, most useful first, listing every "
    "number exactly once. Reply with the array only."
)


def render_prompt(question: Question, hits: Sequence[Hit], max_chars: int) -> str:
    parts = [f"Question:\n{question.text}\n\nPassages:"]
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        heading = f"{c.title} / {c.heading}" if c.heading and c.heading != c.title else c.title
        parts.append(f"[{i}] {heading}\n{c.text[:max_chars]}")
    return "\n\n".join(parts)


def parse_ranking(text: str, n: int) -> tuple[list[int], bool]:
    """0-based order of all n passages, and whether the reply was a complete ranking.
    Valid numbers are kept in the model's order; the rest follow in first-stage order."""
    match = re.search(r"\[[^\[\]]*\]", text)
    numbers: list[int] = []
    if match:
        try:
            raw = json.loads(match.group(0))
            numbers = [int(x) for x in raw if isinstance(x, int | str) and str(x).isdigit()]
        except (json.JSONDecodeError, ValueError):
            numbers = []
    if not numbers:
        numbers = [int(x) for x in re.findall(r"\d+", text)]
    order = list(dict.fromkeys(i - 1 for i in numbers if 1 <= i <= n))
    complete = len(order) == n
    return order + [i for i in range(n) if i not in order], complete


@dataclass
class RerankStats:
    calls: int = 0  # requests actually sent (cache misses)
    cached: int = 0
    fallbacks: int = 0  # replies that were not a complete ranking
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0


class LLMReranker:
    def __init__(
        self,
        base: Retriever,
        provider_factory: Callable[[], LLMProvider],
        *,
        model: str,
        effort: str,
        cache_path: Path,
        depth: int = 20,
        max_chars: int = 1000,
        trial: int = 0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.base = base
        self.name = f"{base.name}+rerank"
        self.chunks = getattr(base, "chunks", [])
        self.provider_factory = provider_factory
        self.model, self.effort = model, effort
        self.depth, self.max_chars, self.trial = depth, max_chars, trial
        self.retry = retry or RetryPolicy()
        self.stats = RerankStats()
        self._loop = asyncio.new_event_loop()
        self._provider: LLMProvider | None = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path, timeout=60)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, text TEXT NOT NULL, "
            "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)"
        )

    def config(self) -> dict[str, object]:
        return {
            "base": {"name": self.base.name, **self.base.config()},
            "reranker": {
                "model": self.model,
                "effort": self.effort,
                "depth": self.depth,
                "max_chars": self.max_chars,
                "trial": self.trial,
                "system_sha256": sha256_text(SYSTEM),
            },
        }

    def search(self, question: Question, k: int) -> list[Hit]:
        hits = self.base.search(question, max(k, self.depth))[: self.depth]
        if len(hits) < 2:
            return hits[:k]
        prompt = render_prompt(question, hits, self.max_chars)
        text = self._reply(prompt)
        order, complete = parse_ranking(text, len(hits))
        if not complete:
            self.stats.fallbacks += 1
        return [Hit(hits[i].chunk, float(len(hits) - r)) for r, i in enumerate(order)][:k]

    def _reply(self, prompt: str) -> str:
        key = sha256_text(json.dumps([self.model, self.effort, SYSTEM, prompt, self.trial]))
        row = self.db.execute("SELECT text FROM replies WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.stats.cached += 1
            return str(row[0])
        response = self._loop.run_until_complete(self._call(prompt))
        self.stats.calls += 1
        usage = response.usage
        self.stats.input_tokens += (
            usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        )
        self.stats.output_tokens += usage.output_tokens
        cost = cost_usd(response.model or self.model, usage.model_dump())
        if cost is None or self.stats.cost_usd is None:
            self.stats.cost_usd = None  # an unknown price is unknown, never $0
        else:
            self.stats.cost_usd += cost
        text = response.text if response.stop_reason != "refusal" else ""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?)",
                (key, text, usage.input_tokens, usage.output_tokens),
            )
        return text

    async def _call(self, prompt: str) -> LLMResponse:
        if self._provider is None:
            self._provider = self.provider_factory()
        provider = self._provider
        messages = [{"role": "user", "content": prompt}]
        retry, waited = 0, 0.0
        while True:
            try:
                return await provider.complete(system=SYSTEM, messages=messages, tools=[])
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
