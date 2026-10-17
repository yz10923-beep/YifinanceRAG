"""A listwise LLM reranker over a first-stage retriever's top N.

One call per question: the question and the top N passages, numbered; the model replies
with a JSON array of all N numbers, most useful first. It can only reorder what the first
stage found, so recall@N is unchanged by design; what it can move is MRR and recall@k<N.

- Calls go through `llm.cached.CachedCaller` (reply cache, trials, retries, cost).
- A reply that is not a usable ranking keeps the first-stage order for the numbers it
  leaves out, and the run counts such fallbacks. API errors that survive the retry
  policy stop the run (an infra failure, never a score).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.llm.cached import CachedCaller, CallStats
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMProvider
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
class RerankStats(CallStats):
    fallbacks: int = 0  # replies that were not a complete ranking


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
        self.model, self.effort = model, effort
        self.depth, self.max_chars, self.trial = depth, max_chars, trial
        self.stats = RerankStats()
        # Same cache key as before the caller was shared, so existing caches still hit.
        self.caller = CachedCaller(
            provider_factory,
            model=model,
            effort=effort,
            system=SYSTEM,
            cache_path=cache_path,
            stats=self.stats,
            trial=trial,
            retry=retry,
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
        text = self.caller.ask(render_prompt(question, hits, self.max_chars))
        order, complete = parse_ranking(text, len(hits))
        if not complete:
            self.stats.fallbacks += 1
        return [Hit(hits[i].chunk, float(len(hits) - r)) for r, i in enumerate(order)][:k]

    def close(self) -> None:
        self.caller.close()
