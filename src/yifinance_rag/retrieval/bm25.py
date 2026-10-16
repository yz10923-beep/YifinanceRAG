"""Okapi BM25, written out so every term of the formula is visible.

    score(q, d) = sum over unique query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

idf is Lucene's form, never negative (the original goes negative for terms in more than
half the docs). k1 sets how fast repeated terms saturate; b how much long docs are
penalised.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence


class BM25:
    def __init__(self, docs: Sequence[Sequence[str]], *, k1: float = 1.2, b: float = 0.75):
        if not docs:
            raise ValueError("BM25 needs at least one document")
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avgdl = sum(self.lengths) / self.n or 1.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> list[float]:
        out = [0.0] * self.n
        k1, b = self.k1, self.b
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = k1 * (1 - b + b * self.lengths[i] / self.avgdl)
                out[i] += idf * tf * (k1 + 1) / (tf + norm)
        return out

    def top(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Best k (index, score), highest first, ties by index; docs scoring 0 are left
        out, so fewer than k come back when few docs share a term with the query."""
        ranked = sorted(
            ((i, s) for i, s in enumerate(self.scores(query)) if s > 0),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:k]
