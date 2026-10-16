"""Hybrid retrieval by reciprocal rank fusion (Cormack et al., 2009).

    rrf(d) = sum over retrievers r of 1 / (k + rank_r(d)),  rank from 1

Only ranks are fused, never raw scores: BM25 scores are unbounded and cosine scores sit
in [-1, 1], so adding them would need a calibration that RRF avoids. k = 60 is the value
from the paper, used as is (not tuned on dev). A chunk a retriever doesn't return in its
top `depth` adds nothing from that retriever (BM25 returns only chunks sharing a term).
"""

from __future__ import annotations

from collections.abc import Sequence

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import Chunk
from yifinance_rag.retrieval.retrievers import Hit, Retriever


class RRFRetriever:
    def __init__(
        self,
        name: str,
        parts: Sequence[Retriever],
        chunks: Sequence[Chunk],
        *,
        k: int = 60,
        depth: int = 100,
    ) -> None:
        if len(parts) < 2:
            raise ValueError("fusion needs at least two retrievers")
        self.name = name
        self.parts = list(parts)
        self.chunks = list(chunks)  # the shared index; also the final tie-break order
        self._order = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.k, self.depth = k, depth

    def config(self) -> dict[str, object]:
        return {
            "rrf_k": self.k,
            "depth": self.depth,
            "parts": [{"name": p.name, **p.config()} for p in self.parts],
        }

    def search(self, question: Question, k: int) -> list[Hit]:
        scores: dict[str, float] = {}
        best: dict[str, int] = {}
        by_id: dict[str, Chunk] = {}
        for part in self.parts:
            for rank, hit in enumerate(part.search(question, self.depth), start=1):
                cid = hit.chunk.chunk_id
                if cid not in self._order:
                    raise ValueError(f"{part.name} returned {cid}, not in the shared index")
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
                best[cid] = min(best.get(cid, rank), rank)
                by_id[cid] = hit.chunk
        ranked = sorted(scores, key=lambda c: (-scores[c], best[c], self._order[c]))
        return [Hit(by_id[c], scores[c]) for c in ranked[:k]]
