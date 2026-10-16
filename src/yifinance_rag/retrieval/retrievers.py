"""Retrievers. Each sees a `Question` only (case id + text), never its label; the oracle
baselines are the deliberate exception, built from labels to check the metrics."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from yifinance_rag.data.benchmark import Label, Question
from yifinance_rag.data.corpus import Chunk
from yifinance_rag.retrieval.bm25 import BM25
from yifinance_rag.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def config(self) -> dict[str, object]: ...

    def search(self, question: Question, k: int) -> list[Hit]: ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.2, b: float = 0.75):
        self.chunks = list(chunks)
        self.index = BM25([tokenize(c.indexed_text()) for c in self.chunks], k1=k1, b=b)

    def config(self) -> dict[str, object]:
        return {"k1": self.index.k1, "b": self.index.b, "tokenizer": "ident+jieba"}

    def search(self, question: Question, k: int) -> list[Hit]:
        return [Hit(self.chunks[i], s) for i, s in self.index.top(tokenize(question.text), k)]


class RandomRetriever:
    """The null baseline: k chunks drawn without replacement, seeded per case."""

    name = "random"

    def __init__(self, chunks: Sequence[Chunk], *, seed: int = 0):
        self.chunks = list(chunks)
        self.seed = seed

    def config(self) -> dict[str, object]:
        return {"seed": self.seed}

    def search(self, question: Question, k: int) -> list[Hit]:
        rng = random.Random(f"{self.seed}:{question.case_id}")
        picked = rng.sample(self.chunks, min(k, len(self.chunks)))
        return [Hit(c, 0.0) for c in picked]


class OracleRetriever:
    """Returns each case's gold doc chunks. With `shift`, returns the gold of another case
    instead (the "wrong answer" check), which must score near 0."""

    name = "oracle"

    def __init__(self, chunks: Sequence[Chunk], labels: Mapping[str, Label], *, shift: int = 0):
        self.by_id = {c.chunk_id: c for c in chunks}
        self.labels = labels
        self.shift = shift
        self.order = sorted(labels)
        if shift:
            self.name = "wrong-oracle"

    def config(self) -> dict[str, object]:
        return {"shift": self.shift}

    def search(self, question: Question, k: int) -> list[Hit]:
        cid = question.case_id
        if self.shift:
            cid = self.order[(self.order.index(cid) + self.shift) % len(self.order)]
        gold = [self.by_id[e] for e in self.labels[cid].doc_evidence if e in self.by_id]
        return [Hit(c, 1.0) for c in gold[:k]]
