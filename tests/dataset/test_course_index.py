"""The course's index, restored by `make course-index`, against our code. Needs the real
dataset and the running pgvector: `make pg-test`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import psycopg
import pytest

from yifinance_rag.config import load_settings
from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.embed.ollama import OllamaEmbedder
from yifinance_rag.retrieval.dense import (
    COURSE_COLLECTION,
    CourseIndex,
    DenseRetriever,
    course_db_url,
    course_style_text,
    load_course_index,
)

pytestmark = [pytest.mark.dataset, pytest.mark.pg]
URL = course_db_url(load_settings().pg_url.get_secret_value())


@pytest.fixture(scope="module")
def index() -> CourseIndex:
    return load_course_index(URL)


def test_course_offsets_point_into_our_doc_text(ds: Dataset, index: CourseIndex) -> None:
    docs = {d.doc_id: d for d in load_docs(ds)}
    assert len(index.chunks) == 712
    for c in index.chunks:
        assert c.text.endswith(docs[c.doc_id].text[c.start : c.end]), c.chunk_id


def test_our_passage_format_reproduces_the_course_text(ds: Dataset, index: CourseIndex) -> None:
    ours = {(c.doc_id, c.start, c.end): c for c in benchmark_chunks(load_docs(ds))}
    same_span = [
        (ours[(c.doc_id, c.start, c.end)], c)
        for c in index.chunks
        if (c.doc_id, c.start, c.end) in ours
    ]
    assert len(same_span) > 300  # most course chunks are whole benchmark sections
    for bench, course in same_span:
        assert course_style_text(bench) == course.text, course.chunk_id


class _Fixed:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def identity(self) -> dict[str, object]:
        return {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


def test_numpy_ranking_equals_the_course_sql(index: CourseIndex) -> None:
    rng = np.random.default_rng(0)
    probes = [index.vectors[i] for i in rng.choice(len(index.vectors), 5, replace=False)]
    probes += [rng.normal(size=1024).tolist() for _ in range(3)]
    with psycopg.connect(URL) as conn:
        for vec in probes:
            rows = conn.execute(
                "SELECT doc_id, chunk_no FROM chunks WHERE collection_id = %s "
                "ORDER BY 1 - (embedding <=> %s::vector) DESC, doc_id, chunk_no LIMIT 10",
                (COURSE_COLLECTION, str(vec)),
            ).fetchall()
            sql = [f"course:{d}#{n:03d}" for d, n in rows]
            dense = DenseRetriever("d", index.chunks, index.vectors, _Fixed(vec), {})
            hits = dense.search(Question(case_id="q", text="x"), 10)
            assert [h.chunk.chunk_id for h in hits] == sql


@pytest.mark.ollama
def test_our_embedder_reproduces_the_course_vectors(index: CourseIndex) -> None:
    """Our client, same model and digest, on the course's own chunk texts, must give the
    vectors stored in its dump; then course-dense differs from the course only in nothing."""
    embedder = OllamaEmbedder(load_settings().ollama_url)
    picks = list(range(0, len(index.chunks), len(index.chunks) // 8))[:8]
    ours = embedder.embed([index.chunks[i].text for i in picks])
    for i, vec in zip(picks, ours, strict=True):
        a, b = np.asarray(vec), np.asarray(index.vectors[i])
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cosine > 0.999, (index.chunks[i].chunk_id, cosine)
