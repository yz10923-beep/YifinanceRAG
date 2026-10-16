"""Dense retrieval: exact cosine over a fixed set of chunk vectors.

Two indexes use it:
- `course-dense`: the course's own collection (712 chunks and their vectors), restored
  from its pgvector dump into the `course_kb` database. Queries are embedded the way the
  course embeds them and ranked as its SQL ranks them: cosine, ties by (doc_id,
  chunk_no). Like the course, the search is exact (no vector index).
- `dense-809`: the benchmark's 809 chunks embedded with the same model and the course's
  passage format, so against course-dense only the chunking differs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import psycopg

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import Chunk
from yifinance_rag.embed.ollama import (
    COURSE_MODEL,
    QUERY_INSTRUCTION,
    Embedder,
    EmbedError,
    query_input,
)
from yifinance_rag.retrieval.retrievers import Hit

COURSE_DB = "course_kb"
COURSE_COLLECTION = "flowpay_docs_v1_qwen3_06b"


def course_style_text(chunk: Chunk) -> str:
    """The course's passage format: title, heading path, blank line, then the excerpt.
    The title section's heading path is the title itself."""
    path = (
        chunk.title
        if not chunk.heading or chunk.heading == chunk.title
        else (f"{chunk.title} / {chunk.heading}")
    )
    return f"{chunk.title}\n{path}\n\n{chunk.text}"


def _normalized(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if (norms == 0).any():
        raise EmbedError("an embedding is all zeros")
    result: np.ndarray = m / norms
    return result


class DenseRetriever:
    def __init__(
        self,
        name: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        embedder: Embedder,
        config: dict[str, object],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.name = name
        self.chunks = list(chunks)  # in tie-break order
        self.matrix = _normalized(vectors)
        self.embedder = embedder
        self._config = config

    def config(self) -> dict[str, object]:
        return {**self._config, "embedder": self.embedder.identity()}

    def search(self, question: Question, k: int) -> list[Hit]:
        query = _normalized(self.embedder.embed([query_input(question.text)]))[0]
        if query.shape[0] != self.matrix.shape[1]:
            raise EmbedError(f"query has {query.shape[0]} dims, index {self.matrix.shape[1]}")
        scores = self.matrix @ query
        order = np.argsort(-scores, kind="stable")[:k]  # stable: ties keep index order
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def model_suffix(embedder: Embedder) -> str:
    """ "" for the course's model, else "-<size>", so run names say which model ran."""
    model = str(embedder.identity().get("model", ""))
    return "" if model == COURSE_MODEL else "-" + model.rpartition(":")[2]


def dense_809(chunks: Sequence[Chunk], embedder: Embedder) -> DenseRetriever:
    vectors = embedder.embed([course_style_text(c) for c in chunks])
    return DenseRetriever(
        "dense-809" + model_suffix(embedder),
        chunks,
        vectors,
        embedder,
        {"chunks": "benchmark-809", "text": "course-style"},
    )


def course_db_url(pg_url: str) -> str:
    parts = urlsplit(pg_url)
    return urlunsplit(parts._replace(path="/" + COURSE_DB))


@dataclass(frozen=True)
class CourseIndex:
    chunks: list[Chunk]
    vectors: list[list[float]]
    build_config: dict[str, object]


def load_course_index(db_url: str) -> CourseIndex:
    """The course's collection, in its tie-break order (doc_id, chunk_no)."""
    with psycopg.connect(db_url) as conn:
        row = conn.execute(
            "SELECT build_config::text FROM collections WHERE collection_id = %s",
            (COURSE_COLLECTION,),
        ).fetchone()
        if row is None:
            raise EmbedError(f"{COURSE_COLLECTION} is not in {COURSE_DB}; run make course-index")
        build_config = json.loads(row[0])
        rows = conn.execute(
            "SELECT doc_id, chunk_no, content, source_locator::text, embedding::text "
            "FROM chunks WHERE collection_id = %s ORDER BY doc_id, chunk_no",
            (COURSE_COLLECTION,),
        ).fetchall()
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for doc_id, chunk_no, content, locator_text, embedding in rows:
        locator = json.loads(locator_text)
        chunks.append(
            Chunk(
                f"course:{doc_id}#{chunk_no:03d}",
                doc_id,
                "",
                "",
                str(locator.get("heading", "")),
                int(locator["start_char"]),
                int(locator["end_char"]),
                content,
            )
        )
        vectors.append(json.loads(embedding))
    return CourseIndex(chunks, vectors, build_config)


def same_vector_space(build_config: dict[str, object], embedder: Embedder) -> list[str]:
    """Why query vectors from `embedder` can't be compared with the collection's (empty
    if they can): the course's `same_vector_space` check, on the fields that matter."""
    built = build_config.get("embedding")
    built = built if isinstance(built, dict) else {}
    ours = embedder.identity()
    problems = [
        f"{key}: collection {built.get(key)!r}, ours {ours.get(key)!r}"
        for key in ("model", "revision", "request_options")
        if built.get(key) != ours.get(key)
    ]
    if build_config.get("query_instruction") != QUERY_INSTRUCTION:
        problems.append("query instruction differs")
    return problems


def course_dense(db_url: str, embedder: Embedder) -> DenseRetriever:
    index = load_course_index(db_url)
    if problems := same_vector_space(index.build_config, embedder):
        raise EmbedError("query embedder is not the collection's: " + "; ".join(problems))
    return DenseRetriever(
        "course-dense",
        index.chunks,
        index.vectors,
        embedder,
        {"collection": COURSE_COLLECTION, "chunks": len(index.chunks)},
    )
