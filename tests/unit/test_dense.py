from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.fake_ollama import fake_ollama, fake_vector
from tests.synthetic import write_dataset
from yifinance_rag.cli import main
from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import Chunk
from yifinance_rag.embed.cache import CachedEmbedder
from yifinance_rag.embed.ollama import (
    COURSE_DIGEST,
    QUERY_INSTRUCTION,
    EmbedError,
    OllamaEmbedder,
    query_input,
)
from yifinance_rag.retrieval.dense import (
    DenseRetriever,
    course_db_url,
    course_style_text,
    same_vector_space,
)


class CountingEmbedder:
    def __init__(self, tag: str = "a") -> None:
        self.tag = tag
        self.calls: list[list[str]] = []

    def identity(self) -> dict[str, object]:
        return {"model": "fake", "revision": self.tag, "request_options": {}}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [fake_vector(t) for t in texts]


def test_identity_is_checked_against_the_pinned_digest() -> None:
    with fake_ollama() as server:
        assert OllamaEmbedder(server.url).identity()["revision"] == COURSE_DIGEST
    with fake_ollama(digest="sha256-other") as server, pytest.raises(EmbedError, match="digest"):
        OllamaEmbedder(server.url).identity()
    with fake_ollama(model="other:1b") as server, pytest.raises(EmbedError, match="pull"):
        OllamaEmbedder(server.url).identity()


def test_embed_batches_and_restores_order() -> None:
    texts = [f"文本{i}号" for i in range(7)]
    with fake_ollama() as server:
        vectors = OllamaEmbedder(server.url, batch_size=3).embed(texts)
    assert [len(b) for b in server.batches] == [3, 3, 1]
    assert vectors == [fake_vector(t) for t in texts]  # the server answered reversed


def test_unreachable_ollama_is_an_infra_error() -> None:
    with pytest.raises(EmbedError, match="api/tags"):
        OllamaEmbedder("http://127.0.0.1:9", timeout=2).identity()


def test_cache_embeds_each_text_once_per_identity(tmp_path: Path) -> None:
    inner = CountingEmbedder("a")
    cached = CachedEmbedder(inner, tmp_path / "e.sqlite")
    first = cached.embed(["甲乙", "丙丁", "甲乙"])
    assert inner.calls == [["甲乙", "丙丁"]]  # duplicates sent once
    assert cached.embed(["丙丁", "甲乙"]) == [first[1], first[0]]
    assert len(inner.calls) == 1  # all from the cache
    reopened = CachedEmbedder(CountingEmbedder("a"), tmp_path / "e.sqlite")
    assert reopened.embed(["甲乙"]) == [first[0]] and reopened.computed == 0
    other = CachedEmbedder(CountingEmbedder("b"), tmp_path / "e.sqlite")
    other.embed(["甲乙"])
    assert other.computed == 1  # another model's vectors are never reused


def test_queries_get_the_course_instruction_and_passages_its_prefix() -> None:
    assert query_input("退款") == f"Instruct: {QUERY_INSTRUCTION}\nQuery: 退款"
    title = Chunk("c1", "D", "t", "幂等键", "幂等键", 0, 1, "# 幂等键\n")
    section = Chunk("c2", "D", "t", "幂等键", "冲突", 1, 2, "## 冲突\n\n返回 FP-4091。\n")
    assert course_style_text(title) == "幂等键\n幂等键\n\n# 幂等键\n"
    assert course_style_text(section) == "幂等键\n幂等键 / 冲突\n\n## 冲突\n\n返回 FP-4091。\n"


def test_dense_ranks_by_cosine_with_ties_in_index_order() -> None:
    chunks = [Chunk(f"c{i}", "D", "t", "", "", i, i + 1, "") for i in range(4)]
    vectors = [[0.0, 1.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]  # c1 and c2 tie on cosine

    class Fixed(CountingEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    r = DenseRetriever("d", chunks, vectors, Fixed(), {})
    hits = r.search(Question(case_id="q", text="x"), 3)
    assert [h.chunk.chunk_id for h in hits] == ["c1", "c2", "c3"]
    assert hits[0].score == pytest.approx(1.0) and hits[2].score == pytest.approx(0.7071, 1e-3)


def test_same_vector_space_names_each_difference() -> None:
    built: dict[str, object] = {
        "embedding": {"model": "fake", "revision": "a", "request_options": {}},
        "query_instruction": QUERY_INSTRUCTION,
    }
    assert same_vector_space(built, CountingEmbedder("a")) == []
    problems = same_vector_space({**built, "query_instruction": "x"}, CountingEmbedder("b"))
    assert problems == ["revision: collection 'a', ours 'b'", "query instruction differs"]


def test_course_db_is_a_sibling_database() -> None:
    url = "postgresql://flowpay:pw@127.0.0.1:5432/flowpay"
    assert course_db_url(url) == "postgresql://flowpay:pw@127.0.0.1:5432/course_kb"


def test_cli_dense_run_embeds_once_then_reads_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = write_dataset(tmp_path / "flowpay-dataset")
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs"
    with fake_ollama() as server:
        monkeypatch.setenv("YFR_OLLAMA_URL", server.url)
        assert main(["retrieval", "--retriever", "dense-809", "--out", str(out)]) == 0
        sent = sum(len(b) for b in server.batches)
        assert sent == 12 + 2  # 12 chunks, 2 dev questions with doc evidence
        assert main(["retrieval", "--retriever", "dense-809", "--out", str(out)]) == 0
        assert sum(len(b) for b in server.batches) == sent  # second run: all cached
    summaries = [json.loads((d / "summary.json").read_text()) for d in sorted(out.iterdir())]
    assert summaries[0]["rows"] == summaries[1]["rows"]
    assert summaries[1]["config"]["embedder"]["revision"] == COURSE_DIGEST
    assert summaries[1]["embeddings_computed"] == 0
    assert summaries[1]["recall_ceiling"] == 1.0


def test_cli_dense_without_ollama_writes_no_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(write_dataset(tmp_path / "flowpay-dataset")))
    monkeypatch.setenv("YFR_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs"
    assert main(["retrieval", "--retriever", "dense-809", "--out", str(out)]) == 3
    assert not out.exists()
