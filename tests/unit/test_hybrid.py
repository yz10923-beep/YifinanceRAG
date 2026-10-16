from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fake_ollama import fake_ollama
from tests.synthetic import write_dataset
from yifinance_rag.cli import main
from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import Chunk
from yifinance_rag.embed.ollama import COURSE_DIGEST, EmbedError, OllamaEmbedder
from yifinance_rag.retrieval.hybrid import RRFRetriever
from yifinance_rag.retrieval.retrievers import Hit

CHUNKS = [Chunk(f"c{i}", "D", "t", "", "", i, i + 1, "") for i in range(5)]
Q = Question(case_id="q", text="x")


class Fixed:
    def __init__(self, name: str, order: list[int]) -> None:
        self.name, self.order = name, order
        self.asked: list[int] = []

    def config(self) -> dict[str, object]:
        return {}

    def search(self, question: Question, k: int) -> list[Hit]:
        self.asked.append(k)
        return [Hit(CHUNKS[i], 0.0) for i in self.order[:k]]


def test_rrf_by_hand() -> None:
    a, b = Fixed("a", [0, 1, 2]), Fixed("b", [2, 3])
    hits = RRFRetriever("h", [a, b], CHUNKS, k=60).search(Q, 10)
    # c2: 1/63 + 1/61; c0: 1/61; c3: 1/62; c1: 1/62 (tie with c3, both best rank 2 -> index)
    assert [h.chunk.chunk_id for h in hits] == ["c2", "c0", "c1", "c3"]
    assert hits[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert a.asked == [100] and b.asked == [100]  # each retriever is asked for `depth`


def test_rrf_cuts_at_k_and_depth() -> None:
    a, b = Fixed("a", [0, 1, 2, 3, 4]), Fixed("b", [4, 3, 2, 1, 0])
    hits = RRFRetriever("h", [a, b], CHUNKS, depth=2).search(Q, 3)
    assert [h.chunk.chunk_id for h in hits] == ["c0", "c4", "c1"]  # c2 is beyond both depths
    assert len(RRFRetriever("h", [a, b], CHUNKS).search(Q, 2)) == 2


def test_rrf_refuses_mixed_indexes_and_single_parts() -> None:
    foreign = Chunk("other", "E", "t", "", "", 0, 1, "")

    class Stranger(Fixed):
        def search(self, question: Question, k: int) -> list[Hit]:
            return [Hit(foreign, 0.0)]

    with pytest.raises(ValueError, match="shared index"):
        RRFRetriever("h", [Fixed("a", [0]), Stranger("s", [])], CHUNKS).search(Q, 5)
    with pytest.raises(ValueError, match="two"):
        RRFRetriever("h", [Fixed("a", [0])], CHUNKS)


def test_embed_models_are_the_qwen3_family_and_only_the_course_model_is_pinned() -> None:
    with pytest.raises(ValueError, match="qwen3-embedding"):
        OllamaEmbedder("http://127.0.0.1:9", model="bge-m3")
    with fake_ollama(model="qwen3-embedding:4b", digest="sha256-four") as server:
        assert OllamaEmbedder(server.url, model="qwen3-embedding:4b").identity()["revision"] == (
            "sha256-four"
        )
    with fake_ollama(digest="sha256-moved") as server, pytest.raises(EmbedError, match="pinned"):
        OllamaEmbedder(server.url).identity()  # the course model must keep the course's digest


def test_cli_hybrid_and_model_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(write_dataset(tmp_path / "flowpay-dataset")))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs"
    with fake_ollama() as server:
        monkeypatch.setenv("YFR_OLLAMA_URL", server.url)
        assert main(["retrieval", "--retriever", "hybrid", "--out", str(out)]) == 0
    with fake_ollama(model="qwen3-embedding:4b", digest="sha256-four") as server:
        monkeypatch.setenv("YFR_OLLAMA_URL", server.url)
        four = ["--embed-model", "qwen3-embedding:4b", "--out", str(out)]
        assert main(["retrieval", "--retriever", "dense-809", *four]) == 0
        assert main(["retrieval", "--retriever", "hybrid", *four]) == 0
        capsys.readouterr()
        assert main(["retrieval", "--retriever", "course-dense", *four]) == 2
        assert "course's model" in capsys.readouterr().out
    runs = {
        d.name.split("_", 1)[1]: json.loads((d / "summary.json").read_text()) for d in out.iterdir()
    }
    assert set(runs) == {"hybrid_dev", "dense-809-4b_dev", "hybrid-4b_dev"}
    hybrid = runs["hybrid_dev"]["config"]
    assert hybrid["rrf_k"] == 60 and [p["name"] for p in hybrid["parts"]] == ["bm25", "dense-809"]
    assert hybrid["parts"][1]["embedder"]["revision"] == COURSE_DIGEST
    four_parts = runs["hybrid-4b_dev"]["config"]["parts"]
    assert four_parts[1]["embedder"]["revision"] == "sha256-four"
    assert runs["hybrid-4b_dev"]["recall_ceiling"] == 1.0
