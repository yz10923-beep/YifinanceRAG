from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.synthetic import write_dataset
from yifinance_rag.cli import main
from yifinance_rag.data.benchmark import Label, Question, load_benchmark, parse_rows
from yifinance_rag.data.corpus import Chunk, EvidenceMap, benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.retrieval.bm25 import BM25
from yifinance_rag.retrieval.evaluate import CaseResult, evaluate, render_table, summarize
from yifinance_rag.retrieval.retrievers import (
    BM25Retriever,
    OracleRetriever,
    RandomRetriever,
    Retriever,
)
from yifinance_rag.retrieval.tokenize import tokenize


def test_identifiers_stay_whole_and_also_match_by_part() -> None:
    assert tokenize("FP-4041 资源不存在") == ["fp-4041", "fp", "4041", "资源", "不", "存在"]
    assert tokenize("pay_bm_001") == ["pay_bm_001", "pay", "bm", "001"]
    assert tokenize("2026-07-01 之后")[:4] == ["2026-07-01", "2026", "07", "01"]
    assert tokenize("Webhook，签名！") == ["webhook", "签名"]  # punctuation dropped
    assert tokenize("幂等键重复使用") == ["幂等键", "重复使用"]


def test_bm25_matches_the_formula_by_hand() -> None:
    docs = [["a", "b"], ["a", "a", "c", "d"], ["e"]]
    bm = BM25(docs, k1=1.2, b=0.75)
    avgdl = 7 / 3
    idf_a = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    tf_part = 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 4 / avgdl))
    assert bm.scores(["a"])[1] == pytest.approx(idf_a * tf_part)
    assert bm.scores(["a", "a"]) == bm.scores(["a"])  # query terms count once
    assert bm.top(["zzz"], 3) == []  # no shared term, nothing returned
    assert [i for i, _ in bm.top(["e", "a"], 3)] == [2, 1, 0]  # rare term outranks common


def test_exact_code_outranks_a_partial_mention() -> None:
    chunks = [
        Chunk("c1", "D1", "t", "错误码", "", 0, 1, "返回 FP-4041 资源不存在"),
        Chunk("c2", "D2", "t", "错误码", "", 0, 1, "返回 FP-4091 幂等冲突，编号 4041 的订单"),
    ]
    r = BM25Retriever(chunks)
    hits = r.search(Question(case_id="q", text="FP-4041 是什么"), 2)
    assert [h.chunk.chunk_id for h in hits] == ["c1", "c2"]


def test_metrics_by_hand() -> None:
    r = CaseResult("q", "multi_document", ("g1", "g2"), (("x",), ("g2",), (), ("g1", "g9")))
    assert r.recall(1) == 0 and r.recall(2) == 0.5 and r.recall(4) == 1.0
    assert r.complete(2) == 0.0 and r.complete(4) == 1.0
    assert r.mrr(4) == 0.5 and r.mrr(1) == 0.0


def _world(
    n_docs: int, n_questions: int, *, unique: bool
) -> tuple[list[Chunk], dict[str, Label], list[Question]]:
    """Docs of 5 sections each; question i's gold is chunk 2i (and 2i+1 for odd i), so
    with `unique` no two questions share a gold chunk."""
    chunks = [
        Chunk(f"doc:D{d}#c{s:03d}", f"D{d}", "t", "", "", s * 10, s * 10 + 10, f"文本 {d} {s}")
        for d in range(n_docs)
        for s in range(1, 6)
    ]
    n = len(chunks)
    if unique:
        assert 2 * n_questions <= n
    rows = [
        {
            "question_id": f"Q{i}",
            "question": f"问题 {i}",
            "question_type": ["simple_retrieval", "multi_document"][i % 2],
            "split": "dev",
            "conversation": {},
            "resolved_evidence_ids": [chunks[2 * i % n].chunk_id]
            + ([chunks[(2 * i + 1) % n].chunk_id] if i % 2 else []),
        }
        for i in range(n_questions)
    ]
    questions, labels = parse_rows(rows)
    return chunks, labels, questions


def _overall(
    retriever: Retriever, chunks: list[Chunk], labels: dict[str, Label], qs: list[Question]
) -> dict[str, object]:
    results, skipped = evaluate(retriever, qs, labels, EvidenceMap(chunks))
    assert skipped == 0
    return summarize(results)[-1]


def test_oracle_passes_and_the_wrong_oracle_fails() -> None:
    chunks, labels, qs = _world(60, 120, unique=True)
    oracle = _overall(OracleRetriever(chunks, labels), chunks, labels, qs)
    assert oracle["recall@10"] == oracle["complete@10"] == oracle["mrr@10"] == 1.0
    wrong = _overall(OracleRetriever(chunks, labels, shift=1), chunks, labels, qs)
    assert wrong["recall@10"] == wrong["mrr@10"] == 0.0


def test_random_is_at_chance() -> None:
    chunks, labels, qs = _world(40, 2000, unique=False)  # 200 chunks: chance = 10/200
    rand = _overall(RandomRetriever(chunks, seed=1), chunks, labels, qs)
    assert rand["recall@10"] == pytest.approx(10 / len(chunks), rel=0.15)  # sd about 5%
    assert _overall(RandomRetriever(chunks, seed=1), chunks, labels, qs) == rand  # seeded


def test_cases_without_doc_evidence_are_counted_not_scored(tmp_path: Path) -> None:
    ds = Dataset(write_dataset(tmp_path / "flowpay-dataset"))
    chunks = benchmark_chunks(load_docs(ds))
    questions, labels = load_benchmark(ds)
    results, skipped = evaluate(
        OracleRetriever(chunks, labels), questions, labels, EvidenceMap(chunks)
    )
    assert skipped == 1 and len(results) == 3
    rows = summarize(results)
    assert [r["type"] for r in rows] == ["simple_retrieval", "multi_document", "all"]
    assert "| all | 3 | 1.000 | 1.000 | 1.000 | 1.000 |" in render_table(rows)


def test_bm25_finds_the_synthetic_gold(tmp_path: Path) -> None:
    ds = Dataset(write_dataset(tmp_path / "flowpay-dataset"))
    chunks = benchmark_chunks(load_docs(ds))
    questions, labels = load_benchmark(ds)
    results, _ = evaluate(BM25Retriever(chunks), questions, labels, EvidenceMap(chunks))
    assert summarize(results)[-1]["recall@10"] == 1.0


def test_cli_writes_a_run_and_guards_the_test_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_dataset(tmp_path / "flowpay-dataset")
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(root))
    monkeypatch.chdir(tmp_path)  # no ./.env from the repo
    out = tmp_path / "runs"
    assert main(["retrieval", "--retriever", "bm25", "--out", str(out)]) == 0
    (run,) = out.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["retriever"] == "bm25" and summary["split"] == "dev"
    assert summary["cases"] == 2 and summary["cases_without_doc_evidence"] == 1
    assert {"release_manifest_sha256", "split_sha256", "chunk_set_sha256"} <= set(summary)
    assert main(["retrieval", "--retriever", "bm25", "--split", "test", "--out", str(out)]) == 2
    assert "--final" in capsys.readouterr().out
