from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.synthetic import CHUNKS, DOCS, QUESTIONS, blueprint, write_dataset
from yifinance_rag.data.benchmark import (
    Question,
    case_id,
    check_benchmark,
    leaks,
    load_benchmark,
    parse_rows,
    select,
    split_hash,
)
from yifinance_rag.data.corpus import (
    Chunk,
    EvidenceMap,
    benchmark_chunks,
    benchmark_sections,
    check_chunks,
    front_matter,
    load_docs,
)
from yifinance_rag.data.dataset import DataError, Dataset


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return Dataset(write_dataset(tmp_path / "flowpay-dataset"))


def test_front_matter_reads_scalars_and_finds_the_body() -> None:
    text = DOCS["data/knowledge/troubleshooting/KB-TS-012_fp_4041.md"]
    meta, body = front_matter(text)
    assert meta == {
        "doc_id": "KB-TS-012",
        "title": "FP-4041 资源不存在或不可见",
        "doc_type": "troubleshooting",
        "rule_ids": "",
    }
    assert text[body:].startswith("# FP-4041")
    with pytest.raises(DataError):
        front_matter("---\ndoc_id: x\n# never closed\n")


def test_sections_follow_the_benchmark_rule() -> None:
    text = "---\ndoc_id: D\n---\n# T\n\nintro\n\n## A\n\na\n### sub\n\nstill A\n\n## B\nb\n"
    sections = benchmark_sections(text)
    assert [(s.ordinal, s.heading) for s in sections] == [(1, "T"), (2, "A"), (3, "B")]
    assert text[sections[0].start : sections[0].end] == "# T\n\nintro\n\n"
    assert "### sub" in text[sections[1].start : sections[1].end]  # only `## ` splits
    assert sections[-1].end == len(text)


def test_rebuild_matches_the_hand_written_chunking(ds: Dataset) -> None:
    chunks = benchmark_chunks(load_docs(ds))
    assert sorted((c.doc_id, c.chunk_id.split("#")[1], c.heading) for c in chunks) == CHUNKS
    assert check_chunks(ds, chunks) == []


def test_chunk_check_reports_drift(ds: Dataset) -> None:
    chunks = benchmark_chunks(load_docs(ds))
    problems = check_chunks(ds, chunks[1:])
    assert any("not rebuilt" in p for p in problems)
    renamed = [replace(c, heading="改名") for c in chunks]
    assert any("headings differ" in p for p in check_chunks(ds, renamed))
    path = ds.path("data/index/knowledge_chunks.jsonl")
    path.write_text(path.read_text() + "\n")
    assert any("not in data/index/index_manifest.json" in p for p in check_chunks(ds, chunks))


def test_release_check_finds_changed_files(ds: Dataset) -> None:
    assert ds.verify_release() == ([], {"generation_sources": "1/1 missing or changed"})
    assert ds.agent_visible_paths() == set(DOCS)
    doc = ds.path(next(iter(DOCS)))
    doc.write_text(doc.read_text().replace("FP-4041", "FP-4042"))
    assert any("sha256 differs" in p for p in ds.verify_release()[0])
    doc.unlink()
    assert any("missing" in p for p in ds.verify_release()[0])


def test_paths_cannot_escape_the_dataset(ds: Dataset) -> None:
    for bad in ("../secret", "/etc/passwd", "data\\x"):
        with pytest.raises(DataError):
            ds.path(bad)


def _chunk(doc: str, start: int, end: int, cid: str = "x") -> Chunk:
    return Chunk(cid, doc, "t", "title", "", start, end, "")


def test_evidence_map_by_span() -> None:
    gold = [_chunk("D", 0, 100, "g1"), _chunk("D", 100, 300, "g2"), _chunk("E", 0, 50, "e1")]
    emap = EvidenceMap(gold)
    assert emap.ids_for(_chunk("D", 0, 100)) == ["g1"]  # same chunking: identity
    assert emap.ids_for(_chunk("D", 100, 180)) == ["g2"]  # a window inside a section
    assert emap.ids_for(_chunk("D", 90, 300)) == ["g2"]  # 10 chars of g1 don't count
    # 70 of its 130 chars in g1; the 60 in g2 are under half of it and under half of g2
    assert emap.ids_for(_chunk("D", 30, 160)) == ["g1"]
    assert emap.ids_for(_chunk("D", 0, 300)) == ["g1", "g2"]  # a whole-doc chunk has both
    assert emap.ids_for(_chunk("E", 0, 50)) == ["e1"]  # never across docs
    assert emap.ids_for(_chunk("F", 0, 50)) == []


def test_evidence_map_tells_repeated_headings_apart(ds: Dataset) -> None:
    chunks = benchmark_chunks(load_docs(ds))
    emap = EvidenceMap(chunks)
    examples = [c for c in chunks if c.heading == "示例"]
    assert [emap.ids_for(c) for c in examples] == [["doc:KB-BR-012#c004"], ["doc:KB-BR-012#c005"]]


def test_question_is_a_whitelist_and_case_ids_are_opaque(ds: Dataset) -> None:
    # Adding a field to Question is a leakage decision: change this test on purpose.
    assert set(Question.model_fields) == {"case_id", "text"}
    with pytest.raises(ValueError):
        Question.model_validate({"case_id": "q", "text": "t", "question_type": "x"})
    questions, labels = load_benchmark(ds)
    for q in questions:
        label = labels[q.case_id]
        assert q.case_id == case_id(label.question_id)
        assert label.question_id not in q.model_dump_json()
        assert label.question_type not in q.case_id
        assert leaks(q, label) == []


def test_leaks_are_found() -> None:
    row = blueprint("Q-1", "simple_retrieval", "dev", "见 doc:KB-TS-012#c003 和 标准答案 Q-1", [])
    questions, labels = parse_rows([row])
    assert leaks(questions[0], labels[questions[0].case_id]) == [
        "Q-1",
        "canonical_answer",
        "doc:KB-TS-012#c003",
    ]


def test_benchmark_checks(ds: Dataset) -> None:
    questions, labels = load_benchmark(ds)
    ids = {c.chunk_id for c in benchmark_chunks(load_docs(ds))}
    assert check_benchmark(ds, questions, labels, ids) == []
    problems = check_benchmark(ds, questions, labels, ids - {"doc:KB-TS-012#c003"})
    assert any("gold doc ids are not chunks" in p for p in problems)
    resolution = ds.path("benchmark/resolved/evidence_resolution.jsonl")
    resolution.write_text(json.dumps({"question_id": "Q-SR-001", "resolved_evidence_ids": []}))
    assert any("differs from" in p for p in check_benchmark(ds, questions, labels, ids))


def test_a_row_without_a_field_names_its_keys() -> None:
    with pytest.raises(DataError, match="keys"):
        parse_rows([{"question_id": "Q-1"}])


def test_split_hash_and_select(ds: Dataset) -> None:
    questions, labels = load_benchmark(ds)
    assert split_hash(labels.values()) == split_hash(reversed(list(labels.values())))
    assert [labels[q.case_id].question_id for q in select(questions, labels, "dev")] == [
        q[0] for q in QUESTIONS if q[2] == "dev"
    ]
    with pytest.raises(ValueError):
        select(questions, labels, "train")
