"""Checks on the real dataset (course material; on the VM only). `make data-test`."""

from __future__ import annotations

import pytest

from yifinance_rag.data import pins
from yifinance_rag.data.benchmark import Question, leaks, load_benchmark
from yifinance_rag.data.check import check_dataset
from yifinance_rag.data.corpus import KNOWLEDGE_DIR, benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset

pytestmark = pytest.mark.dataset


def test_the_dataset_matches_every_pin(ds: Dataset) -> None:
    report = check_dataset(ds)
    assert report.problems == []
    assert report.facts["split_sha256"] == pins.SPLIT_SHA256


def test_the_index_is_built_from_published_knowledge_only(ds: Dataset) -> None:
    docs = load_docs(ds)
    assert docs and all(d.path.startswith(KNOWLEDGE_DIR + "/") for d in docs)
    for chunk in benchmark_chunks(docs):
        assert "event_journal" not in chunk.text


def test_no_question_shows_ground_truth(ds: Dataset) -> None:
    questions, labels = load_benchmark(ds)
    assert set(Question.model_fields) == {"case_id", "text"}
    leaky = {labels[q.case_id].question_id: leaks(q, labels[q.case_id]) for q in questions}
    assert {k: v for k, v in leaky.items() if v} == {}
