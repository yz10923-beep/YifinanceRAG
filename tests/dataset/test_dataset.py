"""Checks on the real dataset (course material; on the VM only). `make data-test`."""

from __future__ import annotations

import pytest

from yifinance_rag.data import pins
from yifinance_rag.data.benchmark import Question, leaks, load_benchmark, select
from yifinance_rag.data.check import check_dataset
from yifinance_rag.data.corpus import KNOWLEDGE_DIR, benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.grade.baselines import build
from yifinance_rag.grade.graders import Context
from yifinance_rag.grade.report import Graded, grade_all, selftest
from yifinance_rag.grade.truth import load_truths

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


def test_graders_pass_their_selftest_on_every_real_question(ds: Dataset) -> None:
    """S3: oracle 1.0, null 0, wrong and rule-only fail, on all 300 questions. Grading
    the test split's labels here runs no system on it; it only checks the graders."""
    questions, labels = load_benchmark(ds)
    truths = load_truths(ds, labels)
    ctx = Context.from_docs(load_docs(ds))
    by_system: dict[str, list[Graded]] = {}
    for split in ("dev", "test"):
        picked = select(questions, labels, split)
        for system in ("oracle", "null", "wrong", "rule-only"):
            by_system.setdefault(system, []).extend(
                grade_all(build(system, picked, truths, ctx), truths, ctx)
            )
    assert selftest(by_system) == []
    # ROADMAP S3 step 1, finding 8: the course scorer passes every rule-only answer.
    rule = [g for g in by_system["rule-only"] if g.truth.dynamic]
    assert len(rule) == 180 and all(g.course.task_pass for g in rule)
