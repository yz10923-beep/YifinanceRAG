"""The course's benchmark scorer (KamaAgenticRAG `scripts/benchmark/run.py` and
`evidence.py`, MIT), reimplemented on our answer records so our numbers compare with
the course's. Its rules are kept as they are, weaknesses included (ROADMAP S3 step 1,
finding 8); it is a second score, never our grade.

One difference: the course maps its own passages to gold chunks by heading; our records
carry gold chunk ids already (mapped by span), so doc evidence is read from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from yifinance_rag.grade.record import AnswerRecord
from yifinance_rag.grade.truth import Truth

# The course's record-id pattern. It has no `re_` (refunds), `in_` (invoices), `key_` or
# `pc_`, so those gold ids can never be found; kept as is, on purpose.
IDENTIFIER = re.compile(r"\b(?:acct|pay|evt|del|we|req|inc|att|cus|sub|inv|ref)_[A-Za-z0-9_]+\b")


def _squash(text: str) -> str:
    return "".join(text.split())


def supported(answer: str, assertion: str) -> bool:
    """Substring, or at least half of the assertion's character bigrams in the answer."""
    text, target = _squash(answer), _squash(assertion)
    if not target:
        return False
    if target in text:
        return True
    grams = {target[i : i + 2] for i in range(len(target) - 1)}
    return bool(grams) and sum(g in text for g in grams) / len(grams) >= 0.5


def violated(answer: str, forbidden: str) -> bool:
    """Only the forbidden conclusion verbatim counts; rewordings pass."""
    return _squash(forbidden) in _squash(answer)


@dataclass(frozen=True)
class CourseScore:
    task_pass: bool
    evidence_pass: bool
    assertion_hit: int
    assertion_total: int
    negative_hit: int
    doc_hit: int
    doc_total: int
    record_hit: int
    record_total: int


def course_score(record: AnswerRecord, truth: Truth) -> CourseScore:
    observed = record.observed()
    docs = list(truth.doc_evidence)
    doc_hit = sum(e in observed for e in docs)
    # Records: ids its pattern finds in the gold ids (negatives excluded), against ids it
    # finds anywhere in what tools returned.
    wanted = sorted(
        {
            m
            for s in truth.selectors
            if not s.negative
            for e in s.evidence_ids
            for m in IDENTIFIER.findall(e)
        }
    )
    seen = {m for e in observed for m in IDENTIFIER.findall(e)}
    forbidden = [a.split("：", 1)[-1] for a in truth.negative_assertions]
    hits = sum(supported(record.answer, text) for _, text in truth.required_assertions)
    breaches = sum(violated(record.answer, f) for f in forbidden)
    return CourseScore(
        task_pass=hits == len(truth.required_assertions) and breaches == 0,
        evidence_pass=bool(docs) and doc_hit == len(docs),
        assertion_hit=hits,
        assertion_total=len(truth.required_assertions),
        negative_hit=breaches,
        doc_hit=doc_hit,
        doc_total=len(docs),
        record_hit=sum(m in seen for m in wanted),
        record_total=len(wanted),
    )
