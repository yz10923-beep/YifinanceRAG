"""The 300 benchmark questions, split into what a retriever or agent may see and what
only graders read.

`Question` is the whole agent-visible view: an opaque case id and the opening message.
Everything else in a blueprint (type, split, gold evidence, ground truth, the simulated
user's hidden facts) goes to `Label`. Case ids are opaque because question ids may encode
the type.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, ValidationError

from yifinance_rag.data.dataset import DataError, Dataset, sha256_text

BLUEPRINTS = "benchmark/resolved/resolved_blueprints.jsonl"
RESOLUTION = "benchmark/resolved/evidence_resolution.jsonl"
TYPES = (
    "simple_retrieval",
    "multi_document",
    "multi_hop",
    "temporal_versioned",
    "multi_source",
    "tool_use_diagnostic",
)
SPLITS = ("dev", "test")
_DOC_EVIDENCE = re.compile(r"doc:[A-Za-z0-9-]+#c\d{3}")


class Question(BaseModel):
    """Everything a retriever or agent sees about a case. Adding a field is a leakage
    decision: the whitelist test fails until it is made on purpose."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    text: str


@dataclass(frozen=True)
class Label:
    """Grader-only. Never passed to a retriever or agent."""

    case_id: str
    question_id: str
    question_type: str
    split: str
    conversation_mode: str
    evidence_ids: tuple[str, ...]
    raw: Mapping[str, object] = field(repr=False)

    @property
    def doc_evidence(self) -> tuple[str, ...]:
        return tuple(e for e in self.evidence_ids if e.startswith("doc:"))


class _Row(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_id: str
    question: str
    question_type: str
    split: str
    conversation: dict[str, object]
    resolved_evidence_ids: list[str]


def case_id(question_id: str) -> str:
    return "q-" + sha256_text(question_id)[:10]


def parse_rows(rows: Iterable[Mapping[str, object]]) -> tuple[list[Question], dict[str, Label]]:
    questions: list[Question] = []
    labels: dict[str, Label] = {}
    for n, raw in enumerate(rows, start=1):
        try:
            row = _Row.model_validate(raw)
        except ValidationError as e:
            raise DataError(f"{BLUEPRINTS} row {n}: keys {sorted(raw)}: {e}") from e
        cid = case_id(row.question_id)
        if cid in labels:
            raise DataError(f"{BLUEPRINTS}: duplicate question_id {row.question_id}")
        questions.append(Question(case_id=cid, text=row.question))
        labels[cid] = Label(
            case_id=cid,
            question_id=row.question_id,
            question_type=row.question_type,
            split=row.split,
            conversation_mode=str(row.conversation.get("mode", "")),
            evidence_ids=tuple(row.resolved_evidence_ids),
            raw=dict(raw),
        )
    return questions, labels


def load_benchmark(ds: Dataset) -> tuple[list[Question], dict[str, Label]]:
    return parse_rows(ds.read_jsonl(BLUEPRINTS))


def with_user_facts(question: Question, label: Label) -> Question:
    """The opening message plus every fact the simulated user would reveal when asked.

    An upper-bound condition, like the oracle: it reads the label (grader-side), so it
    measures how much a perfectly asking agent could gain, never what an agent sees.
    Single-turn questions are returned unchanged."""
    if label.conversation_mode != "progressive_disclosure":
        return question
    conversation = label.raw.get("conversation")
    facts = conversation.get("available_user_facts") if isinstance(conversation, dict) else []
    values = [
        str(f["user_value"]) for f in facts or [] if isinstance(f, dict) and "user_value" in f
    ]
    return Question(case_id=question.case_id, text="\n".join([question.text, *values]))


def split_hash(labels: Iterable[Label]) -> str:
    """Identifies the split: which question is in which split. Pinned in data/pins.py."""
    lines = sorted(f"{lab.question_id}\t{lab.split}" for lab in labels)
    return sha256_text("\n".join(lines))


def select(
    questions: Sequence[Question], labels: Mapping[str, Label], split: str
) -> list[Question]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    return [q for q in questions if labels[q.case_id].split == split]


def leaks(question: Question, label: Label) -> list[str]:
    """Ground truth that shows up in what the agent sees: gold evidence ids (whole), a
    doc evidence pattern, the question id, or the canonical answer."""
    seen = question.model_dump_json()
    found = [e for e in label.evidence_ids if e in seen]
    found += _DOC_EVIDENCE.findall(seen)
    if label.question_id in seen:
        found.append(label.question_id)
    truth = label.raw.get("ground_truth")
    answer = truth.get("canonical_answer") if isinstance(truth, dict) else None
    if isinstance(answer, str) and answer.strip() and answer.strip() in seen:
        found.append("canonical_answer")
    return sorted(set(found))


def check_benchmark(
    ds: Dataset,
    questions: Sequence[Question],
    labels: Mapping[str, Label],
    chunk_ids: set[str],
    *,
    show: int = 5,
) -> list[str]:
    problems: list[str] = []
    if bad := sorted({lab.question_type for lab in labels.values()} - set(TYPES)):
        problems.append(f"unknown question types: {bad}")
    if bad := sorted({lab.split for lab in labels.values()} - set(SPLITS)):
        problems.append(f"unknown splits: {bad}")
    unknown = sorted({e for lab in labels.values() for e in lab.doc_evidence if e not in chunk_ids})
    if unknown:
        problems.append(f"{len(unknown)} gold doc ids are not chunks, e.g. {unknown[:show]}")
    leaky = [
        f"{labels[q.case_id].question_id}: {found}"
        for q in questions
        if (found := leaks(q, labels[q.case_id]))
    ]
    if leaky:
        problems.append(f"{len(leaky)} questions show ground truth, e.g. {leaky[:show]}")
    if ds.path(RESOLUTION).is_file():
        resolved = {
            str(r["question_id"]): r.get("resolved_evidence_ids") for r in ds.read_jsonl(RESOLUTION)
        }
        differ = [
            lab.question_id
            for lab in labels.values()
            if resolved.get(lab.question_id) != list(lab.evidence_ids)
        ]
        if differ:
            problems.append(
                f"{len(differ)} questions' evidence differs from {RESOLUTION}, e.g. {differ[:show]}"
            )
    return problems
