"""What graders know about a case: the label fields, typed, plus the evidence selectors
from `evidence_resolution.jsonl`. Grader-only, like `Label`: never passed to a system.

Facts this relies on (checked on the real data, 2026-10-17): every gold id is a doc chunk
or one of a selector's ids; each dynamic question has one trajectory assertion per
selector; negative selectors carry a hash id no agent can produce, so they are matched by
operation and parameters instead (ROADMAP S3 step 1, findings 4-6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from yifinance_rag.data.benchmark import RESOLUTION, Label
from yifinance_rag.data.dataset import DataError, Dataset

# Payload keys that are not identifying parameters: the call's time scope is graded by
# the as_of check, and `window` is symbolic ("scenario_observation_window").
_NOT_PARAMS = frozenset({"operation", "as_of", "window"})


@dataclass(frozen=True)
class Selector:
    alias: str
    tool: str
    operation: str
    cardinality: str  # exactly_one | at_least_one
    negative: bool  # the right finding is "no such record"
    evidence_ids: tuple[str, ...]
    params: Mapping[str, str]  # identifying parameters, e.g. account_id, payment_id

    @property
    def operation_name(self) -> str:
        return f"{self.tool}.{self.operation}"


@dataclass(frozen=True)
class Trajectory:
    assertion_id: str
    alias: str
    text: str


@dataclass(frozen=True)
class Truth:
    case_id: str
    question_id: str
    question_type: str
    split: str
    mode: str
    family: str
    as_of: str
    canonical_answer: str
    resolved_answer: str | None
    required_assertions: tuple[tuple[str, str], ...]  # (assertion_id, text)
    negative_assertions: tuple[str, ...]  # "不得认定：X"
    trajectory: tuple[Trajectory, ...]
    required_tools: tuple[str, ...]
    required_operations: tuple[str, ...]
    required_categories: tuple[str, ...]
    max_turns: int
    clarification_plan: tuple[tuple[str, tuple[str, ...]], ...]  # (question, categories)
    doc_evidence: tuple[str, ...]
    selectors: tuple[Selector, ...]

    @property
    def dynamic(self) -> bool:
        """Needs the account's records (and asking): the 180 progressive questions."""
        return self.mode == "progressive_disclosure"

    @property
    def gold_ids(self) -> tuple[str, ...]:
        """Every gold id an agent can observe: doc chunks and positive record ids."""
        records = [e for s in self.selectors if not s.negative for e in s.evidence_ids]
        return tuple(dict.fromkeys([*self.doc_evidence, *records]))


class _Assertion(BaseModel):
    model_config = ConfigDict(extra="allow")
    assertion_id: str
    text: str


class _Truth(BaseModel):
    model_config = ConfigDict(extra="allow")
    canonical_answer: str
    required_assertions: list[_Assertion]
    negative_assertions: list[str] = []


class _Trajectory(BaseModel):
    model_config = ConfigDict(extra="allow")
    assertion_id: str
    selector_alias: str
    text: str


class _Step(BaseModel):
    model_config = ConfigDict(extra="allow")
    assistant_question: str
    categories: list[str]


class _Conversation(BaseModel):
    model_config = ConfigDict(extra="allow")
    required_clarification_categories: list[str] = []
    max_clarification_turns: int = 0
    clarification_plan: list[_Step] = []


class _Blueprint(BaseModel):
    model_config = ConfigDict(extra="allow")
    scenario_family_id: str
    as_of: str
    ground_truth: _Truth
    conversation: _Conversation
    resolved_answer: str | None = None
    resolved_trajectory_assertions: list[_Trajectory] = []
    required_tools: list[str] = []
    required_operations: list[str] = []


class _Selector(BaseModel):
    model_config = ConfigDict(extra="allow")
    alias: str
    tool: str
    operation: str
    cardinality: str
    evidence_mode: str
    evidence_ids: list[str]
    normalized_payload: dict[str, object]


class _Resolution(BaseModel):
    model_config = ConfigDict(extra="allow")
    question_id: str
    selector_results: list[_Selector] = []


def selector(raw: _Selector) -> Selector:
    params = {k: str(v) for k, v in raw.normalized_payload.items() if k not in _NOT_PARAMS}
    return Selector(
        alias=raw.alias,
        tool=raw.tool,
        operation=raw.operation,
        cardinality=raw.cardinality,
        negative=raw.evidence_mode == "negative_assertion",
        evidence_ids=tuple(raw.evidence_ids),
        params=params,
    )


def truth(label: Label, selectors: tuple[Selector, ...]) -> Truth:
    try:
        bp = _Blueprint.model_validate(label.raw)
    except ValidationError as e:
        raise DataError(f"{label.question_id}: blueprint fields for grading: {e}") from e
    gt, talk = bp.ground_truth, bp.conversation
    return Truth(
        case_id=label.case_id,
        question_id=label.question_id,
        question_type=label.question_type,
        split=label.split,
        mode=label.conversation_mode,
        family=bp.scenario_family_id,
        as_of=bp.as_of,
        canonical_answer=gt.canonical_answer,
        resolved_answer=bp.resolved_answer,
        required_assertions=tuple((a.assertion_id, a.text) for a in gt.required_assertions),
        negative_assertions=tuple(gt.negative_assertions),
        trajectory=tuple(
            Trajectory(t.assertion_id, t.selector_alias, t.text)
            for t in bp.resolved_trajectory_assertions
        ),
        required_tools=tuple(bp.required_tools),
        required_operations=tuple(bp.required_operations),
        required_categories=tuple(talk.required_clarification_categories),
        max_turns=talk.max_clarification_turns,
        clarification_plan=tuple(
            (s.assistant_question, tuple(s.categories)) for s in talk.clarification_plan
        ),
        doc_evidence=tuple(dict.fromkeys(label.doc_evidence)),
        selectors=selectors,
    )


def load_truths(ds: Dataset, labels: Mapping[str, Label]) -> dict[str, Truth]:
    """Truths keyed by case id. Fails if a gold id is neither a doc chunk nor a selector's
    id: then the per-selector evidence grader would silently miss evidence."""
    by_qid: dict[str, tuple[Selector, ...]] = {}
    for n, row in enumerate(ds.read_jsonl(RESOLUTION), start=1):
        try:
            res = _Resolution.model_validate(row)
        except ValidationError as e:
            raise DataError(f"{RESOLUTION} row {n}: {e}") from e
        by_qid[res.question_id] = tuple(selector(s) for s in res.selector_results)
    out: dict[str, Truth] = {}
    for cid, label in labels.items():
        t = truth(label, by_qid.get(label.question_id, ()))
        covered = {*t.doc_evidence, *(e for s in t.selectors for e in s.evidence_ids)}
        if missing := [e for e in label.evidence_ids if e not in covered]:
            raise DataError(f"{label.question_id}: gold ids outside docs and selectors: {missing}")
        out[cid] = t
    return out
