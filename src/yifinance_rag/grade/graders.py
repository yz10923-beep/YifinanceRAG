"""Deterministic graders: each reads an `AnswerRecord` and the case's `Truth`, never the
system's claims about itself, and returns a score in [0, 1] or None where it doesn't
apply. The LLM judge (required / negative assertions) is separate, in S3 step 5.

| grader          | applies to             | 1.0 means                                          |
|-----------------|------------------------|----------------------------------------------------|
| doc_evidence    | cases with doc gold    | every gold chunk was returned by some search        |
| record_evidence | positive selectors     | every selector's record was returned (per selector) |
| negative_found  | negative selectors     | the right lookup ran and returned nothing           |
| id_recall       | cases with gold ids    | every observable gold id was returned (per id)      |
| tools           | required_tools         | every required source was used (by source, not name)|
| operations      | required_operations    | every required data operation was called            |
| instance_facts  | dynamic cases          | the answer states every atom of every trajectory    |
| as_of           | dynamic, with data calls | every data call was scoped to the case's date     |
| clarification   | all cases              | dynamic: asked for every required category;         |
|                 |                        | single-turn: did not ask                            |
| cited_grounded  | answers that cite      | every cited id was returned by a tool               |
| cited_gold      | answers that cite      | every cited id is gold                              |
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from yifinance_rag.data.corpus import Doc, doc_folder
from yifinance_rag.grade.facts import keys, states
from yifinance_rag.grade.record import SEARCH_PREFIX, AnswerRecord, ToolCall
from yifinance_rag.grade.truth import Selector, Truth

GRADERS = (
    "doc_evidence",
    "record_evidence",
    "negative_found",
    "id_recall",
    "tools",
    "operations",
    "instance_facts",
    "as_of",
    "clarification",
    "cited_grounded",
    "cited_gold",
)


@dataclass(frozen=True)
class Context:
    """Corpus facts graders need: which search source each doc belongs to."""

    doc_tool: Mapping[str, str]  # doc_id -> search_api_docs, search_business_rules, ...

    @classmethod
    def from_docs(cls, docs: Sequence[Doc]) -> Context:
        return cls({d.doc_id: SEARCH_PREFIX + doc_folder(d) for d in docs})

    def source_of(self, evidence_id: str) -> str | None:
        if not evidence_id.startswith("doc:"):
            return None
        return self.doc_tool.get(evidence_id.removeprefix("doc:").split("#", 1)[0])


@dataclass
class CaseGrade:
    case_id: str
    scores: dict[str, float | None]
    details: dict[str, object] = field(default_factory=dict)


def _share(hit: int, total: int) -> float | None:
    return hit / total if total else None


def selector_found(sel: Selector, calls: Sequence[ToolCall], observed: set[str]) -> bool:
    """A positive selector is found when its record was returned (`at_least_one`: any of
    its records). A negative one when a call to the same operation with the same
    identifying parameters succeeded and returned no rows."""
    if not sel.negative:
        ids = set(sel.evidence_ids)
        return bool(ids & observed) if sel.cardinality == "at_least_one" else ids <= observed
    return any(
        c.error is None
        and c.result_count == 0
        and c.tool == sel.tool
        and c.operation == sel.operation
        and all(str(c.args.get(k)) == v for k, v in sel.params.items())
        for c in calls
    )


def _utc_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).date().isoformat()


def grade(record: AnswerRecord, truth: Truth, ctx: Context) -> CaseGrade:
    calls = record.tool_calls
    ok_calls = [c for c in calls if c.error is None]
    observed = record.observed()
    scores: dict[str, float | None] = {}
    details: dict[str, object] = {}

    docs_hit = [e for e in truth.doc_evidence if e in observed]
    scores["doc_evidence"] = _share(len(docs_hit), len(truth.doc_evidence))

    positive = [s for s in truth.selectors if not s.negative]
    negative = [s for s in truth.selectors if s.negative]
    pos_found = [s.alias for s in positive if selector_found(s, calls, observed)]
    neg_found = [s.alias for s in negative if selector_found(s, calls, observed)]
    scores["record_evidence"] = _share(len(pos_found), len(positive))
    scores["negative_found"] = _share(len(neg_found), len(negative))
    gold = truth.gold_ids
    scores["id_recall"] = _share(sum(e in observed for e in gold), len(gold))
    details["selectors_missed"] = [
        s.alias for s in truth.selectors if s.alias not in {*pos_found, *neg_found}
    ]

    sources = {ctx.source_of(e) for e in observed} - {None}
    used = {c.tool for c in ok_calls if not c.is_search}
    tools_hit = [
        t
        for t in truth.required_tools
        if (t in sources if t.startswith(SEARCH_PREFIX) else t in used)
    ]
    scores["tools"] = _share(len(tools_hit), len(truth.required_tools))
    called = {f"{c.tool}.{c.operation}" for c in ok_calls if c.operation}
    ops_hit = [o for o in truth.required_operations if o in called]
    scores["operations"] = _share(len(ops_hit), len(truth.required_operations))
    details["tools_missed"] = [t for t in truth.required_tools if t not in tools_hit]
    details["operations_missed"] = [o for o in truth.required_operations if o not in ops_hit]

    if truth.dynamic and truth.trajectory:
        atoms = [k for t in truth.trajectory for k in keys(t.text)]
        stated = [k for k in atoms if states(record.answer, k)]
        scores["instance_facts"] = _share(len(stated), len(atoms))
        details["facts_missed"] = [k for k in atoms if k not in stated]
    else:
        scores["instance_facts"] = None

    data_calls = [c for c in ok_calls if not c.is_search]
    if truth.dynamic and data_calls:
        day = _utc_date(truth.as_of)
        scoped = [c for c in data_calls if _utc_date(c.args.get("as_of")) == day]
        scores["as_of"] = len(scoped) / len(data_calls)
    else:
        scores["as_of"] = None

    disclosed = {cat for c in record.clarifications for cat in c.disclosed}
    if truth.dynamic:
        need = truth.required_categories
        scores["clarification"] = _share(sum(c in disclosed for c in need), len(need))
        details["turns"] = len(record.clarifications)
        details["over_budget"] = len(record.clarifications) > truth.max_turns
        details["categories_missed"] = [c for c in need if c not in disclosed]
    else:
        scores["clarification"] = 0.0 if record.clarifications else 1.0

    cited = list(dict.fromkeys(record.cited))
    scores["cited_grounded"] = _share(sum(e in observed for e in cited), len(cited))
    scores["cited_gold"] = _share(sum(e in set(gold) for e in cited), len(cited))
    return CaseGrade(record.case_id, scores, details)
