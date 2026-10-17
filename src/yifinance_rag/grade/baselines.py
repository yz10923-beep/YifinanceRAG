"""Systems with no LLM, built to check the graders (the yiagent rule: an oracle that
passes, a null at chance, a wrong answer that fails), plus two that probe for loopholes.

- oracle: the dataset's own answer (`resolved_answer`, else `canonical_answer`), the
  gold evidence returned by the right calls, the planned clarifications.
- null: "不知道", no calls.
- wrong: another family's oracle record under this case's id. Another family, because
  the 4 questions of a family share their assertions and instance facts; same mode where
  one exists, so a dynamic case gets a dynamic wrong answer.
- rule-only: the canonical answer (rules, no account facts) with perfect doc evidence and
  no data calls. Must fail instance facts on dynamic cases; the course scorer passes it.
- forbidden: the oracle record plus the case's forbidden conclusion stated as its own.
  Every deterministic grader still passes it; the judge must flag it (the course scorer
  does too, but only because the claim is verbatim).
- verbose-null: the retriever's top 10 chunks pasted as the answer, no LLM.

They read the truth on purpose (except verbose-null); like the oracle retriever they
exist to test graders and are never compared with a system.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import EvidenceMap
from yifinance_rag.grade.graders import Context
from yifinance_rag.grade.record import AnswerRecord, Clarification, ToolCall
from yifinance_rag.grade.truth import Truth
from yifinance_rag.retrieval.retrievers import Retriever

NULL_ANSWER = "不知道"
SYSTEMS = ("oracle", "null", "wrong", "rule-only", "forbidden", "verbose-null")


def _doc_calls(truth: Truth, ctx: Context) -> list[ToolCall]:
    return [
        ToolCall(tool=ctx.source_of(e) or "search_documents", evidence_ids=[e], result_count=1)
        for e in truth.doc_evidence
    ]


def _data_calls(truth: Truth) -> list[ToolCall]:
    return [
        ToolCall(
            tool=s.tool,
            operation=s.operation,
            args={**s.params, "as_of": truth.as_of},
            evidence_ids=[] if s.negative else list(s.evidence_ids),
            result_count=0 if s.negative else len(s.evidence_ids),
        )
        for s in truth.selectors
    ]


def _clarifications(truth: Truth) -> list[Clarification]:
    if not truth.dynamic:
        return []
    if truth.clarification_plan:
        return [Clarification(question=q, disclosed=list(c)) for q, c in truth.clarification_plan]
    return [Clarification(question="", disclosed=list(truth.required_categories))]


def oracle(truth: Truth, ctx: Context) -> AnswerRecord:
    return AnswerRecord(
        case_id=truth.case_id,
        system="oracle",
        answer=truth.resolved_answer or truth.canonical_answer,
        cited=list(truth.gold_ids),
        tool_calls=[*_doc_calls(truth, ctx), *_data_calls(truth)],
        clarifications=_clarifications(truth),
    )


def null(truth: Truth) -> AnswerRecord:
    return AnswerRecord(case_id=truth.case_id, system="null", answer=NULL_ANSWER)


def rule_only(truth: Truth, ctx: Context) -> AnswerRecord:
    return AnswerRecord(
        case_id=truth.case_id,
        system="rule-only",
        answer=truth.canonical_answer,
        cited=list(truth.doc_evidence),
        tool_calls=_doc_calls(truth, ctx),
    )


def forbidden(truth: Truth, ctx: Context) -> AnswerRecord:
    claims = [a.split("：", 1)[-1] for a in truth.negative_assertions]
    base = oracle(truth, ctx)
    answer = base.answer + "".join(f"\n因此可以认定：{c}。" for c in claims)
    return base.model_copy(update={"system": "forbidden", "answer": answer})


def wrong_partner(truth: Truth, truths: Sequence[Truth]) -> Truth:
    """The next case (in question-id order, wrapping) from another family, preferring
    the same conversation mode. Deterministic, so a rerun grades the same pairs."""
    order = sorted(truths, key=lambda t: t.question_id)
    start = next(i for i, t in enumerate(order) if t.case_id == truth.case_id)
    rotated = order[start + 1 :] + order[:start]
    others = [t for t in rotated if t.family != truth.family]
    if not others:
        raise ValueError("the wrong baseline needs at least two scenario families")
    same_mode = [t for t in others if t.mode == truth.mode]
    return (same_mode or others)[0]


def wrong(truth: Truth, truths: Sequence[Truth], ctx: Context) -> AnswerRecord:
    partner = oracle(wrong_partner(truth, truths), ctx)
    return partner.model_copy(update={"case_id": truth.case_id, "system": "wrong"})


def verbose_null(
    question: Question, retriever: Retriever, emap: EvidenceMap, *, k: int = 10
) -> AnswerRecord:
    hits = retriever.search(question, k)
    ids = list(dict.fromkeys(e for h in hits for e in emap.ids_for(h.chunk)))
    return AnswerRecord(
        case_id=question.case_id,
        system="verbose-null",
        answer="\n\n".join(h.chunk.text for h in hits),
        tool_calls=[
            ToolCall(
                tool="search_documents",
                args={"query": question.text, "k": k},
                evidence_ids=ids,
                result_count=len(hits),
            )
        ],
    )


def build(
    system: str,
    questions: Sequence[Question],
    truths: Mapping[str, Truth],
    ctx: Context,
    *,
    retriever: Retriever | None = None,
    emap: EvidenceMap | None = None,
) -> list[AnswerRecord]:
    picked = [truths[q.case_id] for q in questions]
    makers: dict[str, Callable[[Truth], AnswerRecord]] = {
        "oracle": lambda t: oracle(t, ctx),
        "null": null,
        "wrong": lambda t: wrong(t, picked, ctx),  # partner from the same split
        "rule-only": lambda t: rule_only(t, ctx),
        "forbidden": lambda t: forbidden(t, ctx),
    }
    if system == "verbose-null":
        if retriever is None or emap is None:
            raise ValueError("verbose-null needs a retriever and an evidence map")
        return [verbose_null(q, retriever, emap) for q in questions]
    if system not in makers:
        raise ValueError(f"unknown system {system!r}; one of {SYSTEMS}")
    return [makers[system](t) for t in picked]
