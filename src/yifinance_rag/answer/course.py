"""Answer records from a course-stage run (`scripts/course_stages.py`), so the course's
s1-s4 are graded by the same graders as our systems.

What maps exactly, and so is comparable:
- the answer text (judge, instance facts, the course scorer);
- retrieved passages, mapped to gold chunk ids by character span (`EvidenceMap`, as for
  `course-dense`);
- returned records: the course's log rows carry the dataset's own `evidence_id`, and its
  four SQL relations are keyed by the same ids ours are (`sql:payment:<payment_id>`, ...);
- the simulated user's disclosures, by fact category.

What does not: tool and operation names (the course registers relations, the benchmark
entities), negative selectors (they match our operation and parameters) and `as_of` (the
course has no as-of parameter). Those graders run but their columns are not comparable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from yifinance_rag.data.corpus import Chunk, EvidenceMap
from yifinance_rag.grade.record import AnswerRecord, Clarification, ToolCall
from yifinance_rag.llm.openai_compat import to_usage
from yifinance_rag.llm.pricing import cost_usd

FORMAT = 1
# The course's SQL relations -> (our evidence kind, key column).
SQL_KEYS = {
    "payments": ("payment", "payment_id"),
    "events": ("event", "event_id"),
    "webhook_deliveries": ("webhook_delivery", "delivery_id"),
    "webhook_endpoints": ("webhook_endpoint", "endpoint_id"),
}
SEARCH = {"search": "search_documents", "read": "search_read_document"}
NOT_COMPARABLE = ("tools", "operations", "negative_found", "as_of")


class CourseRunError(ValueError):
    pass


def row_evidence(kind: str, name: str, row: Mapping[str, Any]) -> str | None:
    if kind == "logs":
        value = row.get("evidence_id")
        return str(value) if value else None
    if name not in SQL_KEYS:
        return None
    ours, key = SQL_KEYS[name]
    value = row.get(key) or row.get(f"d.{key}")
    return f"sql:{ours}:{value}" if value else None


def _passage(p: Mapping[str, Any]) -> Chunk | None:
    if not p.get("doc_id") or p.get("start_char") is None or p.get("end_char") is None:
        return None
    return Chunk(
        str(p.get("id") or ""),
        str(p["doc_id"]),
        "",
        "",
        "",
        int(p["start_char"]),
        int(p["end_char"]),
        "",
    )


def tool_calls(calls: Sequence[Mapping[str, Any]], emap: EvidenceMap) -> list[ToolCall]:
    out: list[ToolCall] = []
    for c in calls:
        if c["kind"] in SEARCH:
            chunks = [x for p in c.get("passages") or [] if (x := _passage(p))]
            ids = list(dict.fromkeys(e for ch in chunks for e in emap.ids_for(ch)))
            out.append(ToolCall(tool=SEARCH[c["kind"]], evidence_ids=ids, result_count=len(chunks)))
            continue
        if c["kind"] != "data":
            raise CourseRunError(f"unknown call kind {c['kind']!r}")
        request = c.get("request") or {}
        kind, name = str(request.get("kind")), str(request.get("name"))
        args = dict(request.get("parameters") or request.get("filters") or {})
        for k in ("start", "end"):
            if request.get(k):
                args[k] = request[k]
        rows = c.get("rows") or []
        found = [e for r in rows if (e := row_evidence(kind, name, r))]
        out.append(
            ToolCall(
                tool=name,
                operation=kind,
                args=args,
                evidence_ids=list(dict.fromkeys(found)),
                result_count=int(c.get("result_count") or 0),
                error=c.get("error"),
            )
        )
    return out


def usage(entries: Sequence[Mapping[str, Any]], model: str) -> dict[str, float | int | None]:
    total: dict[str, float | int | None] = {
        "llm_calls": len(entries),
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }
    for e in entries:
        u = to_usage(e.get("usage"))
        for key, value in (
            ("input_tokens", u.input_tokens),
            ("output_tokens", u.output_tokens),
            ("cache_read_tokens", u.cache_read_input_tokens),
        ):
            total[key] = int(total[key] or 0) + value
        cost = cost_usd(str(e.get("model") or model), u.model_dump())
        prior = total["cost_usd"]
        total["cost_usd"] = None if cost is None or prior is None else float(prior) + cost
    return total


def to_record(
    row: Mapping[str, Any], emap: EvidenceMap, case_ids: Mapping[str, str], system: str
) -> AnswerRecord:
    if row.get("format") != FORMAT:
        raise CourseRunError(f"export format {row.get('format')!r}, expected {FORMAT}")
    qid = str(row["question_id"])
    if qid not in case_ids:
        raise CourseRunError(f"unknown question {qid}")
    model = str(row.get("model") or "")
    course = row.get("course_row") or {}
    return AnswerRecord(
        case_id=case_ids[qid],
        system=system,
        answer=str(row["answer"]),
        tool_calls=tool_calls(row.get("calls") or [], emap),
        clarifications=[
            Clarification(
                question=f"[{x.get('signal')}] {', '.join(x.get('asked') or [])}",
                disclosed=list(x.get("disclosed") or []),
            )
            for x in row.get("exchanges") or []
        ],
        usage={
            **usage(row.get("usage") or [], model),
            "seconds": float(row.get("seconds") or 0.0),
            "stop": "answered",
            "deviations": ",".join(row.get("deviations") or []) or None,
            "course_own_task_pass": int(bool(course.get("task_pass"))),
        },
    )


def read_export(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
