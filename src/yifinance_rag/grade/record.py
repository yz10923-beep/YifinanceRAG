"""The answer record: what a system writes for one case and what every grader reads.

The same format for the no-LLM baselines, the one-call RAG baseline and the S4 agent.
`answer` and `cited` are the system's own output. `tool_calls` and `clarifications`
are written by the harness that ran the tools and the simulated user, never parsed out
of the model's text, so graders can trust them the way they trust a trace.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

SEARCH_PREFIX = "search_"


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str  # the name the system exposes: search_api_docs, search_documents, query_payments
    operation: str | None = None  # data tools: get_payment, list_requests, ...
    args: dict[str, object] = Field(default_factory=dict)
    # What the call returned, in the benchmark's vocabulary: gold-chunk ids of returned
    # passages (mapped by EvidenceMap), and the dataset's evidence_id of each returned row.
    evidence_ids: list[str] = Field(default_factory=list)
    result_count: int = 0
    error: str | None = None  # a refused or failed call: it observed nothing

    @property
    def is_search(self) -> bool:
        return self.tool.startswith(SEARCH_PREFIX)


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str  # what the system asked the user
    disclosed: list[str] = Field(default_factory=list)  # fact categories the user revealed


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    system: str
    answer: str
    cited: list[str] = Field(default_factory=list)  # evidence ids the answer cites
    tool_calls: list[ToolCall] = Field(default_factory=list)
    clarifications: list[Clarification] = Field(default_factory=list)
    # tokens, $, seconds, steps; `stop` says how the run ended (submitted, text, budget)
    usage: dict[str, float | int | str | None] = Field(default_factory=dict)

    def observed(self) -> set[str]:
        """Every evidence id some successful tool call returned."""
        return {e for c in self.tool_calls if c.error is None for e in c.evidence_ids}


def write_records(path: Path, records: Iterable[AnswerRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")


def read_records(path: Path) -> list[AnswerRecord]:
    lines: Sequence[str] = path.read_text(encoding="utf-8").splitlines()
    return [AnswerRecord.model_validate_json(line) for line in lines if line.strip()]
