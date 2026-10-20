"""S4b on the real dataset, no model: a fake model that makes exactly the gold lookups.
Through the real agent harness, tools and graders, every dynamic question must get full
record and negative evidence, and nothing the model is sent may carry an evidence id or
the benchmark's bookkeeping."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.fakes import ScriptedProvider, tool_response
from yifinance_rag.answer.agent import ASK, SUBMIT, Agent, AgentConfig, ResponseCache, SearchTool
from yifinance_rag.answer.user import SimulatedUser
from yifinance_rag.data.benchmark import load_benchmark
from yifinance_rag.data.corpus import EvidenceMap, benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.grade.graders import Context, grade
from yifinance_rag.grade.truth import load_truths
from yifinance_rag.llm.types import LLMError, LLMResponse
from yifinance_rag.retrieval.retrievers import BM25Retriever
from yifinance_rag.tools.data import DataStore, DataTools, iso_ms

pytestmark = pytest.mark.dataset

LEAK = re.compile(
    r"evidence_id|(?:doc|sql|runtime|negative):[A-Za-z]|SC-\d{3}|SF-\d{3}|FP-BM-|run_id|scenario_id"
)
ASK_ALL = "请提供：具体问题和现象、账户与环境、发生时间、接口版本、相关编号（ID）。"


def test_oracle_lookups_through_the_harness_get_full_evidence(ds: Dataset, tmp_path: Path) -> None:
    questions, labels = load_benchmark(ds)
    truths = load_truths(ds, labels)
    docs = load_docs(ds)
    chunks = benchmark_chunks(docs)
    ctx = Context.from_docs(docs)
    store = DataStore(ds)
    data = DataTools(store)
    bm25 = BM25Retriever(chunks)
    cache = ResponseCache(tmp_path / "agent.sqlite")
    leaks: list[str] = []
    short: list[str] = []
    for q in questions:
        t = truths[q.case_id]
        plan: list[LLMResponse | LLMError] = (
            [tool_response(("a", ASK, {"question": ASK_ALL}))] if t.dynamic else []
        )
        for n, s in enumerate(t.selectors):
            args = {"operation": s.operation, **s.params, "as_of": t.as_of}
            plan.append(tool_response((f"s{n}", s.tool, args)))
        plan.append(tool_response(("q", "search_documents", {"query": q.text})))
        plan.append(tool_response(("z", SUBMIT, {"answer": "见记录。", "citations": ["R1", "D1"]})))
        provider = ScriptedProvider(plan)
        agent = Agent(
            lambda p=provider: p,  # type: ignore[misc]
            model="oracle-fake",
            options="{}",
            search=[SearchTool("search_documents", "全部文档", bm25)],
            emap=EvidenceMap(chunks),
            data=data,
            cache=cache,
            now=iso_ms(store.freeze_at),
            config=AgentConfig(max_steps=len(plan) + 1),
        )
        conversation = labels[q.case_id].raw["conversation"]
        assert isinstance(conversation, dict)
        user = SimulatedUser.for_conversation(ds, conversation)
        record = agent.answer(q, user, int(conversation.get("max_clarification_turns") or 0))
        agent.close()
        sent = json.dumps(
            [[r.system, r.messages, r.tools] for r in provider.requests], ensure_ascii=False
        )
        leaks += [f"{t.question_id}: {m}" for m in LEAK.findall(sent)]
        scores = grade(record, t, ctx).scores
        if any(
            scores[k] not in (None, 1.0)
            for k in ("record_evidence", "negative_found", "operations")
        ):
            short.append(t.question_id)
    cache.close()
    store.close()
    assert leaks == [] and short == []
