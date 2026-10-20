from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.fakes import ScriptedProvider, text_response, tool_response
from tests.synthetic import write_canonical, write_dataset, write_simulator
from yifinance_rag import cli
from yifinance_rag.answer.agent import (
    ASK,
    LAST_STEP,
    NO_MORE_QUESTIONS,
    NUDGE,
    SUBMIT,
    Agent,
    AgentConfig,
    FolderFilter,
    ResponseCache,
    SearchTool,
)
from yifinance_rag.answer.user import SimulatedUser
from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import EvidenceMap, benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.grade.graders import Context
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Message,
    TextCallback,
    ToolSpec,
)
from yifinance_rag.retrieval.retrievers import BM25Retriever
from yifinance_rag.tools.data import DataStore, DataTools
from yifinance_rag.trace.span import Span
from yifinance_rag.trace.tracer import Tracer

Q = Question(case_id="q-1", text="后台能看到一笔支付，但服务里查询提示找不到。")
SCOPE = {"account_id": "acct_a", "environment": "production"}


@dataclass
class FakeUser:
    asked: list[str] = field(default_factory=list)

    def answer(self, question: str) -> tuple[str, list[str]]:
        self.asked.append(question)
        return "账户 acct_a，支付编号 pay_a_1，正式环境。", ["identifiers", "account_context"]


class Sink:
    def __init__(self) -> None:
        self.spans: list[Span] = []

    def write(self, span: Span) -> None:
        self.spans.append(span)


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    root = write_dataset(tmp_path / "flowpay-dataset", graded=True)
    write_canonical(root)
    return Dataset(write_simulator(root))


@pytest.fixture
def make(ds: Dataset, tmp_path: Path) -> Iterator[Any]:
    chunks = benchmark_chunks(load_docs(ds))
    store = DataStore(ds)
    agents: list[Agent] = []

    def build(provider: LLMProvider, **config: Any) -> Agent:
        agent = Agent(
            lambda: provider,
            model="deepseek-v4-flash",
            options="{}",
            search=[SearchTool("search_documents", "全部文档", BM25Retriever(chunks))],
            emap=EvidenceMap(chunks),
            data=DataTools(store),
            cache=ResponseCache(tmp_path / "agent.sqlite"),
            now="2026-08-27T12:00:00.000Z",
            config=AgentConfig(**config),
            retry=RetryPolicy(base_s=0.001, max_delay_s=0.001),
        )
        agents.append(agent)
        return agent

    yield build
    for a in agents:
        a.close()
        a.cache.close()
    store.close()


def _results(provider: ScriptedProvider, n: int) -> list[dict[str, Any]]:
    """The tool_result blocks sent back after the n-th model call."""
    content = provider.requests[n + 1].messages[-1]["content"]
    assert isinstance(content, list)
    return content


def test_a_run_asks_looks_up_searches_and_submits(make: Any) -> None:
    provider = ScriptedProvider(
        [
            tool_response(("c1", ASK, {"question": "请提供账户编号和支付编号"})),
            tool_response(
                (
                    "c2",
                    "query_payments",
                    {"operation": "get_payment", "payment_id": "pay_a_1", **SCOPE},
                )
            ),
            tool_response(("c3", "search_documents", {"query": "FP-4041 资源不存在"})),
            tool_response(
                (
                    "c4",
                    SUBMIT,
                    {"answer": "支付 pay_a_1 存在且已成功。", "citations": ["R1", "D1", "X9"]},
                )
            ),
        ]
    )
    user = FakeUser()
    record = make(provider).answer(Q, user, turn_budget=2)
    assert record.answer == "支付 pay_a_1 存在且已成功。" and record.usage["stop"] == "submitted"
    assert [c.disclosed for c in record.clarifications] == [["identifiers", "account_context"]]
    assert [c.tool for c in record.tool_calls] == ["query_payments", "search_documents"]  # no ask
    data, search = record.tool_calls
    assert data.evidence_ids == ["sql:payment:pay_a_1"] and data.operation == "get_payment"
    assert search.evidence_ids and search.args == {"query": "FP-4041 资源不存在"}
    assert record.cited[0] == "sql:payment:pay_a_1" and record.cited[1] == search.evidence_ids[0]
    assert record.usage["unknown_citations"] == 1 and record.usage["llm_calls"] == 4
    # every tool_use got a tool_result, and nothing the model saw is an evidence id
    sent = json.dumps([r.messages for r in provider.requests], ensure_ascii=False)
    for marker in ("evidence_id", "sql:payment", "doc:KB-", "runtime:"):
        assert marker not in sent
    assert _results(provider, 1)[0]["tool_use_id"] == "c2"
    assert json.loads(_results(provider, 1)[0]["content"])["rows"][0]["ref"] == "R1"
    assert "当前时间是 2026-08-27" in provider.requests[0].system


def test_the_last_step_offers_only_submit(make: Any) -> None:
    provider = ScriptedProvider(
        [
            tool_response(("c1", "query_payments", {"operation": "list_payments", **SCOPE})),
            tool_response(("c2", SUBMIT, {"answer": "只查到一笔支付。"})),
        ]
    )
    record = make(provider, max_steps=2).answer(Q, None, turn_budget=0)
    assert [t["name"] for t in provider.requests[1].tools] == [SUBMIT]
    last = provider.requests[1].messages[-1]["content"]
    assert [b["type"] for b in last] == ["tool_result", "text"] and last[-1]["text"] == LAST_STEP
    assert record.answer == "只查到一笔支付。"


def test_refusals_are_observations_and_the_run_goes_on(make: Any) -> None:
    provider = ScriptedProvider(
        [
            tool_response(
                ("c1", ASK, {"question": "账户编号是什么？"}),
                ("c2", "query_payments", {"operation": "get_payment", "payment_id": "pay_a_1"}),
            ),
            tool_response(("c3", SUBMIT, {"answer": "缺少账户信息，无法判断。"})),
        ]
    )
    record = make(provider).answer(Q, FakeUser(), turn_budget=0)  # single-turn: no asking
    ask, lookup = _results(provider, 0)
    assert ask["is_error"] and ask["content"] == NO_MORE_QUESTIONS
    assert lookup["is_error"] and "account_id and environment" in lookup["content"]
    assert record.clarifications == [] and all(c.error for c in record.tool_calls)


def test_text_without_tools_is_nudged_once_then_taken(make: Any) -> None:
    provider = ScriptedProvider(
        [text_response("我觉得是权限问题。"), text_response("结论：权限问题。")]
    )
    record = make(provider).answer(Q, None, turn_budget=0)
    assert provider.requests[1].messages[-1]["content"] == NUDGE
    assert record.answer == "结论：权限问题。" and record.usage["stop"] == "text"


def test_a_rerun_is_served_from_the_cache(make: Any) -> None:
    script = [
        tool_response(("c1", "query_accounts", {"operation": "get_account", **SCOPE})),
        tool_response(("c2", SUBMIT, {"answer": "账户套餐为 Starter。", "citations": ["R1"]})),
    ]
    first = make(ScriptedProvider(list(script))).answer(Q, None, turn_budget=0)
    again = make(ScriptedProvider([])).answer(
        Q, None, turn_budget=0
    )  # an empty script fails if called
    assert again.usage["cached_calls"] == 2 and again.usage["llm_calls"] == 0
    assert again.answer == first.answer and again.cited == first.cited == ["sql:account:acct_a"]
    trial = make(ScriptedProvider(list(script)), trial=1).answer(Q, None, turn_budget=0)
    assert trial.usage["llm_calls"] == 2  # a new trial asks again


def test_api_errors_after_retries_propagate(make: Any) -> None:
    busy = LLMError("overloaded", retryable=True, kind="overloaded")
    ok = make(ScriptedProvider([busy, tool_response(("c1", SUBMIT, {"answer": "好"}))]))
    assert ok.answer(Q, None, turn_budget=0).answer == "好"
    bad = LLMError("Insufficient Balance", retryable=False, kind="billing", status=402)
    with pytest.raises(LLMError):
        make(ScriptedProvider([bad])).answer(Question(case_id="q-2", text="别的问题"), None, 0)


def test_spans_cover_the_run_its_model_calls_and_tools(make: Any) -> None:
    sink = Sink()
    agent = make(
        ScriptedProvider(
            [
                tool_response(("c1", "query_payments", {"operation": "list_payments", **SCOPE})),
                tool_response(("c2", SUBMIT, {"answer": "好"})),
            ]
        )
    )
    agent.tracer_factory = lambda cid: Tracer(cid, sink)
    agent.answer(Q, None, turn_budget=0)
    kinds = [s.kind for s in sink.spans]
    assert kinds.count("llm") == 2 and kinds.count("tool") == 1 and kinds[-1] == "agent"
    root = sink.spans[-1]
    assert all(s.parent_id == root.span_id for s in sink.spans[:-1])
    assert root.attrs["stop"] == "submitted" and root.trace_id == "q-1"


def test_folder_filters_and_the_dataset_simulated_user(ds: Dataset) -> None:
    chunks = benchmark_chunks(load_docs(ds))
    doc_tool = Context.from_docs(load_docs(ds)).doc_tool
    rules = FolderFilter(BM25Retriever(chunks), "search_business_rules", doc_tool)
    hits = rules.search(Question(case_id="q", text="退款 时间窗口 FP-4041"), 5)
    assert hits and all(h.chunk.doc_id == "KB-BR-012" for h in hits)
    user = SimulatedUser.for_conversation(
        ds,
        {
            "available_user_facts": [
                {"fact_id": "ids", "category": "identifiers", "user_value": "pay_1"}
            ]
        },
    )
    assert user.answer("请问支付编号是什么？") == ("pay_1", ["identifiers"])
    assert user.answer("编号再说一次？")[1] == []  # each fact is revealed once


@dataclass
class SubmitAtOnce:
    model: str = "deepseek-v4-flash"
    calls: int = 0

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        self.calls += 1
        return tool_response(("c", SUBMIT, {"answer": "不知道"}))


def test_cli_runs_the_agent_and_writes_records_and_traces(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(ds.root))
    monkeypatch.chdir(tmp_path)
    fake = SubmitAtOnce()
    monkeypatch.setattr(cli, "default_chat", lambda settings: lambda: fake)
    out = tmp_path / "answers"
    assert cli.main(["answer", "--system", "agent", "--out", str(out)]) == 0
    (run,) = out.iterdir()
    assert run.name.endswith("_agent-single-bm25_dev")
    records = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines()]
    assert len(records) == 4 and fake.calls == 4
    assert len(list((run / "traces").iterdir())) == 4
    summary = json.loads((run / "summary.json").read_text())
    assert summary["stops"] == {"submitted": 4} and summary["usage"]["cost_usd"] is None
    assert "yfr grade --records" in capsys.readouterr().out
    per_type = tmp_path / "per-type"
    args = [
        "answer",
        "--system",
        "agent",
        "--tool-shape",
        "per-type",
        "--no-ask",
        "--out",
        str(per_type),
    ]
    assert cli.main([*args, "--limit", "2"]) == 0
    (run2,) = per_type.iterdir()
    assert len((run2 / "records.jsonl").read_text().splitlines()) == 2
    config = json.loads((run2 / "summary.json").read_text())["config"]
    assert len(config["search_tools"]) == 5 and config["ask_user"] is False
