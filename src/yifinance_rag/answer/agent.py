"""The S4 agent: our own tool-use loop over the provider interface. No framework.

One question:
1. The model gets the user's first message and the tools: document search, the four data
   tools, `ask_user` (the dataset's simulated user) and `submit_answer`.
2. Each step is one model call. Its tool calls run in order and their results go back as
   one user message (the Messages protocol: every tool_use gets a tool_result).
3. It ends at `submit_answer`, or when the step budget runs out: the last step offers only
   `submit_answer`, so a run always ends with an answer, possibly empty.

What the harness keeps the model from seeing: evidence ids (passages and rows get handles
`D1`, `R1`; citations of handles are mapped back), the label (the simulated user is built
by the runner), and the per-question clarification budget (a refusal says so only when hit).

Every model reply is cached by the full request (model, options, system, messages, tools,
trial): a rerun or a resumed run pays nothing for steps it already took. API errors that
survive the retry policy propagate: the runner records them as infra failures.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import EvidenceMap
from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.grade.record import AnswerRecord, Clarification, ToolCall
from yifinance_rag.llm.pricing import cost_usd
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMError, LLMProvider, LLMResponse, Message, ToolSpec
from yifinance_rag.retrieval.retrievers import Hit, Retriever
from yifinance_rag.tools.data import DataTools, ToolError
from yifinance_rag.trace.tracer import Tracer

SYSTEM = """\
你是 FlowPay 支付平台的技术支持 Agent，帮助商户排查问题。当前时间是 {now}（系统快照时间）。

工作方式：
- 先判断问题属于哪个产品领域。用户的第一句话往往信息不全：需要账户编号、环境（正式/沙箱）、
  发生时间、对象编号（支付、退款、请求、投递等）或接口版本时，{ask}
- {search}；用数据工具查这个账户实际发生了什么。
  数据工具必须带 account_id 和 environment；问题涉及过去某个时间时，传 as_of（ISO 时间，
  例如用户说"8 月 2 日中午前后"就用 2026-08-02T12:00:00Z），不要用当前状态代替历史状态。
- 只根据工具返回的证据下结论。证据不足以确定时，说明还缺什么，不要猜测。历史支持案例只是
  经验，不能当作这个账户的事实。
- 结论先行：先写结论，再写查到的实际记录（编号、状态、金额、错误码、时间）和依据的规则，
  最后写处理建议。用中文回答。
- 用 submit_answer 提交最终回答，并在 citations 里列出支持结论的资料和记录的编号（如 D2、R5）。"""

# The two lines that depend on the tool set, so a variant's prompt names only its own tools.
ASK_LINE = "用 ask_user 有针对性地询问，\n  每次只问当前确实缺少的信息。"
NO_ASK_LINE = "无法向用户追问：用已有信息查证，\n  并在回答中说明还缺哪些信息。"
SINGLE_SEARCH_LINE = "用 search_documents 查规则、API 文档和排障手册"
PER_TYPE_SEARCH_LINE = (
    "用 search_business_rules、search_api_docs、search_troubleshooting、\n"
    "  search_release_notes、search_support_cases 分别查对应类型的文档"
)


def system_prompt(now: str, *, ask_user: bool, single_search: bool) -> str:
    return SYSTEM.format(
        now=now,
        ask=ASK_LINE if ask_user else NO_ASK_LINE,
        search=SINGLE_SEARCH_LINE if single_search else PER_TYPE_SEARCH_LINE,
    )


SUBMIT = "submit_answer"
ASK = "ask_user"
NUDGE = "请调用 submit_answer 提交最终回答。"
LAST_STEP = "步数已用完：请现在调用 submit_answer，根据已有证据给出最终回答。"
NO_MORE_QUESTIONS = "用户现在无法提供更多信息。请根据已有信息作答，并说明还缺少什么。"


def _say(messages: list[Message], text: str) -> None:
    """Add a harness instruction as user text, inside the last user turn if it holds tool
    results: roles must alternate, and every tool_use's result must come first."""
    last = messages[-1]
    if last["role"] == "user" and isinstance(last["content"], list):
        last["content"] = [*last["content"], {"type": "text", "text": text}]
    else:
        messages.append({"role": "user", "content": text})


class User(Protocol):
    def answer(self, question: str) -> tuple[str, list[str]]: ...


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 16
    search_k: int = 10
    passage_chars: int = 1500
    ask_user: bool = True
    trial: int = 0


@dataclass
class SearchTool:
    """One search tool: a name, what it covers, and the retriever behind it."""

    name: str
    description: str
    retriever: Retriever

    def spec(self) -> ToolSpec:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "中文检索语句"}},
                "required": ["query"],
            },
        }


# Per-doc-type search tools (the benchmark's tool names), for the tool-shape A/B.
SEARCH_TOOLS = {
    "search_documents": (
        "检索 FlowPay 全部产品文档：业务规则、API 文档、排障手册、版本说明、历史支持案例。"
    ),
    "search_business_rules": "检索业务规则：权限、状态机、额度、套餐、时间窗口等正式规则。",
    "search_api_docs": "检索 API 文档：接口、参数、字段、错误码和版本差异。",
    "search_troubleshooting": "检索排障手册：错误码含义、排查步骤和解决办法。",
    "search_release_notes": "检索版本说明和迁移指南：版本变化、废弃字段、迁移窗口。",
    "search_support_cases": "检索历史支持案例：只是经验，不是当前账户的事实。",
}


@dataclass
class FolderFilter:
    """A retriever limited to one doc folder: the base's ranking, other folders dropped."""

    base: Retriever
    tool: str  # search_api_docs, ...
    doc_tool: Mapping[str, str]  # doc_id -> its search tool (grade.graders.Context)
    depth: int = 100
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"{self.base.name}|{self.tool}"

    @property
    def chunks(self) -> object:
        return getattr(self.base, "chunks", [])

    def config(self) -> dict[str, object]:
        return {"base": self.base.config(), "folder": self.tool, "depth": self.depth}

    def search(self, question: Question, k: int) -> list[Hit]:
        hits = self.base.search(question, self.depth)
        return [h for h in hits if self.doc_tool.get(h.chunk.doc_id) == self.tool][:k]


class ResponseCache:
    """Model replies keyed by the full request, so a step already paid for is never paid
    again (reruns, resumes)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute("CREATE TABLE IF NOT EXISTS replies (key TEXT PRIMARY KEY, body TEXT)")

    def get(self, key: str) -> LLMResponse | None:
        row = self.db.execute("SELECT body FROM replies WHERE key = ?", (key,)).fetchone()
        return LLMResponse.model_validate_json(row[0]) if row else None

    def put(self, key: str, response: LLMResponse) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?)", (key, response.model_dump_json())
            )

    def close(self) -> None:
        self.db.close()


@dataclass
class _Run:
    """One question's state: handles, the record being built, usage."""

    refs: dict[str, list[str]] = field(default_factory=dict)  # handle -> evidence ids
    calls: list[ToolCall] = field(default_factory=list)
    clarifications: list[Clarification] = field(default_factory=list)
    usage: dict[str, Any] = field(
        default_factory=lambda: {
            "llm_calls": 0,
            "cached_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cost_usd": 0.0,
        }
    )

    def handle(self, prefix: str, evidence: list[str]) -> str:
        ref = f"{prefix}{sum(r.startswith(prefix) for r in self.refs) + 1}"
        self.refs[ref] = evidence
        return ref


class Agent:
    def __init__(
        self,
        provider_factory: Callable[[], LLMProvider],
        *,
        model: str,
        options: str,
        search: Sequence[SearchTool],
        emap: EvidenceMap,
        data: DataTools,
        cache: ResponseCache,
        now: str,
        config: AgentConfig | None = None,
        retry: RetryPolicy | None = None,
        tracer_factory: Callable[[str], Tracer] | None = None,
    ) -> None:
        self.provider_factory = provider_factory
        self.model, self.options = model, options
        self.search = {s.name: s for s in search}
        self.emap, self.data, self.cache = emap, data, cache
        self.config = config or AgentConfig()
        self.system = system_prompt(
            now,
            ask_user=self.config.ask_user,
            single_search=list(self.search) == ["search_documents"],
        )
        self.retry = retry or RetryPolicy()
        self.tracer_factory = tracer_factory or (lambda _: Tracer.noop())
        self._loop = asyncio.new_event_loop()
        self._provider: LLMProvider | None = None

    @property
    def name(self) -> str:
        shape = "single" if list(self.search) == ["search_documents"] else "per-type"
        ask = "" if self.config.ask_user else "-noask"
        return f"agent-{shape}{ask}"

    def specs(self) -> list[ToolSpec]:
        specs = [s.spec() for s in self.search.values()]
        specs += [self.data.spec(t) for t in self.data.tools]
        if self.config.ask_user:
            specs.append(
                {
                    "name": ASK,
                    "description": (
                        "向用户提一个问题，获取缺少的信息（账户、环境、时间、编号、版本等）。"
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                }
            )
        specs.append(
            {
                "name": SUBMIT,
                "description": "提交最终回答，结束本题。",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "citations": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["answer"],
                },
            }
        )
        return specs

    def config_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "options": self.options,
            "search_tools": {n: s.retriever.name for n, s in self.search.items()},
            "max_steps": self.config.max_steps,
            "search_k": self.config.search_k,
            "passage_chars": self.config.passage_chars,
            "ask_user": self.config.ask_user,
            "trial": self.config.trial,
            "system_sha256": sha256_text(self.system),
        }

    # ---- the loop --------------------------------------------------------------------

    def answer(self, question: Question, user: User | None, turn_budget: int) -> AnswerRecord:
        run = _Run()
        tracer = self.tracer_factory(question.case_id)
        started = time.monotonic()
        messages: list[Message] = [{"role": "user", "content": question.text}]
        all_specs = self.specs()
        submit_only = [s for s in all_specs if s["name"] == SUBMIT]
        answer, cited, stop, nudged = "", [], "budget", False
        with tracer.span("agent", "agent", case_id=question.case_id) as span:
            for step in range(self.config.max_steps):
                last = step == self.config.max_steps - 1
                if last:
                    _say(messages, LAST_STEP)
                response = self._complete(tracer, run, messages, submit_only if last else all_specs)
                messages.append({"role": "assistant", "content": response.content})
                calls = response.tool_calls
                submit = next((c for c in calls if c.name == SUBMIT), None)
                if submit is not None:
                    answer = str(submit.input.get("answer") or "")
                    cited = [str(c) for c in submit.input.get("citations") or []]
                    stop = "submitted"
                    break
                if not calls:
                    if nudged or last:
                        answer, stop = response.text, "text"
                        break
                    nudged = True
                    _say(messages, NUDGE)
                    continue
                results = [
                    self._tool(tracer, run, c.id, c.name, c.input, question, user, turn_budget)
                    for c in calls
                ]
                messages.append({"role": "user", "content": results})
            span.set(stop=stop, steps=run.usage["llm_calls"] + run.usage["cached_calls"])
        run.usage |= {"stop": stop, "seconds": round(time.monotonic() - started, 2)}
        known = [e for ref in cited for e in run.refs.get(ref.strip(), [])]
        run.usage["unknown_citations"] = sum(ref.strip() not in run.refs for ref in cited)
        return AnswerRecord(
            case_id=question.case_id,
            system=self.name,
            answer=answer,
            cited=list(dict.fromkeys(known)),
            tool_calls=run.calls,
            clarifications=run.clarifications,
            usage=run.usage,
        )

    def _complete(
        self, tracer: Tracer, run: _Run, messages: list[Message], specs: list[ToolSpec]
    ) -> LLMResponse:
        body = [self.model, self.options, self.system, messages, specs, self.config.trial]
        key = sha256_text(json.dumps(body, ensure_ascii=False, sort_keys=True))
        cached = self.cache.get(key)
        if cached is not None:
            run.usage["cached_calls"] += 1
            return cached
        with tracer.span("llm", "llm", model=self.model) as span:
            response = self._loop.run_until_complete(self._call(messages, specs))
            u = response.usage
            span.set(input_tokens=u.input_tokens, output_tokens=u.output_tokens)
        self.cache.put(key, response)
        run.usage["llm_calls"] += 1
        run.usage["input_tokens"] += u.input_tokens
        run.usage["output_tokens"] += u.output_tokens
        run.usage["cache_read_tokens"] += u.cache_read_input_tokens
        cost = cost_usd(response.model or self.model, u.model_dump())
        run.usage["cost_usd"] = (
            None if cost is None or run.usage["cost_usd"] is None else run.usage["cost_usd"] + cost
        )
        return response

    async def _call(self, messages: list[Message], specs: list[ToolSpec]) -> LLMResponse:
        if self._provider is None:
            self._provider = self.provider_factory()
        retry, waited = 0, 0.0
        while True:
            try:
                return await self._provider.complete(
                    system=self.system, messages=messages, tools=specs
                )
            except LLMError as e:
                retry += 1
                delay = self.retry.delay(e, retry)
                if not self.retry.should_retry(e, retry, waited, delay):
                    raise
                waited += delay
                await asyncio.sleep(delay)

    # ---- tools ---------------------------------------------------------------------

    def _tool(
        self,
        tracer: Tracer,
        run: _Run,
        call_id: str,
        name: str,
        args: Mapping[str, Any],
        question: Question,
        user: User | None,
        turn_budget: int,
    ) -> dict[str, Any]:
        with tracer.span("tool", "tool", tool=name) as span:
            content, error = self._dispatch(run, name, dict(args), question, user, turn_budget)
            span.set(error=error)
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": content
            if isinstance(content, str)
            else json.dumps(content, ensure_ascii=False),
        }
        if error:
            block["is_error"] = True
        return block

    def _dispatch(
        self,
        run: _Run,
        name: str,
        args: dict[str, Any],
        question: Question,
        user: User | None,
        turn_budget: int,
    ) -> tuple[object, str | None]:
        if name in self.search:
            return self._search(run, self.search[name], args, question), None
        if name in self.data.tools:
            try:
                result = self.data.run(name, args)
            except ToolError as e:
                op = args.get("operation")
                run.calls.append(
                    ToolCall(
                        tool=name,
                        operation=op if isinstance(op, str) else None,
                        args=args,
                        error=str(e),
                    )
                )
                return {"error": str(e)}, str(e)
            rows = []
            for row in result.rows:
                shown = {k: v for k, v in row.items() if k != "evidence_id"}
                rows.append({"ref": run.handle("R", [str(row["evidence_id"])]), **shown})
            run.calls.append(
                ToolCall(
                    tool=name,
                    operation=result.operation,
                    args=args,
                    evidence_ids=result.evidence_ids,
                    result_count=len(result.rows),
                )
            )
            return {**result.to_observation(), "rows": rows}, None
        if name == ASK and self.config.ask_user:
            text = str(args.get("question") or "")
            if user is None or len(run.clarifications) >= turn_budget:
                run.calls.append(ToolCall(tool=ASK, args=args, error="no more clarification"))
                return NO_MORE_QUESTIONS, "no more clarification"
            reply, categories = user.answer(text)
            # Recorded as a clarification only: the graders read tool_calls as searches
            # and data lookups (an ask is neither, and has no as_of).
            run.clarifications.append(Clarification(question=text, disclosed=categories))
            return reply, None
        message = f"unknown tool {name!r}"
        run.calls.append(ToolCall(tool=name, args=args, error=message))
        return {"error": message}, message

    def _search(
        self, run: _Run, tool: SearchTool, args: Mapping[str, Any], question: Question
    ) -> object:
        query = str(args.get("query") or "").strip()
        if not query:
            run.calls.append(ToolCall(tool=tool.name, args=dict(args), error="empty query"))
            return {"error": "query is empty"}
        hits: list[Hit] = tool.retriever.search(
            Question(case_id=question.case_id, text=query), self.config.search_k
        )
        passages, observed = [], []
        for h in hits[: self.config.search_k]:
            ids = self.emap.ids_for(h.chunk)
            observed += ids
            c = h.chunk
            heading = f"{c.title} / {c.heading}" if c.heading and c.heading != c.title else c.title
            passages.append(
                {
                    "ref": run.handle("D", ids),
                    "source": heading,
                    "text": c.text[: self.config.passage_chars],
                }
            )
        run.calls.append(
            ToolCall(
                tool=tool.name,
                args={"query": query},
                evidence_ids=list(dict.fromkeys(observed)),
                result_count=len(hits),
            )
        )
        return {"passages": passages}

    def close(self) -> None:
        if self._loop.is_closed():
            return
        aclose = getattr(self._provider, "aclose", None)
        if aclose is not None:
            self._loop.run_until_complete(aclose())
        self._provider = None
        self._loop.close()
