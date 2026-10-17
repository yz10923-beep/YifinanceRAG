"""The one-call RAG baseline (S3 step 6): retrieve the top 10 for the user's first
message, then one chat call. No tools, no asking, no account records: the bar the S4
agent has to beat, and the source of real answers for calibrating the judge.

What the model sees is the opener and the passages, numbered [1]..[k], with their doc
title and heading, never chunk ids (so no gold id can reach it, even by name). It cites
passages by number; the harness maps the numbers back to evidence ids and writes the
answer record, including the search call and what it returned.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import EvidenceMap
from yifinance_rag.grade.record import AnswerRecord, ToolCall
from yifinance_rag.llm.cached import CachedCaller
from yifinance_rag.retrieval.retrievers import Hit, Retriever

SYSTEM = """\
你是 FlowPay 支付平台的技术支持助手，回答商户的问题。
只根据下面给出的资料回答，不使用资料以外的知识。资料不足以确定答案时，直接说明还缺少哪些信息
（例如账户、环境、时间或对象编号），不要编造账户记录、编号、金额或时间。
先给出结论，再说明理由和处理建议。引用资料时在句末标注资料编号，例如 [2] 或 [1][3]。
用中文回答。"""

_CITE = re.compile(r"\[(\d+(?:\s*[,，、]\s*\d+)*)\]")


def render_prompt(question: Question, hits: Sequence[Hit], max_chars: int) -> str:
    parts = [f"商户的问题：\n{question.text}\n\n资料："]
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        heading = f"{c.title} / {c.heading}" if c.heading and c.heading != c.title else c.title
        parts.append(f"[{i}] {heading}\n{c.text[:max_chars]}")
    return "\n\n".join(parts)


def cited_numbers(answer: str, n: int) -> list[int]:
    """1-based passage numbers the answer cites, in order, in range, without repeats."""
    found = [int(x) for m in _CITE.finditer(answer) for x in re.findall(r"\d+", m.group(1))]
    return list(dict.fromkeys(i for i in found if 1 <= i <= n))


@dataclass
class OneCall:
    retriever: Retriever
    caller: CachedCaller
    emap: EvidenceMap
    k: int = 10
    max_chars: int = 3000

    @property
    def name(self) -> str:
        return f"onecall-{self.retriever.name}"

    def config(self) -> dict[str, object]:
        return {
            "retriever": {"name": self.retriever.name, **self.retriever.config()},
            "chat_model": self.caller.model,
            "chat_options": self.caller.effort,
            "k": self.k,
            "max_chars": self.max_chars,
            "trial": self.caller.trial,
        }

    def answer(self, question: Question) -> AnswerRecord:
        hits = self.retriever.search(question, self.k)[: self.k]
        ids = [self.emap.ids_for(h.chunk) for h in hits]
        reply = self.caller.ask_full(render_prompt(question, hits, self.max_chars))
        cited = [e for i in cited_numbers(reply.text, len(hits)) for e in ids[i - 1]]
        return AnswerRecord(
            case_id=question.case_id,
            system=self.name,
            answer=reply.text,
            cited=list(dict.fromkeys(cited)),
            tool_calls=[
                ToolCall(
                    tool="search_documents",
                    args={"query": question.text, "k": self.k},
                    evidence_ids=list(dict.fromkeys(e for group in ids for e in group)),
                    result_count=len(hits),
                )
            ],
            usage={
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "cached": int(reply.cached),
                "cost_usd": reply.cost_usd,
            },
        )
