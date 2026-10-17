from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.fakes import ScriptedProvider, text_response
from tests.synthetic import write_dataset
from yifinance_rag import cli
from yifinance_rag.data.benchmark import Question
from yifinance_rag.data.corpus import Chunk
from yifinance_rag.llm.anthropic_provider import AnthropicProvider
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMError, LLMProvider, LLMResponse, Usage
from yifinance_rag.retrieval.rerank import LLMReranker, parse_ranking, render_prompt
from yifinance_rag.retrieval.retrievers import Hit

CHUNKS = [Chunk(f"c{i}", "D", "t", "幂等键", f"节{i}", i, i + 1, f"正文{i}" * 50) for i in range(4)]
Q = Question(case_id="q", text="幂等冲突返回什么？")


class Fixed:
    name = "fixed"
    chunks = CHUNKS

    def config(self) -> dict[str, object]:
        return {}

    def search(self, question: Question, k: int) -> list[Hit]:
        return [Hit(c, 1.0) for c in CHUNKS[:k]]


def reply(text: str, *, model: str = "claude-haiku-5-5") -> LLMResponse:
    usage = Usage(input_tokens=1000, output_tokens=100)
    return text_response(text).model_copy(update={"usage": usage, "model": model})


def make(provider: ScriptedProvider, tmp_path: Path, **kw: object) -> LLMReranker:
    factory: Callable[[], LLMProvider] = lambda: provider  # noqa: E731
    return LLMReranker(
        Fixed(),
        factory,
        model="claude-haiku-5-5",
        effort="low",
        cache_path=tmp_path / "rerank.sqlite",
        retry=RetryPolicy(base_s=0.001, max_delay_s=0.001),
        **kw,  # type: ignore[arg-type]
    )


def test_parse_ranking() -> None:
    assert parse_ranking("[3, 1, 2]", 3) == ([2, 0, 1], True)
    assert parse_ranking('排序如下：["2", "3", "1"]', 3) == ([1, 2, 0], True)
    assert parse_ranking("[3, 3, 9, 1]", 3) == ([2, 0, 1], False)  # dupes, out of range dropped
    assert parse_ranking("[2]", 3) == ([1, 0, 2], False)  # the rest keep first-stage order
    assert parse_ranking("no idea", 3) == ([0, 1, 2], False)


def test_prompt_numbers_and_trims_passages() -> None:
    prompt = render_prompt(Q, [Hit(c, 1.0) for c in CHUNKS[:2]], max_chars=10)
    assert prompt.startswith("Question:\n幂等冲突返回什么？")
    assert "[1] 幂等键 / 节0\n" + ("正文0" * 50)[:10] in prompt and "[2] 幂等键 / 节1" in prompt
    assert ("正文0" * 50)[:11] not in prompt


def test_rerank_reorders_caches_and_costs(tmp_path: Path) -> None:
    provider = ScriptedProvider([reply("[4, 2, 1, 3]")])
    r = make(provider, tmp_path)
    assert [h.chunk.chunk_id for h in r.search(Q, 3)] == ["c3", "c1", "c0"]
    assert r.stats.calls == 1 and r.stats.fallbacks == 0
    assert r.stats.cost_usd == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    again = make(ScriptedProvider([]), tmp_path)  # an empty script fails if it is called
    assert [h.chunk.chunk_id for h in again.search(Q, 3)] == ["c3", "c1", "c0"]
    assert again.stats.cached == 1 and again.stats.calls == 0
    request = provider.requests[0]
    assert request.tools == [] and "[4] 幂等键 / 节3" in request.messages[0]["content"]


def test_a_new_trial_asks_again(tmp_path: Path) -> None:
    make(ScriptedProvider([reply("[1, 2, 3, 4]")]), tmp_path).search(Q, 4)
    second = make(ScriptedProvider([reply("[2, 1, 3, 4]")]), tmp_path, trial=1)
    assert [h.chunk.chunk_id for h in second.search(Q, 2)] == ["c1", "c0"]
    assert second.stats.calls == 1


def test_unusable_replies_and_unknown_prices_are_counted(tmp_path: Path) -> None:
    r = make(ScriptedProvider([reply("我无法排序", model="some-new-model")]), tmp_path)
    assert [h.chunk.chunk_id for h in r.search(Q, 4)] == ["c0", "c1", "c2", "c3"]
    assert r.stats.fallbacks == 1 and r.stats.cost_usd is None


def test_retries_then_stops_on_errors(tmp_path: Path) -> None:
    busy = LLMError("overloaded", retryable=True, kind="overloaded")
    r = make(ScriptedProvider([busy, reply("[2, 1, 3, 4]")]), tmp_path)
    assert r.search(Q, 1)[0].chunk.chunk_id == "c1"
    bad = LLMError("bad request", retryable=False, kind="invalid_request")
    with pytest.raises(LLMError):
        make(ScriptedProvider([bad]), tmp_path / "other").search(Q, 1)


def test_one_shot_prompts_skip_the_prompt_cache() -> None:
    def req(cache: bool) -> dict[str, object]:
        p = AnthropicProvider(model="claude-haiku-5-5", max_tokens=10, prompt_cache=cache)
        return p.build_request(system="s", messages=[], tools=[])

    assert "cache_control" in req(True) and "cache_control" not in req(False)
    assert "tools" not in req(False)


def test_cli_rerank_records_its_cost_and_refuses_oracles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(write_dataset(tmp_path / "flowpay-dataset")))
    monkeypatch.chdir(tmp_path)
    provider = ScriptedProvider([reply("[2, 1]"), reply("[1, 2]")])
    monkeypatch.setattr(cli, "default_provider", lambda settings, model: lambda: provider)
    out = tmp_path / "runs"
    assert (
        cli.main(["retrieval", "--retriever", "bm25", "--rerank", "haiku", "--out", str(out)]) == 0
    )
    (run,) = out.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert run.name.endswith("_bm25+rerank_dev")
    assert summary["config"]["reranker"]["model"] == "claude-haiku-5-5"
    assert summary["rerank"]["calls"] == len(provider.requests) > 0
    assert "rerank:" in capsys.readouterr().out
    assert cli.main(["retrieval", "--retriever", "oracle", "--rerank", "haiku"]) == 2
