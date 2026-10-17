from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.fakes import ScriptedProvider, text_response
from tests.synthetic import write_dataset
from yifinance_rag import cli
from yifinance_rag.data.benchmark import load_benchmark
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.grade.judge import (
    Judge,
    cohen_kappa,
    forbidden_claims,
    parse_verdicts,
    render_prompt,
)
from yifinance_rag.grade.record import AnswerRecord
from yifinance_rag.grade.truth import Truth, load_truths
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMProvider, LLMResponse, Message, TextCallback, ToolSpec


def _truth(tmp_path: Path, qid: str = "G-MD-002") -> Truth:
    ds = Dataset(write_dataset(tmp_path / "flowpay-dataset", graded=True))
    _, labels = load_benchmark(ds)
    return next(t for t in load_truths(ds, labels).values() if t.question_id == qid)


ANSWER = "支付成功后 180 天内可以发起退款，超过窗口返回 FP-4222。Starter 套餐不支持部分退款。"


def _reply(required: list[tuple[str, str]], forbidden: list[tuple[str, str]]) -> str:
    return json.dumps(
        {
            "required": [
                {"id": f"A{i}", "verdict": v, "quote": q} for i, (v, q) in enumerate(required, 1)
            ],
            "forbidden": [
                {"id": f"N{i}", "verdict": v, "quote": q} for i, (v, q) in enumerate(forbidden, 1)
            ],
        },
        ensure_ascii=False,
    )


def test_prompt_shows_assertions_and_the_claim_without_its_prefix(tmp_path: Path) -> None:
    t = _truth(tmp_path)
    prompt = render_prompt("问题", t, ANSWER)
    assert "A1: 支付成功后 180 天内可以发起退款" in prompt and "A2: Starter" in prompt
    assert "N1: 超过窗口仍可人工退款" in prompt and "不得认定" not in prompt
    assert forbidden_claims(t) == ["超过窗口仍可人工退款"]
    assert prompt.endswith(f"<answer>\n{ANSWER}\n</answer>")


def test_verdicts_need_a_real_quote(tmp_path: Path) -> None:
    t = _truth(tmp_path)
    good = _reply(
        [("supported", "180 天内可以发起退款"), ("supported", "Starter 套餐不支持部分退款")],
        [("not_asserted", "")],
    )
    v = parse_verdicts("判断如下：\n" + good + "\n完毕", t, ANSWER)
    assert v.complete and v.scores() == {
        "judge_assertions": 1.0,
        "judge_forbidden_clear": 1.0,
        "judge_task": 1.0,
    }
    made_up = _reply(
        [("supported", "180天内可以发起退款"), ("supported", "Growth 支持部分退款")],
        [("not_asserted", "")],
    )
    v = parse_verdicts(made_up, t, ANSWER)  # whitespace may differ; invented text may not
    assert v.required == {"A1": True, "A2": False} and v.unverified == ["A2"]
    assert v.scores()["judge_task"] == 0.0


def test_forbidden_claims_flag_the_task(tmp_path: Path) -> None:
    t = _truth(tmp_path)
    answer = ANSWER + "不过超过窗口仍可人工退款。"
    text = _reply(
        [("supported", "180 天内可以发起退款"), ("supported", "Starter 套餐不支持部分退款")],
        [("asserted", "超过窗口仍可人工退款")],
    )
    s = parse_verdicts(text, t, answer).scores()
    assert (
        s["judge_assertions"] == 1.0
        and s["judge_forbidden_clear"] == 0.0
        and s["judge_task"] == 0.0
    )


def test_incomplete_replies_score_none_never_a_guess(tmp_path: Path) -> None:
    t = _truth(tmp_path)
    for text in ("我无法判断", "{not json}", _reply([("supported", "180 天内可以发起退款")], [])):
        v = parse_verdicts(text, t, ANSWER)
        assert not v.complete and v.scores()["judge_task"] is None
    odd = _reply([("maybe", ""), ("not_supported", "")], [("not_asserted", "")])
    assert parse_verdicts(odd, t, ANSWER).required["A1"] is None  # unknown verdict word


def test_cohen_kappa_by_hand() -> None:
    pairs = [(True, True), (True, False), (False, False), (False, False)]
    # observed 0.75; chance 0.5 * 0.25 + 0.5 * 0.75 = 0.5; (0.75 - 0.5) / 0.5
    assert cohen_kappa(pairs) == pytest.approx(0.5)
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, True)] * 3) is None and cohen_kappa([]) is None


def test_judge_caches_counts_and_asks_again_on_a_new_trial(tmp_path: Path) -> None:
    t = _truth(tmp_path)
    record = AnswerRecord(case_id=t.case_id, system="s", answer=ANSWER)
    reply = _reply(
        [("supported", "180 天内可以发起退款"), ("not_supported", "")], [("not_asserted", "")]
    )
    provider = ScriptedProvider([text_response(reply), text_response("???")])

    def make(trial: int) -> Judge:
        factory: Callable[[], LLMProvider] = lambda: provider  # noqa: E731
        return Judge(
            factory,
            model="claude-haiku-5-5",
            cache_path=tmp_path / "j.sqlite",
            trial=trial,
            retry=RetryPolicy(base_s=0.001, max_delay_s=0.001),
        )

    first = make(0)
    assert first.judge("问题", t, record).scores()["judge_assertions"] == 0.5
    again = make(0)
    again.judge("问题", t, record)
    assert again.stats.cached == 1 and again.stats.calls == 0
    third = make(1)
    assert third.judge("问题", t, record).scores()["judge_task"] is None
    assert third.stats.calls == 1 and third.stats.unparsed == 1
    for j in (first, again, third):
        j.close()


@dataclass
class RuleJudge:
    """Answers like a careful judge would on the baselines: an assertion is supported when
    the answer contains it verbatim, a claim is asserted when the answer states it."""

    model: str = "claude-haiku-5-5"
    calls: int = 0
    seen: list[str] = field(default_factory=list)

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        self.calls += 1
        prompt = str(messages[0]["content"])
        answer = prompt.split("<answer>\n", 1)[1].rsplit("\n</answer>", 1)[0]
        req = re.findall(r"^A\d+: (.*)$", prompt, flags=re.M)
        neg = re.findall(r"^N\d+: (.*)$", prompt, flags=re.M)
        squash = "".join(answer.split())
        items = [
            ("supported", a) if "".join(a.split()) in squash else ("not_supported", "") for a in req
        ]
        claims = [
            ("asserted", c) if "可以认定：" + c in answer else ("not_asserted", "") for c in neg
        ]
        text = _reply(items, claims)
        return text_response(text).model_copy(update={"model": self.model})


def test_cli_judged_selftest_writes_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_dataset(tmp_path / "flowpay-dataset", graded=True)
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    fake = RuleJudge()
    monkeypatch.setattr(cli, "default_provider", lambda settings, model: lambda: fake)
    out = tmp_path / "runs"
    assert cli.main(["grade", "--selftest", "--judge", "haiku", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "selftest on dev: ok" in text and "judge claude-haiku-5-5:" in text
    (run,) = out.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert run.name.endswith("judge-selftest-haiku_dev") and summary["problems"] == []
    assert set(summary["systems"]) == {
        "oracle",
        "null",
        "wrong",
        "rule-only",
        "forbidden",
        "verbose-null",
    }
    # 6 systems x 4 dev cases; on single-turn cases oracle and rule-only send the same
    # answer (the canonical one), so the cache serves the second prompt
    judged = summary["judge"]
    assert judged["calls"] == fake.calls and judged["calls"] + judged["cached"] == 6 * 4
    assert summary["systems"]["forbidden"][-1]["judge_forbidden_clear"] == 0.0
    # graded runs carry the judge too, and a rerun is served from the cache
    assert cli.main(["grade", "--system", "oracle", "--judge", "haiku", "--out", str(out)]) == 0
    oracle = next(p for p in out.iterdir() if p.name.endswith("_oracle_dev"))
    s = json.loads((oracle / "summary.json").read_text())
    assert (
        s["rows"][-1]["judge_task"] == 1.0
        and s["judge"]["cached"] == 4
        and s["judge"]["calls"] == 0
    )


def test_a_judge_that_passes_everything_fails_its_selftest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_dataset(tmp_path / "flowpay-dataset", graded=True)
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(root))
    monkeypatch.chdir(tmp_path)

    class Lenient(RuleJudge):
        async def complete(
            self,
            *,
            system: str,
            messages: list[Message],
            tools: list[ToolSpec],
            on_text: TextCallback | None = None,
        ) -> LLMResponse:
            prompt = str(messages[0]["content"])
            answer = prompt.split("<answer>\n", 1)[1].rsplit("\n</answer>", 1)[0]
            n = len(re.findall(r"^A\d+: ", prompt, flags=re.M))
            m = len(re.findall(r"^N\d+: ", prompt, flags=re.M))
            quote = answer[:2] or "?"
            return text_response(_reply([("supported", quote)] * n, [("not_asserted", "")] * m))

    fake = Lenient()
    monkeypatch.setattr(cli, "default_provider", lambda settings, model: lambda: fake)
    assert cli.main(["grade", "--selftest", "--judge", "haiku", "--out", str(tmp_path / "r")]) == 1
    out = capsys.readouterr().out
    assert "judge on wrong: judge_task" in out and "judge on forbidden" in out
