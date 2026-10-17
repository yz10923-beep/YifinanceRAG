from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.fakes import text_response
from tests.synthetic import write_dataset
from tests.unit.test_judge import RuleJudge
from yifinance_rag import cli
from yifinance_rag.answer.onecall import cited_numbers
from yifinance_rag.grade.calibrate import agreement, read_labels
from yifinance_rag.llm.types import LLMError, LLMResponse, Message, TextCallback, ToolSpec


@dataclass
class FakeChat:
    """Answers with the first passage it was shown, citing it; or fails, if told to."""

    model: str = "deepseek-v4-flash"
    prompts: list[str] = field(default_factory=list)
    fail: LLMError | None = None

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_text: TextCallback | None = None,
    ) -> LLMResponse:
        if self.fail:
            raise self.fail
        prompt = str(messages[0]["content"])
        self.prompts.append(prompt)
        first = prompt.split("[1] ", 1)[1].split("\n\n[2]", 1)[0]
        return text_response(f"结论如下。{first} [1]，另见 [2、9] 和 [99]。").model_copy(
            update={"model": self.model}
        )


def test_citations_are_passage_numbers_in_range() -> None:
    assert cited_numbers("见 [2]，以及 [1][3]、[2, 5]，[12] 不在范围", 10) == [2, 1, 3, 5]
    assert cited_numbers("没有引用", 10) == []


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = write_dataset(tmp_path / "flowpay-dataset", graded=True)
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_onecall_writes_records_and_shows_no_ids(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    chat = FakeChat()
    monkeypatch.setattr(cli, "default_chat", lambda settings: lambda: chat)
    out = env / "answers"
    assert cli.main(["answer", "--out", str(out)]) == 0
    (run,) = out.iterdir()
    assert run.name.endswith("_onecall-bm25_dev")
    records = [json.loads(x) for x in (run / "records.jsonl").read_text().splitlines()]
    assert len(records) == 4 and len(chat.prompts) == 4
    for r in records:
        observed = set(r["tool_calls"][0]["evidence_ids"])
        assert r["cited"] and set(r["cited"]) <= observed  # [1] and [2] mapped, [99] dropped
    for p in chat.prompts:  # no chunk ids, question ids or answers in what the model sees
        assert "doc:" not in p and "G-" not in p and "#c0" not in p
    summary = json.loads((run / "summary.json").read_text())
    assert summary["chat"]["calls"] == 4 and summary["chat"]["cost_usd"] is None  # no price
    assert "yfr grade --records" in capsys.readouterr().out
    again = FakeChat()
    monkeypatch.setattr(cli, "default_chat", lambda settings: lambda: again)
    assert cli.main(["answer", "--out", str(out)]) == 0 and again.prompts == []  # cached
    assert cli.main(["answer", "--retriever", "oracle", "--out", str(out)]) == 2


def test_an_api_failure_is_recorded_and_stops_the_run(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = FakeChat(
        fail=LLMError("Insufficient Balance", retryable=False, kind="billing", status=402)
    )
    monkeypatch.setattr(cli, "default_chat", lambda settings: lambda: chat)
    out = env / "answers"
    assert cli.main(["answer", "--out", str(out)]) == 3
    (run,) = out.iterdir()
    error = json.loads((run / "errors.jsonl").read_text())
    assert error["kind"] == "billing" and not (run / "records.jsonl").exists()


def test_no_key_is_a_clear_config_error(env: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["answer", "--out", str(env / "a")]) == 1
    assert "DEEPSEEK_API_KEY" in capsys.readouterr().err


def test_label_sheet_then_agreement_with_a_judge(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "default_chat", lambda settings: lambda: FakeChat())
    monkeypatch.setattr(cli, "default_provider", lambda settings, model: lambda: RuleJudge())
    assert cli.main(["answer", "--out", str(env / "answers")]) == 0
    (answers,) = (env / "answers").iterdir()
    records = str(answers / "records.jsonl")
    assert (
        cli.main(["grade", "--records", records, "--judge", "haiku", "--out", str(env / "g")]) == 0
    )
    (judged,) = (env / "g").iterdir()
    assert cli.main(["label-sheet", str(answers)]) == 0
    sheet = (answers / "labels.md").read_text()
    assert "blind" in sheet and "**A1**" in sheet and "judge" not in sheet.split("\n", 1)[1]
    # Yi's labels: the judge's own verdicts, with one flipped
    cases = [json.loads(x) for x in (judged / "cases.jsonl").read_text().splitlines()]
    verdict = {
        (c["case_id"], item): v
        for c in cases
        for section in ("required", "forbidden")
        for item, v in c["judge"][section].items()
    }
    tsv = answers / "labels.tsv"
    rows = list(csv.DictReader(tsv.open(), delimiter="\t"))
    assert {(r["case_id"], r["item"]) for r in rows} == set(verdict)
    for i, r in enumerate(rows):
        v = verdict[(r["case_id"], r["item"])]
        r["label"] = ("y" if v else "n") if i else ("n" if v else "y")
    with tsv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "item", "label", "note"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    result = agreement(read_labels(tsv), cases)
    assert result["all_items"]["n"] == len(rows) and len(result["disagreements"]) == 1  # type: ignore[index, arg-type]
    assert cli.main(["agree", str(tsv), str(judged)]) == 0
    assert "| all_items |" in capsys.readouterr().out
    assert json.loads((judged / "agreement.json").read_text())["all_items"]["n"] == len(rows)


def test_labels_must_be_y_or_n(tmp_path: Path) -> None:
    tsv = tmp_path / "labels.tsv"
    tsv.write_text("case_id\titem\tlabel\tnote\nq-1\tA1\tyes\t\nq-1\tN1\t\t\n")
    with pytest.raises(Exception, match="not y or n"):
        read_labels(tsv)
    tsv.write_text("case_id\titem\tlabel\tnote\nq-1\tA1\tY\t\nq-1\tN1\t\t\n")
    assert read_labels(tsv) == {("q-1", "A1"): True}  # blanks are not labelled yet


def test_label_sheets_refuse_test_answers(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "default_chat", lambda settings: lambda: FakeChat())
    out = env / "answers"
    assert cli.main(["answer", "--split", "test", "--final", "--out", str(out)]) == 0
    (run,) = out.iterdir()
    assert cli.main(["label-sheet", str(run)]) == 2
