from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.synthetic import F1_AS_OF, NEGATIVE_PAYMENT, write_dataset
from yifinance_rag.cli import main
from yifinance_rag.data.benchmark import load_benchmark, select
from yifinance_rag.data.corpus import load_docs
from yifinance_rag.data.dataset import DataError, Dataset
from yifinance_rag.grade.baselines import SYSTEMS, build, wrong_partner
from yifinance_rag.grade.course import course_score, supported, violated
from yifinance_rag.grade.facts import keys, normalize, states
from yifinance_rag.grade.graders import Context, grade
from yifinance_rag.grade.record import AnswerRecord, Clarification, ToolCall, read_records
from yifinance_rag.grade.report import grade_all, selftest
from yifinance_rag.grade.truth import Truth, load_truths


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return Dataset(write_dataset(tmp_path / "flowpay-dataset", graded=True))


def _truths(ds: Dataset) -> tuple[dict[str, Truth], Context]:
    _, labels = load_benchmark(ds)
    return load_truths(ds, labels), Context.from_docs(load_docs(ds))


def _by_qid(truths: dict[str, Truth], qid: str) -> Truth:
    return next(t for t in truths.values() if t.question_id == qid)


def test_dates_and_times_normalize_to_one_form() -> None:
    forms = ["2026 年 08 月 02 日 11:50 UTC", "8月2日 11:50", "2026-08-02T11:50:00Z"]
    assert {normalize(f).replace(" utc", "") for f in forms} == {"08-02 11:50"}
    assert normalize("9:05") == "09:05"


def test_atoms_are_values_not_template_words() -> None:
    text = (
        "请求 req_bm_001_01 在 2026 年 08 月 02 日 11:50 UTC 以 POST 调用 /v1/payments，"
        "返回 HTTP 404、错误码 FP-4041。"
    )
    assert keys(text) == ["req_bm_001_01", "fp-4041", "08-02", "11:50", "404", "payments"]
    # the resolver's scope note: ids stay, as_of, parameter and operation names go
    scope = (
        "在限定范围（as_of=2026-08-02T12:00:00Z、payment_id=pay_bm_001）"
        "内没有查到 get_payment 对应记录。"
    )
    assert keys(scope) == ["pay_bm_001"]


def test_atoms_match_whole_tokens_only() -> None:
    assert states("地区 NA。", "na") and not states("final", "na")
    assert states("金额 50 元", "50") and not states("2050 年", "50")
    assert states("错误码 fp-4041", "fp-4041") and states("在 8月2日", "08-02")
    assert not states("req_bm_001_011", "req_bm_001_01")


def test_truths_carry_selectors_and_refuse_unexplained_gold(ds: Dataset) -> None:
    truths, _ = _truths(ds)
    ms = _by_qid(truths, "G-MS-001")
    assert ms.dynamic and [s.negative for s in ms.selectors] == [False, True, False]
    assert ms.selectors[1].params == {
        "account_id": "acct_bm_001",
        "environment": "production",
        "payment_id": "pay_bm_001",
    }
    assert NEGATIVE_PAYMENT not in ms.gold_ids  # no agent can observe a hash id
    path = ds.path("benchmark/resolved/evidence_resolution.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        r["selector_results"] = []
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _, labels = load_benchmark(ds)
    with pytest.raises(DataError, match="outside docs and selectors"):
        load_truths(ds, labels)


def test_negative_evidence_needs_the_same_lookup_returning_nothing(ds: Dataset) -> None:
    truths, ctx = _truths(ds)
    ms = _by_qid(truths, "G-MS-001")
    args: dict[str, object] = {
        "account_id": "acct_bm_001",
        "environment": "production",
        "payment_id": "pay_bm_001",
    }

    def found(call: ToolCall) -> float | None:
        r = AnswerRecord(case_id=ms.case_id, system="t", answer="", tool_calls=[call])
        return grade(r, ms, ctx).scores["negative_found"]

    call = ToolCall(tool="query_payments", operation="get_payment", args=args)
    assert found(call) == 1.0
    assert found(call.model_copy(update={"result_count": 1})) == 0.0  # it found a row
    assert found(call.model_copy(update={"error": "timeout"})) == 0.0  # it never looked
    other = {**args, "account_id": "acct_bm_002"}
    assert found(call.model_copy(update={"args": other})) == 0.0  # another tenant


def test_list_selectors_need_any_record_single_ones_their_record(ds: Dataset) -> None:
    truths, ctx = _truths(ds)
    mh = _by_qid(truths, "G-MH-002")
    one = ToolCall(tool="get_runtime_state", evidence_ids=["runtime:req:req_bm_036_02"])
    r = AnswerRecord(case_id=mh.case_id, system="t", answer="", tool_calls=[one])
    g = grade(r, mh, ctx)
    assert g.scores["record_evidence"] == 0.5  # list_requests found, get_refund not
    assert g.details["selectors_missed"] == ["get_refund"]
    assert g.scores["id_recall"] == pytest.approx(1 / 4)  # 1 doc + 3 record ids


def test_sources_count_by_corpus_folder_not_tool_name(ds: Dataset) -> None:
    truths, ctx = _truths(ds)
    sr = _by_qid(truths, "G-SR-001")
    call = ToolCall(tool="search_documents", evidence_ids=["doc:KB-TS-012#c003"], result_count=1)
    r = AnswerRecord(case_id=sr.case_id, system="t", answer="", tool_calls=[call])
    assert grade(r, sr, ctx).scores["tools"] == 1.0  # one search tool, right source


def test_as_of_is_the_case_date_in_utc(ds: Dataset) -> None:
    truths, ctx = _truths(ds)
    ms = _by_qid(truths, "G-MS-001")

    def scoped(*values: object) -> float | None:
        calls = [
            ToolCall(tool="query_accounts", operation="get_account", args={"as_of": v})
            for v in values
        ]
        r = AnswerRecord(case_id=ms.case_id, system="t", answer="", tool_calls=calls)
        return grade(r, ms, ctx).scores["as_of"]

    assert scoped("2026-08-02T20:00:00+08:00") == 1.0  # 12:00 UTC the same day
    assert scoped("2026-08-03T09:00:00+08:00") == 0.0  # 01:00 UTC on 08-03
    assert scoped(F1_AS_OF, None) == 0.5  # an unscoped call reads current state
    assert scoped("2026-08-27T00:00:00Z") == 0.0  # the snapshot's freeze, not the case


def test_single_turn_questions_should_not_ask(ds: Dataset) -> None:
    truths, ctx = _truths(ds)
    sr = _by_qid(truths, "G-SR-001")
    asked = AnswerRecord(
        case_id=sr.case_id, system="t", answer="", clarifications=[Clarification(question="?")]
    )
    assert grade(asked, sr, ctx).scores["clarification"] == 0.0


def test_every_grader_passes_its_selftest(ds: Dataset) -> None:
    truths, ctx = _truths(ds)
    questions, labels = load_benchmark(ds)
    dev = select(questions, labels, "dev")
    by_system = {
        s: grade_all(build(s, dev, truths, ctx), truths, ctx)
        for s in SYSTEMS
        if s != "verbose-null"
    }
    assert selftest(by_system) == []
    # A grader that can't tell a wrong answer from a right one is caught.
    lenient = {**by_system, "wrong": by_system["oracle"]}
    assert any("wrong answer passes" in p for p in selftest(lenient))
    blind = {**by_system, "null": by_system["oracle"]}
    assert any("null above 0" in p for p in selftest(blind))


def test_wrong_answers_come_from_another_family_same_mode(ds: Dataset) -> None:
    truths, _ = _truths(ds)
    pool = [t for t in truths.values() if t.split == "dev"]
    for t in pool:
        partner = wrong_partner(t, pool)
        assert partner.family != t.family and partner.mode == t.mode


def test_course_scorer_rules_and_known_gaps(ds: Dataset) -> None:
    assert supported("退款需在180天内发起", "退款需在 180 天内发起")
    assert violated("可以，把 FP-4041 直接解释为拼写错误", "把 FP-4041 直接解释为拼写错误")
    assert not violated("FP-4041 就是拼错了", "把 FP-4041 直接解释为拼写错误")  # rewording
    truths, ctx = _truths(ds)
    questions, _ = load_benchmark(ds)
    mh = _by_qid(truths, "G-MH-002")
    (rule,) = [r for r in build("rule-only", questions, truths, ctx) if r.case_id == mh.case_id]
    assert course_score(rule, mh).task_pass  # it passes a rule-only answer on a dynamic case
    assert (grade(rule, mh, ctx).scores["instance_facts"] or 0) < 0.5  # ours does not
    (oracle,) = [r for r in build("oracle", questions, truths, ctx) if r.case_id == mh.case_id]
    s = course_score(oracle, mh)
    assert (s.record_hit, s.record_total) == (2, 2)  # the req ids: it can't see re_bm_036


def test_cli_grades_baselines_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_dataset(tmp_path / "flowpay-dataset", graded=True)
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs"
    assert main(["grade", "--selftest"]) == 0
    assert "selftest on dev: ok" in capsys.readouterr().out
    assert main(["grade", "--system", "oracle", "--out", str(out)]) == 0
    (run,) = out.iterdir()
    summary = json.loads((run / "summary.json").read_text())
    assert summary["system"] == "oracle" and summary["cases"] == 4
    assert summary["rows"][-1]["instance_facts"] == 1.0
    records = read_records(run / "records.jsonl")
    assert len(records) == 4 and {
        json.loads(x)["case_id"] for x in (run / "cases.jsonl").read_text().splitlines()
    } == {r.case_id for r in records}
    # a system's records go through the same graders
    assert (
        main(
            ["grade", "--records", str(run / "records.jsonl"), "--name", "again", "--out", str(out)]
        )
        == 0
    )
    again = next(p for p in out.iterdir() if p.name.endswith("again_dev"))
    assert json.loads((again / "summary.json").read_text())["rows"] == summary["rows"]
    assert main(["grade", "--system", "verbose-null", "--out", str(out)]) == 0
    assert main(["grade", "--system", "null", "--split", "test", "--out", str(out)]) == 2
