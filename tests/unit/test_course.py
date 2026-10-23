from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from tests.synthetic import write_dataset
from yifinance_rag.answer.course import row_evidence, to_record, tool_calls
from yifinance_rag.cli import main
from yifinance_rag.data.benchmark import case_id
from yifinance_rag.data.corpus import EvidenceMap, benchmark_chunks, load_docs
from yifinance_rag.data.dataset import Dataset

REQUEST = "runtime:req:req_bm_001_01"
DEV = ("G-SR-001", "G-MS-001", "G-MD-002", "G-MH-002")


def _export(gold: Any, **extra: Any) -> dict[str, Any]:
    """One question as scripts/course_stages.py writes it."""
    return {
        "format": 1,
        "question_id": "G-MS-001",
        "stage": "s4",
        "model": "deepseek-v4-flash",
        "answer": "请求 req_bm_001_01 返回 HTTP 404、错误码 FP-4041。",
        "exchanges": [
            {"signal": "ask_user_tool", "asked": ["identifiers"], "disclosed": ["identifiers"]},
            {"signal": "text", "asked": ["occurrence_time"], "disclosed": ["occurrence_time"]},
        ],
        "deviations": [],
        "seconds": 3.5,
        "course_row": {"task_pass": True},
        "calls": [
            {
                "kind": "search",
                "passages": [
                    # the course's chunk covers the gold section exactly
                    {
                        "id": "x",
                        "doc_id": gold.doc_id,
                        "start_char": gold.start,
                        "end_char": gold.end,
                    },
                    {"id": "y", "doc_id": None, "start_char": None, "end_char": None},
                ],
            },
            {
                "kind": "data",
                "request": {
                    "kind": "logs",
                    "name": "api_requests",
                    "filters": {"request_id": "req_bm_001_01"},
                    "start": "2026-08-02T00:00:00Z",
                },
                "result_count": 1,
                "rows": [
                    {"request_id": "req_bm_001_01", "evidence_id": "runtime:req:req_bm_001_01"}
                ],
            },
            {
                "kind": "data",
                "request": {"kind": "sql", "name": "payments", "parameters": {"payment_id": "p"}},
                "error": "ToolInputError: bad filter",
            },
        ],
        "usage": [
            {
                "model": "deepseek-v4-flash",
                "usage": {
                    "prompt_tokens": 1000,
                    "prompt_cache_hit_tokens": 600,
                    "completion_tokens": 100,
                },
            }
        ],
        "failures": [],
        **extra,
    }


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    return Dataset(write_dataset(tmp_path / "flowpay-dataset", graded=True))


def _gold(ds: Dataset) -> Any:
    (gold,) = [c for c in benchmark_chunks(load_docs(ds)) if c.chunk_id == "doc:KB-TS-012#c003"]
    return gold


def test_rows_map_to_our_evidence_ids() -> None:
    assert row_evidence("sql", "payments", {"payment_id": "pay_1"}) == "sql:payment:pay_1"
    assert row_evidence("sql", "webhook_deliveries", {"d.delivery_id": "del_1"}) == (
        "sql:webhook_delivery:del_1"
    )
    assert row_evidence("logs", "usage_buckets", {"evidence_id": "runtime:usage:a"}) == (
        "runtime:usage:a"
    )
    assert row_evidence("sql", "unknown", {"x": 1}) is None


def test_a_course_question_becomes_an_answer_record(ds: Dataset) -> None:
    emap = EvidenceMap(benchmark_chunks(load_docs(ds)))
    record = to_record(_export(_gold(ds)), emap, {"G-MS-001": "q-1"}, "course-s4")
    search, logs, failed = record.tool_calls
    assert search.tool == "search_documents" and search.evidence_ids == ["doc:KB-TS-012#c003"]
    assert search.result_count == 1  # the passage with no span maps to nothing
    assert logs.evidence_ids == ["runtime:req:req_bm_001_01"] and logs.operation == "logs"
    assert logs.args == {"request_id": "req_bm_001_01", "start": "2026-08-02T00:00:00Z"}
    assert failed.error and failed.evidence_ids == []
    assert record.observed() == {"doc:KB-TS-012#c003", "runtime:req:req_bm_001_01"}
    assert [c.disclosed for c in record.clarifications] == [["identifiers"], ["occurrence_time"]]
    u = record.usage
    assert (u["input_tokens"], u["cache_read_tokens"], u["output_tokens"]) == (400, 600, 100)
    assert u["cost_usd"] == pytest.approx((400 * 0.15 + 600 * 0.003 + 100 * 0.6) / 1e6)
    unknown = _export(_gold(ds), usage=[{"model": "other-model", "usage": {"prompt_tokens": 1}}])
    assert to_record(unknown, emap, {"G-MS-001": "q-1"}, "x").usage["cost_usd"] is None
    with pytest.raises(ValueError, match="unknown question"):
        to_record(_export(_gold(ds)), emap, {}, "x")
    with pytest.raises(ValueError, match="format"):
        to_record(_export(_gold(ds), format=2), emap, {"G-MS-001": "q-1"}, "x")
    with pytest.raises(ValueError, match="call kind"):
        tool_calls([{"kind": "other"}], emap)


def test_cli_imports_an_export_and_grades_it(
    ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FLOWPAY_DATASET_ROOT", str(ds.root))
    monkeypatch.chdir(tmp_path)
    export = tmp_path / "course" / "dev" / "s4.jsonl"
    export.parent.mkdir(parents=True)
    answers, grades = tmp_path / "answers", tmp_path / "grades"
    export.write_text(json.dumps(_export(_gold(ds)), ensure_ascii=False) + "\n")
    assert main(["import-course", str(export), "--out", str(tmp_path / "partial")]) == 0
    assert "partial run" in capsys.readouterr().out
    rows = [_export(_gold(ds), question_id=q) for q in DEV]
    export.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert main(["import-course", str(export), "--out", str(answers)]) == 0
    (run,) = answers.iterdir()
    assert run.name.endswith("_course-s4_dev") and "partial run" not in capsys.readouterr().out
    summary = json.loads((run / "summary.json").read_text())
    assert summary["cases"] == 4 and summary["split"] == "dev" and summary["infra_errors"] == 0
    assert "as_of" in summary["not_comparable"] and summary["usage"]["cost_usd"] > 0
    records = run / "records.jsonl"
    first = json.loads(records.read_text().splitlines()[1])
    assert first["case_id"] == case_id("G-MS-001")
    args = ["grade", "--records", str(records), "--name", "course-s4", "--out", str(grades)]
    assert main(args) == 0
    (graded,) = grades.iterdir()
    cases = [json.loads(x) for x in (graded / "cases.jsonl").read_text().splitlines()]
    (row,) = [c for c in cases if c["case_id"] == case_id("G-MS-001")]
    scores = row["scores"]
    assert scores["doc_evidence"] == 1.0 and scores["clarification"] == 1.0
    assert scores["record_evidence"] == 0.5  # the request, not the account (no such tool)
    # a stage mixed into another stage's export is refused
    mixed = tmp_path / "mixed.jsonl"
    rows = [_export(_gold(ds)), _export(_gold(ds), stage="s3", question_id="G-SR-001")]
    mixed.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    assert main(["import-course", str(mixed), "--out", str(answers)]) == 1


def _runner() -> Any:
    path = Path(__file__).resolve().parents[2] / "scripts" / "course_stages.py"
    spec = importlib.util.spec_from_file_location("course_stages", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_runner_export_is_what_the_importer_reads(ds: Dataset) -> None:
    gold = _gold(ds)

    def ev(kind: str, payload: dict[str, Any], call: str | None = None) -> dict[str, Any]:
        return {"type": kind, "payload": payload, "call_id": call}

    passage = {
        "id": "doc:KB-TS-012#c002",
        "text": "...",
        "source": {
            "doc_id": gold.doc_id,
            "locator": {"start_char": gold.start, "end_char": gold.end},
        },
    }
    events = [
        ev("model.delta", {"text": "streamed, ignored"}),
        ev("knowledge.response", {"collection_id": "c", "results": [passage]}, "k1"),
        ev("tool.started", {"name": "query_logs", "arguments": "{}"}, "t1"),
        ev("source.request", {"kind": "logs", "name": "api_requests", "filters": {}}, "s1"),
        ev("source.response", {"rows": [{"evidence_id": REQUEST}], "truncated": False}, "s1"),
        ev("source.request", {"kind": "sql", "name": "payments", "parameters": {}}, "s2"),
        ev("source.error", {"error_type": "Timeout", "message": "slow"}, "s2"),
        ev(
            "model.response",
            {
                "completion": {
                    "model": "deepseek-v4-flash",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                }
            },
            "m1",
        ),
    ]
    exported = _runner().export(events)
    assert exported["failures"] == [] and len(exported["usage"]) == 1
    row = {**_export(gold), **exported}
    emap = EvidenceMap(benchmark_chunks(load_docs(ds)))
    record = to_record(row, emap, {"G-MS-001": "q-1"}, "course-s4")
    assert record.observed() == {"doc:KB-TS-012#c003", REQUEST}
    assert record.tool_calls[2].error == "Timeout: slow"
    failed = _runner().export([ev("run.failed", {"error_type": "APIError", "message": "402"})])
    assert failed["failures"] == [
        {"type": "run.failed", "error_type": "APIError", "message": "402"}
    ]
