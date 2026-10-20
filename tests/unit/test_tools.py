from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.synthetic import write_canonical
from yifinance_rag.data.dataset import Dataset
from yifinance_rag.tools.data import DataStore, DataTools, ToolError, load_policy

A = {"account_id": "acct_a", "environment": "production"}


@pytest.fixture
def tools(tmp_path: Path) -> Iterator[DataTools]:
    t = DataTools(DataStore(Dataset(write_canonical(tmp_path / "flowpay-dataset"))))
    yield t
    t.store.close()


def ids(t: DataTools, tool: str, **args: object) -> list[str]:
    return t.run(tool, args).evidence_ids


def test_scope_is_enforced_in_sql_not_by_the_prompt(tools: DataTools) -> None:
    assert ids(tools, "query_payments", operation="get_payment", payment_id="pay_a_1", **A) == [
        "sql:payment:pay_a_1"
    ]
    # another tenant's payment: no row, and nothing says it exists
    other = tools.run("query_payments", {"operation": "get_payment", "payment_id": "pay_b_1", **A})
    assert other.rows == [] and "pay_b_1" not in json.dumps(other.to_observation())
    sandbox = {**A, "environment": "sandbox"}
    assert (
        ids(tools, "query_payments", operation="get_payment", payment_id="pay_a_1", **sandbox) == []
    )
    # a parameter is data, never SQL
    assert (
        ids(
            tools,
            "query_payments",
            operation="list_payments",
            **{**A, "account_id": "x' OR 1=1 --"},
        )
        == []
    )


def test_refused_calls_are_tool_errors_with_a_reason(tools: DataTools) -> None:
    with pytest.raises(ToolError, match="account_id and environment"):
        tools.run("query_payments", {"operation": "list_payments", "account_id": "acct_a"})
    with pytest.raises(ToolError, match="needs payment_id"):
        tools.run("query_payments", {"operation": "get_payment", **A})
    with pytest.raises(ToolError, match="no operation 'drop_table'"):
        tools.run("query_payments", {"operation": "drop_table", **A})
    with pytest.raises(ToolError, match="ISO 8601"):
        tools.run("query_payments", {"operation": "list_payments", **A, "as_of": "上周"})


def test_hidden_columns_and_tables_never_come_out(tools: DataTools) -> None:
    seen = ""
    for tool in tools.tools:
        for op, rule in tools.policy["operations"][tool].items():
            args: dict[str, object] = {"operation": op, **A}
            args |= {p: "x" for p in rule["params"]}
            if op == "get_service_status":
                args |= {"service": "payments", "region": "NA", "as_of": "2026-08-20T12:00:00Z"}
            seen += json.dumps(tools.run(tool, args).to_observation(), ensure_ascii=False)
    for tool, op in [
        ("get_runtime_state", "list_requests"),
        ("get_runtime_state", "get_usage"),
        ("query_accounts", "list_api_keys"),
    ]:
        listing = {"operation": op, **A, "as_of": "2026-08-20T12:00:00Z"}
        seen += json.dumps(tools.run(tool, listing).to_observation())
    for secret in (
        "run_id",
        "run_canonical",
        "scenario_id",
        "SC-001",
        "SECRET-FP",
        "HASH",
        "THE ANSWERS",
    ):
        assert secret not in seen
    tables = set(load_policy()["columns"])
    assert not tables & {"event_journal", "scenario_executions", "simulation_runs"}


def test_the_database_is_read_only(tools: DataTools) -> None:
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        tools.store.sql.execute("DELETE FROM payments")


def test_as_of_scopes_reads_and_defaults_to_the_freeze(tools: DataTools) -> None:
    get = {"operation": "get_payment", "payment_id": "pay_a_1", **A}
    assert (
        ids(tools, "query_payments", **get, as_of="2026-07-31T00:00:00Z") == []
    )  # not yet created
    assert ids(tools, "query_payments", **get) == ["sql:payment:pay_a_1"]  # no as_of: now
    sub = {"operation": "get_subscription", "subscription_id": "sub_a_1", **A}
    assert ids(tools, "query_payments", **sub, as_of="2026-08-20T00:00:00Z") == [
        "sql:subscription:sub_a_1",
        "sql:invoice:in_a_1",
    ]  # in_a_2 is finalized later
    account = tools.run(
        "query_accounts", {"operation": "get_account", **A, "as_of": "2026-08-01T00:00:00Z"}
    )
    assert account.rows[0]["plan_at_as_of"] == "Growth"  # before the downgrade took effect


def test_request_logs_follow_the_plan_in_force(tools: DataTools) -> None:
    old = {"operation": "get_request", "request_id": "req_a_old", **A}
    # on 08-02 the account was Growth (30 days): a 1-day-old log is visible
    assert ids(tools, "get_runtime_state", **old, as_of="2026-08-02T12:00:00Z") == [
        "runtime:req:req_a_old"
    ]
    # on 08-20 it is Starter (7 days): the same log is gone, and the note says why
    late = tools.run("get_runtime_state", {**old, "as_of": "2026-08-20T12:00:00Z"})
    assert late.rows == [] and "Starter" in str(late.note) and "2026-08-13" in str(late.note)
    # a scheduled change that never took effect doesn't count
    assert tools.store.plan_at("acct_b", "2026-08-27T00:00:00.000Z") == "Growth"


def test_request_lists_are_a_24_hour_window_capped_at_20(tools: DataTools) -> None:
    listing = {"operation": "list_requests", **A, "as_of": "2026-08-20T12:00:00Z"}
    first = tools.run("get_runtime_state", listing)
    assert len(first.rows) == 20 and first.truncated
    assert "runtime:req:req_a_day_before" not in first.evidence_ids  # 26 hours earlier
    wider = tools.run("get_runtime_state", {**listing, "since": "2026-08-19T00:00:00Z"})
    assert wider.evidence_ids[0] == "runtime:req:req_a_day_before"


def test_delivery_timelines_and_incidents(tools: DataTools) -> None:
    timeline = {"operation": "get_delivery_timeline", "delivery_id": "del_a_1", **A}
    assert ids(tools, "get_runtime_state", **timeline, as_of="2026-08-20T12:00:00Z") == [
        "runtime:attempt:att_a_1",
        "runtime:attempt:att_a_2",
    ]  # the benchmark's timelines include later attempts
    assert ids(tools, "get_runtime_state", **{**timeline, "account_id": "acct_b"}) == []
    status = {"operation": "get_service_status", "service": "payments", "region": "NA"}
    assert ids(tools, "get_runtime_state", **status, as_of="2026-08-20T12:00:00Z") == [
        "runtime:incident:inc_1"
    ]
    assert ids(tools, "get_runtime_state", **status, as_of="2026-08-21T12:00:00Z") == []
    with pytest.raises(ToolError, match="incident_id, or service and region"):
        tools.run("get_runtime_state", {"operation": "get_service_status"})


def test_specs_list_operations_and_their_parameters(tools: DataTools) -> None:
    spec = tools.spec("query_payments")
    props = spec["input_schema"]["properties"]
    assert "get_invoice_timeline" in props["operation"]["enum"] and "payment_id" in props
    assert props["environment"]["enum"] == ["production", "sandbox"]
    assert "Read-only" in spec["description"]
