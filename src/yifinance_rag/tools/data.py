"""The agent's read-only data tools: four tools, 20 operations, governed by `policy.toml`.

- The business tables are opened read-only (`mode=ro`). Runtime logs are loaded into an
  in-memory SQLite with only the policy's columns, so `scenario_id` and `run_id` are
  dropped as they are read: no operation can return them.
- Every query is fixed in code with bound parameters; the agent picks an operation and
  fills its parameters, never SQL. Scope (account_id, environment) is required and bound
  in every tenant query.
- `as_of` scopes reads to what existed then (default: the snapshot's freeze time). Request
  logs and usage are also limited by the log retention of the plan in force at `as_of`.
- Each returned row carries the dataset's `evidence_id`, so what a call returned is
  comparable with gold evidence without a mapping.
"""

from __future__ import annotations

import json
import sqlite3
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

from yifinance_rag.data.dataset import Dataset

POLICY_PATH = Path(__file__).with_name("policy.toml")
SQLITE = "data/generated/canonical/flowpay.sqlite"
RUNTIME = "data/generated/canonical/runtime"
MANIFEST = "data/generated/canonical/manifest.json"
ENVIRONMENTS = ("production", "sandbox")
# Runtime log -> (table, file); loaded with the policy's columns plus evidence_id.
_LOGS = {
    "api_requests": "api_requests.jsonl",
    "usage_buckets": "usage_buckets.jsonl",
    "delivery_attempts": "webhook_delivery_attempts.jsonl",
    "service_incidents": "service_incidents.jsonl",
}


class ToolError(Exception):
    """A refused call (unknown operation, missing or bad parameter). The agent sees the
    message as the tool's result; it is not an infra failure."""


@dataclass(frozen=True)
class ToolResult:
    tool: str
    operation: str
    rows: list[dict[str, Any]]
    truncated: bool = False
    note: str | None = None

    @property
    def evidence_ids(self) -> list[str]:
        return [str(r["evidence_id"]) for r in self.rows]

    def to_observation(self) -> dict[str, Any]:
        """What the agent reads."""
        out: dict[str, Any] = {"operation": self.operation, "result_count": len(self.rows)}
        if self.truncated:
            out["truncated"] = True
        if self.note:
            out["note"] = self.note
        out["rows"] = self.rows
        return out


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def iso_ms(moment: datetime) -> str:
    """The dataset's timestamp format, so text comparison in SQL is time order."""
    return (
        moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
    )


def parse_time(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("as_of must be an ISO 8601 time, e.g. 2026-08-02T12:00:00Z")
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as e:
        raise ToolError(f"as_of {value!r} is not an ISO 8601 time") from e
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


@dataclass
class DataStore:
    """Read-only access to one dataset's business tables and runtime logs."""

    ds: Dataset
    policy: dict[str, Any] = field(default_factory=load_policy)

    @cached_property
    def sql(self) -> sqlite3.Connection:
        db = sqlite3.connect(f"file:{self.ds.path(SQLITE)}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    @cached_property
    def logs(self) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        for table, name in _LOGS.items():
            cols = [*self.policy["columns"][table], "evidence_id"]
            db.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
            path = self.ds.path(f"{RUNTIME}/{name}")
            rows = []
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw = json.loads(line)
                        rows.append([_cell(raw.get(c)) for c in cols])
            db.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(cols))})", rows)
        return db

    @cached_property
    def freeze_at(self) -> datetime:
        manifest = json.loads(self.ds.path(MANIFEST).read_text(encoding="utf-8"))
        return parse_time(manifest["freeze_at"])

    def plan_at(self, account_id: str, at: str) -> str | None:
        """The plan in force at `at`: the current plan, with effective changes that took
        effect later undone. Scheduled changes never applied don't count."""
        row = self.sql.execute(
            "SELECT current_plan FROM accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
        if row is None:
            return None
        plan = str(row["current_plan"])
        later = self.sql.execute(
            "SELECT from_plan FROM plan_changes WHERE account_id = ? AND status = 'effective' "
            "AND effective_at > ? ORDER BY effective_at DESC",
            (account_id, at),
        ).fetchall()
        for change in later:
            plan = str(change["from_plan"])
        return plan

    def retention_cutoff(self, account_id: str, at: datetime) -> str | None:
        plan = self.plan_at(account_id, iso_ms(at))
        days = self.policy["retention_days"].get(plan) if plan else None
        return iso_ms(at - timedelta(days=int(days))) if days is not None else None

    def close(self) -> None:
        for name in ("sql", "logs"):
            if name in self.__dict__:
                self.__dict__.pop(name).close()


def _cell(value: object) -> object:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict | list) else value


@dataclass(frozen=True)
class Call:
    """A validated call: scope and as_of normalized."""

    tool: str
    operation: str
    params: Mapping[str, str]
    account_id: str
    environment: str
    as_of: datetime

    @property
    def at(self) -> str:
        return iso_ms(self.as_of)


Op = Callable[["DataTools", Call], ToolResult]


class DataTools:
    def __init__(self, store: DataStore) -> None:
        self.store = store
        self.policy = store.policy
        self.max_rows = int(self.policy["max_rows"])
        self._ops: dict[tuple[str, str], Op] = {
            ("query_accounts", "get_account"): DataTools._get_account,
            ("query_accounts", "get_plan_timeline"): DataTools._get_plan_timeline,
            ("query_accounts", "list_api_keys"): DataTools._list_api_keys,
            ("query_accounts", "get_feature_flags"): DataTools._get_feature_flags,
            ("query_payments", "get_payment"): DataTools._get_payment,
            ("query_payments", "list_payments"): DataTools._list_payments,
            ("query_payments", "get_refund"): DataTools._get_refund,
            ("query_payments", "get_customer"): DataTools._get_customer,
            ("query_payments", "list_customers"): DataTools._list_customers,
            ("query_payments", "get_subscription"): DataTools._get_subscription,
            ("query_payments", "get_invoice_timeline"): DataTools._get_invoice_timeline,
            ("query_webhooks", "get_endpoint"): DataTools._get_endpoint,
            ("query_webhooks", "list_endpoints"): DataTools._list_endpoints,
            ("query_webhooks", "get_event"): DataTools._get_event,
            ("query_webhooks", "get_delivery"): DataTools._get_delivery,
            ("get_runtime_state", "get_request"): DataTools._get_request,
            ("get_runtime_state", "list_requests"): DataTools._list_requests,
            ("get_runtime_state", "get_usage"): DataTools._get_usage,
            ("get_runtime_state", "get_delivery_timeline"): DataTools._get_delivery_timeline,
            ("get_runtime_state", "get_service_status"): DataTools._get_service_status,
        }
        declared = {(t, o) for t, ops in self.policy["operations"].items() for o in ops}
        if declared != set(self._ops):
            raise ValueError(f"policy and code disagree on operations: {declared ^ set(self._ops)}")

    @property
    def tools(self) -> list[str]:
        return list(self.policy["operations"])

    def spec(self, tool: str) -> dict[str, Any]:
        """The tool's description and JSON schema for the model."""
        ops = self.policy["operations"][tool]
        lines = [f"- {name}: {o['description']}" for name, o in ops.items()]
        needed = sorted({p for o in ops.values() for p in [*o["params"], *o.get("optional", [])]})
        props: dict[str, Any] = {
            "operation": {"type": "string", "enum": list(ops)},
            "account_id": {"type": "string"},
            "environment": {"type": "string", "enum": list(ENVIRONMENTS)},
            "as_of": {
                "type": "string",
                "description": "ISO 8601 time the question is about; omit for now",
            },
        }
        props |= {p: {"type": "string"} for p in needed}
        return {
            "name": tool,
            "description": "Read-only. Operations:\n" + "\n".join(lines),
            "input_schema": {"type": "object", "properties": props, "required": ["operation"]},
        }

    def run(self, tool: str, args: Mapping[str, object]) -> ToolResult:
        operation = str(args.get("operation") or "")
        op = self._ops.get((tool, operation))
        if op is None:
            known = ", ".join(self.policy["operations"].get(tool, {})) or "none"
            raise ToolError(f"{tool} has no operation {operation!r}; operations: {known}")
        rule = self.policy["operations"][tool][operation]
        scoped = rule.get("scoped", True)
        account_id = str(args.get("account_id") or "")
        environment = str(args.get("environment") or "")
        if scoped and (not account_id or environment not in ENVIRONMENTS):
            raise ToolError("account_id and environment (production or sandbox) are required")
        missing = [p for p in rule["params"] if not str(args.get(p) or "").strip()]
        if missing:
            raise ToolError(f"{operation} needs {', '.join(missing)}")
        allowed = [*rule["params"], *rule.get("optional", [])]
        params = {p: str(args[p]).strip() for p in allowed if str(args.get(p) or "").strip()}
        raw_as_of = args.get("as_of")
        as_of = self.store.freeze_at if raw_as_of in (None, "") else parse_time(raw_as_of)
        call = Call(tool, operation, params, account_id, environment, as_of)
        return op(self, call)

    # ---- helpers -------------------------------------------------------------------

    def _cols(self, table: str) -> list[str]:
        return list(self.policy["columns"][table])

    def _select(self, table: str, prefix: str = "") -> str:
        return ", ".join(f"{prefix}{c}" for c in self._cols(table))

    def _rows(
        self,
        db: sqlite3.Connection,
        table: str,
        sql: str,
        args: Sequence[object],
        evidence: Callable[[dict[str, Any]], str] | None = None,
        *,
        cap: bool = False,
    ) -> tuple[list[dict[str, Any]], bool]:
        limit = self.max_rows + 1 if cap else None
        query = sql + (f" LIMIT {limit}" if limit else "")
        out: list[dict[str, Any]] = []
        for r in db.execute(query, args).fetchall():
            row = {c: r[c] for c in self._cols(table)}
            row["evidence_id"] = evidence(row) if evidence else r["evidence_id"]
            out.append(row)
        truncated = cap and len(out) > self.max_rows
        return out[: self.max_rows] if truncated else out, truncated

    def _result(
        self, c: Call, rows: list[dict[str, Any]], truncated: bool = False, note: str | None = None
    ) -> ToolResult:
        if not rows and note is None:
            note = "no record in this account and environment at as_of"
        return ToolResult(c.tool, c.operation, rows, truncated, note)

    def _one(self, c: Call, table: str, key: str, time_col: str, kind: str) -> ToolResult:
        rows, _ = self._rows(
            self.store.sql,
            table,
            f"SELECT {self._select(table)} FROM {table} WHERE {key} = ? AND account_id = ? "
            f"AND environment = ? AND {time_col} <= ?",
            (c.params[key], c.account_id, c.environment, c.at),
            lambda r: f"sql:{kind}:{r[key]}",
        )
        return self._result(c, rows)

    def _list(self, c: Call, table: str, key: str, time_col: str, kind: str) -> ToolResult:
        rows, truncated = self._rows(
            self.store.sql,
            table,
            f"SELECT {self._select(table)} FROM {table} WHERE account_id = ? AND environment = ? "
            f"AND {time_col} <= ? ORDER BY {time_col}, {key}",
            (c.account_id, c.environment, c.at),
            lambda r: f"sql:{kind}:{r[key]}",
            cap=True,
        )
        return self._result(c, rows, truncated)

    def _window(self, c: Call) -> tuple[str, str | None]:
        """Start of a log listing: `since` or the default window before as_of, never
        before the plan's retention. The 24-hour default is the window consistent with
        every benchmark selector (one excludes a request 25 hours old)."""
        cutoff, note = self._retained(c)
        start = (
            parse_time(c.params["since"])
            if "since" in c.params
            else c.as_of - timedelta(hours=int(self.policy["window_hours"]))
        )
        return max(cutoff, iso_ms(start)), note

    def _retained(self, c: Call) -> tuple[str, str | None]:
        cutoff = self.store.retention_cutoff(c.account_id, c.as_of)
        plan = self.store.plan_at(c.account_id, c.at)
        note = (
            f"logs before {cutoff} are outside the {plan} plan's retention"
            if cutoff and plan
            else None
        )
        return cutoff or "", note

    # ---- query_accounts ----------------------------------------------------------

    def _get_account(self, c: Call) -> ToolResult:
        rows, _ = self._rows(
            self.store.sql,
            "accounts",
            f"SELECT {self._select('accounts')} FROM accounts WHERE account_id = ? "
            "AND created_at <= ?",
            (c.account_id, c.at),
            lambda r: f"sql:account:{r['account_id']}",
        )
        if rows:
            rows[0]["plan_at_as_of"] = self.store.plan_at(c.account_id, c.at)
        return self._result(c, rows)

    def _get_plan_timeline(self, c: Call) -> ToolResult:
        rows, _ = self._rows(
            self.store.sql,
            "plan_changes",
            f"SELECT {self._select('plan_changes')} FROM plan_changes WHERE account_id = ? "
            "AND requested_at <= ? ORDER BY requested_at",
            (c.account_id, c.at),
            lambda r: f"sql:account_plan:{r['plan_change_id']}",
        )
        note = None if rows else "no plan change requested up to as_of"
        return self._result(c, rows, note=note)

    def _list_api_keys(self, c: Call) -> ToolResult:
        return self._list(c, "api_keys", "key_id", "created_at", "api_key")

    def _get_feature_flags(self, c: Call) -> ToolResult:
        rows, _ = self._rows(
            self.store.sql,
            "feature_flags",
            f"SELECT {self._select('feature_flags')} FROM feature_flags WHERE account_id = ? "
            "AND environment = ? AND effective_at <= ? ORDER BY feature_name",
            (c.account_id, c.environment, c.at),
            lambda r: f"sql:feature_flag:{r['account_id']}:{r['environment']}:{r['feature_name']}",
        )
        return self._result(c, rows)

    # ---- query_payments ----------------------------------------------------------

    def _get_payment(self, c: Call) -> ToolResult:
        return self._one(c, "payments", "payment_id", "created_at", "payment")

    def _list_payments(self, c: Call) -> ToolResult:
        return self._list(c, "payments", "payment_id", "created_at", "payment")

    def _get_refund(self, c: Call) -> ToolResult:
        return self._one(c, "refunds", "refund_id", "created_at", "refund")

    def _get_customer(self, c: Call) -> ToolResult:
        return self._one(c, "customers", "customer_id", "created_at", "customer")

    def _list_customers(self, c: Call) -> ToolResult:
        return self._list(c, "customers", "customer_id", "created_at", "customer")

    def _get_subscription(self, c: Call) -> ToolResult:
        sub = self._one(c, "subscriptions", "subscription_id", "created_at", "subscription")
        if not sub.rows:
            return sub
        invoices, _ = self._rows(
            self.store.sql,
            "invoices",
            f"SELECT {self._select('invoices')} FROM invoices WHERE subscription_id = ? "
            "AND account_id = ? AND environment = ? "
            "AND (finalized_at IS NULL OR finalized_at <= ?) "
            "ORDER BY period_start, invoice_id",
            (c.params["subscription_id"], c.account_id, c.environment, c.at),
            lambda r: f"sql:invoice:{r['invoice_id']}",
        )
        return self._result(c, sub.rows + invoices)

    def _get_invoice_timeline(self, c: Call) -> ToolResult:
        invoice, _ = self._rows(
            self.store.sql,
            "invoices",
            f"SELECT {self._select('invoices')} FROM invoices WHERE invoice_id = ? "
            "AND account_id = ? AND environment = ?",
            (c.params["invoice_id"], c.account_id, c.environment),
            lambda r: f"sql:invoice:{r['invoice_id']}",
        )
        if not invoice:
            return self._result(c, [])
        attempts, _ = self._rows(
            self.store.sql,
            "invoice_payment_attempts",
            f"SELECT {self._select('invoice_payment_attempts')} FROM invoice_payment_attempts "
            "WHERE invoice_id = ? ORDER BY attempt_number",
            (c.params["invoice_id"],),
            lambda r: f"sql:invoice_attempt:{r['invoice_id']}:{r['attempt_number']}",
        )
        return self._result(c, invoice + attempts)

    # ---- query_webhooks ----------------------------------------------------------

    def _get_endpoint(self, c: Call) -> ToolResult:
        return self._one(c, "webhook_endpoints", "endpoint_id", "created_at", "webhook_endpoint")

    def _list_endpoints(self, c: Call) -> ToolResult:
        return self._list(c, "webhook_endpoints", "endpoint_id", "created_at", "webhook_endpoint")

    def _get_event(self, c: Call) -> ToolResult:
        return self._one(c, "events", "event_id", "occurred_at", "event")

    def _get_delivery(self, c: Call) -> ToolResult:
        # Deliveries have no tenant columns: scope through their event.
        rows, _ = self._rows(
            self.store.sql,
            "webhook_deliveries",
            f"SELECT {self._select('webhook_deliveries', 'd.')} FROM webhook_deliveries d "
            "JOIN events e ON e.event_id = d.event_id WHERE d.delivery_id = ? "
            "AND e.account_id = ? AND e.environment = ? "
            "AND COALESCE(d.first_attempt_at, e.occurred_at) <= ?",
            (c.params["delivery_id"], c.account_id, c.environment, c.at),
            lambda r: f"sql:webhook_delivery:{r['delivery_id']}",
        )
        return self._result(c, rows)

    # ---- get_runtime_state -------------------------------------------------------

    def _get_request(self, c: Call) -> ToolResult:
        cutoff, note = self._retained(c)
        rows, _ = self._rows(
            self.store.logs,
            "api_requests",
            f"SELECT {self._select('api_requests')}, evidence_id FROM api_requests "
            "WHERE request_id = ? AND account_id = ? AND environment = ? "
            "AND started_at <= ? AND started_at >= ?",
            (c.params["request_id"], c.account_id, c.environment, c.at, cutoff),
        )
        return self._result(c, rows, note=None if rows else note)

    def _list_requests(self, c: Call) -> ToolResult:
        cutoff, note = self._window(c)
        rows, truncated = self._rows(
            self.store.logs,
            "api_requests",
            f"SELECT {self._select('api_requests')}, evidence_id FROM api_requests "
            "WHERE account_id = ? AND environment = ? AND started_at <= ? AND started_at >= ? "
            "ORDER BY started_at, request_id",
            (c.account_id, c.environment, c.at, cutoff),
            cap=True,
        )
        return self._result(c, rows, truncated, note if not rows else None)

    def _get_usage(self, c: Call) -> ToolResult:
        cutoff, note = self._window(c)
        rows, truncated = self._rows(
            self.store.logs,
            "usage_buckets",
            f"SELECT {self._select('usage_buckets')}, evidence_id FROM usage_buckets "
            "WHERE account_id = ? AND environment = ? AND bucket_start <= ? "
            "AND bucket_start >= ? ORDER BY bucket_start",
            (c.account_id, c.environment, c.at, cutoff),
            cap=True,
        )
        return self._result(c, rows, truncated, note if not rows else None)

    def _get_delivery_timeline(self, c: Call) -> ToolResult:
        owner = self.store.sql.execute(
            "SELECT 1 FROM webhook_deliveries d JOIN events e ON e.event_id = d.event_id "
            "WHERE d.delivery_id = ? AND e.account_id = ? AND e.environment = ?",
            (c.params["delivery_id"], c.account_id, c.environment),
        ).fetchone()
        if owner is None:
            return self._result(c, [])
        rows, _ = self._rows(
            self.store.logs,
            "delivery_attempts",
            f"SELECT {self._select('delivery_attempts')}, evidence_id FROM delivery_attempts "
            "WHERE delivery_id = ? ORDER BY attempt_number",
            (c.params["delivery_id"],),
        )
        return self._result(c, rows)

    def _get_service_status(self, c: Call) -> ToolResult:
        if "incident_id" in c.params:
            sql = "incident_id = ?"
            args: tuple[str, ...] = (c.params["incident_id"],)
        elif "service" in c.params and "region" in c.params:
            sql = (
                "service = ? AND region = ? AND started_at <= ? "
                "AND (resolved_at IS NULL OR resolved_at >= ?)"
            )
            args = (c.params["service"], c.params["region"], c.at, c.at)
        else:
            raise ToolError("get_service_status needs incident_id, or service and region")
        rows, _ = self._rows(
            self.store.logs,
            "service_incidents",
            f"SELECT {self._select('service_incidents')}, evidence_id FROM service_incidents "
            f"WHERE {sql} ORDER BY started_at",
            args,
        )
        return self._result(c, rows, note=None if rows else "no incident matches at as_of")
