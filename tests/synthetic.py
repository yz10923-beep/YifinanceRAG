"""A tiny dataset in the FlowPay layout, written to a temp dir, so loaders, checks and the
retrieval eval run end to end in `make verify` without the real (course-only) data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yifinance_rag.data.dataset import RELEASE_MANIFEST

DOCS = {
    "data/knowledge/troubleshooting/KB-TS-012_fp_4041.md": """---
doc_id: KB-TS-012
title: FP-4041 资源不存在或不可见
doc_type: troubleshooting
rule_ids:
  - BR-001
---
# FP-4041 资源不存在或不可见

## 现象

请求返回 FP-4041，提示资源不存在。

## 常见原因

使用了沙箱密钥查询生产环境的 pay_bm_001，或资源属于其他账户。

## 处理步骤

确认环境与账户一致后重试。
""",
    "data/knowledge/business_rules/KB-BR-012_refund_window.md": """---
doc_id: KB-BR-012
title: 退款资格、金额与时间窗口
doc_type: business_rule
---
# 退款资格、金额与时间窗口

适用于所有套餐。

## 时间窗口

支付成功后 180 天内可以发起退款，超过窗口返回 FP-4222。

## 部分退款

Starter 套餐不支持部分退款，Growth 及以上支持。

## 示例

## 示例

同一标题可以出现两次。
""",
    "data/knowledge/api_docs/KB-API-033_idempotency_keys.md": """---
doc_id: KB-API-033
title: 幂等键
doc_type: api_doc
---
# 幂等键

## 用法

在请求头 Idempotency-Key 中传入幂等键，24 小时内重复请求返回同一结果。

## 冲突

同一幂等键配合不同参数时返回 FP-4091 幂等冲突。
""",
}

# The benchmark's chunking of the docs above, written by hand from the rule (not by our
# code), so the rebuild is checked against an independent answer.
CHUNKS = [
    ("KB-API-033", "c001", "幂等键"),
    ("KB-API-033", "c002", "用法"),
    ("KB-API-033", "c003", "冲突"),
    ("KB-BR-012", "c001", "退款资格、金额与时间窗口"),
    ("KB-BR-012", "c002", "时间窗口"),
    ("KB-BR-012", "c003", "部分退款"),
    ("KB-BR-012", "c004", "示例"),
    ("KB-BR-012", "c005", "示例"),
    ("KB-TS-012", "c001", "FP-4041 资源不存在或不可见"),
    ("KB-TS-012", "c002", "现象"),
    ("KB-TS-012", "c003", "常见原因"),
    ("KB-TS-012", "c004", "处理步骤"),
]

QUESTIONS = [
    (
        "Q-SR-001",
        "simple_retrieval",
        "dev",
        "查询支付时返回 FP-4041 是什么原因？",
        ["doc:KB-TS-012#c003"],
    ),
    (
        "Q-MD-001",
        "multi_document",
        "dev",
        "退款超过时间窗口会怎样，Starter 能部分退款吗？",
        ["doc:KB-BR-012#c002", "doc:KB-BR-012#c003", "sql:payments:pay_bm_001"],
    ),
    ("Q-MS-001", "multi_source", "dev", "我的 webhook 为什么没收到？", ["runtime:req_1"]),
    ("Q-SR-002", "simple_retrieval", "test", "幂等键冲突返回什么错误码？", ["doc:KB-API-033#c003"]),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blueprint(
    qid: str, qtype: str, split: str, text: str, evidence: list[str]
) -> dict[str, object]:
    return {
        "question_id": qid,
        "question": text,
        "question_type": qtype,
        "split": split,
        "conversation": {"mode": "single_turn_complete", "available_user_facts": []},
        "ground_truth": {"canonical_answer": f"标准答案 {qid}", "required_assertions": []},
        "resolved_evidence_ids": evidence,
    }


# A second question set for the S3 graders, in the real blueprint and resolution layout:
# two scenario families (the wrong baseline needs another family), each with a
# single-turn and a dynamic question, plus one test question. Written instead of
# QUESTIONS by write_dataset(root, graded=True).
F1_AS_OF = "2026-08-02T12:00:00Z"
F2_AS_OF = "2026-08-10T12:00:00Z"
NEGATIVE_PAYMENT = "negative:sql:payment:294c61a6865e5f11ad07"


def _selector(
    alias: str, tool: str, ids: list[str], params: dict[str, str], as_of: str, **kw: str
) -> dict[str, object]:
    negative = ids[0].startswith("negative:")
    return {
        "alias": alias,
        "tool": tool,
        "operation": alias,
        "cardinality": kw.get("cardinality", "exactly_one"),
        "evidence_mode": "negative_assertion" if negative else "records",
        "evidence_ids": ids,
        "normalized_payload": {"operation": alias, "as_of": as_of, **params},
        "ok": True,
    }


F1_SCOPE = {"account_id": "acct_bm_001", "environment": "production"}
F2_SCOPE = {"account_id": "acct_bm_036", "environment": "production"}
GRADED: list[dict[str, object]] = [
    {
        "question_id": "G-SR-001",
        "question_type": "simple_retrieval",
        "split": "dev",
        "family": "SF-001-tenant",
        "question": "查询另一个账户的支付返回 FP-4041，能说明支付不存在吗？",
        "as_of": F1_AS_OF,
        "assertions": ["资源属于其他账户时返回 FP-4041，不能证明资源不存在。"],
        "negative": "不得认定：把 FP-4041 直接解释为 Payment ID 拼写错误",
        "docs": ["doc:KB-TS-012#c003"],
        "tools": ["search_troubleshooting"],
    },
    {
        "question_id": "G-MS-001",
        "question_type": "multi_source",
        "split": "dev",
        "family": "SF-001-tenant",
        "question": "后台能看到一笔支付，但服务里查询提示找不到。",
        "as_of": F1_AS_OF,
        "assertions": ["资源属于其他账户时返回 FP-4041，不能证明资源不存在。"],
        "negative": "不得认定：把 FP-4041 直接解释为 Payment ID 拼写错误",
        "docs": ["doc:KB-TS-012#c003"],
        "tools": [
            "search_troubleshooting",
            "query_accounts",
            "query_payments",
            "get_runtime_state",
        ],
        "categories": ["occurrence_time", "identifiers"],
        "selectors": [
            _selector(
                "get_account", "query_accounts", ["sql:account:acct_bm_001"], F1_SCOPE, F1_AS_OF
            ),
            _selector(
                "get_payment",
                "query_payments",
                [NEGATIVE_PAYMENT],
                {**F1_SCOPE, "payment_id": "pay_bm_001"},
                F1_AS_OF,
            ),
            _selector(
                "get_request",
                "get_runtime_state",
                ["runtime:req:req_bm_001_01"],
                {**F1_SCOPE, "request_id": "req_bm_001_01"},
                F1_AS_OF,
            ),
        ],
        "trajectory": [
            "账户 acct_bm_001 当时套餐为 Growth、主地区为 NA，Production 状态为 active。",
            "在限定范围（as_of=2026-08-02T12:00:00Z、payment_id=pay_bm_001、"
            "account_id=acct_bm_001）内没有查到 get_payment 对应记录。",
            "请求 req_bm_001_01 在 2026 年 08 月 02 日 11:50 UTC 以 POST 调用 /v1/payments，"
            "返回 HTTP 404、错误码 FP-4041。",
        ],
    },
    {
        "question_id": "G-MD-002",
        "question_type": "multi_document",
        "split": "dev",
        "family": "SF-002-refund",
        "question": "支付成功 200 天后发起退款会怎样，Starter 套餐能部分退款吗？",
        "as_of": F2_AS_OF,
        "assertions": [
            "支付成功后 180 天内可以发起退款，超过窗口返回 FP-4222。",
            "Starter 套餐不支持部分退款。",
        ],
        "negative": "不得认定：超过窗口仍可人工退款",
        "docs": ["doc:KB-BR-012#c002", "doc:KB-BR-012#c003"],
        "tools": ["search_business_rules"],
    },
    {
        "question_id": "G-MH-002",
        "question_type": "multi_hop",
        "split": "dev",
        "family": "SF-002-refund",
        "question": "客户的退款一直失败，不知道是不是规则问题。",
        "as_of": F2_AS_OF,
        "assertions": ["支付成功后 180 天内可以发起退款，超过窗口返回 FP-4222。"],
        "negative": "不得认定：超过窗口仍可人工退款",
        "docs": ["doc:KB-BR-012#c002"],
        "tools": ["search_business_rules", "query_payments", "get_runtime_state"],
        "categories": ["identifiers"],
        "selectors": [
            _selector(
                "get_refund",
                "query_payments",
                ["sql:refund:re_bm_036"],
                {**F2_SCOPE, "refund_id": "re_bm_036"},
                F2_AS_OF,
            ),
            _selector(
                "list_requests",
                "get_runtime_state",
                ["runtime:req:req_bm_036_01", "runtime:req:req_bm_036_02"],
                F2_SCOPE,
                F2_AS_OF,
                cardinality="at_least_one",
            ),
        ],
        "trajectory": [
            "退款 re_bm_036 状态 failed，金额 5000，错误码 FP-4222。",
            "请求 req_bm_036_01 在 2026 年 08 月 10 日 09:05 UTC 返回 HTTP 422。",
        ],
    },
    {
        "question_id": "G-SR-003",
        "question_type": "simple_retrieval",
        "split": "test",
        "family": "SF-003-idempotency",
        "question": "幂等键冲突返回什么错误码？",
        "as_of": F2_AS_OF,
        "assertions": ["同一幂等键配合不同参数时返回 FP-4091。"],
        "negative": "不得认定：幂等冲突可以换参数重试",
        "docs": ["doc:KB-API-033#c003"],
        "tools": ["search_api_docs"],
    },
]


def graded_blueprint(spec: dict[str, object]) -> dict[str, object]:
    """A resolved blueprint with every field the graders read."""
    qid = str(spec["question_id"])
    selectors = list(spec.get("selectors", []))  # type: ignore[call-overload]
    trajectory = list(spec.get("trajectory", []))  # type: ignore[call-overload]
    dynamic = bool(selectors)
    assertions = [str(a) for a in spec["assertions"]]  # type: ignore[attr-defined]
    canonical = " ".join(assertions)
    categories = list(spec.get("categories", []))  # type: ignore[call-overload]
    ids = [*spec["docs"], *(e for s in selectors for e in s["evidence_ids"])]  # type: ignore[misc]
    return {
        "question_id": qid,
        "question": spec["question"],
        "question_type": spec["question_type"],
        "split": spec["split"],
        "scenario_family_id": spec["family"],
        "as_of": spec["as_of"],
        "conversation": {
            "mode": "progressive_disclosure" if dynamic else "single_turn_complete",
            "initial_user_message": spec["question"],
            "available_user_facts": [],
            "required_clarification_categories": categories,
            "max_clarification_turns": 2 if dynamic else 0,
            "clarification_plan": (
                [
                    {
                        "assistant_question": "什么时候发生的？相关编号是什么？",
                        "categories": categories,
                    }
                ]
                if dynamic
                else []
            ),
        },
        "ground_truth": {
            "canonical_answer": canonical,
            "required_assertions": [
                {"assertion_id": f"{qid}-A{i}", "text": a} for i, a in enumerate(assertions, 1)
            ],
            "negative_assertions": [spec["negative"]],
            "acceptable_variants": [],
        },
        "resolved_answer": ("\n".join([*trajectory, canonical]) if dynamic else None),
        "resolved_trajectory_assertions": [
            {
                "assertion_id": f"{qid}-T{i}",
                "selector_alias": s["alias"],
                "text": text,
                "evidence_ids": s["evidence_ids"],
            }
            for i, (s, text) in enumerate(zip(selectors, trajectory, strict=True), 1)
        ],
        "required_tools": spec["tools"],
        "required_operations": [f"{s['tool']}.{s['operation']}" for s in selectors],
        "forbidden_shortcuts": [
            "support_case_as_current_fact",
            "current_state_used_for_historical_question",
            "raw_sql_or_hidden_audit_access",
        ],
        "resolved_evidence_ids": ids,
    }


def write_dataset(root: Path, *, graded: bool = False) -> Path:
    for rel, text in DOCS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    index = root / "data/index"
    index.mkdir(parents=True)
    rows = [
        {"evidence_id": f"doc:{d}#{c}", "doc_id": d, "section": h, "tool": "search"}
        for d, c, h in CHUNKS
    ]
    chunks_file = index / "knowledge_chunks.jsonl"
    chunks_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (index / "index_manifest.json").write_text(
        json.dumps({"chunks": {"path": "knowledge_chunks.jsonl", "sha256": _sha(chunks_file)}})
    )
    bench = root / "benchmark/resolved"
    bench.mkdir(parents=True)
    if graded:
        blueprints = [graded_blueprint(s) for s in GRADED]
        resolution = [
            {
                "question_id": s["question_id"],
                "resolved_evidence_ids": b["resolved_evidence_ids"],
                "selector_results": s.get("selectors", []),
            }
            for s, b in zip(GRADED, blueprints, strict=True)
        ]
    else:
        blueprints = [blueprint(*q) for q in QUESTIONS]
        resolution = [{"question_id": q[0], "resolved_evidence_ids": q[4]} for q in QUESTIONS]
    (bench / "resolved_blueprints.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blueprints)
    )
    (bench / "evidence_resolution.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in resolution)
    )
    write_release(root)
    return root


def write_release(root: Path) -> None:
    """The real release's group ids and visibility labels; groups with no synthetic
    files are listed empty. `generation_sources` lists a file the zip doesn't ship, as the
    real course-member zip does."""
    files = {
        ("published_knowledge", "agent_knowledge"): list(DOCS),
        ("knowledge_index", "retrieval_runtime"): ["data/index/knowledge_chunks.jsonl"],
        ("benchmark_design", "hidden_evaluation"): [],
        ("compiled_scenarios", "generator_internal"): [],
        ("canonical_agent_data", "agent_via_typed_tools"): [],
        ("canonical_audit_data", "forbidden_to_agent"): [],
        ("resolved_evidence", "hidden_evaluation"): [
            "benchmark/resolved/resolved_blueprints.jsonl"
        ],
    }
    groups: list[dict[str, object]] = [
        {
            "group_id": gid,
            "visibility": vis,
            "file_count": len(rels),
            "artifacts": [
                {"path": rel, "sha256": _sha(root / rel), "bytes": (root / rel).stat().st_size}
                for rel in rels
            ],
        }
        for (gid, vis), rels in files.items()
    ]
    groups.append(
        {
            "group_id": "generation_sources",
            "visibility": "provenance_source",
            "file_count": 1,
            "artifacts": [{"path": "src/not_shipped.py", "sha256": "0" * 64, "bytes": 1}],
        }
    )
    release = {"status": "frozen", "release_id": "synthetic", "artifact_groups": groups}
    path = root / RELEASE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(release))


# A tiny canonical snapshot for the data tools: two tenants, a plan downgrade (retention
# shrinks from 30 to 7 days), more requests than one page, a delivery attempt after as_of,
# and the things no tool may return (run_id, scenario_id, the audit journal).
CANONICAL = "data/generated/canonical"


def _table(db: object, name: str, rows: list[dict[str, object]]) -> None:
    import sqlite3

    assert isinstance(db, sqlite3.Connection)
    cols = list(rows[0])
    db.execute(f"CREATE TABLE {name} ({', '.join(cols)})")
    db.executemany(
        f"INSERT INTO {name} VALUES ({', '.join('?' * len(cols))})",
        [[r[c] for c in cols] for r in rows],
    )


def write_canonical(root: Path) -> Path:
    import sqlite3

    base = root / CANONICAL
    (base / "runtime").mkdir(parents=True, exist_ok=True)
    (base / "manifest.json").write_text(json.dumps({"freeze_at": "2026-08-27T12:00:00.000Z"}))
    db = sqlite3.connect(base / "flowpay.sqlite")
    run = {"run_id": "run_canonical_x"}
    _table(
        db,
        "accounts",
        [
            {
                **run,
                "account_id": a,
                "name": a,
                "primary_region": "NA",
                "default_api_version": "2026-07-01",
                "production_status": "active",
                "current_plan": plan,
                "billing_cycle_anchor": None,
                "created_at": "2026-07-01T00:00:00.000Z",
                "updated_at": "2026-08-05T00:00:00.000Z",
            }
            for a, plan in (("acct_a", "Starter"), ("acct_b", "Growth"))
        ],
    )
    _table(
        db,
        "plan_changes",
        [
            {
                **run,
                "plan_change_id": "pc_a_1",
                "account_id": "acct_a",
                "from_plan": "Growth",
                "to_plan": "Starter",
                "requested_at": "2026-08-05T00:00:00.000Z",
                "effective_at": "2026-08-05T00:00:00.000Z",
                "status": "effective",
            },
            {
                **run,
                "plan_change_id": "pc_b_1",
                "account_id": "acct_b",
                "from_plan": "Growth",
                "to_plan": "Scale",
                "requested_at": "2026-08-06T00:00:00.000Z",
                "effective_at": "2026-08-26T00:00:00.000Z",
                "status": "scheduled",
            },
        ],
    )
    _table(
        db,
        "api_keys",
        [
            {
                **run,
                "key_id": "key_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "key_type": "secret",
                "key_prefix": "sk_live_",
                "key_fingerprint": "SECRET-FP",
                "scopes_json": "[]",
                "status": "active",
                "created_at": "2026-07-02T00:00:00.000Z",
                "expires_at": None,
                "revoked_at": None,
                "replaced_by_key_id": None,
            },
        ],
    )
    _table(
        db,
        "feature_flags",
        [
            {
                **run,
                "account_id": "acct_a",
                "environment": "production",
                "feature_name": "subscriptions",
                "enabled": 1,
                "effective_at": "2026-08-01T00:00:00.000Z",
                "updated_at": "2026-08-01T00:00:00.000Z",
            },
        ],
    )
    _table(
        db,
        "customers",
        [
            {
                **run,
                "customer_id": "cus_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "email": "a@example.com",
                "name": "A",
                "metadata_json": "{}",
                "status": "active",
                "created_at": "2026-07-05T00:00:00.000Z",
                "updated_at": "2026-07-05T00:00:00.000Z",
                "deleted_at": None,
            },
        ],
    )
    pay = {
        "customer_id": "cus_a_1",
        "currency": "USD",
        "status": "succeeded",
        "api_version": "2026-07-01",
        "payment_method_profile_id": None,
        "failure_code": None,
        "failure_category": None,
        "idempotency_key_hash": "HASH",
        "created_by_request_id": None,
        "updated_at": None,
        "succeeded_at": None,
        "terminal_at": None,
    }
    _table(
        db,
        "payments",
        [
            {
                **run,
                **pay,
                "payment_id": "pay_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "amount": 10000,
                "created_at": "2026-08-01T10:00:00.000Z",
            },
            {
                **run,
                **pay,
                "payment_id": "pay_b_1",
                "account_id": "acct_b",
                "environment": "production",
                "amount": 5000,
                "created_at": "2026-08-01T10:00:00.000Z",
            },
        ],
    )
    _table(
        db,
        "refunds",
        [
            {
                **run,
                "refund_id": "re_a_1",
                "payment_id": "pay_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "amount": 2000,
                "currency": "USD",
                "status": "pending",
                "mode": "standard",
                "failure_code": None,
                "created_by_request_id": None,
                "created_at": "2026-08-10T00:00:00.000Z",
                "updated_at": None,
                "completed_at": None,
            },
        ],
    )
    _table(
        db,
        "subscriptions",
        [
            {
                **run,
                "subscription_id": "sub_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "customer_id": "cus_a_1",
                "status": "active",
                "amount": 2500,
                "currency": "USD",
                "interval": "month",
                "trial_end_at": None,
                "cancel_at_period_end": 0,
                "current_period_start": None,
                "current_period_end": None,
                "created_at": "2026-07-10T00:00:00.000Z",
                "updated_at": None,
                "canceled_at": None,
            },
        ],
    )
    inv = {
        "subscription_id": "sub_a_1",
        "account_id": "acct_a",
        "environment": "production",
        "amount_due": 2500,
        "currency": "USD",
        "period_start": None,
        "period_end": None,
    }
    _table(
        db,
        "invoices",
        [
            {
                **run,
                **inv,
                "invoice_id": "in_a_1",
                "status": "paid",
                "amount_paid": 2500,
                "finalized_at": "2026-08-10T00:00:00.000Z",
                "paid_at": "2026-08-10T00:00:00.000Z",
            },
            {
                **run,
                **inv,
                "invoice_id": "in_a_2",
                "status": "open",
                "amount_paid": 0,
                "finalized_at": "2026-08-25T00:00:00.000Z",
                "paid_at": None,
            },
        ],
    )
    _table(
        db,
        "invoice_payment_attempts",
        [
            {
                **run,
                "invoice_id": "in_a_1",
                "attempt_number": 0,
                "payment_id": "pay_a_1",
                "scheduled_at": None,
                "executed_at": "2026-08-10T00:00:00.000Z",
                "outcome": "succeeded",
            },
        ],
    )
    _table(
        db,
        "webhook_endpoints",
        [
            {
                **run,
                "endpoint_id": "we_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "url_host": "a.example.com",
                "url_path_hash": "HASH",
                "status": "enabled",
                "api_version": "2026-07-01",
                "secret_version": 1,
                "secret_fingerprint": "SECRET-FP",
                "retry_policy_id": "default",
                "created_at": "2026-07-03T00:00:00.000Z",
                "updated_at": None,
                "disabled_at": None,
            },
        ],
    )
    _table(
        db,
        "events",
        [
            {
                **run,
                "event_id": "evt_a_1",
                "account_id": "acct_a",
                "environment": "production",
                "event_type": "payment.succeeded",
                "api_version": "2026-07-01",
                "object_type": "payment",
                "object_id": "pay_a_1",
                "occurred_at": "2026-08-20T11:00:00.000Z",
                "payload_hash": "HASH",
                "caused_by_request_id": None,
            },
        ],
    )
    _table(
        db,
        "webhook_deliveries",
        [
            {
                **run,
                "delivery_id": "del_a_1",
                "event_id": "evt_a_1",
                "endpoint_id": "we_a_1",
                "status": "retrying",
                "attempt_count": 2,
                "first_attempt_at": "2026-08-20T11:01:00.000Z",
                "last_attempt_at": "2026-08-21T11:01:00.000Z",
                "next_attempt_at": None,
                "delivered_at": None,
                "terminal_reason": None,
            },
        ],
    )
    _table(db, "event_journal", [{**run, "event_sequence": 1, "payload_json": "THE ANSWERS"}])
    _table(db, "scenario_executions", [{**run, "scenario_id": "SC-001-x", "split": "dev"}])
    db.commit()
    db.close()

    def req(rid: str, account: str, started: str) -> dict[str, object]:
        return {
            "request_id": rid,
            "account_id": account,
            "environment": "production",
            "region": "NA",
            "service": "payments",
            "method": "POST",
            "route": "/v1/payments",
            "api_version": "2026-07-01",
            "http_status": 200,
            "error_code": None,
            "latency_ms": 10,
            "started_at": started,
            "completed_at": started,
            "response_object_type": None,
            "response_object_id": None,
            "retry_parent_request_id": None,
            "idempotency_key_hash": None,
            "request_details": {},
            "evidence_id": f"runtime:req:{rid}",
            "scenario_id": "SC-001-x",
            "run_id": "run_canonical_x",
            "schema_version": "1.0",
        }

    requests = [
        req("req_a_old", "acct_a", "2026-08-01T11:50:00.000Z"),
        req("req_a_day_before", "acct_a", "2026-08-19T10:00:00.000Z"),
    ]
    requests += [
        req(f"req_a_{i:02d}", "acct_a", f"2026-08-20T11:{i:02d}:00.000Z") for i in range(22)
    ]
    requests += [req("req_b_1", "acct_b", "2026-08-20T11:30:00.000Z")]
    usage = [
        {
            "account_id": "acct_a",
            "environment": "production",
            "bucket_start": "2026-08-20T11:50:00.000Z",
            "bucket_width_seconds": 60,
            "request_count": 22,
            "success_count": 22,
            "error_count": 0,
            "rate_limited_count": 0,
            "burst_peak_10s": 3,
            "plan_effective_at_bucket": "Starter",
            "evidence_id": "runtime:usage:acct_a:production:2026-08-20T11:50:00.000Z",
            "run_id": "run_canonical_x",
        }
    ]
    attempts: list[dict[str, object]] = [
        {
            "attempt_id": f"att_a_{n}",
            "attempt_number": n,
            "delivery_id": "del_a_1",
            "event_id": "evt_a_1",
            "endpoint_id": "we_a_1",
            "attempted_at": at,
            "http_status": 500,
            "error_code": "delivery_failed",
            "response_kind": "http",
            "resulting_delivery_status": "retrying",
            "next_retry_at": None,
            "latency_ms": 5,
            "secret_version": 1,
            "signature_timestamp": at,
            "verification_details": {},
            "evidence_id": f"runtime:attempt:att_a_{n}",
            "scenario_id": "SC-001-x",
            "run_id": "run_canonical_x",
        }
        for n, at in ((1, "2026-08-20T11:01:00.000Z"), (2, "2026-08-21T11:01:00.000Z"))
    ]
    incidents = [
        {
            "incident_id": "inc_1",
            "service": "payments",
            "region": "NA",
            "status": "resolved",
            "impact": "elevated_errors",
            "affected_error_codes": ["FP-5031"],
            "started_at": "2026-08-20T11:00:00.000Z",
            "identified_at": "2026-08-20T11:05:00.000Z",
            "resolved_at": "2026-08-20T12:30:00.000Z",
            "timeline": [],
            "evidence_id": "runtime:incident:inc_1",
            "run_id": "run_canonical_x",
        }
    ]
    for name, rows in (
        ("api_requests", requests),
        ("usage_buckets", usage),
        ("webhook_delivery_attempts", attempts),
        ("service_incidents", incidents),
    ):
        (base / "runtime" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    return root


# A stand-in for the dataset's src/flowpay_simulator/simulated_user.py (course material,
# not in this repo): same class name and interface, minimal keyword matching.
SIMULATOR = """
from dataclasses import dataclass, field

KEYWORDS = {"identifiers": ("编号", "id"), "occurrence_time": ("时间", "什么时候")}


@dataclass
class DeterministicSimulatedUser:
    conversation: dict
    disclosed_fact_ids: set = field(default_factory=set)

    def answer(self, assistant_question):
        q = assistant_question.lower()
        asked = {c for c, words in KEYWORDS.items() if any(w in q for w in words)}
        facts = [f for f in self.conversation.get("available_user_facts", [])
                 if f.get("category") in asked and f["fact_id"] not in self.disclosed_fact_ids]
        if not facts:
            return "这些信息我现在还没整理出来。", []
        for f in facts:
            self.disclosed_fact_ids.add(f["fact_id"])
        return " ".join(f["user_value"] for f in facts), facts
"""


def write_simulator(root: Path) -> Path:
    path = root / "src/flowpay_simulator/simulated_user.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SIMULATOR, encoding="utf-8")
    return root
