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


def write_dataset(root: Path) -> Path:
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
    blueprints = [blueprint(*q) for q in QUESTIONS]
    (bench / "resolved_blueprints.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in blueprints)
    )
    (bench / "evidence_resolution.jsonl").write_text(
        "".join(
            json.dumps({"question_id": q[0], "resolved_evidence_ids": q[4]}) + "\n"
            for q in QUESTIONS
        )
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
