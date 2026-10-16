"""`yfr` command line.

yfr data check                          verify the dataset; prints facts and problems
yfr retrieval --retriever bm25          dev table for one retriever (no LLM, no cost)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from yifinance_rag.config import REPO_ROOT, ConfigError, load_settings
from yifinance_rag.data.benchmark import load_benchmark, select, split_hash
from yifinance_rag.data.check import check_dataset
from yifinance_rag.data.corpus import EvidenceMap, benchmark_chunks, chunk_set_hash, load_docs
from yifinance_rag.data.dataset import RELEASE_MANIFEST, DataError, Dataset, sha256_file
from yifinance_rag.retrieval.evaluate import KS, evaluate, render_table, summarize
from yifinance_rag.retrieval.retrievers import (
    BM25Retriever,
    OracleRetriever,
    RandomRetriever,
    Retriever,
)

RETRIEVERS = ("bm25", "random", "oracle", "wrong-oracle")


def _git_commit() -> str | None:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if head.returncode != 0:
        return None
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def cmd_data_check(ds: Dataset) -> int:
    report = check_dataset(ds)
    print(json.dumps(report.facts, ensure_ascii=False, indent=2))
    if report.problems:
        print(f"\n{len(report.problems)} problem(s):")
        for p in report.problems:
            print(f"- {p}")
        return 1
    print("\nok")
    return 0


def cmd_retrieval(ds: Dataset, args: argparse.Namespace) -> int:
    if args.split == "test" and not args.final:
        print("the test split is for a stage's final report only; Yi passes --final")
        return 2
    chunks = benchmark_chunks(load_docs(ds))
    questions, labels = load_benchmark(ds)
    picked = select(questions, labels, args.split)
    retriever: Retriever
    if args.retriever == "bm25":
        retriever = BM25Retriever(chunks, k1=args.k1, b=args.b)
    elif args.retriever == "random":
        retriever = RandomRetriever(chunks, seed=args.seed)
    else:
        retriever = OracleRetriever(chunks, labels, shift=int(args.retriever == "wrong-oracle"))

    results, skipped = evaluate(retriever, picked, labels, EvidenceMap(chunks), k=max(KS))
    rows = summarize(results)
    table = render_table(rows)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) / f"{stamp}_{retriever.name}_{args.split}"
    out.mkdir(parents=True)
    summary = {
        "retriever": retriever.name,
        "config": retriever.config(),
        "split": args.split,
        "cases": len(results),
        "cases_without_doc_evidence": skipped,
        "rows": rows,
        "release_manifest_sha256": sha256_file(ds.path(RELEASE_MANIFEST)),
        "split_sha256": split_hash(labels.values()),
        "chunk_set_sha256": chunk_set_hash(chunks),
        "git_commit": _git_commit(),
        "created": stamp,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "table.md").write_text(table + "\n")
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            row: dict[str, object] = {"case_id": r.case_id, "type": r.question_type}
            row |= {"gold": r.gold, "ranked": r.ranked}
            row |= {"recall@10": r.recall(10), "mrr@10": r.mrr(10)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{retriever.name} on {args.split}: {len(results)} cases ({skipped} without doc evidence)"
    )
    print(table)
    print(f"\nwritten to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yfr")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="dataset checks")
    data.add_argument("action", choices=["check"])
    ret = sub.add_parser("retrieval", help="score a retriever against gold doc evidence")
    ret.add_argument("--retriever", choices=RETRIEVERS, required=True)
    ret.add_argument("--split", choices=["dev", "test"], default="dev")
    ret.add_argument("--final", action="store_true", help="allow the test split (Yi only)")
    ret.add_argument("--k1", type=float, default=1.2)
    ret.add_argument("--b", type=float, default=0.75)
    ret.add_argument("--seed", type=int, default=0)
    ret.add_argument("--out", default=str(REPO_ROOT / "evals" / "results" / "retrieval"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ds = Dataset.from_settings(load_settings())
        if args.command == "data":
            return cmd_data_check(ds)
        return cmd_retrieval(ds, args)
    except (ConfigError, DataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
