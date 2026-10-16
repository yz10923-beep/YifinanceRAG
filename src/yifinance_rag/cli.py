"""`yfr` command line.

yfr data check                          verify the dataset; prints facts and problems
yfr data path postgres/x.dump           absolute path of a dataset file (for make targets)
yfr retrieval --retriever bm25          dev table for one retriever (no LLM, no cost)
yfr retrieval --retriever course-dense  needs Ollama and `make course-index` (S1b)
yfr compare RUN_A RUN_B                 paired per-case comparison of two runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from yifinance_rag.config import REPO_ROOT, ConfigError, Settings, cache_dir, load_settings
from yifinance_rag.data.benchmark import load_benchmark, select, split_hash
from yifinance_rag.data.check import check_dataset
from yifinance_rag.data.corpus import EvidenceMap, benchmark_chunks, chunk_set_hash, load_docs
from yifinance_rag.data.dataset import RELEASE_MANIFEST, DataError, Dataset, sha256_file
from yifinance_rag.embed.cache import CachedEmbedder
from yifinance_rag.embed.ollama import Embedder, EmbedError, OllamaEmbedder
from yifinance_rag.retrieval.compare import compare, load_run, render
from yifinance_rag.retrieval.dense import course_db_url, course_dense, dense_809
from yifinance_rag.retrieval.evaluate import KS, CaseResult, evaluate, render_table, summarize
from yifinance_rag.retrieval.retrievers import (
    BM25Retriever,
    OracleRetriever,
    RandomRetriever,
    Retriever,
)

RETRIEVERS = ("bm25", "random", "oracle", "wrong-oracle", "dense-809", "course-dense")
DENSE = ("dense-809", "course-dense")


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


def default_embedder(settings: Settings) -> Embedder:
    return CachedEmbedder(
        OllamaEmbedder(settings.ollama_url), cache_dir(settings) / "embeddings.sqlite"
    )


def recall_ceiling(results: Sequence[CaseResult], reachable: set[str]) -> float:
    """Mean recall a perfect ranker over this index could reach: gold chunks that no
    index chunk maps to can never be retrieved."""
    if not results:
        return 0.0
    return sum(len(set(r.gold) & reachable) / len(r.gold) for r in results) / len(results)


def cmd_retrieval(
    ds: Dataset, settings: Settings, args: argparse.Namespace, embedder: Embedder | None = None
) -> int:
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
    elif args.retriever in DENSE:
        embedder = embedder or default_embedder(settings)
        if args.retriever == "dense-809":
            retriever = dense_809(chunks, embedder)
        else:
            retriever = course_dense(course_db_url(settings.pg_url.get_secret_value()), embedder)
    else:
        retriever = OracleRetriever(chunks, labels, shift=int(args.retriever == "wrong-oracle"))

    emap = EvidenceMap(chunks)
    results, skipped = evaluate(retriever, picked, labels, emap, k=max(KS))
    rows = summarize(results)
    table = render_table(rows)
    index = getattr(retriever, "chunks", chunks)
    reachable = {e for c in index for e in emap.ids_for(c)}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = Path(args.out) / f"{stamp}_{retriever.name}_{args.split}"
    out, n = base, 1
    while out.exists():  # two runs in the same second
        n += 1
        out = base.with_name(f"{base.name}_{n}")
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
        "index_chunk_set_sha256": chunk_set_hash(index),
        "index_chunks": len(index),
        "recall_ceiling": recall_ceiling(results, reachable),
        "embeddings_computed": getattr(embedder, "computed", None),
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
    print(f"recall ceiling of this index: {summary['recall_ceiling']:.3f}")
    print(f"\nwritten to {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    summary_a, a = load_run(args.run_a)
    summary_b, b = load_run(args.run_b)
    for key in ("split", "split_sha256", "release_manifest_sha256"):
        if summary_a.get(key) != summary_b.get(key):
            print(f"error: the runs differ in {key}; a paired comparison needs the same cases")
            return 1
    for name in args.metric or ["recall@10", "mrr@10"]:
        rows = compare(a, b, name)
        print(render(rows, str(summary_a["retriever"]), str(summary_b["retriever"]), name))
        print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yfr")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="dataset checks")
    data.add_argument("action", choices=["check", "path"])
    data.add_argument("relative", nargs="?", help="for `path`: a dataset-relative file")
    ret = sub.add_parser("retrieval", help="score a retriever against gold doc evidence")
    ret.add_argument("--retriever", choices=RETRIEVERS, required=True)
    ret.add_argument("--split", choices=["dev", "test"], default="dev")
    ret.add_argument("--final", action="store_true", help="allow the test split (Yi only)")
    ret.add_argument("--k1", type=float, default=1.2)
    ret.add_argument("--b", type=float, default=0.75)
    ret.add_argument("--seed", type=int, default=0)
    ret.add_argument("--out", default=str(REPO_ROOT / "evals" / "results" / "retrieval"))
    cmp = sub.add_parser("compare", help="paired comparison of two runs (B against A)")
    cmp.add_argument("run_a", type=Path)
    cmp.add_argument("run_b", type=Path)
    cmp.add_argument("--metric", action="append", help="default: recall@10 and mrr@10")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        return cmd_compare(args)
    try:
        settings = load_settings()
        ds = Dataset.from_settings(settings)
        if args.command == "data" and args.action == "path":
            path = ds.path(args.relative or "")
            if not path.is_file():
                raise DataError(f"no file {args.relative!r} in the dataset")
            print(path)
            return 0
        if args.command == "data":
            return cmd_data_check(ds)
        return cmd_retrieval(ds, settings, args)
    except (ConfigError, DataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (EmbedError, psycopg.Error) as e:
        # an infra failure: the run stops and writes no scores
        print(f"infra error, no scores written: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
