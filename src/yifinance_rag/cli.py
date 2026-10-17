"""`yfr` command line.

yfr data check                          verify the dataset; prints facts and problems
yfr data path postgres/x.dump           absolute path of a dataset file (for make targets)
yfr retrieval --retriever bm25          dev table for one retriever (no LLM, no cost)
yfr retrieval --retriever course-dense  needs Ollama and `make course-index` (S1b)
yfr retrieval --retriever hybrid        RRF over BM25 + dense-809 (S2); --embed-model picks
                                        the Qwen3-Embedding size for dense-809 and hybrid
yfr retrieval --retriever hybrid --embed-model qwen3-embedding:4b --rerank haiku
                                        reorders the top 20 with Claude Haiku (costs money)
yfr compare RUN_A RUN_B                 paired per-case comparison of two runs
yfr grade --selftest                    every grader against the no-LLM baselines (S3)
yfr grade --system oracle|null|wrong|rule-only|verbose-null
                                        grade a baseline; verbose-null takes the retrieval
                                        flags (default bm25)
yfr grade --records FILE --name NAME    grade a system's answer records
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from yifinance_rag.config import REPO_ROOT, ConfigError, Settings, cache_dir, load_settings
from yifinance_rag.data.benchmark import (
    Label,
    Question,
    load_benchmark,
    select,
    split_hash,
    with_user_facts,
)
from yifinance_rag.data.check import check_dataset
from yifinance_rag.data.corpus import (
    Chunk,
    EvidenceMap,
    benchmark_chunks,
    chunk_set_hash,
    load_docs,
)
from yifinance_rag.data.dataset import RELEASE_MANIFEST, DataError, Dataset, sha256_file
from yifinance_rag.embed.cache import CachedEmbedder
from yifinance_rag.embed.ollama import (
    COURSE_MODEL,
    EMBED_MODELS,
    Embedder,
    EmbedError,
    OllamaEmbedder,
)
from yifinance_rag.grade.baselines import SYSTEMS, build
from yifinance_rag.grade.graders import Context
from yifinance_rag.grade.record import AnswerRecord, read_records, write_records
from yifinance_rag.grade.report import Graded, grade_all, selftest
from yifinance_rag.grade.report import render_table as render_grades
from yifinance_rag.grade.report import summarize as summarize_grades
from yifinance_rag.grade.truth import load_truths
from yifinance_rag.llm.anthropic_provider import AnthropicProvider
from yifinance_rag.llm.types import LLMError, LLMProvider
from yifinance_rag.retrieval.compare import compare, load_run, render
from yifinance_rag.retrieval.dense import course_db_url, course_dense, dense_809, model_suffix
from yifinance_rag.retrieval.evaluate import KS, CaseResult, evaluate, render_table, summarize
from yifinance_rag.retrieval.hybrid import RRFRetriever
from yifinance_rag.retrieval.rerank import LLMReranker
from yifinance_rag.retrieval.retrievers import (
    BM25Retriever,
    OracleRetriever,
    RandomRetriever,
    Retriever,
)

RETRIEVERS = (
    "bm25",
    "random",
    "oracle",
    "wrong-oracle",
    "dense-809",
    "course-dense",
    "hybrid",
)
DENSE = ("dense-809", "course-dense", "hybrid")


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


def default_embedder(settings: Settings, model: str = COURSE_MODEL) -> Embedder:
    return CachedEmbedder(
        OllamaEmbedder(settings.ollama_url, model=model), cache_dir(settings) / "embeddings.sqlite"
    )


RERANK_MODELS = {"haiku": "claude-haiku-5-5"}


def default_provider(settings: Settings, model: str) -> Callable[[], LLMProvider]:
    key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None

    def make() -> LLMProvider:
        # effort low: a ranking judgment, not a hard problem; no cache: every prompt is new
        return AnthropicProvider(
            model=model,
            max_tokens=8000,
            effort="low",
            refusal_fallback=False,
            api_key=key,
            prompt_cache=False,
        )

    return make


def recall_ceiling(results: Sequence[CaseResult], reachable: set[str]) -> float:
    """Mean recall a perfect ranker over this index could reach: gold chunks that no
    index chunk maps to can never be retrieved."""
    if not results:
        return 0.0
    return sum(len(set(r.gold) & reachable) / len(r.gold) for r in results) / len(results)


def build_retriever(
    args: argparse.Namespace,
    settings: Settings,
    chunks: Sequence[Chunk],
    labels: Mapping[str, Label],
    embedder: Embedder | None = None,
) -> tuple[Retriever, LLMReranker | None, Embedder | None]:
    """The retriever the shared retrieval flags describe (see `add_retriever_args`)."""
    retriever: Retriever
    if args.retriever == "bm25":
        retriever = BM25Retriever(chunks, k1=args.k1, b=args.b)
    elif args.retriever == "random":
        retriever = RandomRetriever(chunks, seed=args.seed)
    elif args.retriever in DENSE:
        embedder = embedder or default_embedder(settings, args.embed_model)
        if args.retriever == "dense-809":
            retriever = dense_809(chunks, embedder)
        elif args.retriever == "hybrid":
            parts: list[Retriever] = [
                BM25Retriever(chunks, k1=args.k1, b=args.b),
                dense_809(chunks, embedder),
            ]
            name = "hybrid" + model_suffix(embedder)
            retriever = RRFRetriever(name, parts, chunks, k=args.rrf_k, depth=args.depth)
        else:
            retriever = course_dense(course_db_url(settings.pg_url.get_secret_value()), embedder)
    else:
        retriever = OracleRetriever(chunks, labels, shift=int(args.retriever == "wrong-oracle"))

    reranker: LLMReranker | None = None
    if args.rerank != "none":
        if args.retriever in ("oracle", "wrong-oracle", "random"):
            raise ValueError(f"--rerank needs a real first stage, not {args.retriever}")
        model = RERANK_MODELS[args.rerank]
        reranker = LLMReranker(
            retriever,
            default_provider(settings, model),
            model=model,
            effort="low",
            cache_path=cache_dir(settings) / "rerank.sqlite",
            depth=args.rerank_depth,
            trial=args.trial,
        )
        retriever = reranker
    return retriever, reranker, embedder


def cmd_retrieval(
    ds: Dataset, settings: Settings, args: argparse.Namespace, embedder: Embedder | None = None
) -> int:
    if args.split == "test" and not args.final:
        print("the test split is for a stage's final report only; Yi passes --final")
        return 2
    if args.retriever == "course-dense" and args.embed_model != COURSE_MODEL:
        print("course-dense is the course's index: its vectors exist only for the course's model")
        return 2
    chunks = benchmark_chunks(load_docs(ds))
    questions, labels = load_benchmark(ds)
    picked = select(questions, labels, args.split)
    if args.user_facts:
        picked = [with_user_facts(q, labels[q.case_id]) for q in picked]
    try:
        retriever, reranker, embedder = build_retriever(args, settings, chunks, labels, embedder)
    except ValueError as e:
        print(e)
        return 2

    emap = EvidenceMap(chunks)
    try:
        results, skipped = evaluate(retriever, picked, labels, emap, k=max(KS))
    finally:
        if reranker:
            reranker.close()
    rows = summarize(results)
    table = render_table(rows)
    index = getattr(retriever, "chunks", chunks)
    reachable = {e for c in index for e in emap.ids_for(c)}

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    facts = "+facts" if args.user_facts else ""
    base = Path(args.out) / f"{stamp}_{retriever.name}{facts}_{args.split}"
    out, n = base, 1
    while out.exists():  # two runs in the same second
        n += 1
        out = base.with_name(f"{base.name}_{n}")
    out.mkdir(parents=True)
    summary = {
        "retriever": retriever.name,
        "config": retriever.config(),
        "split": args.split,
        "query": "opener + all simulated-user facts (upper bound)" if args.user_facts else "opener",
        "cases": len(results),
        "cases_without_doc_evidence": skipped,
        "rows": rows,
        "release_manifest_sha256": sha256_file(ds.path(RELEASE_MANIFEST)),
        "split_sha256": split_hash(labels.values()),
        "chunk_set_sha256": chunk_set_hash(chunks),
        "index_chunk_set_sha256": chunk_set_hash(index),
        "index_chunks": len(index),
        "depth": max(KS),  # hits kept per case; metrics at k > depth are not defined
        "recall_ceiling": recall_ceiling(results, reachable),
        "embeddings_computed": getattr(embedder, "computed", None),
        "rerank": reranker.stats.__dict__ if reranker else None,
        "git_commit": _git_commit(),
        "created": stamp,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "table.md").write_text(table + "\n")
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            row: dict[str, object] = {"case_id": r.case_id, "type": r.question_type, "mode": r.mode}
            row |= {"gold": r.gold, "ranked": r.ranked}
            row |= {"recall@10": r.recall(10), "mrr@10": r.mrr(10)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"{retriever.name} on {args.split}: {len(results)} cases ({skipped} without doc evidence)"
    )
    print(table)
    print(f"recall ceiling of this index: {summary['recall_ceiling']:.3f}")
    if reranker:
        st = reranker.stats
        cost = "unknown" if st.cost_usd is None else f"${st.cost_usd:.4f}"
        print(
            f"rerank: {st.calls} calls, {st.cached} cached, {st.fallbacks} fallbacks, "
            f"{st.input_tokens} in / {st.output_tokens} out tokens, {cost}"
        )
    print(f"\nwritten to {out}")
    return 0


def _run_dir(base: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out, n = base / f"{stamp}_{name}", 1
    while out.exists():  # two runs in the same second
        n += 1
        out = base / f"{stamp}_{name}_{n}"
    out.mkdir(parents=True)
    return out


def _case_row(g: Graded) -> dict[str, object]:
    return {
        "case_id": g.record.case_id,
        "type": g.truth.question_type,
        "mode": g.truth.mode,
        "scores": g.grade.scores,
        "details": g.grade.details,
        "course": g.course.__dict__,
    }


def cmd_grade(
    ds: Dataset, settings: Settings, args: argparse.Namespace, embedder: Embedder | None = None
) -> int:
    if args.split == "test" and not args.final:
        print("the test split is for a stage's final report only; Yi passes --final")
        return 2
    docs = load_docs(ds)
    chunks = benchmark_chunks(docs)
    questions, labels = load_benchmark(ds)
    truths = load_truths(ds, labels)
    ctx = Context.from_docs(docs)
    picked: list[Question] = select(questions, labels, args.split)

    if args.selftest:
        by_system = {
            s: grade_all(build(s, picked, truths, ctx), truths, ctx)
            for s in SYSTEMS
            if s != "verbose-null"
        }
        for system, graded in by_system.items():
            rows = [
                r for r in summarize_grades(graded) if str(r["type"]).startswith(("mode", "all"))
            ]
            print(f"{system}\n{render_grades(rows)}\n")
        problems = selftest(by_system)
        for p in problems:
            print(f"- {p}")
        print(
            f"selftest on {args.split}: "
            + ("ok" if not problems else f"{len(problems)} problem(s)")
        )
        return 1 if problems else 0

    reranker: LLMReranker | None = None
    config: dict[str, object] = {}
    records: list[AnswerRecord]
    if args.records:
        name = args.name or Path(args.records).stem
        wanted = {q.case_id for q in picked}
        records = [r for r in read_records(Path(args.records)) if r.case_id in wanted]
        if missing := len(wanted) - len(records):
            print(f"{missing} of {len(wanted)} {args.split} cases have no record; none graded")
            return 1
    elif args.system == "verbose-null":
        try:
            retriever, reranker, embedder = build_retriever(
                args, settings, chunks, labels, embedder
            )
        except ValueError as e:
            print(e)
            return 2
        name, config = f"verbose-null-{retriever.name}", retriever.config()
        try:
            records = build(
                "verbose-null", picked, truths, ctx, retriever=retriever, emap=EvidenceMap(chunks)
            )
        finally:
            if reranker:
                reranker.close()
    elif args.system:
        name = args.system
        records = build(args.system, picked, truths, ctx)
    else:
        print("give --system, --records or --selftest")
        return 2

    graded = grade_all(records, truths, ctx)
    rows = summarize_grades(graded)
    table = render_grades(rows)
    out = _run_dir(Path(args.out), f"{name}_{args.split}")
    summary = {
        "system": name,
        "config": config,
        "split": args.split,
        "cases": len(graded),
        "rows": rows,
        "release_manifest_sha256": sha256_file(ds.path(RELEASE_MANIFEST)),
        "split_sha256": split_hash(labels.values()),
        "rerank": reranker.stats.__dict__ if reranker else None,
        "git_commit": _git_commit(),
        "created": out.name.split("_", 1)[0],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (out / "table.md").write_text(table + "\n")
    write_records(out / "records.jsonl", records)
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for g in graded:
            f.write(json.dumps(_case_row(g), ensure_ascii=False) + "\n")
    print(f"{name} on {args.split}: {len(graded)} cases")
    print(table)
    print(f"\nwritten to {out}")
    return 0


def _label(summary: dict[str, object]) -> str:
    facts = "" if summary.get("query", "opener") == "opener" else "+facts"
    return f"{summary['retriever']}{facts}"


def cmd_compare(args: argparse.Namespace) -> int:
    summary_a, a = load_run(args.run_a)
    summary_b, b = load_run(args.run_b)
    for key in ("split", "split_sha256", "release_manifest_sha256"):
        if summary_a.get(key) != summary_b.get(key):
            print(f"error: the runs differ in {key}; a paired comparison needs the same cases")
            return 1
    # Runs recorded before recall@20 kept only their top 10.
    depth = min(int(str(s.get("depth", 10))) for s in (summary_a, summary_b))
    for name in args.metric or ["recall@10", "mrr@10", "recall@20"]:
        if int(name.partition("@")[2]) > depth:
            print(f"{name}: skipped, a run kept only its top {depth}\n")
            continue
        rows = compare(a, b, name)
        print(render(rows, _label(summary_a), _label(summary_b), name))
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
    add_retriever_args(ret)
    ret.add_argument(
        "--user-facts",
        action="store_true",
        help="upper bound: append every fact the simulated user would reveal (grader-side)",
    )
    ret.add_argument("--out", default=str(REPO_ROOT / "evals" / "results" / "retrieval"))
    grd = sub.add_parser("grade", help="grade answer records (S3 graders)")
    what = grd.add_mutually_exclusive_group()
    what.add_argument("--system", choices=SYSTEMS, help="a no-LLM baseline")
    what.add_argument("--records", help="a records.jsonl written by a system")
    what.add_argument("--selftest", action="store_true", help="graders vs the baselines")
    grd.add_argument("--name", help="with --records: the system's name in the run")
    grd.add_argument("--retriever", choices=RETRIEVERS, default="bm25", help="verbose-null")
    add_retriever_args(grd)
    grd.add_argument("--out", default=str(REPO_ROOT / "evals" / "results" / "grade"))
    cmp = sub.add_parser("compare", help="paired comparison of two runs (B against A)")
    cmp.add_argument("run_a", type=Path)
    cmp.add_argument("run_b", type=Path)
    cmp.add_argument("--metric", action="append", help="default: recall@10 and mrr@10")
    return parser


def add_retriever_args(p: argparse.ArgumentParser) -> None:
    """Flags `build_retriever` reads, shared by `retrieval` and `grade`."""
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--final", action="store_true", help="allow the test split (Yi only)")
    p.add_argument("--k1", type=float, default=1.2)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--embed-model", choices=sorted(EMBED_MODELS), default=COURSE_MODEL)
    p.add_argument("--rrf-k", type=int, default=60, help="RRF constant (paper value)")
    p.add_argument("--depth", type=int, default=100, help="hits fused per retriever")
    p.add_argument("--rerank", choices=["none", *RERANK_MODELS], default="none")
    p.add_argument("--rerank-depth", type=int, default=20, help="first-stage hits reranked")
    p.add_argument("--trial", type=int, default=0, help="rerank again (variance), not cached")


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
        if args.command == "grade":
            return cmd_grade(ds, settings, args)
        return cmd_retrieval(ds, settings, args)
    except (ConfigError, DataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (EmbedError, psycopg.Error, LLMError) as e:
        # an infra failure: the run stops and writes no scores
        print(f"infra error, no scores written: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
