"""Run the course's stages (s1-s4) on our split and export what our graders need.

Runs inside the course's own environment (its backend venv, its .env, its knowledge
database and Ollama), not ours: it imports the course's code and drives each question with
the course's own `play()` from scripts/benchmark/run.py, i.e. the dataset's simulated
user and the course's clarification detection. No course code is modified or copied.

It differs from the course's runner in three ways: questions come from our split (dev by
default; test needs --final, which only Yi passes), every question's events are exported
in a small stable form for `yfr import-course`, and a failed run stops the stage (infra
failures are not scores). Re-running resumes: questions already exported are skipped.

    ~/KamaAgenticRAG/backend/.venv/bin/python scripts/course_stages.py \
        --course ~/KamaAgenticRAG --stages s1,s2,s3,s4 [--split dev] [--limit N]

Output: <out>/<stage>.jsonl, one line per question; errors in <out>/<stage>.errors.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

FORMAT = 1
SEARCH_EVENTS = {"knowledge.response": "search", "document.response": "read"}
FAILED = {"run.failed", "run.interrupted", "run.canceled"}


def load_env(course: Path) -> None:
    """What the course's Makefile `with-env` does: its .env, then the derived settings."""
    env_file = course / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    root = os.environ.get("FLOWPAY_DATASET_ROOT")
    if root:
        os.environ.setdefault("FLOWPAY_DATASET_PATH", f"{root}/data")
    password = os.environ.setdefault("FLOWPAY_DB_PASSWORD", "flowpay_local")
    host = "127.0.0.1:55432/flowpay_knowledge"
    for var, role in (
        ("FLOWPAY_KNOWLEDGE_ADMIN_DSN", "flowpay_admin"),
        ("FLOWPAY_KNOWLEDGE_DSN", "flowpay_knowledge_reader"),
        ("FLOWPAY_KNOWLEDGE_BUILDER_DSN", "flowpay_knowledge_builder"),
    ):
        os.environ.setdefault(var, f"postgresql://{role}:{password}@{host}")


def split_ids(dataset: Path, split: str) -> list[str]:
    path = dataset / "benchmark/resolved/resolved_blueprints.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["question_id"] for r in rows if r["split"] == split]


def export(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The parts of a question's events the importer reads, in a stable form."""
    calls: list[dict[str, Any]] = []
    started: dict[str, dict[str, Any]] = {}
    usage: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for e in events:
        kind, payload = e["type"], e["payload"]
        if kind in SEARCH_EVENTS:
            items = payload.get("results") or payload.get("passages") or []
            passages = []
            for item in items:
                source = item.get("source") or {}
                locator = source.get("locator") or {}
                passages.append(
                    {
                        "id": item.get("id"),
                        "doc_id": source.get("doc_id"),
                        "start_char": locator.get("start_char"),
                        "end_char": locator.get("end_char"),
                    }
                )
            calls.append({"kind": SEARCH_EVENTS[kind], "passages": passages})
        elif kind == "source.request":
            # every data access goes through the course's reader, whichever tool asked
            started[e["call_id"]] = {"kind": "data", "request": payload}
        elif kind in {"source.response", "source.error"} and e["call_id"] in started:
            call = started.pop(e["call_id"])
            if kind == "source.error":
                call["error"] = f"{payload.get('error_type')}: {payload.get('message')}"
            else:
                rows = payload.get("rows")
                call["result_count"] = len(rows) if isinstance(rows, list) else 0
                call["rows"] = rows if isinstance(rows, list) else []
            calls.append(call)
        elif kind == "model.response":
            completion = payload.get("completion") or {}
            usage.append({"model": completion.get("model"), "usage": completion.get("usage")})
        elif kind in FAILED or kind == "model.error":
            failures.append({"type": kind, **payload})
    return {"calls": calls, "usage": usage, "failures": failures}


async def run_stage(
    stage: str, cases: list[dict[str, Any]], out: Path, config: Path, alignment: dict[str, str]
) -> int:
    from flowpay.bootstrap import create_workflow
    from flowpay.core.config import load_settings
    from flowpay.core.event_log.runtime import open_store
    from flowpay.core.execution.manager import SessionManager
    from flowpay.core.llm.embedding.ollama import ollama_identity
    from scripts.benchmark.run import play, score

    path = out / f"{stage}.jsonl"
    done = (
        {json.loads(x)["question_id"] for x in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    todo = [c for c in cases if c["question_id"] not in done]
    print(f"{stage}: {len(done)} done, {len(todo)} to run", flush=True)
    settings = replace(load_settings(config), data_dir=out / f"{stage}.events", workflow=stage)
    if settings.embedding is None:
        raise SystemExit("the course config has no embedding settings")
    await ollama_identity(settings.embedding)  # the course's vector-space check
    with open_store(settings) as store:
        manager = SessionManager(store, create_workflow(settings), settings)
        try:
            await manager.recover()
            for n, case in enumerate(todo, 1):
                started = perf_counter()
                answer, events, script = await play(manager, case, store.read)
                facts = {
                    f["fact_id"]: f["category"]
                    for f in case["conversation"].get("available_user_facts") or []
                }
                row = {
                    "format": FORMAT,
                    "question_id": case["question_id"],
                    "stage": stage,
                    "model": settings.model.name,
                    "answer": answer,
                    "exchanges": [
                        {
                            "signal": x.signal,
                            "asked": x.categories,
                            "disclosed": sorted({facts.get(f, "") for f in x.disclosed} - {""}),
                        }
                        for x in script.exchanges
                    ],
                    "deviations": list(script.deviations),
                    "seconds": round(perf_counter() - started, 1),
                    "course_row": score(case, answer, events, alignment),
                    **export([dict(e) for e in events]),
                }
                if row["failures"] or not answer.strip():
                    with (out / f"{stage}.errors.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(f"{stage} {case['question_id']}: failed, stage stopped", flush=True)
                    return 1
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  {stage} {n}/{len(todo)} {row['seconds']}s", flush=True)
        finally:
            await manager.shutdown()
    return 0


OUT = Path(__file__).resolve().parents[1] / "evals/results/course"


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--course", type=Path, required=True, help="the KamaAgenticRAG checkout")
    p.add_argument("--stages", default="s1,s2,s3,s4")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--final", action="store_true", help="allow the test split (Yi only)")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke run)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    if args.split == "test" and not args.final:
        print("error: the test split runs once, for the final report; pass --final (Yi only)")
        return 2
    course = args.course.expanduser().resolve()
    load_env(course)
    sys.path.insert(0, str(course))  # the course's `scripts` package
    from flowpay.dataset import require_dataset
    from scripts.benchmark import evidence
    from scripts.benchmark.run import blueprints

    dataset = require_dataset()
    wanted = split_ids(dataset, args.split)
    by_id = {c["question_id"]: c for c in blueprints()}
    cases = [by_id[q] for q in wanted][: args.limit or None]
    out = args.out / args.split
    out.mkdir(parents=True, exist_ok=True)
    alignment = evidence.chunk_alignment()
    print(f"{len(cases)} {args.split} questions, stages {args.stages}, out {out}", flush=True)
    for stage in args.stages.split(","):
        if await run_stage(stage, cases, out, course / "backend/configs/local.toml", alignment):
            return 1
    print(f"done; next: uv run yfr import-course {out}/<stage>.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
