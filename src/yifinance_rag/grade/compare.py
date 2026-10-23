"""Paired comparison of two grade runs (`yfr grade` output) on the same cases: per case
wins / losses / ties, the exact sign test and a paired bootstrap interval, per question
type and mode, for any grader column. A case counts only where both runs have a value.

Judge columns are recomputed from the stored verdicts (cases.jsonl keeps them), so a run
graded with `--judge` can be compared on judge_task like on any deterministic grader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from yifinance_rag.data.benchmark import TYPES
from yifinance_rag.grade.judge import Verdicts
from yifinance_rag.grade.report import MODES
from yifinance_rag.retrieval.compare import Comparison, bootstrap_ci, sign_test

DEFAULT_METRICS = ("judge_task", "instance_facts", "record_evidence", "doc_evidence")


@dataclass(frozen=True)
class GradedCase:
    case_id: str
    question_type: str
    mode: str
    values: Mapping[str, float | None]

    def groups(self) -> tuple[str, ...]:
        mode = MODES.get(self.mode)
        return (self.question_type, *([mode] if mode else []), "all")


def is_grade_run(run: Path) -> bool:
    first = (run / "cases.jsonl").read_text(encoding="utf-8").split("\n", 1)[0]
    return bool(first.strip()) and "scores" in json.loads(first)


def load_grade_run(run: Path) -> tuple[dict[str, object], dict[str, GradedCase]]:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    cases: dict[str, GradedCase] = {}
    for line in (run / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values: dict[str, float | None] = dict(row["scores"])
        values["course_task"] = float(row["course"]["task_pass"])
        if row.get("judge"):
            values |= Verdicts(**row["judge"]).scores()
        cases[row["case_id"]] = GradedCase(row["case_id"], row["type"], row["mode"], values)
    return summary, cases


def compare_grades(
    a: Mapping[str, GradedCase], b: Mapping[str, GradedCase], name: str
) -> list[Comparison]:
    if set(a) != set(b):
        raise ValueError(f"runs cover different cases ({len(a)} vs {len(b)})")
    out: list[Comparison] = []
    for group in [*TYPES, *MODES.values(), "all"]:
        pairs = [
            (va, vb)
            for c in sorted(a)
            if group in a[c].groups()
            and (va := a[c].values.get(name)) is not None
            and (vb := b[c].values.get(name)) is not None
        ]
        if not pairs:
            continue
        diffs = [y - x for x, y in pairs]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        n = len(pairs)
        out.append(
            Comparison(
                group,
                n,
                sum(x for x, _ in pairs) / n,
                sum(y for _, y in pairs) / n,
                wins,
                losses,
                n - wins - losses,
                sign_test(wins, losses),
                bootstrap_ci(diffs),
            )
        )
    return out
