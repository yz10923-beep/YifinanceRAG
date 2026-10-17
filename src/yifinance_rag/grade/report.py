"""Grade a set of records: per-case grades, tables per question type and mode, and the
selftest that every grader must pass on the baselines before it grades a real system."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from yifinance_rag.data.benchmark import TYPES
from yifinance_rag.grade.course import CourseScore, course_score
from yifinance_rag.grade.graders import GRADERS, CaseGrade, Context, grade
from yifinance_rag.grade.judge import Verdicts
from yifinance_rag.grade.record import AnswerRecord
from yifinance_rag.grade.truth import Truth

MODES = {"progressive_disclosure": "mode=progressive", "single_turn_complete": "mode=single"}
COURSE = ("course_task", "course_evidence")
JUDGE = ("judge_assertions", "judge_forbidden_clear", "judge_task")
COLUMNS = (*GRADERS, *COURSE, *JUDGE)


@dataclass
class Graded:
    record: AnswerRecord
    truth: Truth
    grade: CaseGrade
    course: CourseScore
    verdicts: Verdicts | None = None  # None: not judged in this run

    def value(self, name: str) -> float | None:
        if name in JUDGE:
            return self.verdicts.scores()[name] if self.verdicts else None
        if name == "course_task":
            return float(self.course.task_pass)
        if name == "course_evidence":
            return float(self.course.evidence_pass) if self.course.doc_total else None
        return self.grade.scores[name]

    def groups(self) -> tuple[str, ...]:
        mode = MODES.get(self.truth.mode)
        return (self.truth.question_type, *([mode] if mode else []), "all")


def grade_all(
    records: Sequence[AnswerRecord],
    truths: Mapping[str, Truth],
    ctx: Context,
    verdicts: Mapping[str, Verdicts] | None = None,
) -> list[Graded]:
    seen: set[str] = set()
    out: list[Graded] = []
    for r in records:
        if r.case_id in seen:
            raise ValueError(f"two records for case {r.case_id}")
        seen.add(r.case_id)
        t = truths[r.case_id]
        v = verdicts.get(r.case_id) if verdicts is not None else None
        out.append(Graded(r, t, grade(r, t, ctx), course_score(r, t), v))
    return out


def summarize(graded: Sequence[Graded]) -> list[dict[str, object]]:
    """One row per type, mode and "all": each column's mean over the cases it applies
    to, and `n:<column>`, how many that is."""
    rows: list[dict[str, object]] = []
    for name in [*TYPES, *MODES.values(), "all"]:
        group = [g for g in graded if name in g.groups()]
        if not group:
            continue
        row: dict[str, object] = {"type": name, "n": len(group)}
        for col in COLUMNS:
            values = [v for g in group if (v := g.value(col)) is not None]
            row[col] = sum(values) / len(values) if values else None
            row[f"n:{col}"] = len(values)
        rows.append(row)
    return rows


def render_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| type | n | " + " | ".join(COLUMNS) + " |",
        "|---|---:|" + "---:|" * len(COLUMNS),
    ]
    for row in rows:
        cells = [f"{v:.3f}" if isinstance(v := row[c], float) else "-" for c in COLUMNS]
        lines.append(f"| {row['type']} | {row['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Graders that score finding or stating the right things; a null or wrong answer must
# not pass them. (clarification and as_of score behaviour, cited_* need citations.)
EVIDENCE = ("doc_evidence", "record_evidence", "negative_found", "id_recall", "tools", "operations")
FAIL_BELOW = 0.5  # a wrong answer's mean must stay under this, on every answer grader


def _mean(graded: Sequence[Graded], col: str) -> float | None:
    values = [v for g in graded if (v := g.value(col)) is not None]
    return sum(values) / len(values) if values else None


def selftest(by_system: Mapping[str, Sequence[Graded]]) -> list[str]:
    """The yiagent rule for every grader: the oracle passes, the null scores 0, a wrong
    answer fails; plus rule-only must fail instance facts. Returns the problems."""
    problems: list[str] = []
    for g in by_system.get("oracle", []):
        bad = [c for c in (*GRADERS, *COURSE) if (v := g.value(c)) is not None and v != 1.0]
        if bad:
            problems.append(f"oracle below 1.0 on {g.truth.question_id}: {bad}")
    for g in by_system.get("null", []):
        bad = [c for c in (*EVIDENCE, "instance_facts", "course_task") if g.value(c)]
        if bad:
            problems.append(f"null above 0 on {g.truth.question_id}: {bad}")
    wrong = by_system.get("wrong", [])
    for col in ("record_evidence", "negative_found", "instance_facts", "course_task"):
        mean = _mean(wrong, col)
        if mean is not None and mean >= FAIL_BELOW:
            problems.append(f"wrong answer passes {col}: mean {mean:.3f}")
    rule = [g for g in by_system.get("rule-only", []) if g.truth.dynamic]
    if any(g.value("instance_facts") == 1.0 for g in rule):
        problems.append("rule-only states every instance fact on some dynamic case")
    mean = _mean(rule, "instance_facts")
    if mean is not None and mean >= FAIL_BELOW:
        problems.append(f"rule-only passes instance_facts: mean {mean:.3f}")
    if any(g.course.task_pass for g in by_system.get("forbidden", [])):
        problems.append("the course scorer missed a verbatim forbidden claim")
    return problems + judge_selftest(by_system)


# The judge is a model: it gets thresholds, not exact values. Chosen before its first run
# (2026-10-17) and recorded in ROADMAP S3 step 5; changing them is a decision, not tuning.
JUDGE_ORACLE_MIN = 0.90  # judge_task on the dataset's own answers
JUDGE_NULL_MAX = 0.05  # judge_assertions on "不知道"
JUDGE_WRONG_MAX = 0.20  # judge_task on another family's answer
JUDGE_FORBIDDEN_MAX = 0.10  # judge_forbidden_clear when the claim is stated
JUDGE_UNPARSED_MAX = 0.05  # share of judged answers with incomplete verdicts


def judge_selftest(by_system: Mapping[str, Sequence[Graded]]) -> list[str]:
    """Only for systems that were judged in this run."""
    problems: list[str] = []
    checks = (
        ("oracle", "judge_task", ">=", JUDGE_ORACLE_MIN),
        ("null", "judge_assertions", "<=", JUDGE_NULL_MAX),
        ("wrong", "judge_task", "<=", JUDGE_WRONG_MAX),
        ("forbidden", "judge_forbidden_clear", "<=", JUDGE_FORBIDDEN_MAX),
    )
    for system, col, op, bound in checks:
        judged = [g for g in by_system.get(system, []) if g.verdicts is not None]
        mean = _mean(judged, col)
        if mean is not None and (mean < bound if op == ">=" else mean > bound):
            problems.append(f"judge on {system}: {col} {mean:.3f}, needs {op} {bound}")
    judged = [g for gs in by_system.values() for g in gs if g.verdicts is not None]
    incomplete = sum(not g.verdicts.complete for g in judged if g.verdicts)
    if judged and incomplete / len(judged) > JUDGE_UNPARSED_MAX:
        problems.append(f"judge left {incomplete} of {len(judged)} answers incomplete")
    return problems
