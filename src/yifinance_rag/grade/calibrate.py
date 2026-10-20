"""Judge calibration (S3 step 6): Yi labels real answers by hand, blind to the judge, and
each judge's verdicts are compared with those labels item by item.

- `labels.md`: per case, the question, the required assertions (A1..), the forbidden
  claims (N1..) and the answer. It carries benchmark text: never committed.
- `labels.tsv`: one row per item with an empty `label` column for Yi: `y` = the answer
  supports the assertion / asserts the forbidden claim, `n` = it does not. Case ids and
  y/n only, so it can be committed under evals/records/.
- `agreement`: per item kind and overall, the share of agreement and Cohen's kappa, plus
  the task level (all assertions supported and no claim asserted) and the disagreements.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from yifinance_rag.data.dataset import DataError
from yifinance_rag.grade.judge import cohen_kappa, forbidden_claims
from yifinance_rag.grade.record import AnswerRecord
from yifinance_rag.grade.truth import Truth

TSV_FIELDS = ("case_id", "item", "label", "note")
_LABELS = {"y": True, "n": False}


def items(truth: Truth) -> list[tuple[str, str, str]]:
    """(item, kind, text): A1.. required assertions, N1.. forbidden claims."""
    req = [(f"A{i}", "required", text) for i, (_, text) in enumerate(truth.required_assertions, 1)]
    neg = [(f"N{i}", "forbidden", c) for i, c in enumerate(forbidden_claims(truth), 1)]
    return [*req, *neg]


def write_label_sheet(
    records: Sequence[AnswerRecord],
    truths: Mapping[str, Truth],
    questions: Mapping[str, str],
    md_path: Path,
    tsv_path: Path,
) -> int:
    """Writes both files; returns the number of items to label."""
    lines = [
        "# Label sheet (blind: no judge verdicts here)",
        "",
        "For each item, put `y` or `n` in the `label` column of labels.tsv:",
        "- A (required): `y` if the answer itself states it for this question (wording may",
        "  differ; numbers, codes, states and conditions must match).",
        "- N (forbidden): `y` if the answer concludes, states or recommends the claim.",
        "",
    ]
    rows: list[dict[str, str]] = []
    for r in records:
        t = truths[r.case_id]
        lines += [f"## {r.case_id} ({t.question_type}, {t.mode})", "", "**问题**", ""]
        lines += [questions[r.case_id], ""]
        for item, kind, text in items(t):
            label = "要求" if kind == "required" else "禁止结论"
            lines.append(f"- **{item}** ({label}): {text}")
            rows.append({"case_id": r.case_id, "item": item, "label": "", "note": ""})
        lines += ["", "**回答**", "", r.answer.strip() or "(空)", "", "---", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_labels(path: Path) -> dict[tuple[str, str], bool]:
    """Labelled items only; blank labels are skipped, anything but y/n is an error."""
    if path.is_dir() or not path.is_file():
        raise DataError(f"{path} is not a labels.tsv file")
    out: dict[tuple[str, str], bool] = {}
    in_notes = 0  # y/n typed one column too far right (an extra tab)
    with path.open(encoding="utf-8", newline="") as f:
        for n, row in enumerate(csv.DictReader(f, delimiter="\t"), start=2):
            raw = (row.get("label") or "").strip().lower()
            if not raw:
                in_notes += (row.get("note") or "").strip().lower() in _LABELS
                continue
            if raw not in _LABELS:
                raise DataError(f"{path} line {n}: label {raw!r} is not y or n")
            out[(row["case_id"], row["item"])] = _LABELS[raw]
    if in_notes:
        raise DataError(
            f"{path}: {in_notes} rows have y/n in the note column and an empty label: "
            "the labels are one column too far right (an extra tab)"
        )
    return out


def _stats(pairs: Sequence[tuple[bool, bool]]) -> dict[str, object]:
    n = len(pairs)
    return {
        "n": n,
        "agree": sum(a == b for a, b in pairs) / n if n else None,
        "kappa": cohen_kappa(pairs),
        "yi_yes": sum(a for a, _ in pairs),
        "judge_yes": sum(b for _, b in pairs),
    }


def agreement(
    labels: Mapping[tuple[str, str], bool], cases: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """`cases`: rows of a judged run's cases.jsonl. Pairs are (Yi, judge)."""
    by_kind: dict[str, list[tuple[bool, bool]]] = {"required": [], "forbidden": []}
    disagree: list[dict[str, object]] = []
    tasks: list[tuple[bool, bool]] = []
    missing = 0
    for row in cases:
        judge = row.get("judge")
        if not isinstance(judge, dict):
            continue
        cid = str(row["case_id"])
        yi_case: list[tuple[str, bool]] = []
        judge_case: list[tuple[str, bool]] = []
        complete = True
        for kind, section in (("required", "required"), ("forbidden", "forbidden")):
            verdicts = judge.get(section) or {}
            for item, verdict in verdicts.items():
                if (cid, item) not in labels:
                    complete = False
                    continue
                yi = labels[(cid, item)]
                if verdict is None:
                    missing += 1
                    complete = False
                    continue
                by_kind[kind].append((yi, bool(verdict)))
                yi_case.append((kind, yi))
                judge_case.append((kind, bool(verdict)))
                if yi != bool(verdict):
                    disagree.append({"case_id": cid, "item": item, "yi": yi, "judge": verdict})
        if complete and yi_case:

            def passed(xs: list[tuple[str, bool]]) -> bool:
                return all(v for k, v in xs if k == "required") and not any(
                    v for k, v in xs if k == "forbidden"
                )

            tasks.append((passed(yi_case), passed(judge_case)))
    every = by_kind["required"] + by_kind["forbidden"]
    return {
        "required": _stats(by_kind["required"]),
        "forbidden": _stats(by_kind["forbidden"]),
        "all_items": _stats(every),
        "task": _stats(tasks),
        "judge_incomplete_items": missing,
        "disagreements": disagree,
    }


def render(name: str, result: Mapping[str, object]) -> str:
    lines = [
        f"### {name}",
        "",
        "| level | n | agree | kappa | Yi yes | judge yes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for level in ("required", "forbidden", "all_items", "task"):
        s = result[level]
        assert isinstance(s, dict)
        agree = "-" if s["agree"] is None else f"{s['agree']:.3f}"
        kappa = "-" if s["kappa"] is None else f"{s['kappa']:.3f}"
        lines.append(
            f"| {level} | {s['n']} | {agree} | {kappa} | {s['yi_yes']} | {s['judge_yes']} |"
        )
    dis = result["disagreements"]
    assert isinstance(dis, list)
    lines += [
        "",
        f"{len(dis)} disagreements; judge incomplete on "
        f"{result['judge_incomplete_items']} labelled items.",
    ]
    return "\n".join(lines)
