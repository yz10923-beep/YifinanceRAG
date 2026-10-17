"""Paired comparison of two retrieval runs on the same cases.

With 60 dev questions, a difference in means says little on its own: this reports the
per-case wins, losses and ties, an exact two-sided sign test over the non-ties, and a
paired bootstrap 95% interval for the mean difference (seeded, so reruns match).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from yifinance_rag.data.benchmark import TYPES
from yifinance_rag.retrieval.evaluate import MODES, CaseResult


def load_run(run_dir: Path) -> tuple[dict[str, object], dict[str, CaseResult]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    cases: dict[str, CaseResult] = {}
    for line in (run_dir / "cases.jsonl").read_text().splitlines():
        row = json.loads(line)
        cases[row["case_id"]] = CaseResult(
            row["case_id"],
            row["type"],
            tuple(row["gold"]),
            tuple(tuple(ids) for ids in row["ranked"]),
            row.get("mode", ""),
        )
    return summary, cases


def metric(result: CaseResult, name: str) -> float:
    kind, _, k = name.partition("@")
    if kind == "recall":
        return result.recall(int(k))
    if kind == "complete":
        return result.complete(int(k))
    if kind == "mrr":
        return result.mrr(int(k))
    raise ValueError(f"unknown metric {name!r}")


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial test of wins vs losses at p = 0.5 (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / float(2**n)
    return min(1.0, 2 * tail)


def bootstrap_ci(
    diffs: Sequence[float], *, rounds: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


@dataclass(frozen=True)
class Comparison:
    group: str
    n: int
    mean_a: float
    mean_b: float
    wins: int  # b better than a
    losses: int
    ties: int
    p_sign: float
    ci: tuple[float, float]

    @property
    def diff(self) -> float:
        return self.mean_b - self.mean_a


def compare(
    a: Mapping[str, CaseResult], b: Mapping[str, CaseResult], name: str
) -> list[Comparison]:
    if set(a) != set(b):
        raise ValueError(f"runs cover different cases ({len(a)} vs {len(b)})")
    out: list[Comparison] = []
    for group in [*TYPES, *MODES.values(), "all"]:
        ids = sorted(c for c in a if group in a[c].groups())
        if not ids:
            continue
        va = [metric(a[c], name) for c in ids]
        vb = [metric(b[c], name) for c in ids]
        diffs = [y - x for x, y in zip(va, vb, strict=True)]
        wins = sum(d > 0 for d in diffs)
        losses = sum(d < 0 for d in diffs)
        out.append(
            Comparison(
                group,
                len(ids),
                sum(va) / len(ids),
                sum(vb) / len(ids),
                wins,
                losses,
                len(ids) - wins - losses,
                sign_test(wins, losses),
                bootstrap_ci(diffs),
            )
        )
    return out


def render(rows: Sequence[Comparison], label_a: str, label_b: str, name: str) -> str:
    lines = [
        f"{name}: B = {label_b} vs A = {label_a} (B wins / losses / ties per case)",
        "| type | n | A | B | B-A | 95% CI | W/L/T | sign p |",
        "|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.group} | {r.n} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.diff:+.3f} | "
            f"[{r.ci[0]:+.3f}, {r.ci[1]:+.3f}] | {r.wins}/{r.losses}/{r.ties} | {r.p_sign:.3f} |"
        )
    return "\n".join(lines)
