"""Retrieval metrics against gold doc evidence, per case and per question type.

Only `doc:` evidence counts here; sql / runtime / negative evidence is found by data tools
and graded in S3. Cases with no doc evidence are counted and left out of the means.

- recall@k: share of a case's gold chunks among the evidence ids of the top k chunks.
- complete@k: 1 if every gold chunk is in the top k (what multi-document questions need).
- mrr@k: 1 / rank of the first chunk that maps to any gold id, 0 if none in the top k.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from yifinance_rag.data.benchmark import TYPES, Label, Question
from yifinance_rag.data.corpus import EvidenceMap
from yifinance_rag.retrieval.retrievers import Retriever

# recall@k for these k; the run keeps the top max(KS). recall@20 is the ceiling of any
# reranker over the top 20. complete@ and mrr@ stay at 10, the agent's reading budget.
KS = (5, 10, 20)
HEAD_K = 10
# Conversation modes, reported as extra rows next to the question types.
MODES = {"progressive_disclosure": "mode=progressive", "single_turn_complete": "mode=single"}


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    question_type: str
    gold: tuple[str, ...]
    ranked: tuple[tuple[str, ...], ...]  # evidence ids of each retrieved chunk, in rank order
    mode: str = ""  # conversation mode; "" in runs recorded before it was stored

    def groups(self) -> tuple[str, ...]:
        """The table rows this case counts in: its type, its mode, and "all"."""
        return (self.question_type, *([MODES[self.mode]] if self.mode in MODES else []), "all")

    def recall(self, k: int) -> float:
        found = {e for ids in self.ranked[:k] for e in ids}
        return len(found & set(self.gold)) / len(self.gold)

    def complete(self, k: int) -> float:
        return 1.0 if self.recall(k) == 1.0 else 0.0

    def mrr(self, k: int) -> float:
        gold = set(self.gold)
        for rank, ids in enumerate(self.ranked[:k], start=1):
            if gold.intersection(ids):
                return 1.0 / rank
        return 0.0


def evaluate(
    retriever: Retriever,
    questions: Sequence[Question],
    labels: Mapping[str, Label],
    emap: EvidenceMap,
    *,
    k: int = max(KS),
) -> tuple[list[CaseResult], int]:
    """Results for cases with doc evidence, and how many cases had none (skipped)."""
    results: list[CaseResult] = []
    skipped = 0
    for q in questions:
        label = labels[q.case_id]
        if not label.doc_evidence:
            skipped += 1
            continue
        hits = retriever.search(q, k)
        ranked = tuple(tuple(emap.ids_for(h.chunk)) for h in hits[:k])
        gold = tuple(dict.fromkeys(label.doc_evidence))
        results.append(
            CaseResult(q.case_id, label.question_type, gold, ranked, label.conversation_mode)
        )
    return results, skipped


def metric_names(ks: Sequence[int] = KS) -> list[str]:
    return [*(f"recall@{k}" for k in ks), f"complete@{HEAD_K}", f"mrr@{HEAD_K}"]


def _means(results: Sequence[CaseResult], ks: Sequence[int]) -> dict[str, float]:
    n = len(results)
    out = {f"recall@{k}": sum(r.recall(k) for r in results) / n for k in ks}
    out[f"complete@{HEAD_K}"] = sum(r.complete(HEAD_K) for r in results) / n
    out[f"mrr@{HEAD_K}"] = sum(r.mrr(HEAD_K) for r in results) / n
    return out


def summarize(results: Sequence[CaseResult], ks: Sequence[int] = KS) -> list[dict[str, object]]:
    """One row per question type present, then "all" (mean over cases, not over types)."""
    rows: list[dict[str, object]] = []
    for name in [*TYPES, *MODES.values(), "all"]:
        group = [r for r in results if name in r.groups()]
        if group:
            rows.append({"type": name, "n": len(group), **_means(group, ks)})
    return rows


def render_table(rows: Sequence[Mapping[str, object]], ks: Sequence[int] = KS) -> str:
    names = metric_names(ks)
    lines = [
        "| type | n | " + " | ".join(names) + " |",
        "|---|---:|" + "---:|" * len(names),
    ]
    for row in rows:
        cells = [f"{v:.3f}" if isinstance(v := row[m], float) else str(v) for m in names]
        lines.append(f"| {row['type']} | {row['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
