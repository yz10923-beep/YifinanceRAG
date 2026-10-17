"""The LLM judge for the free-text assertions (S3 step 5): one call per answer, a verdict
per required assertion (`supported` / `not_supported`) and per forbidden claim
(`asserted` / `not_asserted`), as JSON.

Checks that don't trust the judge:
- Every positive verdict must quote the answer; a quote that is not in the answer
  (whitespace aside) does not count, and the case records it as `unverified`.
- A reply that can't be parsed, or leaves an assertion out, scores None for that case
  and is counted (`unparsed`), never guessed.
- Its own selftest on the baselines' records (report.selftest), and agreement with Yi's
  hand labels (Cohen's kappa) before its numbers are used (S3 step 6).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.grade.record import AnswerRecord
from yifinance_rag.grade.truth import Truth
from yifinance_rag.llm.cached import CachedCaller, CallStats
from yifinance_rag.llm.retry import RetryPolicy
from yifinance_rag.llm.types import LLMProvider

JUDGE_MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
FORBIDDEN_PREFIX = "不得认定："

SYSTEM = """\
You grade a support assistant's answer to a merchant's question about FlowPay, a payments \
platform, against reference assertions written by the benchmark's authors. The texts are \
in Chinese.

For each required assertion (A1, A2, ...): "supported" only if the answer itself states \
the same fact or conclusion for this question. Wording may differ; numbers, error codes, \
states, ids and conditions must match. Otherwise "not_supported": missing, contradicted, \
only hedged, or only present in documentation the answer quotes without applying it to \
the question.

For each forbidden claim (N1, ...): "asserted" if the answer concludes, states or \
recommends it, even reworded; otherwise "not_asserted". Rejecting the claim is \
"not_asserted".

Judge only from the answer text, not from your own knowledge of payments. For every \
"supported" or "asserted" verdict, copy into "quote" the shortest span of the answer, \
verbatim, that shows it; otherwise "quote" is "".

Reply with JSON only, in this shape:
{"required": [{"id": "A1", "verdict": "supported", "quote": "..."}], \
"forbidden": [{"id": "N1", "verdict": "not_asserted", "quote": ""}]}"""


def forbidden_claims(truth: Truth) -> list[str]:
    return [a.removeprefix(FORBIDDEN_PREFIX) for a in truth.negative_assertions]


def render_prompt(question: str, truth: Truth, answer: str) -> str:
    required = "\n".join(
        f"A{i}: {text}" for i, (_, text) in enumerate(truth.required_assertions, 1)
    )
    forbidden = "\n".join(f"N{i}: {c}" for i, c in enumerate(forbidden_claims(truth), 1))
    return (
        f"Question (the user's first message):\n{question}\n\n"
        f"Required assertions:\n{required or '(none)'}\n\n"
        f"Forbidden claims:\n{forbidden or '(none)'}\n\n"
        f"Answer to grade:\n<answer>\n{answer}\n</answer>"
    )


@dataclass
class Verdicts:
    """One answer's verdicts. None in a map = the judge left that item out."""

    required: dict[str, bool | None]
    forbidden: dict[str, bool | None]
    unverified: list[str] = field(default_factory=list)  # positive verdicts with no real quote
    parsed: bool = True

    @property
    def complete(self) -> bool:
        return self.parsed and None not in {*self.required.values(), *self.forbidden.values()}

    def scores(self) -> dict[str, float | None]:
        if not self.complete:
            return {"judge_assertions": None, "judge_forbidden_clear": None, "judge_task": None}
        req = [bool(v) for v in self.required.values()]
        clear = not any(self.forbidden.values())
        return {
            "judge_assertions": sum(req) / len(req) if req else None,
            "judge_forbidden_clear": float(clear) if self.forbidden else None,
            "judge_task": float(all(req) and clear),
        }


def _squash(text: str) -> str:
    return "".join(text.split())


def parse_verdicts(text: str, truth: Truth, answer: str) -> Verdicts:
    req_ids = [f"A{i}" for i in range(1, len(truth.required_assertions) + 1)]
    neg_ids = [f"N{i}" for i in range(1, len(truth.negative_assertions) + 1)]
    empty = Verdicts({i: None for i in req_ids}, {i: None for i in neg_ids}, parsed=False)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return empty
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return empty
    if not isinstance(raw, dict):
        return empty
    body = _squash(answer)
    out = Verdicts({i: None for i in req_ids}, {i: None for i in neg_ids})
    for section, ids, positive, negative in (
        ("required", req_ids, "supported", "not_supported"),
        ("forbidden", neg_ids, "asserted", "not_asserted"),
    ):
        target = out.required if section == "required" else out.forbidden
        items = raw.get(section)
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("id") not in ids:
                continue
            verdict, quote = item.get("verdict"), str(item.get("quote") or "")
            if verdict == negative:
                target[str(item["id"])] = False
            elif verdict == positive:
                real = bool(_squash(quote)) and _squash(quote) in body
                if not real:
                    out.unverified.append(str(item["id"]))
                target[str(item["id"])] = real
    return out


@dataclass
class JudgeStats(CallStats):
    unparsed: int = 0  # answers whose verdicts are incomplete: scored None
    unverified: int = 0  # positive verdicts dropped for a quote not in the answer


class Judge:
    def __init__(
        self,
        provider_factory: Callable[[], LLMProvider],
        *,
        model: str,
        cache_path: Path,
        effort: str = "low",
        trial: int = 0,
        retry: RetryPolicy | None = None,
    ) -> None:
        self.model, self.effort, self.trial = model, effort, trial
        self.stats = JudgeStats()
        self.caller = CachedCaller(
            provider_factory,
            model=model,
            effort=effort,
            system=SYSTEM,
            cache_path=cache_path,
            stats=self.stats,
            trial=trial,
            retry=retry,
        )

    def config(self) -> dict[str, object]:
        return {
            "model": self.model,
            "effort": self.effort,
            "trial": self.trial,
            "system_sha256": sha256_text(SYSTEM),
        }

    def judge(self, question: str, truth: Truth, record: AnswerRecord) -> Verdicts:
        reply = self.caller.ask(render_prompt(question, truth, record.answer))
        verdicts = parse_verdicts(reply, truth, record.answer)
        if not verdicts.complete:
            self.stats.unparsed += 1
        self.stats.unverified += len(verdicts.unverified)
        return verdicts

    def judge_all(
        self,
        records: Sequence[AnswerRecord],
        truths: Mapping[str, Truth],
        questions: Mapping[str, str],
    ) -> dict[str, Verdicts]:
        return {r.case_id: self.judge(questions[r.case_id], truths[r.case_id], r) for r in records}

    def close(self) -> None:
        self.caller.close()


def cohen_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Agreement between two binary raters beyond chance. None when it is undefined
    (no pairs, or both raters gave one label to everything)."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    chance = pa * pb + (1 - pa) * (1 - pb)
    return None if chance == 1 else (observed - chance) / (1 - chance)
