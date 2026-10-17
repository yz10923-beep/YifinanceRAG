"""Instance facts: the checkable atoms (ids, error codes, dates, times, amounts, enum
words) of a trajectory assertion, and whether an answer states them.

The benchmark's trajectory assertions are generated from the account's records
("支付 pay_bm_026 状态 succeeded，金额 10000 USD"), so their atoms are exact values. Both
texts go through the same normalization, so "2026 年 08 月 02 日 11:50 UTC", "8月2日 11:50"
and "2026-08-02T11:50:00Z" all become "08-02 11:50". Matching is whole-token: "na" never
matches inside "final", "50" never inside "2050".
"""

from __future__ import annotations

import re
import unicodedata

# The resolver's query scope, "（as_of=…、payment_id=pay_bm_001、…）": the looked-up ids
# are facts ("pay_bm_001 was not found"), but the grader-side as_of is never stated by the
# user, and parameter names and operation names are not facts about the case.
_AS_OF = re.compile(r"as_of=[^、，,）)\s]*")
_PARAM = re.compile(r"[a-z_]+=")
_OPERATION = re.compile(r"(?<![a-z0-9_])(?:get|list)_[a-z_]+")
_ISO = re.compile(r"\d{4}-(\d{2})-(\d{2})(?:t(\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?z?)?")
_CN_DATE = re.compile(r"(?:\d{4}\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")

_ID = re.compile(r"[a-z][a-z0-9]*_[a-z0-9_]*\d[a-z0-9_]*")
_CODE = re.compile(r"fp-\d{4}")
_DATE = re.compile(r"(?<![\d-])\d{2}-\d{2}(?![\d-])")
_CLOCK = re.compile(r"(?<![\d:])\d{2}:\d{2}(?![\d:])")
_NUMBER = re.compile(r"(?<![\d.])\d{2,}(?:\.\d+)?(?![\d.])")
_WORD = re.compile(r"(?<![a-z0-9_])[a-z][a-z_]*[a-z](?![a-z0-9_])")
# Words in every assertion's template, not facts about the case.
STOP_WORDS = frozenset({"utc", "post", "get", "http", "id", "api", "usd", "scenario"})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()

    def iso(m: re.Match[str]) -> str:
        day = f"{m[1]}-{m[2]}"
        return f"{day} {m[3]}:{m[4]}" if m[3] else day

    text = _ISO.sub(iso, text)
    text = _CN_DATE.sub(lambda m: f"{int(m[1]):02d}-{int(m[2]):02d}", text)
    return _TIME.sub(lambda m: f"{int(m[1]):02d}:{m[2]}", text)


def keys(assertion: str) -> list[str]:
    """The atoms of one assertion, in order, without repeats."""
    text = _PARAM.sub(" ", _AS_OF.sub(" ", assertion.lower()))
    text = normalize(_OPERATION.sub(" ", text))
    found: list[str] = []
    for pattern in (_ID, _CODE, _DATE, _CLOCK):
        found += pattern.findall(text)
        text = pattern.sub(" ", text)
    found += _NUMBER.findall(text)
    found += [w for w in _WORD.findall(_NUMBER.sub(" ", text)) if w not in STOP_WORDS]
    return list(dict.fromkeys(found))


def states(answer: str, key: str) -> bool:
    """Whether the normalized answer contains the atom as a whole token."""
    edge = r"[a-z0-9_]" if key[0].isalpha() or key[-1].isalpha() else r"[\d.]"
    pattern = rf"(?<!{edge}){re.escape(key)}(?!{edge})"
    return re.search(pattern, normalize(answer)) is not None
