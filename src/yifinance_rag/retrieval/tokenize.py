"""Tokens for BM25 over Chinese text with ASCII identifiers in it.

jieba alone splits `FP-4041` into `FP`, `-`, `4041` and `pay_bm_001` into five pieces, so
a query for the error code matches every doc that mentions the number 4041. ASCII
identifier runs are taken out first and kept whole (lowercased), plus their parts, so an
exact code scores highest and a partial mention still matches; jieba segments the rest.
"""

from __future__ import annotations

import logging
import re

import jieba

jieba.setLogLevel(logging.WARNING)  # it logs its dictionary build at INFO on first use

_IDENT = re.compile(r"[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)*")
_SEPARATORS = re.compile(r"[-_.:/]")


def identifier_tokens(ident: str) -> list[str]:
    whole = ident.lower()
    parts = _SEPARATORS.split(whole)
    return [whole, *parts] if len(parts) > 1 else [whole]


def _segment(text: str) -> list[str]:
    if not text.strip():
        return []
    return [t for t in jieba.lcut(text) if any(ch.isalnum() for ch in t)]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    for m in _IDENT.finditer(text):
        tokens += _segment(text[pos : m.start()])
        tokens += identifier_tokens(m.group())
        pos = m.end()
    tokens += _segment(text[pos:])
    return tokens
