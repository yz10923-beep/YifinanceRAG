"""S4a on the real dataset: the data tools reproduce every benchmark selector, and no
tool output carries the benchmark's bookkeeping. `make data-test` (VM or container)."""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from yifinance_rag.data.dataset import Dataset
from yifinance_rag.tools.data import DataStore, DataTools

pytestmark = pytest.mark.dataset

LEAK = re.compile(r"SC-\d{3}|SF-\d{3}|FP-BM-|scenario_id|run_id|run_canonical|event_journal")


@pytest.fixture(scope="module")
def selector_calls(ds: Dataset) -> list[tuple[dict[str, object], DataTools]]:
    tools = DataTools(DataStore(ds))
    rows = [
        json.loads(x)
        for x in ds.read_text("benchmark/resolved/evidence_resolution.jsonl").splitlines()
    ]
    return [(s, tools) for r in rows for s in r["selector_results"]]


def _call(s: dict[str, object], tools: DataTools) -> tuple[list[str], str]:
    payload = s["normalized_payload"]
    assert isinstance(payload, dict)
    # `window` is the resolver's symbolic name for the 24-hour default
    args = {k: v for k, v in payload.items() if k != "window"}
    result = tools.run(str(s["tool"]), args)
    return result.evidence_ids, json.dumps(result.to_observation(), ensure_ascii=False)


def test_tools_reproduce_all_535_selectors(
    selector_calls: list[tuple[dict[str, object], DataTools]],
) -> None:
    misses: Counter[str] = Counter()
    for s, tools in selector_calls:
        want = sorted(s["evidence_ids"]) if s["evidence_mode"] == "records" else []  # type: ignore[call-overload]
        got, _ = _call(s, tools)
        if sorted(got) != want:
            misses[str(s["operation"])] += 1
    assert len(selector_calls) == 535 and dict(misses) == {}


def test_no_tool_output_carries_benchmark_bookkeeping(
    selector_calls: list[tuple[dict[str, object], DataTools]],
) -> None:
    leaks = Counter(m for s, tools in selector_calls for m in LEAK.findall(_call(s, tools)[1]))
    assert dict(leaks) == {}
