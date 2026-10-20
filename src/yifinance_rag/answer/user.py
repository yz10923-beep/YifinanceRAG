"""The simulated user: the dataset's own `DeterministicSimulatedUser`, loaded from the
dataset by file path (course material: not copied into this repo, not reimplemented), so
its disclosure rules are exactly the benchmark's.

The harness builds one per question from the label's conversation (grader-side data):
the agent never sees the facts, only the replies to what it asks.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Protocol, cast

from yifinance_rag.data.dataset import DataError, Dataset

MODULE = "src/flowpay_simulator/simulated_user.py"


class _Inner(Protocol):
    def answer(self, assistant_question: str) -> tuple[str, list[dict[str, Any]]]: ...


@cache
def _factory(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("flowpay_simulated_user", path)
    if spec is None or spec.loader is None:
        raise DataError(f"cannot load the simulated user from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before running: dataclasses with postponed annotations look their
    # module up in sys.modules while the class is being created.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DeterministicSimulatedUser


@dataclass
class SimulatedUser:
    """Answers one question at a time; reveals only the fact categories asked about."""

    inner: _Inner

    @classmethod
    def for_conversation(cls, ds: Dataset, conversation: Mapping[str, object]) -> SimulatedUser:
        path = ds.path(MODULE)
        if not path.is_file():
            raise DataError(f"the dataset has no simulated user at {MODULE}")
        return cls(cast(_Inner, _factory(path)(conversation=dict(conversation))))

    def answer(self, question: str) -> tuple[str, list[str]]:
        """The reply and the categories it disclosed (empty when nothing matched)."""
        reply, facts = self.inner.answer(question)
        return reply, [str(f.get("category")) for f in facts]
