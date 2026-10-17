"""A disk cache of embeddings keyed by (embedder identity, text), so the 809 chunks and
300 questions are embedded once per model and Ollama can be stopped afterwards
(docs/ROADMAP.md S0: memory next to Elasticsearch).

Long runs (the 4B model on CPU takes about an hour) must survive interruption and must
not run twice at once:
- vectors are saved after every `flush_every` texts, so a timeout or Ctrl-C loses at most
  one flush and a rerun resumes where it stopped;
- computing holds an exclusive lock on `<cache>.lock`; a second process waits for it, then
  re-reads the cache and embeds only what is still missing (concurrent runs on a CPU
  Ollama queue behind each other until requests time out).
"""

from __future__ import annotations

import fcntl
import json
import sqlite3
import sys
import time
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.embed.ollama import Embedder


def identity_key(identity: dict[str, object]) -> str:
    return sha256_text(json.dumps(identity, sort_keys=True))


class CachedEmbedder:
    def __init__(self, inner: Embedder, path: Path, *, flush_every: int = 32) -> None:
        self.inner = inner
        self.path, self.flush_every = path, flush_every
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=60)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "identity TEXT NOT NULL, text_sha256 TEXT NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY (identity, text_sha256))"
        )
        self.computed = 0  # texts sent to the inner embedder in this process

    def identity(self) -> dict[str, object]:
        return self.inner.identity()

    @contextmanager
    def _computing(self) -> Iterator[None]:
        with self.path.with_name(self.path.name + ".lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("another run is embedding into this cache; waiting for it", file=sys.stderr)
                fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _lookup(self, ident: str, keys: Sequence[str], found: dict[str, list[float]]) -> None:
        for key in set(keys) - set(found):
            row = self.db.execute(
                "SELECT vector FROM vectors WHERE identity = ? AND text_sha256 = ?", (ident, key)
            ).fetchone()
            if row is not None:
                found[key] = array("f", row[0]).tolist()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ident = identity_key(self.identity())
        keys = [sha256_text(t) for t in texts]
        found: dict[str, list[float]] = {}
        self._lookup(ident, keys, found)
        if len(found) < len(set(keys)):
            with self._computing():
                self._lookup(ident, keys, found)  # another run may have filled it meanwhile
                missing = list(dict.fromkeys(t for t in texts if sha256_text(t) not in found))
                self._compute(ident, missing, found)
        return [found[k] for k in keys]

    def _compute(self, ident: str, missing: list[str], found: dict[str, list[float]]) -> None:
        start = time.monotonic()
        for i in range(0, len(missing), self.flush_every):
            batch = missing[i : i + self.flush_every]
            vectors = self.inner.embed(batch)
            with self.db:  # one commit per flush: an interruption loses at most this batch
                for text, vec in zip(batch, vectors, strict=True):
                    key = sha256_text(text)
                    found[key] = vec
                    self.db.execute(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                        (ident, key, array("f", vec).tobytes()),
                    )
            self.computed += len(batch)
            if len(missing) > self.flush_every:
                done = i + len(batch)
                print(
                    f"embedded {done}/{len(missing)} in {time.monotonic() - start:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )
