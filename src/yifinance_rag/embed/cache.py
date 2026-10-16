"""A disk cache of embeddings keyed by (embedder identity, text), so the 809 chunks and
300 questions are embedded once per model and Ollama can be stopped afterwards
(docs/ROADMAP.md S0: memory next to Elasticsearch)."""

from __future__ import annotations

import json
import sqlite3
from array import array
from collections.abc import Sequence
from pathlib import Path

from yifinance_rag.data.dataset import sha256_text
from yifinance_rag.embed.ollama import Embedder


def identity_key(identity: dict[str, object]) -> str:
    return sha256_text(json.dumps(identity, sort_keys=True))


class CachedEmbedder:
    def __init__(self, inner: Embedder, path: Path) -> None:
        self.inner = inner
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "identity TEXT NOT NULL, text_sha256 TEXT NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY (identity, text_sha256))"
        )
        self.computed = 0  # texts sent to the inner embedder in this process

    def identity(self) -> dict[str, object]:
        return self.inner.identity()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ident = identity_key(self.identity())
        keys = [sha256_text(t) for t in texts]
        found: dict[str, list[float]] = {}
        for key in set(keys):
            row = self.db.execute(
                "SELECT vector FROM vectors WHERE identity = ? AND text_sha256 = ?", (ident, key)
            ).fetchone()
            if row is not None:
                found[key] = array("f", row[0]).tolist()
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys, strict=True) if k not in found))
        if missing:
            vectors = self.inner.embed(missing)
            self.computed += len(missing)
            with self.db:
                for text, vec in zip(missing, vectors, strict=True):
                    key = sha256_text(text)
                    found[key] = vec
                    self.db.execute(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                        (ident, key, array("f", vec).tobytes()),
                    )
        return [found[k] for k in keys]
