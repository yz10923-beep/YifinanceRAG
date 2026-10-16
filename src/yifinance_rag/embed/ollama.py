"""Embeddings from a local Ollama, exactly as the course requests them, so query vectors
land in the same space as the vectors in the course's dump.

Same model tag, checked by digest (a tag can be re-pointed upstream; the course pins
the digest for that reason), the same OpenAI-compatible endpoint and request body, and
the same query instruction. Passages are embedded as given; only queries get the
instruction.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

COURSE_MODEL = "qwen3-embedding:0.6b"
COURSE_DIGEST = "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d"
QUERY_INSTRUCTION = "Given a technical support question, retrieve relevant passages that answer it."


class EmbedError(Exception):
    """Ollama unreachable, the wrong model, or a malformed response: an infra failure."""


def query_input(question: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery: {question}"


class Embedder(Protocol):
    def identity(self) -> dict[str, object]:
        """What decides the vector space: two embedders with equal identities are
        interchangeable, and cached vectors are keyed by it."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        *,
        model: str = COURSE_MODEL,
        digest: str = COURSE_DIGEST,
        batch_size: int = 8,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model, self.digest = model, digest
        self.batch_size, self.timeout = batch_size, timeout
        self._identity: dict[str, object] | None = None

    def _call(self, path: str, body: dict[str, object] | None = None) -> object:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise EmbedError(f"Ollama {path} at {self.base_url}: {e}") from e

    def identity(self) -> dict[str, object]:
        if self._identity is None:
            tags = self._call("/api/tags")
            models = tags.get("models", []) if isinstance(tags, dict) else []
            found = next((m for m in models if m.get("name") == self.model), None)
            if found is None:
                raise EmbedError(f"{self.model} is not pulled; run `ollama pull {self.model}`")
            if found.get("digest") != self.digest:
                raise EmbedError(
                    f"{self.model} has digest {found.get('digest')}, pinned {self.digest}: "
                    "its vectors would not be comparable with the course's"
                )
            self._identity = {
                "endpoint": "openai-compatible /v1/embeddings",
                "model": self.model,
                "revision": self.digest,
                "request_options": {},
            }
        return self._identity

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.identity()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i : i + self.batch_size])
            reply = self._call(
                "/v1/embeddings",
                {"model": self.model, "input": batch, "encoding_format": "float"},
            )
            data = reply.get("data") if isinstance(reply, dict) else None
            if not isinstance(data, list) or len(data) != len(batch):
                raise EmbedError(f"expected {len(batch)} embeddings, got {reply!r:.200}")
            ordered = sorted(data, key=lambda d: int(d["index"]))
            if [int(d["index"]) for d in ordered] != list(range(len(batch))):
                raise EmbedError("embedding indices do not cover the batch")
            out += [[float(x) for x in d["embedding"]] for d in ordered]
        return out
