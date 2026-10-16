"""A stand-in Ollama on localhost: /api/tags, /api/version and the OpenAI-compatible
/v1/embeddings, with deterministic vectors (hashed character bigrams), so the real HTTP
client runs in tests."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from yifinance_rag.embed.ollama import COURSE_DIGEST, COURSE_MODEL

DIMS = 64


def fake_vector(text: str) -> list[float]:
    v = [0.0] * DIMS
    for a, b in zip(text, text[1:], strict=False):
        h = int(hashlib.md5((a + b).encode()).hexdigest()[:8], 16)
        v[h % DIMS] += 1.0
    return v if any(v) else [1.0] + [0.0] * (DIMS - 1)


@dataclass
class FakeOllama:
    url: str = ""
    model: str = COURSE_MODEL
    digest: str = COURSE_DIGEST
    batches: list[list[str]] = field(default_factory=list)
    reverse: bool = True  # answer out of index order, as a real server may


@contextmanager
def fake_ollama(**kwargs: object) -> Iterator[FakeOllama]:
    state = FakeOllama(**kwargs)  # type: ignore[arg-type]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: object) -> None:
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/api/tags":
                self._send({"models": [{"name": state.model, "digest": state.digest}]})
            else:
                self._send({"version": "0.0-fake"})

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = body["input"]
            state.batches.append(texts)
            data = [
                {"index": i, "embedding": fake_vector(t), "object": "embedding"}
                for i, t in enumerate(texts)
            ]
            self._send({"data": data[::-1] if state.reverse else data})

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
