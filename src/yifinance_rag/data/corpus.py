"""The knowledge corpus: the 100 docs, the benchmark's chunking rebuilt from them, and the
map from any chunk back to the benchmark's gold evidence ids.

The benchmark's rule (the course's `scripts/benchmark/evidence.py`): drop the front
matter, split the body at lines starting with `## `, skip blank sections. The first
section (the `# Title` line and any intro) is `c001`. Offsets here are character offsets
into the whole file, so chunks from any chunking map onto the same sections.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from yifinance_rag.data.dataset import DataError, Dataset, sha256_file, sha256_text

KNOWLEDGE_DIR = "data/knowledge"
CHUNKS_FILE = "data/index/knowledge_chunks.jsonl"
INDEX_MANIFEST = "data/index/index_manifest.json"


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    doc_type: str
    path: str  # dataset-relative
    text: str  # the whole file, front matter included, so offsets match the file


@dataclass(frozen=True)
class Section:
    ordinal: int  # 1-based: c001, c002, ...
    heading: str  # text after `## `; "" for the title section
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # for the benchmark's chunking, the gold evidence id
    doc_id: str
    doc_type: str
    title: str
    heading: str
    start: int
    end: int
    text: str

    def indexed_text(self) -> str:
        """What a retriever indexes: the doc title carries much of a section's topic."""
        return f"{self.title}\n{self.text}"


def front_matter(text: str) -> tuple[dict[str, str], int]:
    """Scalar `key: value` lines of the YAML front matter, and the offset where the body
    starts. Lists and nested values are skipped: only doc_id, title and doc_type are read."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, 0
    meta: dict[str, str] = {}
    offset = len(lines[0])
    for line in lines[1:]:
        offset += len(line)
        if line.strip() == "---":
            return meta, offset
        key, sep, value = line.partition(":")
        if sep and key.strip() and not key.startswith((" ", "\t", "-")):
            meta[key.strip()] = value.strip().strip("'\"")
    raise DataError("front matter is not closed")


def benchmark_sections(text: str) -> list[Section]:
    _, body = front_matter(text)
    spans: list[tuple[int, int, str]] = []  # start, end, first line
    start = offset = body
    first = ""
    for line in text[body:].splitlines(keepends=True):
        if line.startswith("## ") and offset > start:
            spans.append((start, offset, first))
            start = offset
        if offset == start:
            first = line
        offset += len(line)
    if offset > start:
        spans.append((start, offset, first))
    sections: list[Section] = []
    for s, e, first_line in spans:
        if not text[s:e].strip():
            continue
        heading = first_line[3:].strip() if first_line.startswith("## ") else ""
        sections.append(Section(len(sections) + 1, heading, s, e))
    return sections


def evidence_id(doc_id: str, ordinal: int) -> str:
    return f"doc:{doc_id}#c{ordinal:03d}"


def parse_doc(path: str, text: str) -> Doc:
    try:
        meta, _ = front_matter(text)
    except DataError as e:
        raise DataError(f"{path}: {e}") from e
    missing = [k for k in ("doc_id", "title", "doc_type") if not meta.get(k)]
    if missing:
        raise DataError(f"{path}: front matter has no {', '.join(missing)}")
    return Doc(meta["doc_id"], meta["title"], meta["doc_type"], path, text)


def load_docs(ds: Dataset) -> list[Doc]:
    folder = ds.path(KNOWLEDGE_DIR)
    if not folder.is_dir():
        raise DataError(f"missing {KNOWLEDGE_DIR}/ under {ds.root}")
    docs = [
        parse_doc(p.relative_to(ds.root).as_posix(), p.read_text(encoding="utf-8"))
        for p in sorted(folder.rglob("*.md"))
    ]
    dupes = [d for d, n in Counter(doc.doc_id for doc in docs).items() if n > 1]
    if dupes:
        raise DataError(f"duplicate doc_id: {dupes}")
    return docs


def benchmark_chunks(docs: Iterable[Doc]) -> list[Chunk]:
    return [
        Chunk(
            evidence_id(doc.doc_id, s.ordinal),
            doc.doc_id,
            doc.doc_type,
            doc.title,
            s.heading,
            s.start,
            s.end,
            doc.text[s.start : s.end],
        )
        for doc in docs
        for s in benchmark_sections(doc.text)
    ]


def doc_folder(doc: Doc) -> str:
    """The folder under data/knowledge/ (api_docs, business_rules, ...)."""
    return PurePosixPath(doc.path).relative_to(KNOWLEDGE_DIR).parts[0]


def chunk_set_hash(chunks: Sequence[Chunk]) -> str:
    """Identifies an index build's input: ids, spans and indexed text, in order."""
    lines = [
        json.dumps([c.chunk_id, c.start, c.end, c.indexed_text()], ensure_ascii=False)
        for c in chunks
    ]
    return sha256_text("\n".join(lines))


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def check_chunks(ds: Dataset, chunks: Sequence[Chunk], *, show: int = 5) -> list[str]:
    """Problems between our rebuilt chunking and the benchmark's knowledge_chunks.jsonl:
    the file's hash against index_manifest.json, the ids, the headings and (if the file
    carries it) the text."""
    problems: list[str] = []
    digest = sha256_file(ds.path(CHUNKS_FILE))
    manifest = json.loads(ds.read_text(INDEX_MANIFEST))
    if digest not in set(_strings(manifest)):
        problems.append(f"{CHUNKS_FILE} sha256 {digest} is not in {INDEX_MANIFEST}")

    rows = ds.read_jsonl(CHUNKS_FILE)
    if rows and "evidence_id" not in rows[0]:
        return [*problems, f"{CHUNKS_FILE} rows have no evidence_id; keys: {sorted(rows[0])}"]
    theirs = {str(r["evidence_id"]): r for r in rows}
    ours = {c.chunk_id: c for c in chunks}
    if len(theirs) != len(rows):
        problems.append(f"{CHUNKS_FILE}: duplicate evidence_id")
    if missing := sorted(set(theirs) - set(ours)):
        problems.append(f"{len(missing)} benchmark chunks not rebuilt, e.g. {missing[:show]}")
    if extra := sorted(set(ours) - set(theirs)):
        problems.append(f"{len(extra)} rebuilt chunks not in the benchmark, e.g. {extra[:show]}")

    heading_diffs: list[str] = []
    text_diffs: list[str] = []
    for cid in sorted(set(theirs) & set(ours)):
        row, chunk = theirs[cid], ours[cid]
        section = row.get("section")
        if isinstance(section, str) and section.strip() != chunk.heading:
            heading_diffs.append(f"{cid}: {section.strip()!r} vs ours {chunk.heading!r}")
        text = row.get("text", row.get("content"))
        if isinstance(text, str) and "".join(text.split()) != "".join(chunk.text.split()):
            text_diffs.append(cid)
    if heading_diffs:
        problems.append(f"{len(heading_diffs)} headings differ, e.g. {heading_diffs[:show]}")
    if text_diffs:
        problems.append(f"{len(text_diffs)} chunk texts differ, e.g. {text_diffs[:show]}")
    return problems


class EvidenceMap:
    """Maps a chunk from any chunking to the benchmark evidence ids it stands for.

    A chunk counts for a gold section when at least half of the chunk lies in that
    section, or it covers at least half of the section. Spans, not headings: headings can
    repeat inside a doc, and windowed chunks can straddle sections.
    """

    def __init__(self, gold: Iterable[Chunk]) -> None:
        self._by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in gold:
            self._by_doc[c.doc_id].append(c)

    def ids_for(self, chunk: Chunk) -> list[str]:
        size = chunk.end - chunk.start
        out: list[str] = []
        for g in self._by_doc.get(chunk.doc_id, []):
            overlap = min(chunk.end, g.end) - max(chunk.start, g.start)
            if overlap <= 0:
                continue
            if 2 * overlap >= size or 2 * overlap >= g.end - g.start:
                out.append(g.chunk_id)
        return out
