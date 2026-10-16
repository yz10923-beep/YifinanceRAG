"""One pass over the dataset: hashes, counts, the chunk rebuild, the split and leakage.
`yfr data check` prints it; the `dataset` tests assert on it."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from yifinance_rag.data import pins
from yifinance_rag.data.benchmark import check_benchmark, load_benchmark, split_hash
from yifinance_rag.data.corpus import (
    benchmark_chunks,
    check_chunks,
    chunk_set_hash,
    doc_folder,
    load_docs,
)
from yifinance_rag.data.dataset import RELEASE_MANIFEST, Dataset, sha256_file


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)


def check_dataset(ds: Dataset) -> Report:
    r = Report()

    release_sha = sha256_file(ds.path(RELEASE_MANIFEST))
    r.facts["release_manifest_sha256"] = release_sha
    if release_sha != pins.RELEASE_MANIFEST_SHA256:
        r.problems.append("release manifest differs from the course's pinned release")
    r.problems += ds.verify_release()

    docs = load_docs(ds)
    by_folder = dict(sorted(Counter(doc_folder(d) for d in docs).items()))
    r.facts["docs_by_folder"] = by_folder
    if by_folder != pins.DOCS_BY_FOLDER:
        r.problems.append(f"docs by folder {by_folder} != pinned {pins.DOCS_BY_FOLDER}")
    r.facts["doc_types"] = dict(sorted(Counter(d.doc_type for d in docs).items()))

    chunks = benchmark_chunks(docs)
    r.facts["chunks"] = len(chunks)
    r.facts["chunk_set_sha256"] = chunk_set_hash(chunks)
    if len(chunks) != pins.N_CHUNKS:
        r.problems.append(f"{len(chunks)} chunks rebuilt, pinned {pins.N_CHUNKS}")
    r.problems += check_chunks(ds, chunks)

    questions, labels = load_benchmark(ds)
    labs = list(labels.values())
    by_type = dict(Counter(lab.question_type for lab in labs))
    dev_by_type = dict(Counter(lab.question_type for lab in labs if lab.split == "dev"))
    splits = Counter(lab.split for lab in labs)
    r.facts["questions_by_type"] = by_type
    r.facts["dev_by_type"] = dev_by_type
    r.facts["splits"] = dict(splits)
    r.facts["conversation_modes"] = dict(Counter(lab.conversation_mode for lab in labs))
    r.facts["evidence_kinds"] = dict(
        Counter(e.split(":", 1)[0] for lab in labs for e in lab.evidence_ids)
    )
    r.facts["no_doc_evidence_by_type"] = dict(
        Counter(lab.question_type for lab in labs if not lab.doc_evidence)
    )
    if by_type != pins.QUESTIONS_BY_TYPE:
        r.problems.append(f"questions by type {by_type} != pinned {pins.QUESTIONS_BY_TYPE}")
    if dev_by_type != pins.DEV_BY_TYPE:
        r.problems.append(f"dev by type {dev_by_type} != pinned {pins.DEV_BY_TYPE}")
    if (splits["dev"], splits["test"]) != (pins.N_DEV, pins.N_TEST):
        r.problems.append(f"split sizes {dict(splits)} != pinned {pins.N_DEV}/{pins.N_TEST}")

    r.facts["split_sha256"] = digest = split_hash(labs)
    if pins.SPLIT_SHA256 is None:
        r.problems.append(f"split hash not pinned yet; Yi: pin SPLIT_SHA256 = {digest!r}")
    elif digest != pins.SPLIT_SHA256:
        r.problems.append(f"split hash {digest} != pinned {pins.SPLIT_SHA256}")

    r.problems += check_benchmark(ds, questions, labels, {c.chunk_id for c in chunks})
    return r
