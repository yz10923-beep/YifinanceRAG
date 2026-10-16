"""Frozen facts about the dataset. `yfr data check` and the `dataset` tests compare the
files on disk with these; a mismatch means a different dataset, not a bug to patch here.
Only Yi changes a pin, after looking at why it moved."""

from __future__ import annotations

# The course pins the same file (backend/configs/knowledge.toml): a match means our copy
# is the release the course's own numbers were made on.
RELEASE_MANIFEST_SHA256 = "2e50d09e8898ce0aa4eb3f02698ac0ef76484f9e9c90c4ffac318d12d3e77a5f"

# sha256 over sorted `question_id<TAB>split` lines (benchmark.split_hash). None until Yi's
# first `yfr data check` on the VM prints it.
SPLIT_SHA256: str | None = None

# Counts checked by hand on 2026-10-16 (CLAUDE.md "Facts checked about the dataset").
N_CHUNKS = 809
DOCS_BY_FOLDER = {
    "api_docs": 34,
    "business_rules": 16,
    "release_notes": 10,
    "support_cases": 16,
    "troubleshooting": 24,
}
QUESTIONS_BY_TYPE = {
    "simple_retrieval": 70,
    "multi_source": 70,
    "multi_hop": 60,
    "multi_document": 50,
    "temporal_versioned": 25,
    "tool_use_diagnostic": 25,
}
DEV_BY_TYPE = {
    "simple_retrieval": 14,
    "multi_source": 14,
    "multi_hop": 12,
    "multi_document": 10,
    "temporal_versioned": 5,
    "tool_use_diagnostic": 5,
}
N_DEV, N_TEST = 60, 240
