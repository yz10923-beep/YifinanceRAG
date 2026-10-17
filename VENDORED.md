# Vendored code

These modules were copied from yiagent (github.com/yz10923-beep/yiagent, branch
`claude/jolly-euler-45xyyc`, commit `3502431631b52ad92f88fdff7e64a4fa1c198a9b`,
2026-10-15, fetched 2026-10-16 from GitHub, not from the local Desktop/yiagent checkout).
Most of them came to yiagent from kama (github.com/yz10923-beep/project-1); yiagent's
VENDORED.md has that history and its own changes (e.g. the pricing rewrite). They are
owned here from now on; yiagent is not a dependency. Record every change below, so a fix
can be ported in either direction.

Every file had the import path changed (`yi_agent` -> `yifinance_rag`); the table lists
changes beyond that.

| Here | yiagent | Changes |
|---|---|---|
| `src/yifinance_rag/llm/types.py` | `src/yi_agent/llm/types.py` | none |
| `src/yifinance_rag/llm/anthropic_provider.py` | `src/yi_agent/llm/anthropic_provider.py` | S2 (2026-10-17): `prompt_cache` switch (default on, unchanged behaviour); off for one-shot prompts such as the reranker, where an unread cache write costs 1.25x input. An empty tool list is no longer sent as `tools: []`. `aclose()` closes the HTTP client (the reranker runs many calls on one loop and closes it at the end) |
| `src/yifinance_rag/llm/pricing.py` | `src/yi_agent/llm/pricing.py` | none (rates checked 2026-10-11 there) |
| `src/yifinance_rag/llm/retry.py` | `src/yi_agent/llm/retry.py` | none |
| `src/yifinance_rag/trace/span.py` | `src/yi_agent/trace/span.py` | none |
| `src/yifinance_rag/trace/tracer.py` | `src/yi_agent/trace/tracer.py` | ContextVar renamed `yfr_current_span` |
| `src/yifinance_rag/trace/analyze.py` | `src/yi_agent/trace/analyze.py` | one comment (no `yi trace` CLI here) |
| `src/yifinance_rag/context.py` | `src/yi_agent/context.py` | none |
| `src/yifinance_rag/outputs.py` | `src/yi_agent/outputs.py` | none |
| `src/yifinance_rag/config.py` | `src/yi_agent/config.py` | `YFR_` prefix, `~/.yfr/.env`; this repo's fields (`FLOWPAY_DATASET_ROOT` unprefixed, `YFR_PG_URL` as a secret; no ES); `dataset_root()` check (set, unzipped, expected layout, outside the repo); S1b: `ollama_url`, `cache_dir` + `cache_dir()` |
| `tests/fakes.py` | same | none |
| `tests/conftest.py` | same | also sets `USERPROFILE`, so the isolated home works on Windows too |
| `tests/unit/test_anthropic_provider.py` | same | one annotation so tests pass `mypy --strict` (here `make type` checks tests too) |
| `tests/unit/test_pricing.py`, `test_trace.py`, `test_context.py` | same | none |
| `tests/unit/test_config.py` | same | rewritten for this repo's fields and `dataset_root()` |

The span format is kept identical on purpose: `trace/analyze.py` reads traces from kama,
yiagent or this repo.
