# Roadmap

Built from CLAUDE.md's "Proposed stages" (2026-10-16). Time box: about 3 weeks in total,
alongside job hunting; yiagent's S5 (Bank) closes first or in parallel. Decisions are
recorded here with their date when made; open ones say what would decide them.

Rules for every stage (CLAUDE.md "Invariants"): no ground truth reaches the agent; graders
have oracle / null / wrong checks inside `make verify`; tune on dev (60), report on test
(240) once; one condition per variant; infra failures go to `errors.jsonl`, not scores.

## S0 scaffold (done 2026-10-16, pending Yi's first VM run)

- uv + Python 3.12, `make verify` = ruff check + ruff format --check + mypy --strict
  (src and tests) + pytest.
- Vendored from yiagent: LLM provider (Anthropic), retry, pricing, tracer + analyzer,
  context meter, output store, `tests/fakes.py`; config loader rewritten for this repo
  (VENDORED.md).
- `docker-compose.yml`: `pgvector/pgvector:0.8.2-pg17` on 127.0.0.1:5432.
- Config: `FLOWPAY_DATASET_ROOT` (the unpacked `flowpay-dataset/` folder, checked to be
  outside the repo), `YFR_PG_URL`, `ANTHROPIC_API_KEY`; `.env` and data gitignored.

**Done when** `make verify` passes on the VM and `make pg-up` brings pgvector up healthy.

### Decided: where things run (2026-10-16)
Yi: **everything runs on the VM** (Ubuntu 24.04, 4 vCPU, 7.8 GiB RAM, no swap). The laptop
is for editing only.

### Open: fitting pgvector + Ollama next to yiagent's Elasticsearch on 7.8 GiB
Rough sizes: ES with a 2 GiB heap uses about 3 GiB resident; pgvector with 809 chunks is
a few hundred MiB; Ollama with `qwen3-embedding:0.6b` about 1-1.5 GiB while loaded; a
0.6B reranker about the same. All at once is near the limit with no swap, and an OOM kill
mid-run is an infra failure.

Options:
1. **Stop ES while working here** (`docker compose down` in yiagent). Simple; blocks
   working on both repos in the same hour.
2. **Run both, ES heap at 1 GiB** (`YI_ES_HEAP=1g`). Probably fits; yiagent's frozen eval
   numbers were made with 2 GiB, so don't run yiagent evals in this mode.
3. **Embed once, cache to disk.** 809 chunks + 300 questions per embedding model take
   minutes on CPU. Store vectors in a file keyed by (model, chunk sha256); Ollama runs
   only while building an index, then exits (`keep_alive: 0`). pgvector then holds a
   frozen collection, and eval runs need no Ollama at all.

Recommendation: 3, plus 1 during index builds. It also makes every eval number point to
an exact index build (the course's immutable-collection idea). Add a 2 GiB swap file as a
safety net either way. Decide before S1's first index build.

## S1 data + retrieval eval

- Loaders for docs, `knowledge_chunks.jsonl`, `index_manifest.json`, the 300 blueprints,
  with sha256 checks against the manifest; the split's hash pinned in a test.
- Leakage test: nothing the agent can see contains `ground_truth`, `resolved_*`,
  `canonical_answer`, evidence ids or anything from `audit/event_journal.jsonl`.
- Chunk-id mapping: retrieved passage -> gold `evidence_id` by doc_id + section heading,
  tested on the benchmark's own chunking and on the course's (title-only `c001`, split
  long sections).
- Metrics: recall@5 / recall@10 / MRR against gold doc evidence, overall and **per
  question type**; only `doc:` evidence counts for retrieval (sql/runtime/negative are
  tool evidence, scored in S3).
- Baselines on dev: oracle (gold chunks; must score 1.0), null (random chunks), course
  dense top-10 (qwen3-embedding:0.6b, 1000-char heading chunks), BM25.

**Done when** the metric code passes its oracle/null checks in `make verify` and the dev
table for dense vs BM25 vs oracle exists.

### Open: BM25 via Postgres full-text or in-process with jieba
- **Postgres full-text.** Postgres has no Chinese parser; the pgvector image has no
  zhparser / pg_jieba. Workaround: segment with jieba in Python, store space-joined
  tokens in a `tsvector` with the `simple` config. But `ts_rank` is not BM25 (no IDF, no
  length saturation), so it is a different baseline from the one the literature means.
  Gains: one store, SQL filters (doc_type, version, effective dates) in the same query.
- **In-process.** About 60 lines of our own BM25 (k1, b tunable) over 809 chunks, built in
  well under a second; exact BM25; tokenization fully under test. Metadata filters done in
  Python before scoring.

Recommendation: **in-process, own implementation**, with a jieba user dictionary for
FlowPay tokens (error codes like `FP-4041`, ids like `pay_bm_001`; check first how jieba
splits them, a test should pin it). Explaining BM25 from its formula is an interview
question; "I called ts_rank" is not an answer. Postgres FTS can be added later as a
variant if a deploy needs a single store.

## S2 retrieval ablations (dev only)

One change per variant, each against the S1 table: hybrid (RRF over dense + BM25),
reranker (a Qwen3 or bge reranker over the top-k), chunking variants (overlap, size,
heading-aware), metadata filters (doc_type, version, effective dates for
`temporal_versioned`). Hypothesis: BM25 / hybrid wins on exact tokens, dense on
paraphrase; report per type either way.

**Done when** a dev table lists every variant per question type, and one retrieval
config is chosen for S4 with the reason.

## S3 answer graders

- Deterministic: evidence ids cited vs gold, trajectory assertions, forbidden shortcuts.
- LLM judge for `required_assertions` / `negative_assertions` (Chinese free text),
  calibrated against about 50 dev answers Yi labels by hand; report agreement.
- The course's `scripts/benchmark/` scorer run on our outputs as a second score.
- Oracle / null / wrong checks for every grader.

**Done when** every grader passes its oracle/null/wrong checks and the judge's agreement
with Yi's labels is reported.

## S4 agent

Own loop over the vendored provider; an OpenAI-compatible provider (DeepSeek / Qwen) next
to the Anthropic one; data tools under a policy file (column whitelists, scope enforced in
SQL, row caps); ask-user for progressive-disclosure questions via the dataset's simulated
user. A/Bs on dev: tool shape (per-doc-type search tools vs one `search_documents`),
`disputed()` bounce, budgets. Course s1-s4 run as baselines on the same chat model.

**Done when** the dev table compares our agent with course s1-s4 on the same model.

## S5 test once + deploy

Final numbers on the 240 test questions, run once; a small deployed demo; README with the
results table and cost per question.
