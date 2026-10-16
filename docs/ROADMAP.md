# Roadmap

Built from CLAUDE.md's "Proposed stages" (2026-10-16). Time box: about 3 weeks in total,
alongside job hunting; yiagent's S5 (Bank) closes first or in parallel. Decisions are
recorded here with their date when made; open ones say what would decide them.

Rules for every stage (CLAUDE.md "Invariants"): no ground truth reaches the agent; graders
have oracle / null / wrong checks inside `make verify`; tune on dev (60), report on test
(240) once; one condition per variant; infra failures go to `errors.jsonl`, not scores.

## S0 scaffold (done 2026-10-16, closed on the VM)

- uv + Python 3.12, `make verify` = ruff check + ruff format --check + mypy --strict
  (src and tests) + pytest.
- Vendored from yiagent: LLM provider (Anthropic), retry, pricing, tracer + analyzer,
  context meter, output store, `tests/fakes.py`; config loader rewritten for this repo
  (VENDORED.md).
- `docker-compose.yml`: `pgvector/pgvector:0.8.2-pg17` on 127.0.0.1:5432.
- Config: `FLOWPAY_DATASET_ROOT` (the unpacked `flowpay-dataset/` folder, checked to be
  outside the repo), `YFR_PG_URL`, `ANTHROPIC_API_KEY`; `.env` and data gitignored.

**Result (VM, 2026-10-16):** `make verify` 61 passed; `make pg-up` healthy; dataset
unpacked at `~/flowpay-dataset`.

### Decided: where things run (2026-10-16)
Yi: **everything runs on the VM** (Ubuntu 24.04, 4 vCPU, 7.8 GiB RAM, no swap). The laptop
is for editing only. Claude's cloud container has no dataset; it writes code against the
formats in the course code, and the VM checks them (`make data-check`, `make data-test`).

### Decided: memory next to yiagent's Elasticsearch (2026-10-16)
Measured on the VM with both stacks up: ES 3.8 GiB resident, pgvector 26 MiB, 4.1 GiB
available, no swap. Ollama with `qwen3-embedding:0.6b` (about 1-1.5 GiB loaded) fits in
that, with little margin for a reranker.

Decision (Yi accepted the recommendation when starting S1): **embed once, cache to
disk**, keyed by (embedding model identity, text sha256); Ollama runs only while
embedding (`keep_alive: 0`); plus a **2 GiB swap file** as a safety net, added before
Ollama is installed (S1b). Caveat: the cache covers S1-S2 (300 fixed questions). In S4
the agent writes its own queries, so something must embed them live: Ollama next to ES,
or a hosted copy of the exact same model (a different model is a new collection and a new
condition). Decide that at S4's start, with a measurement.

## S1 data + retrieval eval

Split into S1a (no embeddings; done 2026-10-16) and S1b (dense retrieval, needs Ollama on
the VM).

### S1a: data, blinding, metrics, BM25

- Loaders: the release manifest (`data/releases/flowpay_dataset_v1/release_manifest.json`,
  sha256 pinned to the value the course pins in `backend/configs/knowledge.toml`, so our
  copy is the release the course's numbers were made on); every artifact it lists is
  checked for size and sha256. `knowledge_chunks.jsonl` is checked against
  `index_manifest.json`.
- The benchmark's chunking is **rebuilt from the 100 docs** (front matter dropped; split
  at `## ` lines; blank sections skipped; the title section is `c001`, the course's
  `scripts/benchmark/evidence.py` rule) and cross-checked against
  `knowledge_chunks.jsonl`: same ids, same headings. Chunk text comes from the docs, so
  S2's chunking variants use the same code path.
- **Mapping retrieved chunks to gold ids by character span, not by heading** (decided
  2026-10-16): a chunk counts for a gold section when at least half of the chunk lies in
  that section, or it covers at least half of the section. Headings can repeat inside a
  doc, and windowed chunks can straddle sections; spans handle both. Tested on synthetic
  docs with split and straddling chunks.
- **Label blinding:** a retriever (later the agent) sees `Question(case_id, text)` only.
  `case_id` is opaque (`q-` + sha256 of the question id), because question ids may encode
  the type. Labels (type, split, gold evidence, ground truth) live in `Label`, read only
  by graders. Leakage tests: the `Question` fields are a whitelist; no question text
  contains a gold evidence id or its canonical answer; the index is built from
  `data/knowledge/` only.
- **Split hash:** sha256 over sorted `question_id<TAB>split` lines; printed by
  `yfr data check`, pinned in `src/yifinance_rag/data/pins.py` once Yi sends it.
- **Metrics** (`doc:` evidence only; sql / runtime / negative evidence is tool evidence,
  scored in S3): recall@5, recall@10, complete@10 (all gold chunks in the top 10),
  MRR@10, overall and per question type. Questions without doc evidence are counted
  and excluded. Checks in `make verify`: oracle = 1.0, a wrong oracle (another question's
  gold) = 0, a seeded random retriever at chance (k / N).
- **BM25** (decided 2026-10-16: in-process, own implementation; Lucene's IDF,
  k1 = 1.2, b = 0.75). Tokens: ASCII identifier runs first (`FP-4041`, `pay_bm_001`,
  `2026-07-01`), kept whole and lowercased plus their parts; jieba on the rest (jieba
  alone splits `FP-4041` into `FP`, `-`, `4041`). Indexed text = doc title + section.
- `yfr retrieval --retriever bm25|oracle|random --split dev` writes
  `evals/results/retrieval/<run>/summary.json` (with the release hash, split hash, chunk
  set hash and git commit) and a Markdown table. `--split test` needs `--final`, which
  only Yi passes.

**Done when** `make data-test` passes on the VM (hashes, counts, chunk rebuild, leakage),
the split hash is pinned, and the dev table for oracle / random / BM25 exists.

**Result (2026-10-16, on the zip Yi uploaded; confirmed on the VM the same day: same split
hash, `data-test` 3 passed, identical tables):**
`yfr data check` ok, `make data-test` 3 passed, `make verify` 83 passed. Found on the real
data, and fixed:
- The release check failed on 45 files. All are in groups we never read
  (`evidence_audit`, `offline_fixture_validation`, `generation_sources`, part of
  `dataset_documentation`), which the course-member zip ships trimmed or newer. Every group
  we read passes, including the audit journal's hash. Now strict on those 7 groups
  (`REQUIRED_GROUPS`) and reporting the rest.
- The benchmark names `c001` after the doc title, not `""`. With that, all 809 ids,
  headings and texts match `knowledge_chunks.jsonl`.
- Indexed docs are now checked against the release's own `visibility` labels
  (`agent_knowledge`, `agent_via_typed_tools`): a leakage rule from the dataset itself.

Dev table (60 questions, all with doc evidence; runs under `evals/results/retrieval/`,
commit `3a9e555`):

| retriever | recall@5 | recall@10 | complete@10 | MRR@10 |
|---|---:|---:|---:|---:|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| wrong-oracle | 0.000 | 0.000 | 0.000 | 0.000 |
| random (chance 10/809 = 0.012) | 0.000 | 0.000 | 0.000 | 0.000 |
| **BM25** | **0.290** | **0.456** | **0.317** | **0.287** |

BM25 by type (recall@10 / MRR@10): simple_retrieval 0.714 / 0.214, multi_document
0.600 / 0.510, multi_hop 0.292 / 0.209, temporal_versioned 0.400 / 0.192, multi_source
0.345 / 0.330, tool_use_diagnostic 0.200 / 0.200. Small n per type (5-14): read
differences under about 0.15 as noise.

What the misses look like (dev, BM25): 24 of 60 cases have no gold chunk in the top 10;
when gold is found it is often at rank 2-10, not 1 (simple_retrieval: 71% found, MRR
0.21). 30 of 300 top-5 slots are the right doc's wrong section. The title chunks are not
the cause (1 of 300 slots); the hypothesis that short title sections win on length was
wrong. Leads for S2, not acted on: section-level ranking inside the right doc, dense
retrieval for paraphrase, query rewriting for the progressive-disclosure openers (vague
by design).

### S1b: dense baselines (code done 2026-10-16; the embedding runs are Yi's, on the VM)

Claude's container can't reach Ollama's registry (403), so everything except the
embedding calls was built and checked here; tests use a fake Ollama HTTP server.

- `OllamaEmbedder`: the course's exact request (`/v1/embeddings`, `encoding_format:
  float`, no options), `qwen3-embedding:0.6b` **refused unless its digest is the course's
  pinned `ac6da0df...`** (a re-pointed tag would put queries in another vector space).
  Queries get the course's instruction (`Instruct: ...\nQuery: ...`); passages don't.
- `CachedEmbedder`: SQLite at `~/.yfr/cache/embeddings.sqlite`, keyed by (embedder
  identity, text sha256); a second run sends nothing to Ollama (tested).
- `course-dense`: the course's collection, restored by `make course-index` into database
  `course_kb` (tables and data only; its guard triggers need functions the dump lacks).
  Its `build_config` is checked against our embedder (`same_vector_space`, borrowed from
  the course) before any query.
- `dense-809`: the benchmark's chunks embedded with the same model in the course's
  passage format, so against course-dense **only the chunking differs**.
- Both rank by exact cosine in numpy, ties in (doc_id, chunk_no) order, like the course's
  SQL. Embedding failures stop the run with exit 3 and write no scores.

Checked on the real data (`make pg-test`, 3 passed):
- All 712 course chunk offsets point into our doc text (the course and we count the same
  characters), so `EvidenceMap` maps course chunks to gold ids by span.
- `course_style_text` reproduces the course's stored chunk text exactly for every course
  chunk that is a whole benchmark section.
- numpy's top 10 equals pgvector's (`ORDER BY 1 - (embedding <=> q) DESC, doc_id,
  chunk_no`) on 8 probe vectors.

Found: the course chunker drops all 100 title-only `c001` sections, and 9 of 126 dev gold
ids are `c001`. **Recall ceiling of the course index on dev: 0.942** (simple_retrieval
1.0, temporal_versioned 1.0, multi_source 0.964, multi_document 0.9,
tool_use_diagnostic 0.9, multi_hop 0.875). The course's own scorer can't hit `c001`
either. Runs print and record this ceiling (`recall_ceiling` in summary.json), so the
course row is read against what its index could reach.

**Done when** the dev table has oracle, random, BM25, course-dense and dense-809, per
type, each pointing to an exact index build.

**Result (VM, 2026-10-16, commit `d8cf4a1`):** model digest = the course's pin;
`make course-index` 712 chunks; `make pg-test` 3 passed. Embedding the 809 chunks took
about 14 minutes on the VM's 4 vCPUs (then cached); with ES and Ollama loaded, 3.0 GiB
available, swap 22 MiB used.

| dev (60) | recall@5 | recall@10 | complete@10 | MRR@10 | ceiling |
|---|---:|---:|---:|---:|---:|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| random | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| **BM25** | **0.290** | **0.456** | **0.317** | **0.287** | 1.000 |
| course-dense (the course's index) | 0.206 | 0.293 | 0.117 | 0.222 | 0.942 |
| dense-809 (same model, benchmark chunks) | 0.193 | 0.276 | 0.117 | 0.222 | 1.000 |

recall@10 by type (BM25 / course-dense / dense-809): simple_retrieval 0.714 / 0.286 /
0.286; multi_document 0.600 / 0.500 / 0.500; multi_hop 0.292 / 0.292 / 0.250;
temporal_versioned 0.400 / 0.283 / 0.283; multi_source 0.345 / 0.202 / 0.167;
tool_use_diagnostic 0.200 / 0.167 / 0.167.

Reading (provisional until `yfr compare` gives the paired numbers):
- **The hypothesis "dense wins on paraphrase" is not supported on dev.** BM25 beats both
  dense indexes overall (recall@10 +0.16) and on no type does dense come out ahead; the
  largest gap is simple_retrieval (10 vs 4 of 14 questions found).
- **Chunking is not the story:** course-dense and dense-809 are within 0.02 everywhere,
  and the course index's 0.942 ceiling never binds (both sit far below it).
- Open, to check before believing it: whether our query path matches the course's
  (`make ollama-test` re-embeds course chunks and compares with the dump's vectors; the
  passage path is already shown equal by dense-809 tracking course-dense), and whether
  the gaps survive pairing (`yfr compare`: wins/losses/ties, sign test, bootstrap CI).
  Hypotheses, not findings: questions carry exact codes, versions and dates that BM25
  matches and a 0.6B embedder blurs; many docs share structure (API docs, cases that
  restate rules), which hurts dense more.

What it means for the plan: the course's dense-only retrieval loses to a 60-line BM25 on
its own benchmark (dev; test is the claim). S2's first variant is hybrid (RRF over BM25 +
dense), then a reranker over the hybrid top 20-50.

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
