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

Dev table (60 questions, all with doc evidence; runs under `evals/records/retrieval/`,
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

**S1 closed (2026-10-16).** `make ollama-test` passed: our client reproduces the course's
stored vectors, so the dense rows are the course's retrieval, not a broken setup. Paired
(`yfr compare`, all tables in `evals/records/retrieval/s1_dev_paired.md`):

| B vs A on dev (60) | recall@10 B-A [95% CI] | W/L/T | sign p | MRR@10 B-A [95% CI] |
|---|---|---|---:|---|
| course-dense vs BM25 | -0.163 [-0.276, -0.053] | 11/21/28 | 0.110 | -0.064 [-0.171, +0.044] |
| dense-809 vs BM25 | -0.179 [-0.292, -0.071] | 10/22/28 | 0.050 | -0.064 [-0.173, +0.045] |
| dense-809 vs course-dense | -0.017 [-0.042, +0.000] | 0/2/58 | 0.500 | -0.000 [-0.007, +0.007] |

What the dev data supports, and how strongly:
- **BM25 finds more gold chunks in the top 10 than either dense index**: the bootstrap
  interval excludes 0 for both; the sign test is borderline (p 0.05 / 0.11) because 28 of
  60 cases tie. Clearest on simple_retrieval: 0 wins / 6 losses / 8 ties, p 0.031.
- **Ranking (MRR) is not distinguishable** at n = 60: say "finds more", not "ranks better".
- **Chunking does not matter here**: dense-809 and course-dense differ on 2 of 60 cases.
- **The exact-code hypothesis was wrong.** Only 2 of 14 simple_retrieval questions share
  an identifier (`FP-xxxx`, versions, ids) with their gold chunk; BM25 reaches 0.632 on
  single-turn questions with no shared identifier. Its edge is ordinary Chinese word
  overlap between questions and the docs they were written from (a property of a
  benchmark generated from its corpus; say so when reporting).
- **36 of 60 dev questions are progressive-disclosure openers** (vague by design): BM25
  0.315 there vs 0.65 on single-turn. No retriever fixes a question that lacks the facts;
  that is S4's job (ask the user), and S2 gains should be read per mode as well as per type.

Interview line: "On the course's benchmark its dense-only retrieval found fewer gold
chunks than a BM25 I wrote (recall@10 0.29 vs 0.46 on dev, paired CI excludes zero);
chunking made no difference; and the reason was lexical overlap, not error codes as I
had guessed." Test numbers come at S5.

## S2 retrieval ablations (dev only)

One change per variant, each against the S1 rows, read with `yfr compare` (paired) per
type and per conversation mode. Order, cheapest and most likely first:

1. **Hybrid: RRF over BM25 + dense (0.6B)**: no new compute (cached vectors). RRF needs no
   score calibration; k = 60 as in the literature, not tuned.
2. **A stronger embedder, as one condition**: dense-only and hybrid rows with it.
   Proposed `qwen3-embedding:4b` (same family, prompt format and Ollama path as the
   course's 0.6B, so only model size changes; digest pinned at first pull). 8B does not
   fit next to Elasticsearch; bge-m3 changes family (a second condition, only if 4B is
   inconclusive); a hosted API costs money and sends course docs to a third party.
   It answers "is dense weak here, or is the 0.6B model weak?" (an interviewer will
   ask); it does not change the course baseline row, which stays the course's config.
   Cost: RAM about 3 GB while embedding (stop ES for that hour), roughly 1-1.5 h of CPU
   for the 809 chunks once, then cached. Decision: Yi.
3. **Reranker** over the hybrid top 20-50 (a Qwen3 or bge reranker).
4. **Metadata filters** (doc_type, version, effective dates) for temporal_versioned.
5. Chunking variants last: S1 showed chunking moves almost nothing.

**Done when** a dev table lists every variant per question type and mode, with paired
comparisons, and one retrieval config is chosen for S4 with the reason.

### Decided (Yi, 2026-10-16): hybrid first, plus a model option for the dense runs

Built (code, 2026-10-16; `make verify` 100 passed):
- `yfr retrieval --retriever hybrid`: `RRFRetriever` fuses BM25 and dense-809 over the
  same 809 chunks by rank only; k = 60 (paper value, untuned), each part's top 100
  (`--depth`); ties by best single rank, then chunk order. Run summaries record every
  part's config, including the embedder digest.
- `--embed-model qwen3-embedding:{0.6b,4b,8b}` for dense-809 and hybrid. Only the
  Qwen3-Embedding family is allowed, so a switch changes model size and nothing else
  (same query instruction, same Ollama path). The course's 0.6B stays pinned to the
  course's digest; for 4B/8B the installed digest is recorded in the identity, the run
  summary and the cache key. Run names carry the size (`dense-809-4b`, `hybrid-4b`);
  course-dense refuses other models (its vectors exist only for the course's).
- Comparisons to run: hybrid vs BM25 (does fusion add anything over the stronger part?),
  dense-809-4b vs dense-809 (model size), hybrid-4b vs hybrid.

**Hybrid result (VM, 2026-10-17, commit `994c638`;** run and paired tables in
`evals/records/retrieval/`, `s2_dev_paired.md`):

| dev (60) | recall@10 | complete@10 | MRR@10 |
|---|---:|---:|---:|
| BM25 | 0.456 | 0.317 | 0.287 |
| dense-809 | 0.276 | 0.117 | 0.222 |
| **hybrid (RRF, BM25 + dense-809)** | **0.549** | **0.367** | **0.302** |

- **Hybrid vs BM25: recall@10 +0.094**, 20 wins / 8 losses / 32 ties, sign p 0.036,
  bootstrap 95% CI [-0.004, +0.191] (touches 0). Borderline on 60 questions: "likely
  better", to be settled on test. MRR +0.015, not distinguishable.
- **Hybrid vs dense-809: +0.273**, 28/3/29, CI [+0.180, +0.371]: clearly better.
- Reading: dense alone is the weaker retriever, but it finds *different* gold chunks than
  BM25, so fusion recovers some of BM25's misses. Gains sit on the harder types
  (multi_hop +0.21, multi_source +0.17, tool_use_diagnostic +0.15, temporal +0.11);
  simple_retrieval, where BM25 is already strong, loses a little (-0.07, 1/2/11).
- Hybrid is now the S2 baseline to beat; each next variant is compared with it.

**The 4B rows (VM, 2026-10-17, commit `3fde484`;** tables in `s2_dev_paired.md`):

| dev (60) | recall@10 | recall@20 | complete@10 | MRR@10 |
|---|---:|---:|---:|---:|
| dense-809 (0.6B) | 0.276 | n/a | 0.117 | 0.222 |
| dense-809-4b | 0.380 | n/a | 0.200 | 0.273 |
| hybrid (BM25 + 0.6B) | 0.549 | 0.682 | 0.367 | 0.302 |
| **hybrid-4b (BM25 + 4B)** | **0.604** | **0.761** | **0.433** | **0.363** |

- Model size matters for dense alone: 4B vs 0.6B recall@10 +0.104 (13/4/43, CI
  [+0.033, +0.182]). Answer to "is the 0.6B model the problem?": partly; even 4B dense
  stays below BM25 (0.380 vs 0.456).
- Inside hybrid the gain shrinks: recall@10 +0.055 (CI [-0.024, +0.137]); recall@20
  +0.079 (CI [+0.022, +0.142]) and MRR +0.061 (CI [+0.006, +0.120]) do clear 0.
- hybrid-4b is the best first stage. Its gap between recall@20 (0.761) and recall@10
  (0.604) is headroom a reranker can use, so the reranker rule below is met.
- Cost to remember for S4: hybrid-4b needs the 4B model to embed every new agent query
  live (about 3 GB of RAM next to ES on the VM). Dev queries are cached; live agent
  queries are not. Decide at S4 between 4B live, 0.6B live, or BM25-only for the agent.

**How much would asking the user help retrieval? (2026-10-17, container, BM25)**
`--user-facts` appends every fact the simulated user would reveal to the 36 dev
progressive-disclosure openers (an upper bound, grader-side, like the oracle; runs and
tables in `evals/records/retrieval/`, `s2_user_facts_upper_bound.md`):

| BM25, dev | recall@10 | MRR@10 | complete@10 |
|---|---:|---:|---:|
| openers only (36) | 0.315 | 0.253 | 0.139 |
| openers + all user facts (36) | **0.462** | **0.412** | 0.083 |
| single-turn questions (24, unchanged) | 0.667 | 0.337 | 0.583 |

- recall@10 +0.147 on the openers, 15/3/18, sign p 0.008, CI [+0.052, +0.243]: perfect
  clarification is worth about as much as hybrid search (+0.094 overall, +0.147 here).
- complete@10 drops (0.139 to 0.083): the facts pull in the account- or record-specific
  chunks and push out a second gold chunk; worth a look when the agent writes its queries.
- What it means: the biggest remaining retrieval lever for 60% of the questions is the
  agent asking for the missing facts (S4), not another retriever variant. That argues for
  closing S2 soon. Upper bound only: a real agent asks for some facts, not all.

**Reranker (Yi chose Claude Haiku 5.5, 2026-10-17).** Built: `--rerank haiku` reorders
the first stage's top 20 in one listwise call per question (passages numbered, trimmed
to 1,000 characters; the reply is a JSON array of all 20 numbers). Effort `low`, no
prompt caching (each prompt is new; a cache write would cost 1.25x), no tools field.
Unusable replies keep the first-stage order and are counted (`fallbacks`); replies are
cached on disk by (model, effort, prompt, trial), so reruns are free, and `--trial 1`
asks again to measure run-to-run noise (Haiku 5.5 takes no fixed temperature). Tokens
and dollars go into the run summary. Condition: hybrid-4b + rerank vs hybrid-4b; recall@20
cannot change by design, MRR and recall@5/10 can.

**Reranker result (VM, 2026-10-17, commit `3125900`;** runs and paired tables in
`evals/records/retrieval/`, `s2_rerank_dev_paired.md`):

| dev (60) | recall@5 | recall@10 | recall@20 | complete@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| hybrid-4b | n/a | 0.604 | 0.761 | 0.433 | 0.363 |
| **hybrid-4b + Haiku rerank, trial 0** | 0.501 | **0.714** | 0.761 | **0.533** | 0.449 |
| hybrid-4b + Haiku rerank, trial 1 | 0.496 | 0.697 | 0.761 | 0.500 | 0.410 |

- **recall@10 +0.110**, 15 wins / 2 losses / 43 ties, sign p 0.002, CI [+0.054, +0.171]:
  the clearest gain in S2. The reranker pulls gold chunks from ranks 11-20 into the top 10.
  Largest on multi_source (+0.196, 7/1/6) and on progressive openers (+0.127, CI
  [+0.056, +0.204]).
- **MRR@10 +0.085, not distinguishable** (30/22/8, CI [-0.032, +0.205]); trial 1 alone
  gives +0.047. So it finds more gold but does not reliably put it first.
- recall@20 unchanged in every row (0/0/60): the sanity check that it only reorders.
- **Run-to-run noise (trial 1 vs trial 0, same prompts):** recall@10 -0.017 (0/2/58),
  MRR -0.038 (7/13/40, CI [-0.084, +0.003]). The recall@10 gain is about 6x the noise;
  the MRR gain is about 2x and its CI crosses 0. Report recall@10 for the reranker and
  treat its MRR as noise-level.
- **Cost:** one dev run (60 calls) $0.0286, 187K in / 20K out tokens: about $0.0005 a
  search, about 3.1K input tokens a call. 1 unusable reply in 120 (kept first-stage order).
- Worth a look later, not investigated: simple_retrieval has recall@10 0.786 but MRR
  0.27, so its gold usually sits around rank 4 (perhaps several chunks state the same
  rule and the benchmark marks one).
- The first attempt printed "Event loop is closed" tracebacks: client cleanup after each
  call, not failed calls (reproduced against a local server; the 46 replies it cached are
  valid). Fixed in `3125900`: one loop and one client per run, closed at the end.

### S2 closed (2026-10-17): the retrieval config for S4

**Chosen: hybrid-4b (RRF k = 60 over BM25 + dense-809 with qwen3-embedding:4b, each
part's top 100) + Claude Haiku 5.5 listwise rerank of the top 20 (effort low, 1,000
characters a passage), keeping the top 10.** Dev recall@10 0.714 vs the course's
dense-only 0.293 (same 60 questions, same gold); every step is a paired comparison in
`evals/records/retrieval/`.

The dev ladder, recall@10 / MRR@10: course-dense 0.293 / 0.222, dense-809 0.276 / 0.222,
BM25 0.456 / 0.287, hybrid 0.549 / 0.302, hybrid-4b 0.604 / 0.363, **+ rerank 0.714 /
0.449**. All on queries = the question openers; the agent will write its own queries.

Why, and what it costs: the reranker's gain is the largest and clearest in S2, for about
$0.0005 a search. Costs to keep in view in S4:
- The 4B embedder must embed every live agent query (about 3 GB of RAM next to ES on the
  VM). If that hurts, the fallback is a new condition to measure, not an assumption
  (hybrid 0.6B + rerank was not run).
- Rerank latency was not measured (one Haiku call a search); record it in S4's traces.
- Not done, by decision: metadata filters and chunking variants (temporal_versioned has
  5 dev questions, so no effect is measurable; S1 showed chunking moves nothing); a
  second reranker model.

The bigger lever left is not retrieval: on the 36 progressive openers, giving the
question all the simulated user's facts added +0.147 recall@10 (BM25). S4's ask-user
loop is where that is won. It is measured only after S3: retrieval metrics can score the
agent's queries, but whether asking produces better answers needs S3's graders.

Interview line: "Dense-only retrieval, as in the reference course, found 29% of gold
chunks in the top 10. BM25 with jieba found 46%, fusion 55%, a larger embedder 60%, and
a $0.0005 Haiku reranker 71%, each step a paired test on the 60 dev
questions with a noise run for the LLM step."

Recommended rest of S2 (as planned before the reranker run; all done or decided above) (time box: about 2 more days, then S3):
1. Finish the 4B rows (running on the VM).
2. One reranker condition over the hybrid top 20, for MRR (the metric hybrid did not
   move), **only if there is headroom**: a reranker can only reorder what the first stage
   found, so its ceiling is hybrid's recall@20 (runs now report recall@20; complete@ and
   MRR stay at 10). BM25 alone: recall@10 0.456, recall@20 0.553 (container, 2026-10-17).
   Decision rule: if hybrid recall@20 is barely above its recall@10 (0.549), skip the
   reranker and start S3. Proposed: an LLM reranker with Claude Haiku 5.5 through the vendored provider
   (about $0.04 per dev run at $0.10/MTok input; Yi runs it); an open reranker needs
   torch on the VM's CPU.
3. Skip metadata filters and chunking variants for now: temporal_versioned has 5 dev
   questions (no measurable effect possible), and S1 showed chunking moves nothing.
4. Choose the S4 retrieval config (hybrid, plus the reranker if it earns its cost).

**4B run, first attempt (2026-10-17): no scores (infra failure, as designed).** Three
runs hit Ollama at once (the background run was started twice and hybrid-4b began while
both were embedding); a CPU Ollama serves one request at a time, so a batch waited past
the 120 s client timeout. Fixed in code: the embedding cache now commits every 32
texts (an interruption loses at most one batch; a rerun resumes), computing holds an
exclusive file lock (a second run waits, then reuses the first run's vectors), progress
goes to stderr, and the Ollama request timeout is 600 s. Tested (`make verify` 102).

## S3 answer graders

- Deterministic: evidence ids cited vs gold, trajectory assertions, forbidden shortcuts.
- LLM judge for `required_assertions` / `negative_assertions` (Chinese free text),
  calibrated against about 50 dev answers Yi labels by hand; report agreement.
- The course's `scripts/benchmark/` scorer run on our outputs as a second score.
- Oracle / null / wrong checks for every grader.

**Done when** every grader passes its oracle/null/wrong checks and the judge's agreement
with Yi's labels is reported.

**Order for S3 (steps 1-5 done, step 6 built 2026-10-17; see below):**
1. ~~Read before writing graders~~: done, see "S3 step 1" below.
2. An answer record format (answer text, cited evidence ids, tool calls with their
   observed evidence ids, clarification turns) that the graders read and the S4 agent
   will write; grader-only labels stay in `Label`.
3. Deterministic graders first: evidence per selector (doc / sql / runtime / negative,
   finding 5-6), source and operation coverage (finding 7), instance facts (finding 9),
   the `as_of` shortcut (finding 3), clarification accounting (finding 10).
4. Baselines with no LLM, inside `make verify` on synthetic data: oracle
   (`resolved_answer` where it exists, else `canonical_answer`, plus gold ids), null
   (empty / "不知道"), wrong (an answer from **another family**: finding 2), the rule-only
   answer (`canonical_answer` on the 180 dynamic questions: must fail instance facts),
   and the "verbose null" (the S2 config's top-10 chunks pasted as the answer), scored by
   our graders and the course scorer.
5. The LLM judge for required / negative assertions (one call per answer, per-assertion
   verdicts as JSON, replies cached like the reranker's). Judge model: decided below.
6. The OpenAI-compatible provider for `deepseek-v4-flash` (moved here from S4 by the
   chat-model decision), then calibration on real answers: a one-call RAG baseline (S2
   config top 10 + one chat call, no tools, no asking) answers the dev questions (paid,
   Yi runs it); Yi labels about 50 of its answers by hand; report judge-vs-Yi agreement
   (per assertion, with Cohen's kappa). The one-call baseline is also S4's bar, as in
   yiagent (0.93 there).

### Decided (Yi, 2026-10-17): the models

- **Chat model: `deepseek-v4-flash`** for every answer (one-call baseline, our agent, the
  course's s1-s4), the course's own model, so the course stages run unchanged and the
  comparison is one condition. Consequence: the OpenAI-compatible provider is built in S3
  (step 6), not S4. Its price is None until read from DeepSeek's live pricing page.
- **Judge: Claude Haiku 5.5 or Sonnet 5.5**, a different family from the answering model.
  Proposed rule: run both on Yi's ~50 labelled answers and keep the cheaper one unless the
  other's agreement with Yi is clearly higher (kappa difference larger than the
  trial-to-trial noise of re-asking the same judge). Estimate at an assumed ~3k input /
  ~400 output tokens an answer, from `llm/pricing.py`'s rates: Haiku ~$0.0005 an answer,
  Sonnet ~$0.01 an answer; calibrating both on 50 answers is under $1.

### S3 step 1 (2026-10-17): what the labels and the course scorer say

Read: the benchmark spec (`docs/02_benchmark/FlowPay_300题Benchmark设计规范.md`), all
fields of `resolved_blueprints.jsonl` and `evidence_resolution.jsonl`, the simulated user,
the SQLite schema and runtime logs, and the course scorer (`scripts/benchmark/`, repo
unchanged at 4 commits). Every number below was computed on the uploaded zip.

The spec's own scoring dimensions (section 11): answer correctness (required assertions
hit, negative not triggered), evidence recall, citation precision, tool selection, hop
completion, temporal accuracy, stop accuracy, unnecessary calls, cost / latency. Section
5.1: numbers, enums, entity relations and times are graded deterministically; an LLM judge
only for free-text explanation. Section 5.4: on the 180 dynamic questions a rule-only
answer fails; the answer must state what the account's records show.

Findings:
1. **Label shape.** `required_assertions`: 1 or 2 per question (230 have 2).
   `negative_assertions`: exactly 1 per question, always `不得认定：X`.
   `acceptable_variants`: empty everywhere. On the 180 `progressive_disclosure`
   questions only: `resolved_trajectory_assertions` (2-5 each; each has the selector alias
   and evidence ids), `resolved_answer` (the oracle with instance facts and an
   `Evidence IDs：[...]` line), `resolved_timeline`, and `instance_diagnosis_contract`
   with `static_rule_only_answer_is_accepted: false`. The 120 single-turn questions have
   none of these. 535 `evidence_selectors` (361 `exactly_one`, 174 `at_least_one`).
   `max_clarification_turns`: 2 (87), 3 (63), 4 (30).
2. **The 4 questions of a family share their required assertions** (e.g. SF-001's
   simple_retrieval, multi_document, multi_hop and multi_source questions), and its
   dynamic questions share instance facts. A "wrong" baseline must come from another
   family; a same-family answer scores as right on every grader.
3. **`forbidden_shortcuts` is the same 3 values on all 300 questions**: global rules, not
   per-question labels. `raw_sql_or_hidden_audit_access` is prevented by construction (no
   such tool) and checked by a leakage test, not graded. `current_state_used_for_historical_question`
   is checkable from the trajectory: a data tool call with no `as_of`, or one not on the
   label's `as_of` date. `support_case_as_current_fact` needs the judge (low priority).
4. **Evidence id formats are in the data already.** Runtime rows carry `evidence_id`
   (`runtime:req:…`, `runtime:attempt:…`, `runtime:usage:<acct>:<env>:<minute>`,
   `runtime:incident:…`); SQL ids are `sql:<entity>:<primary key>`, composite for
   `feature_flag` (`<acct>:<env>:<feature>`) and `invoice_attempt` (`<invoice>:<n>`).
   So S4's data tools return the dataset's own id on every row, and observed ids compare
   with gold without a mapping (unlike doc chunks, which need `EvidenceMap`).
5. **Negative evidence ids (50) can't be produced by an agent.** They are hashes:
   sha256 of the selector payload (sorted, compact JSON, without `operation`), first 20
   hex chars, reproduces 37 of 50; the other 13 (`list_api_keys`, `get_plan_timeline`,
   `get_service_status`, one `get_request`) hash fields the resolution file doesn't show,
   and the resolver isn't in the course-member zip. Proposed: grade a negative by
   selector: found when the trajectory has a call to that tool and operation with the
   selector's identifying parameters that returned zero rows.
6. **List selectors expand to many ids** (one `list_requests` resolves to 20 request ids).
   Per-id recall would weight it 20x. Proposed: score record evidence per selector
   (`exactly_one`: its id observed; `at_least_one`: any of its ids observed), and report
   per-id recall next to it.
7. **Tool names: measure sources, not names.** Every chunk carries the `tool` that should
   find it (api_docs 237, troubleshooting 213, support_cases 160, business_rules 126,
   release_notes 73 chunks). A required `search_*` tool is satisfied when a passage tagged
   with it was observed, which works for per-doc-type tools and for one
   `search_documents` alike (S4's tool-shape A/B needs both scored the same way). Data
   tools: `required_operations` names 20 operations over the 4 data tools
   (`query_payments.get_payment` 71 … `query_payments.list_customers` 4), satisfied by a
   call to that operation.
8. **The course scorer, measured:**
   - It reads evidence from what tools returned, not from what the answer cites.
   - Its record-id regex (`acct|pay|evt|del|we|req|inc|att|cus|sub|inv|ref`) misses
     refunds (`re_…`), invoices (`in_…`), API keys (`key_…`) and plan changes (`pc_…`):
     **74 of 680 gold record ids (11%), in 45 questions, can never be found**; 20 more
     (feature flags, usage buckets) count as found whenever the account id appears.
   - It drops negative evidence entirely.
   - It never grades `resolved_trajectory_assertions`: **the rule-only `canonical_answer`
     passes all 180 dynamic questions**, which the spec says must fail.
   - **Pasting the question's gold doc chunks, no LLM, passes 87/300** (dev 12/60;
     simple_retrieval 28/70, multi_document 7/50, multi_hop 16/60, multi_source 20/70,
     temporal 9/25, diagnostic 7/25): the bigram rule rewards long text.
   - What does hold: oracle `canonical_answer` 300/300, another family's answer 0/300,
     "不知道" 0/300, the canonical answer plus the forbidden claim verbatim 0/300.
   - It reads `design/` blueprints; their ground truth equals `resolved/` on 300/300, so
     that is harmless.
   So it stays a second score (to compare with the course's numbers), never our grade.
9. **Instance facts can be graded deterministically.** Prototype: take the ids, codes,
   numbers and enum words in each trajectory assertion and check them in the answer.
   Mean key coverage: oracle `resolved_answer` 1.00 (180/180 complete), rule-only
   `canonical_answer` 0.11 (0/180), another family's `resolved_answer` 0.38 (0/180; shared
   tokens like `2026`, `production`, `NA`). To build: drop generic tokens, normalize dates
   (`08 月 02 日 11:50` vs `8月2日`), count a key only where the trajectory observed it.
10. **The simulated user matches keywords** (`CATEGORY_KEYWORDS`, 6 categories): one
    question naming every category gets every fact in one turn, so the clarification
    turn budget alone doesn't measure asking well. Record turns used, the categories asked
    and the ones beyond `required_clarification_categories`; the course driver also logs
    `answered_without_clarification`, `max_clarifications_exceeded`, `repeated_category`.
11. **Leakage hazards for S4's data tools:** `flowpay.sqlite` itself holds an
    `event_journal` table (134,055 rows: the audit journal), `scenario_executions` (with
    the split) and `simulation_runs`; runtime rows carry `scenario_id` (4,204 of 54,604
    API requests, e.g. `SC-054-downgrade_restricts_newer_endpoints`, which names the
    scenario) and `run_id`. Data tools: a table and column whitelist, `scenario_id` and
    `run_id` stripped; the leakage test scans tool outputs for `SC-`, `SF-`, question ids
    and the event journal. Today's leakage test only checks chunks for `event_journal`.

Interview line: "The reference scorer passes a rule-only answer on every one of the 180
questions that require looking up the account, and passes pasted source text on 29% of
all questions with no model at all. So I graded instance facts deterministically and
used the LLM judge only for the free-text assertions, calibrated against my own labels."

### S3 steps 2-4 (2026-10-17): records, deterministic graders, no-LLM baselines

Built (`src/yifinance_rag/grade/`, `yfr grade`), all without money:
- **Answer record** (`record.py`): `answer` and `cited` are the system's output;
  `tool_calls` (tool, operation, args, the evidence ids each call returned, result count,
  error) and `clarifications` (question, categories disclosed) are written by the harness
  that ran the tools and the simulated user, never parsed from model text. One format for
  the baselines, the one-call RAG baseline and the S4 agent.
- **Truth** (`truth.py`): the grader-only view, with the selectors from
  `evidence_resolution.jsonl`; loading fails if a gold id is neither a doc chunk nor a
  selector's id (0 such on the real data).
- **Graders** (`graders.py`), each a score in [0, 1] or "doesn't apply":
  `doc_evidence`, `record_evidence` (per selector; `at_least_one` = any of its ids),
  `negative_found` (same tool and operation, same identifying parameters, success, zero
  rows), `id_recall` (per id, for reference), `tools` (search sources by corpus folder,
  so one `search_documents` and per-type tools score alike), `operations`,
  `instance_facts` (atoms of the trajectory assertions stated in the answer: `facts.py`),
  `as_of` (share of data calls scoped to the case's UTC date; unscoped = reading current
  state), `clarification` (dynamic: required categories disclosed; single-turn: 1 if it
  did not ask), `cited_grounded`, `cited_gold`.
- **Course scorer** (`course.py`): its rules on our records, flaws kept (VENDORED.md).
- **Baselines** (`baselines.py`): oracle, null, wrong (next case from another family,
  same mode, same split), rule-only, verbose-null (top 10 chunks of any retriever pasted).
- **Selftest** (`yfr grade --selftest`; in `make verify` on synthetic data and in
  `make data-test` on all 300 real questions): oracle 1.0 on every grader on every case;
  null 0 on every evidence grader, instance facts and the course task; the wrong answer's
  mean under 0.5 on record evidence, negatives, instance facts and the course task;
  rule-only never complete on instance facts and under 0.5 on average. A test checks
  the selftest catches a grader that can't tell wrong from right.

Decided while building (2026-10-17):
- `tools`, `operations`, `clarification` and `as_of` score **behaviour**, not
  correctness: a wrong answer that took a plausible route scores high on them (dev:
  tools 0.75, clarification 0.98 on dynamic questions), so the selftest does not ask a
  wrong answer to fail them. Correctness is the evidence graders, instance facts and
  (step 5) the judge.
- `doc_evidence` is not in the wrong-answer check either: families reuse documents, so
  another family's chunks are sometimes this case's gold (0.015 dev, 0.08 on all 300).
- Instance-fact atoms were designed on the label text of all 300 questions (no system
  output involved): ids, error codes, dates and times normalized, amounts of 2+ digits,
  enum words minus template words (`utc post get http id api usd scenario`), the
  resolver's `as_of=` and operation names dropped. On all 180 dynamic questions: oracle
  1.000, rule-only 0.110, another family's oracle 0.152 (max 0.667), null 0. Two of the
  535 trajectory assertions have no atoms and are skipped.

Dev table (`evals/records/grade/s3_dev_baselines.md`; all 60 dev questions):

| system | doc_evidence | record_evidence | instance_facts | course_task |
|---|---:|---:|---:|---:|
| oracle | 1.000 | 1.000 | 1.000 | 1.000 |
| null | 0 | 0 | 0 | 0 |
| wrong (other family) | 0.015 | 0 | 0.224 | 0 |
| rule-only | 1.000 | 0 | 0.119 | **1.000** |
| verbose-null (BM25 top 10, no LLM) | 0.456 | 0 | 0.173 | **0.433** |

verbose-null's `doc_evidence` equals S1's BM25 recall@10 on dev (0.456), a cross-check
that records and the retrieval eval agree. On all 300 questions verbose-null passes the
course's task check on 53% (single-turn 82%) with no model at all.

**Confirmed on the VM (2026-10-17):** `make verify` and `make data-test` pass,
`yfr grade --selftest` ok with tables identical to the container's. verbose-null with the
S2 config (hybrid-4b + Haiku rerank, replies cached), dev, all 60: doc_evidence 0.714
(= S2's recall@10 0.714), instance_facts 0.255, **course_task 0.567** (single-turn 0.750,
progressive 0.444; simple_retrieval 0.857). So with the S4 retrieval config and no chat
model, the course scorer already passes 57% of dev; any LLM system must be read against
that floor, and against the 0.255 on instance facts.

Interview line, extended: "Pasting the top 10 BM25 chunks with no model passes the
reference scorer's task check on 53% of questions; my instance-fact grader gives it 0.17."

Next: step 5 (the judge) and step 6 (the DeepSeek provider, the one-call baseline, Yi's
labels). The judge is the only grader of the free-text required / negative assertions.

### S3 step 5 (2026-10-17): the LLM judge, built; its paid selftest is Yi's

Built (`grade/judge.py`, `yfr grade ... --judge haiku|sonnet`), tested with fake judges
only (no API calls in `make verify`):
- One call per answer: the question (opener), the required assertions `A1..`, the
  forbidden claims `N1..` (the `不得认定：` prefix stripped) and the answer. The reply is
  JSON: `supported` / `not_supported` per assertion, `asserted` / `not_asserted` per claim,
  and a verbatim `quote` for every positive verdict. Effort low, no prompt cache, replies
  cached in `~/.yfr/cache/judge.sqlite` keyed by model, effort, prompt and trial;
  `--judge-trial N` asks again (variance).
- Checks that don't trust the judge: a positive verdict whose quote is not in the
  answer (whitespace aside) counts as negative and is counted as `unverified`; a reply
  that can't be parsed or leaves an item out scores None and is counted as incomplete,
  never guessed; API errors after retries stop the run.
- Scores: `judge_assertions` (share supported), `judge_forbidden_clear` (no claim
  asserted), `judge_task` (all supported and none asserted; the judged counterpart of
  `course_task`).
- A new baseline, **forbidden**: the oracle record plus "因此可以认定：<claim>。". Every
  deterministic grader still passes it; the judge must flag it.
- `cohen_kappa` for step 6's agreement with Yi's labels.
- The reranker and the judge now share `llm/cached.py` (reply cache, trials, retries,
  cost); the reranker's cache key is unchanged, so existing caches still hit.

**Judge thresholds, fixed before its first run** (changing them later is a decision to
record, not tuning): on the dev baselines, oracle `judge_task` >= 0.90; null
`judge_assertions` <= 0.05; wrong `judge_task` <= 0.20; forbidden
`judge_forbidden_clear` <= 0.10; incomplete verdicts <= 5% of judged answers.
verbose-null and rule-only are reported, not thresholded: rule-only is a correct answer on
the free-text assertions (instance facts fail it), and verbose-null is the probe: how
much less than the course scorer's 0.567 a judge gives pasted chunks.

Run (Yi, VM, paid): `yfr grade --selftest --judge haiku` and `--judge sonnet`, with the
S2 retrieval flags for verbose-null. Estimate from prompt sizes (360 prompts, ~720k
characters on dev): Haiku ~$0.10, Sonnet ~$2; the run prints the measured cost.

**Judged selftest on dev (VM, 2026-10-17): both judges pass every threshold**
(`evals/records/grade/s3_judge_selftest_dev.md`). Haiku 5.5: oracle judge_task 1.000, null
0, wrong 0, forbidden caught on 60/60, 0 incomplete, 0 unverified quotes, $0.069 for 336
calls. Sonnet 5.5: oracle 0.950 (misses only on progressive questions), forbidden_clear
0.050, 9 incomplete replies, 18 unverified quotes, $1.14 (16x Haiku).
- The judge rates pasted chunks (verbose-null, S2 config) far lower than the course
  scorer: judge_task 0.367 (Haiku) / 0.308 (Sonnet) vs course_task 0.567; on progressive
  questions 0.167 / 0.206 vs 0.444.
- Not yet a verdict on which judge is better: these baselines are easy (the oracle is the
  dataset's own answer), and Sonnet's misses may be our checks rather than its judgment
  (quotes that don't match the answer verbatim, replies the parser rejects). Diagnose
  from the cached replies (case rows now keep each raw reply), then decide by agreement
  with Yi's labels on real answers (step 6), as decided.
- **Diagnosed (2026-10-17, from Sonnet's cached replies): both are our checks, not its
  judgment.** (1) The 18 unverified quotes: Sonnet writes full-width punctuation as ASCII
  (`；` as `;`, `，` as `,`), otherwise verbatim. (2) The 9 incomplete replies: Sonnet
  writes a verdict, reconsiders in prose ("Wait…", "Correction…"), then writes a corrected
  verdict; the parser took the span from the first `{` to the last `}`, which is not JSON.
  **Fixed:** quotes compare after NFKC normalization; the verdict is the last JSON object
  in the reply that has a verdict section. Neither changes a prompt or a threshold, so
  both judges are re-scored from their cached replies at no cost.

### S3 step 6 (2026-10-17): DeepSeek provider, one-call baseline, calibration tools

Built (container; the paid runs and the labels are Yi's):
- **`llm/openai_compat.py`**: an OpenAI-compatible Chat Completions provider over
  httpx2 (no vendor SDK). Translates our Anthropic-format history both ways, tool calls
  included (S4 needs them); usage maps cache hits apart (`input_tokens` = uncached, as in
  the Anthropic API); HTTP errors classified like the Anthropic provider's (429 / 5xx
  retryable, 402 billing, 400 "context length" = context_overflow). Settings:
  `DEEPSEEK_API_KEY` (in `~/.yfr/.env`), `YFR_DEEPSEEK_URL`.
- **Chat config = the course's** (`backend/configs/local.toml`): `deepseek-v4-flash` at
  `https://api.deepseek.com`, `thinking: disabled`, `max_tokens` 4096. Price: **unknown
  (None)**: DeepSeek's docs were unreachable from the container; tokens are recorded,
  so cost is computable once Yi reads the live price into `llm/pricing.py`.
- **`answer/onecall.py`** (`yfr answer --system onecall`): the S2 retrieval top 10 for
  the opener, one chat call, no tools, no asking. The model sees the opener and passages
  numbered [1]..[10] with doc title and heading, never chunk ids; it cites by number and
  the harness maps citations to evidence ids. Replies cached (`answers.sqlite`); an API
  failure writes `errors.jsonl` and stops the run (no partial scores; a rerun resumes from
  the cache). Answering and grading are separate commands. Dev prompts: 60, ~137k
  characters in all. A dataset test checks every question's prompt carries no evidence
  id, question id or scenario name.
- **Calibration** (`grade/calibrate.py`): `yfr label-sheet RUN` writes `labels.md` (the
  texts; never committed) and `labels.tsv` (case id, item, empty label: y/n; committable),
  dev only, blind to the judge. `yfr agree labels.tsv RUN...` gives agreement and Cohen's
  kappa per item kind (required / forbidden), over all items and at the task level, plus
  the disagreements, for each judged run.
- The reranker, judge and one-call share `llm/cached.py`, which now also returns each
  call's usage (records carry tokens per answer).

Decision rule for the judge (from "Decided: the models", made concrete before the
labels exist): compare kappa over all items; keep Haiku unless Sonnet's kappa is higher
by more than the gap between two Haiku trials (`--judge-trial 1` on the same records).
Report kappa per kind and at the task level either way.

## S4 agent

Own loop over the vendored provider, answering with `deepseek-v4-flash` through the
OpenAI-compatible provider built in S3; data tools under a policy file (column whitelists, scope enforced in
SQL, row caps); ask-user for progressive-disclosure questions via the dataset's simulated
user. A/Bs on dev: tool shape (per-doc-type search tools vs one `search_documents`),
`disputed()` bounce, budgets. Course s1-s4 run as baselines on the same chat model.

**Done when** the dev table compares our agent with course s1-s4 on the same model.

## S5 test once + deploy

Final numbers on the 240 test questions, run once; a small deployed demo; README with the
results table and cost per question.
