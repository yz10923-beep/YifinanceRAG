# CLAUDE.md

Guidance for Claude Code when working in this repository. It is also the handoff from
the session that chose the project (2026-10-14 to 2026-10-16, run from the Desktop
folder): who Yi is, the goal, the decisions already made, the facts checked about the
data, and what to do first. Nothing is built yet; the repo was empty when this was written.

## Who I am
I'm Yi Zhou, an MSIS student at NYU (graduating December 2026), with an undergraduate
degree from the University of Washington. I'm working toward a role as an LLM application
engineer (LLM应用工程师), ideally building AI agents for finance. **I'm applying mainly in
China**, so Chinese-language data and Chinese models (Qwen, DeepSeek) are a plus, not a
problem.

- Python and SQL are my working languages. I read Java and C++; I don't know Go or TS.
- Two finished projects this one builds on:
  - **kama** (github.com/yz10923-beep/project-1): a staged local coding-agent runtime in
    the style of Claude Code, S0-S6 measured with A/B evals.
  - **yiagent** (github.com/yz10923-beep/yiagent, branch `claude/jolly-euler-45xyyc`):
    an incident-triage agent over RCAEval telemetry in Elasticsearch, with a frozen eval
    harness. On the 283-case test split, run once: AC@1 0.98 vs BARO 0.29, fault-type
    F1 RE2 0.75 / RE3 0.40, $0.041 a case; a one-call (no tools) baseline 0.93, so the
    loop earns its cost. S5 (OpenRCA Bank) is in progress there.
- Where I'm weak, and what this project exists to fix: **retrieval engineering**
  (chunking, BM25 / hybrid search, reranking, pgvector) and **RAG evaluation** (retrieval
  metrics against gold evidence, a calibrated LLM judge).

## How I want you to help
- Treat me as a capable engineer who learns fastest by building. Concrete code,
  architecture sketches and trade-offs over general explanations.
- Python only. No new languages in this repo unless a specific role requires it (a small
  TypeScript trace/report viewer is allowed later, time-boxed to about a week).
- Teach fundamentals that won't churn (raw LLM API mechanics, retrieval, evals,
  observability) before frameworks. No LangChain / LlamaIndex / LangGraph.
- Push me toward finished, deployed, measured work. If I scatter, say so directly.
- Be honest when my approach is weak. Point out gaps a Chinese interviewer would notice.
- Connect advice to how it reads on a resume or in an interview.

## Time budget and priority
Yi graduates in December 2026 and is job-hunting now. This project is time-boxed to
**about 3 weeks**. Applications, LeetCode and interview prep run alongside it every week.
yiagent's S5 (Bank) should close first or in parallel; don't let this repo stall it.

---

## This project

**Goal.** A RAG + agent support assistant for **FlowPay**, a simulated Stripe-like
payments platform. It answers merchant questions from 100 product documents and from
the account's own records (SQLite tables + runtime logs). It is **measured, not demoed**:
300 frozen benchmark questions with gold evidence, a dev/test split, and the course's
reference implementations as baselines.

**Where the data comes from.** The FlowPay dataset belongs to 程序员Carl's 卡码网 course
*KamaAgenticRAG* (code: github.com/youngyangyang04/KamaAgenticRAG, MIT; 4 commits and a
full README as of 2026-10-16; its index has 712 chunks, the benchmark's has 809). The
dataset is distributed separately to course members:
`C:\Users\micha\Downloads\flowpay-dataset.zip` (15 MB zipped, 165 MB unpacked).

**Why this, not the course itself.** Many 卡码 students will have the course repo on their
resumes, and Chinese interviewers recognize it. This repo is Yi's own implementation, and
its value is **what the course leaves out**:
1. **Retrieval engineering, measured.** The course retrieves by dense vectors only
   (qwen3-embedding:0.6b, 1024-d, pgvector cosine), with fixed 1000-char chunks by
   Markdown heading, no overlap. No BM25, no hybrid search, no reranker, no retrieval
   metrics. Here: dense vs BM25 (jieba segmentation, since the text is Chinese) vs hybrid
   (RRF) vs hybrid + reranker, plus chunking variants, each scored by recall@k / MRR
   against gold chunk ids, **per question type**. Hypothesis to test: BM25 / hybrid wins
   on exact tokens (error codes like `FP-4041`, ids like `pay_bm_001`); dense wins on
   paraphrase.
2. **Evaluation rigor** in the yiagent style: grade against hidden ground truth, never
   the agent's claims; oracle / null / wrong baselines for every grader; tune on dev,
   report on test once; infra failures are not scores.
3. **The course's stages s1-s4 as baselines**, run by Yi on the same split with **the
   same chat model**, the way BARO is yiagent's bar: "I beat the reference
   implementation on 240 held-out questions, and here is which change did it."

## Facts checked about the dataset (2026-10-16)

Layout inside the zip (`flowpay-dataset/`):
- `data/knowledge/`: 100 Markdown docs in 5 types: api_docs 34, business_rules 16,
  release_notes 10, support_cases 16, troubleshooting 24. Docs carry metadata
  (doc_type, version, effective_from / effective_to, rule_ids, product_area).
- `data/index/knowledge_chunks.jsonl`: **the benchmark's own chunking**, 809 chunks,
  sha256 pinned in `data/index/index_manifest.json`. Each row has `evidence_id`
  (`doc:KB-BR-001#c001`), doc_id, section, title, version, effective dates, rule_ids, and
  the `tool` that should find it (`search_business_rules`, `search_api_docs`, ...).
- `data/generated/canonical/`: already generated (153 MB), although the dataset README
  says to generate it with the simulator: `flowpay.sqlite` (business tables), `runtime/`
  (`api_requests.jsonl` 32 MB, `usage_buckets.jsonl` 20 MB,
  `webhook_delivery_attempts.jsonl` 9 MB, `service_incidents.jsonl`), `manifest.json`.
  **`audit/event_journal.jsonl` must never reach the agent** (the dataset's own rule; it
  is the full event log, i.e. the answers).
- `benchmark/resolved/resolved_blueprints.jsonl`: the 300 questions. Fields include
  `question`, `question_type`, `split`, `conversation` (initial message, facts the
  simulated user reveals only when asked), `ground_truth` (`canonical_answer`,
  `required_assertions`, `negative_assertions`), `resolved_evidence_ids`,
  `resolved_trajectory_assertions`, `required_tools`, `forbidden_shortcuts`, `as_of`,
  `api_version_context`, `difficulty`.
- `benchmark/readable/`: the questions as readable Markdown, one file per type.
- `postgres/flowpay_knowledge.dump`: the course's built pgvector collection.
- `src/flowpay_simulator/` incl. `simulated_user.py`: deterministic, stdlib only.
- `docs/`: design specs in Chinese (business rules master table, corpus constraints,
  simulator design, the 300-question benchmark spec). Read the benchmark spec before
  writing graders.

The benchmark:
- **Split: dev 60 / test 240, grouped by scenario family** (75 families x 4 questions;
  0 families appear in both splits, so tuning on dev doesn't leak templates into test).
  Stratified by type: dev has 14 simple_retrieval, 14 multi_source, 12 multi_hop,
  10 multi_document, 5 temporal_versioned, 5 tool_use_diagnostic.
- All 300 by type: simple_retrieval 70, multi_source 70, multi_hop 60, multi_document 50,
  temporal_versioned 25, tool_use_diagnostic 25.
- Conversation mode: 180 `progressive_disclosure` (the agent must ask for missing facts;
  a simulated user answers), 120 `single_turn_complete`.
- Gold evidence kinds across all questions: doc 633, sql 365, runtime 315, negative 50
  (`negative:sql:...` = the correct finding is "no such record").
- `required_tools` names a tool set **different from the course backend's**: one search
  tool per doc type (`search_api_docs` 225, `search_business_rules` 221,
  `search_troubleshooting` 89, `search_support_cases` 25, `search_release_notes` 14) and
  `query_payments` 106, `query_accounts` 70, `query_webhooks` 58, `get_runtime_state` 137.
  The course backend uses one `search_documents` plus four SQL and three log tools.
  Which tool shape works better is an A/B question for this repo, not a given.
- `forbidden_shortcuts` example: `support_case_as_current_fact`,
  `current_state_used_for_historical_question`, `raw_sql_or_hidden_audit_access`.

Gotchas:
- **Gold chunk ids follow the benchmark's chunking.** The benchmark keeps a title-only
  first section as `c001`; the course's chunker skips it and also splits long sections.
  If our chunking differs, map retrieved passages back to gold ids, or retrieval metrics
  silently go wrong whenever chunking changes. The course maps by doc_id + heading
  (`scripts/benchmark/evidence.py`); we map **by character span** (`EvidenceMap`,
  decided 2026-10-16), since headings repeat inside docs and windows straddle sections.
- **The corpus is small (809 chunks).** Dense top-10 may already find most gold chunks;
  gains from hybrid may show only in some question types. Report per type, either way.
- Questions, docs and assertions are Chinese. `required_assertions` are free text, so
  answer grading needs an LLM judge; evidence-id and trajectory checks are deterministic.

## The course code worth knowing (read it, don't copy it wholesale)
Clone github.com/youngyangyang04/KamaAgenticRAG outside this repo for reference. Ideas
already reviewed as worth borrowing:
- `backend/policies/sources.toml`: data tools governed by policy, not by the prompt:
  per-table column whitelists, allowed filters, required scope (`account_id`,
  `environment`) enforced **in SQL**, row caps, timeouts. Tenant isolation by
  construction is an interview topic in finance.
- `backend/migrations/knowledge.sql`: immutable knowledge snapshots: a collection records
  its build config + dataset manifest hash; triggers forbid edits once ready and refuse
  to publish with missing chunks or vectors; a new chunking or embedding model = a new
  collection. Every eval number then points to an exact index build.
- `core/agent/turn.py` `disputed()`: before accepting an answer, any record id it cites
  that no tool returned is bounced back to the model (no extra LLM call). Use it here as
  an A/B, not as a grader; graders stay outside the agent.
- s2's shared tool budget + dedup across tools, with refusals returned to the model as
  observations, not errors.
- `scripts/benchmark/`: the course's own scorer. Run it on our outputs too, so numbers
  compare with the course's. Read it as a heuristic, not a judge (checked 2026-10-16):
  an assertion is "supported" if >=50% of its character bigrams occur anywhere in the
  answer (long answers score higher); a forbidden conclusion counts only as an exact
  substring (rewordings pass); it reads all 300 `design/` blueprints with no dev/test
  split ("课程不调参"); it does not check `required_tools` because its tool names differ.
  So: report course stages only on our test split, run by Yi; add a "verbose null"
  (top-10 chunks pasted as the answer, no LLM) to S3 and score it with both graders.
- `core/llm/embedding` `same_vector_space`: search refuses to run if the query embedder's
  identity (provider, model, revision digest, dimensions, options) differs from the
  collection's. Borrow it: it ties every number to an exact index build.
- Course model config: chat `deepseek-v4-flash` (OpenAI-compatible), embeddings
  `qwen3-embedding:0.6b` via Ollama.
If any code is copied, keep the MIT notice and record it in `VENDORED.md`.

## Decisions already made (don't re-litigate)
- FlowPay over FinanceBench, because Yi applies mainly in China (2026-10-16).
- Own implementation and own loop; the course's s1-s4 are baselines, not the base.
- Raw SDKs behind our own provider interface, no agent frameworks. Reuse yiagent's
  domain-free layers (LLM provider, retry, pricing, tracer + trace analyzer, context meter,
  output store, config loader, `tests/fakes.py`) **from GitHub branch
  `claude/jolly-euler-45xyyc`**, not from the stale local `Desktop/yiagent` checkout.
  Record the source commit in `VENDORED.md`, as yiagent did with kama.
- Add an OpenAI-compatible provider (DeepSeek / Qwen) next to the Anthropic one: needed
  to run the course baselines on the same model, and Chinese models read well in China.
- Postgres + pgvector for vectors (the course's image `pgvector/pgvector:0.8.2-pg17`);
  BM25 in-process, our own implementation, jieba plus an identifier pre-pass (2026-10-16).
- Embeddings computed once and cached; a 2 GiB swap file on the VM (2026-10-16; ROADMAP S0).
- Retrieval eval before the agent; graders before the agent (yiagent's S2-before-S3 order).

## Proposed stages (refine in `docs/ROADMAP.md`; record decisions there)
- **S0 scaffold:** uv, Python 3.12, `make verify` (ruff, mypy --strict, pytest), vendored
  layers, docker compose with pgvector, config loader with `FLOWPAY_DATASET_ROOT`.
- **S1 data + retrieval eval:** load docs/chunks/manifest with hash checks; dataset path
  outside the repo; leakage test (no audit journal, no ground truth in anything the agent
  sees); retrieval metrics (recall@5/10, MRR, per type) against gold ids, with the
  chunk-id mapping tested; baselines: course dense top-10, BM25, oracle.
- **S2 retrieval ablations:** hybrid RRF, reranker (e.g. a Qwen3 or bge reranker),
  chunking variants, metadata filters (doc_type, version, effective dates for
  `temporal_versioned`). One change per variant; dev only.
- **S3 answer graders:** deterministic evidence-id and trajectory checks; LLM judge for
  required / negative assertions, **calibrated against about 50 dev answers Yi labels by
  hand** (report agreement); forbidden-shortcut checks; course scorer as a second score.
- **S4 agent:** own loop, multi-source tools under a policy file, ask-user for
  progressive-disclosure questions via the dataset's simulated user; A/Bs (tool shape,
  `disputed()`, budgets). Course s1-s4 run as baselines on the same model.
- **S5 test once + deploy:** final numbers on the 240 test questions, run once; a small
  deployed demo; README with the results table and cost per question.

## Commands

```bash
make sync / make verify  # install; ruff + mypy --strict + pytest (synthetic data only)
make pg-up / pg-down     # pgvector on 127.0.0.1:5432
make data-check          # real dataset: hashes, counts, chunk rebuild, split, leakage
make data-test           # the same as pytest assertions (marker `dataset`, VM only)
uv run yfr retrieval --retriever bm25|random|oracle|wrong-oracle [--split dev]
```

Layout: `src/yifinance_rag/data/` (dataset files and hashes, corpus + `EvidenceMap`,
benchmark `Question` vs `Label`, pins, `check_dataset`), `src/yifinance_rag/retrieval/`
(tokenizer, BM25, retrievers, metrics), `src/yifinance_rag/cli.py` (`yfr`),
`tests/synthetic.py` (a tiny dataset in the real layout, for `make verify`).

## Invariants (keep these true)
- **No ground truth reaches the agent**: not `ground_truth`, `resolved_*`,
  `canonical_answer`, the audit journal, or the evidence ids. A leakage test checks
  prompts and tool outputs.
- **Grade against hidden labels, never the agent's claims.** Every grader has an oracle
  that passes, a null at chance and a wrong answer that fails, inside `make verify`.
- **Tune on dev, report on test.** The benchmark's split is used as given; its hash is
  pinned in a test; test runs only for a stage's final report.
- **A variant is one condition.** Comparisons with course baselines use the same chat
  model, the same embedding model unless that is the variant, and the same questions.
- **Infra failures are not scores** (API errors, timeouts, DB failures go to
  `errors.jsonl`).
- **Data tools are read-only and scoped.** Account and environment scope enforced in SQL.
- Unknown prices are None, never $0. Secrets never reach a trace, a log or the repo.

## Licence and publishing (important)
The dataset zip has **no licence file** and is course material.

**2026-10-16: Yi checked the course terms; publishing results from the FlowPay dataset is
allowed.** The exact wording and source were not recorded; Yi, add them here. Publishing
results is not redistributing the data, and the dataset is handed out to course members
only, so these rules stay:
- Never commit the dataset, the 300 questions, or the docs to this repo (gitignore the
  data dir; the dataset lives outside the repo at `FLOWPAY_DATASET_ROOT`).
- Publish code and results (aggregate numbers, per-type tables); the README says
  "requires the FlowPay dataset from the KamaAgenticRAG course".
- Don't paste benchmark questions into the README or blog posts unless the recorded terms
  allow quoting them.

## Working with Yi
- Yi runs anything that costs money (paid eval runs). Give exact commands and say what to
  send back. Never pass an approval flag on Yi's behalf.
- Machines: this Windows laptop (Git Bash + PowerShell; no uv, make or Python 3.12
  installed) and Yi's DigitalOcean VM (Ubuntu 24.04, 4 vCPU, 7.8 GiB RAM, no swap;
  yiagent's Elasticsearch lives there). **Yi tests and runs everything on the VM
  (2026-10-16)**; the laptop is for editing. Give VM commands, and ask Yi to send back the
  output. Memory next to yiagent's ES: docs/ROADMAP.md S0 (measured and decided).
- Git (decided 2026-10-16): **one long-lived working branch, `claude/yifinancerag`**,
  merged into `main` at stage ends, as in kama and yiagent. Commit there, and **push when
  each task Yi assigned is finished; don't ask first** (Yi, 2026-10-16). `make verify`
  passes before every push.
- Claude's cloud container has no dataset (course material stays on the VM). Code is
  written against the formats read from the course code and checked on the VM by
  `make data-check` / `make data-test`; Yi sends back their output.
- Security: a file in `Desktop/learn-claude-code/` is named with an Anthropic API key;
  Yi was told to revoke it. Never put keys in file names, traces or commits.

## Status (2026-10-16)
**S0 done** (closed on the VM 2026-10-16): `make verify` 61 passed, `make pg-up` healthy,
dataset unpacked at `~/flowpay-dataset`. RAM measured with yiagent's ES running: ES
3.8 GiB, pgvector 26 MiB, 4.1 GiB available, no swap. S0/S1 open questions decided
(docs/ROADMAP.md): embed once and cache, plus a 2 GiB swap file; BM25 in-process with jieba.

**Current: S1 part 1 (S1a)** built: dataset loaders with hash checks, label blinding
(opaque case ids), leakage tests, the benchmark's chunking rebuilt and cross-checked,
span-based mapping to gold ids, retrieval metrics with oracle / null / wrong checks, BM25
(jieba + an identifier pre-pass), `yfr data check` and `yfr retrieval`.
**Next:** Yi runs `make data-check` and `make data-test` on the VM; pin the split hash
from that output; then the BM25 / oracle / random dev table; then S1b (dense).
