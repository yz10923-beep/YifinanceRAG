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
  the `tool` that should find it (`search_business_rules`, `search_api_docs`, ...). The
  chunk text is in `content`; also `source_path`, `api_versions`, `product_area`.
  Front-matter `doc_type` has 6 values: the release_notes folder holds 4 `release_note`
  and 6 `migration_guide` docs.
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
- `data/releases/flowpay_dataset_v1/release_manifest.json` (sha256 = the course's pin):
  11 artifact groups, each with a `visibility` (`agent_knowledge`,
  `agent_via_typed_tools`, `hidden_evaluation`, `forbidden_to_agent`, ...) and an
  `access_policy`; the dataset's own leakage rules. Its `quality_gates` claim every
  question's doc evidence is retrievable in some top 20 (`fully_retrievable_top20`
  300/300). The course-member zip ships 4 groups we don't read trimmed or newer
  (`evidence_audit`, `offline_fixture_validation`, `generation_sources`, half of
  `dataset_documentation`); the 7 groups we read are intact.
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
- `forbidden_shortcuts` is the same three values on all 300 questions (global rules, not
  labels): `support_case_as_current_fact`, `current_state_used_for_historical_question`,
  `raw_sql_or_hidden_audit_access`. More label facts (checked 2026-10-17): ROADMAP
  "S3 step 1".
- **`flowpay.sqlite` also holds the audit journal** as table `event_journal`, plus
  `scenario_executions` (with the split); runtime rows carry `scenario_id`, which names
  the scenario. Data tools whitelist tables and columns and strip both (ROADMAP S3 step 1).

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
  Measured 2026-10-17 (ROADMAP S3 step 1): it never grades instance facts (a rule-only
  answer passes all 180 dynamic questions), pasting the gold chunks passes 87/300, and
  its id regex can't find 11% of gold record ids (refunds `re_`, invoices `in_`, ...).
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
- **Chat model `deepseek-v4-flash`** for all answers (one-call baseline, our agent, the
  course stages); **judge Claude Haiku 5.5 or Sonnet 5.5**, chosen by agreement with Yi's
  labels (Yi, 2026-10-17; rule in ROADMAP S3).

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
make course-index        # the course's pgvector dump -> database course_kb (after pg-up)
make pg-test             # course index vs our code (offsets, passage format, ranking)
make ollama-test         # our embedder reproduces the course's stored vectors (VM)
uv run yfr compare RUN_A RUN_B   # paired: wins/losses/ties, sign test, bootstrap CI
uv run yfr retrieval --retriever bm25|random|oracle|wrong-oracle [--split dev]
uv run yfr retrieval --retriever course-dense|dense-809|hybrid   # need Ollama (VM)
    [--embed-model qwen3-embedding:4b]   # dense-809 / hybrid only; Qwen3 family only
uv run yfr retrieval --retriever hybrid --embed-model qwen3-embedding:4b --rerank haiku \
    [--trial 1]   # the S4 config; paid (~$0.03 a dev run), replies cached; --trial asks again
uv run yfr grade --selftest      # every grader vs the no-LLM baselines (S3)
uv run yfr grade --system oracle|null|wrong|rule-only|verbose-null [retrieval flags]
uv run yfr grade --records FILE --name NAME   # grade a system's answer records
uv run yfr grade ... --judge haiku|sonnet [--judge-trial N]   # + LLM judge (paid, cached)
uv run yfr answer --system onecall [retrieval flags]   # deepseek-v4-flash, paid, cached
uv run yfr answer --system agent [--tool-shape single|per-type] [--no-ask] [--max-steps N] \
    [--limit N] [retrieval flags]   # the S4 agent; paid; replies cached, traces per case
~/KamaAgenticRAG/backend/.venv/bin/python scripts/course_stages.py --course ~/KamaAgenticRAG \
    [--stages s1,s2,s3,s4] [--limit N]   # course stages on our split, in the course's venv; paid
uv run yfr import-course evals/results/course/dev/s1.jsonl   # -> answer records for yfr grade
uv run yfr label-sheet RUN       # blind labels.md + labels.tsv for a run's dev answers
uv run yfr agree LABELS.tsv RUN [RUN ...]   # judge vs Yi's labels: agreement, kappa
```

Layout: `src/yifinance_rag/data/` (dataset files and hashes, corpus + `EvidenceMap`,
benchmark `Question` vs `Label`, pins, `check_dataset`), `src/yifinance_rag/retrieval/`
(tokenizer, BM25, retrievers, metrics, `dense.py`), `src/yifinance_rag/tools/` (data tools:
`policy.toml`, `data.py`), `src/yifinance_rag/grade/` (answer
record, grader-side `Truth`, deterministic graders, instance-fact atoms, the course scorer
port, no-LLM baselines, selftest, judge, calibration), `src/yifinance_rag/answer/` (systems
that answer: `onecall.py`, `agent.py`, `user.py`), `src/yifinance_rag/llm/` (Anthropic and OpenAI-compatible
providers, `cached.py`, pricing, retry), `src/yifinance_rag/embed/` (Ollama
client pinned to the course's model digest, SQLite embedding cache),
`src/yifinance_rag/cli.py` (`yfr`), `tests/synthetic.py` (a tiny dataset in the real
layout; `graded=True` writes a second question set with selectors and trajectories) and `tests/fake_ollama.py` (a local HTTP stand-in), both for `make verify`.

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
- Run outputs (`evals/results/`) are machine-local and gitignored. The runs a report cites
  are committed as copies under `evals/records/` (decided 2026-10-17, after committing
  VM summaries at their original paths blocked Yi's `git pull`).
- Data in Claude's cloud container: when Yi uploads the zip to a session, Claude unpacks
  it in its scratchpad (never in the repo) and runs `yfr data check`, `make data-test`
  and the no-cost retrieval runs itself. The VM stays the source of truth: anything that
  costs money, and confirming results, runs there.
- Security: a file in `Desktop/learn-claude-code/` is named with an Anthropic API key;
  Yi was told to revoke it. Never put keys in file names, traces or commits.

## Status (2026-10-17)
**S0 done** (closed on the VM 2026-10-16): `make verify` 61 passed, `make pg-up` healthy,
dataset unpacked at `~/flowpay-dataset`. RAM measured with yiagent's ES running: ES
3.8 GiB, pgvector 26 MiB, 4.1 GiB available, no swap. S0/S1 open questions decided
(docs/ROADMAP.md): embed once and cache, plus a 2 GiB swap file; BM25 in-process with jieba.

**S1a done** (2026-10-16, on the uploaded zip): dataset checks pass (release hash = the
course's pin, all counts, all 809 chunks rebuilt exactly, split hash pinned, leakage
checks), `make verify` 83 passed, `make data-test` 3 passed. Dev table: oracle 1.0,
wrong-oracle 0, random 0, **BM25 recall@10 0.456, MRR@10 0.287** (per type in ROADMAP).
**Confirmed on the VM** (2026-10-16): `make data-check` ok with the same split hash,
`make data-test` 3 passed, identical dev tables.
**S1b code done** (2026-10-16): Ollama client (digest-pinned), embedding cache,
course-dense (the course's own index, restored from its dump) and dense-809; `make verify`
93 passed, `make pg-test` 3 passed on the real data. Course index recall ceiling on dev
0.942 (it drops the `c001` title sections).
**S1 dev table done on the VM** (2026-10-16): recall@10 BM25 0.456, course-dense 0.293,
dense-809 0.276 (MRR@10 0.287 / 0.222 / 0.222). Dense loses to BM25 overall and on no
type wins; chunking barely matters (ROADMAP S1b).
**S1 closed** (2026-10-16): `make ollama-test` passed (our embedder reproduces the
course's vectors). Paired on dev: BM25 finds more gold than both dense indexes (recall@10
-0.16 / -0.18 for dense, CIs exclude 0; MRR not distinguishable); chunking moves nothing;
BM25's edge is Chinese word overlap, not exact codes; 36/60 dev questions are vague
progressive-disclosure openers (ROADMAP S1b).
**S2 started** (2026-10-16): Yi chose hybrid first plus a model option for the dense
runs. Built: `--retriever hybrid` (RRF over BM25 + dense-809, k = 60) and `--embed-model`
(Qwen3-Embedding 0.6b/4b/8b; course model digest-pinned). `make verify` 100 passed.
**Hybrid on dev (VM, 2026-10-17):** recall@10 0.549 vs BM25 0.456 (+0.094, 20/8/32,
sign p 0.036, CI just touches 0) and vs dense-809 0.276 (+0.273, CI excludes 0); MRR not
distinguishable. Hybrid is the S2 baseline now. The first 4B run failed as an infra error
(three concurrent runs on a CPU Ollama); the cache now commits per batch, locks against
concurrent runs and reports progress; request timeout 600 s. `make verify` 102 passed.
**Asking the user, upper bound (container, 2026-10-17):** `--user-facts` adds every
simulated-user fact to the 36 dev openers: BM25 recall@10 0.315 to 0.462 (+0.147, p
0.008), as large as hybrid's gain, so S4's ask-user loop is the next big lever. Tables
and comparisons now have per-mode rows. `make verify` 104 passed.
Runs now report recall@20 (a reranker's ceiling): BM25 0.553 vs 0.456 at 10.
**4B rows (VM, 2026-10-17):** dense-809-4b recall@10 0.380 (+0.104 over 0.6B, CI clears
0); hybrid-4b recall@10 0.604, recall@20 0.761, MRR 0.363 (best first stage; +0.055 /
+0.079 / +0.061 over hybrid). Headroom between recall@10 and @20, so Yi chose a Claude
Haiku 5.5 reranker: built (`--rerank haiku`, listwise top 20, cached replies, trials,
cost in the summary). `make verify` 113 passed.
**Reranker on dev (VM, 2026-10-17):** hybrid-4b + Haiku rerank recall@10 0.714 vs 0.604
(+0.110, 15/2/43, p 0.002, CI [+0.054, +0.171]); MRR 0.449 vs 0.363, not distinguishable
(and twice the trial-to-trial noise); recall@20 unchanged by design; $0.0286 a dev run
($0.0005 a search). The first attempt's "Event loop is closed" tracebacks were client
cleanup, not failed calls; fixed (one loop and one client per run). `make verify` 115.
**S2 closed (2026-10-17).** S4 retrieval config: **hybrid-4b + Haiku 5.5 rerank of the
top 20, keep the top 10** (ROADMAP "S2 closed" has the reasons, costs and the dev ladder:
course-dense 0.293, BM25 0.456, hybrid 0.549, hybrid-4b 0.604, + rerank 0.714 recall@10).

**S3 step 1 done** (2026-10-17, container, on the uploaded zip): read the benchmark spec,
every label field, the simulated user, the SQLite schema and the course scorer; findings
with numbers and the grader design they lead to are in ROADMAP "S3 step 1". Yi decided
the models: chat `deepseek-v4-flash`, judge Haiku 5.5 or Sonnet 5.5.

**S3 steps 2-4 done** (2026-10-17, container, on the uploaded zip): answer record format,
deterministic graders (evidence per selector incl. negatives, sources, operations,
instance facts, `as_of`, clarification, citations), the course scorer port, and the
oracle / null / wrong / rule-only / verbose-null baselines; `yfr grade`. Selftest passes
in `make verify` (128 passed) and on all 300 real questions (`make data-test` 4 passed).
Dev: rule-only passes the course task 1.000 but instance facts 0.119; verbose-null (BM25
top 10 pasted, no LLM) passes the course task 0.433 (53% on all 300), instance facts 0.173
(table: ROADMAP "S3 steps 2-4", `evals/records/grade/`). **Confirmed on the VM**
(2026-10-17): all tests pass, selftest tables identical; verbose-null with the S2 config
(no chat model): course task 0.567 on dev, instance facts 0.255, doc evidence 0.714.

**S3 step 5 built** (2026-10-17, container): the LLM judge (`--judge haiku|sonnet`):
per-assertion JSON verdicts with verbatim quotes checked against the answer, incomplete
replies scored None, cached replies and trials, a `forbidden` baseline, thresholds fixed
before the first run (ROADMAP "S3 step 5"). `make verify` 136 passed, `make data-test` 4.
**Judged selftest on dev (VM, 2026-10-17): both pass every threshold.** Haiku: oracle
1.000, 0 incomplete, 0 unverified, $0.069; Sonnet: oracle 0.950, 9 incomplete, 18
unverified quotes, $1.14. Verbose-null judge_task 0.367 / 0.308 vs course 0.567.
Sonnet's misses were our checks (ASCII punctuation in quotes, a corrected second JSON):
fixed (NFKC quotes, last JSON verdict). **Re-scored on the VM (cached, $0): step 5
closed.** Sonnet oracle 1.000, 0 incomplete, 0 unverified, same as Haiku on every check;
verbose-null judge_task 0.367 / 0.350 vs course 0.567. The judge is chosen by Yi's labels.

**S3 step 6 built** (2026-10-17, container): OpenAI-compatible provider (httpx2) with
the course's chat config (`deepseek-v4-flash`, thinking disabled, 4096 tokens; price
unknown = None until Yi reads it), the one-call baseline (`yfr answer`), blind label
sheets and agreement (`yfr label-sheet`, `yfr agree`). `make verify` 153 passed,
`make data-test` 5 passed (incl. a no-ids check on every one-call prompt).
**One-call on dev (VM, 2026-10-17):** judge_task 0.383 (Haiku and Sonnet; Haiku trial 2
0.333), single-turn 0.667, progressive 0.194; instance facts 0.208; course task 0.383.
About equal to pasting the chunks with no model (0.367 / 0.350).
**Judge vs Yi's labels (VM, 2026-10-20): Haiku chosen** (Sonnet's +0.025 all-items kappa
is inside Haiku's 0.027 trial gap). Agreement only fair: required-item kappa 0.36 (Haiku)
/ 0.40 (Sonnet); judges stricter than Yi (49 vs 76 of 106 supported; 23 vs 41 of 60
answers pass). Labels frozen in `evals/records/grade/s3_labels_dev.tsv`.
**S3 closed (2026-10-20).** Yi chose the strict rubric (hedged or partial answers don't
support an assertion). Judge Haiku 5.5; after the ellipsis-quote fix the one-call bar is
judge_task 0.400 (single 0.708, progressive 0.194), instance facts 0.208 (ROADMAP "S3
closed": graders, numbers, interview lines). `make verify` 154 passed.

**S4a done (2026-10-20, container):** data tools (`src/yifinance_rag/tools/`: `policy.toml`
+ `data.py`; 4 tools, 20 operations, read-only, scope bound in SQL, `as_of`, retention by
plan, 24-hour log window). **Reproduce all 535 benchmark selectors exactly**; 0 leaks in
real outputs. `make verify` 163 passed, `make data-test` 7 passed (ROADMAP "S4a done").
**S4b done (2026-10-20, container):** the agent loop (`answer/agent.py`, `answer/user.py`;
`yfr answer --system agent`): search + data tools + `ask_user` (the dataset's simulated
user) + `submit_answer`; handles instead of evidence ids; cached replies; traces. An
oracle-driven fake model through the real harness gets full evidence on all 300 questions
with no leaks. `make verify` 172 passed, `make data-test` 8 passed.
**S4c first run (VM, 2026-10-23):** agent judge_task 0.672 vs the one-call bar 0.400
(progressive 0.559 vs 0.194); instance facts 0.934 vs 0.208 (36/0/0); doc evidence
unchanged. A grader fix for all systems (thousands separators in instance facts: agent
0.919 -> 0.934, one-call unchanged). Flaws: asks on 22/24 single-turn questions (refused);
one question never asked. `yfr compare` now pairs grade runs. `make verify` 174 passed
(ROADMAP "S4c first run"). **Paired on the VM:** judge_task +0.259, CI [+0.10, +0.40],
19/4/35, p 0.003; multi_source stays low (0.357).
**S4d started (2026-10-23):** `disputed()` dropped (would fire on 1/60, a false positive);
the prompt now names only each variant's tools (default byte-identical, hash pinned).
A/Bs: `--no-ask` and `--tool-shape per-type`, each vs S4c. `make verify` 175 passed.
**S4d results (VM, 2026-10-23):** no-ask judge_task 0.684 -> 0.456 (-0.228, CI
[-0.35, -0.11], p 0.002; back to the one-call bar): asking is the loop's gain. Per-type
tools +0.089, CI [-0.02, +0.20], p 0.227: not distinguishable (multi_source +0.286,
multi_hop -0.273). DeepSeek price recorded from Yi's bill ($0.15 / $0.003 / $0.60 per M,
off-peak; peak 2x); S4c chat cost $0.0018 a question. `make verify` 176 passed.
**S4d decided:** per-type tools for S5 (Yi, 2026-10-23).
**S4e built (2026-10-23, container):** `scripts/course_stages.py` runs the course's s1-s4
in its own venv with its own `play()` on our split; `yfr import-course` maps the export to
answer records (passages by span, records by the dataset's ids), graded like ours.
Not comparable: tools, operations, negative_found, as_of. `make verify` 180 passed.
**S4e results (VM, 2026-10-23):** judge_task on dev: ours (per-type) 0.741 vs course s1
0.300, s2 0.550, s3 0.350, s4 0.183; vs s2 +0.190 [+0.05, +0.33], p 0.013 (all CIs
exclude 0). Adapter checks hold (s1 doc evidence 0.306 = S1's course-index recall 0.293).
s4 read in its export: 17/24 single-turn answers are only a request for ids (the course's
protocol grades that as the answer), 6/36 progressive asks missed by its ？ detector;
quote s4 on progressive only (0.194 vs 0.588) and s2 as the headline bar. Caveats: dev
favours us (tuned there).
**Agent trial 1 on dev (VM, 2026-10-23):** judge_task 0.741 -> 0.690 (paired -0.052, CI
[-0.17, +0.07]): our own run-to-run noise is about 0.05; per-type vs single is noise-sized.
DeepSeek's served model id is `deepseek-flash`: added to pricing (costs were None).
**S5 plan fixed** (ROADMAP "S5 plan"): ours (per-type), one-call, course s1-s4, once each on
test; headline ours vs course s2.
**S5 test results (VM, 2026-10-24, run once):** judge_task ours 0.662 vs course s1 0.458,
s2 0.567, s3 0.382, s4 0.315, one-call 0.544; vs s2 +0.097 [+0.03, +0.17], p 0.009. Gain is
all progressive (0.521 vs 0.354); single-turn level; doc evidence level with s2 (0.649 vs
0.620). Dev gap (+0.19) shrank on test as expected (ROADMAP "S5 results").

## Next session: start here (S5: cost, README, demo)
1. `git pull` on `claude/yifinancerag`; read docs/ROADMAP.md "S3 closed" and "S4" (plan
   S4a-S4e). The graders are closed: don't change them to make S4 look better.
2. If Yi uploads the dataset zip, unpack it in the scratchpad (never the repo); S4a
   needs `data/generated/canonical/` and `evidence_resolution.jsonl`.
3. S5 test run done (ROADMAP "S5 results"). Open: cost per question from the answer-run
   summaries, the README results table (done, cost pending), a small deployed demo.
4. No `main` branch exists on GitHub yet (2026-10-20); ask Yi before creating it.
- Run records the next session may need: `evals/records/retrieval/` holds every S1/S2
  run summary and the paired tables (`s1_dev_paired.md`, `s2_dev_paired.md`,
  `s2_rerank_dev_paired.md`, `s2_user_facts_upper_bound.md`). The two rerank runs'
  `cases.jsonl` stay on the VM (`evals/results/retrieval/20261017T0722*_hybrid-4b+rerank_dev`).
- Rerank replies and embeddings are cached on the VM in `~/.yfr/cache` (`rerank.sqlite`,
  the embedding caches), so repeating an S2 run costs nothing.
