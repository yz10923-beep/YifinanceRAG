# YifinanceRAG

A support agent for **FlowPay**, a simulated Stripe-like payments platform. It answers
merchant questions from 100 product documents and from the merchant's own account
records (payments, refunds, subscriptions, webhooks, API logs), asking the merchant for
missing facts first. It is **measured, not demoed**: 300 frozen benchmark questions with
gold evidence, a dev/test split, a calibrated LLM judge, and the reference
implementation of the KamaAgenticRAG course (程序员Carl, 卡码网) run as the baseline on the
same chat model.

**Status (2026-10-24):** stages S0-S5 done; the test split has been run once, with the
systems and settings fixed in advance. Not built: a deployed demo. Full log of decisions
and numbers: [docs/ROADMAP.md](docs/ROADMAP.md).

## Results: 240 held-out test questions, run once

Every system uses the same chat model (`deepseek-v4-flash`, thinking off) and the
dataset's own simulated user; the course's stages run from its own code. "Task pass" is
an LLM judge (Claude Haiku 5.5) checking every required conclusion and no forbidden one.

| system | task pass | document-only questions (96) | questions needing the account's records (144) | records found | $ / question |
|---|---:|---:|---:|---:|---:|
| course s1 (one retrieval) | 0.458 | 0.760 | 0.257 | 0.000 | 0.0007 |
| course s2 (agentic search, its best) | 0.567 | 0.885 | 0.354 | 0.000 | 0.0017 |
| course s3 (+ data tools) | 0.382 | 0.777 | 0.125 | 0.054 | 0.0010 |
| course s4 (full agent) | 0.315 | 0.302 | 0.324 | 0.336 | 0.0013 |
| one-call (our retrieval, no loop) | 0.544 | 0.823 | 0.357 | 0.000 | 0.0009 |
| **this agent** | **0.662** | **0.874** | **0.521** | **0.974** | **0.0037** |

- **vs the course's best stage: +0.097 task pass, 95% CI [+0.03, +0.17], p 0.009**
  (paired over questions; 47 wins, 24 losses). vs our own one-call baseline: +0.114,
  CI [+0.05, +0.18]. Every course stage is beaten with a CI that excludes 0.
- **The gain is on questions that need the account's own records** (0.521 vs 0.354):
  the agent asks for the account and ids, then looks the records up under tenant scope
  (it finds 97% of the gold records; the course's best stage has no data tools). On
  document-only questions the systems are level (0.874 vs 0.885).
- **Cost:** $0.0037 a question at DeepSeek's off-peak price ($0.0023 chat, $0.0014 for the
  Haiku reranker), about 2x the course's best stage. Grading is not included.

## What made the difference (ablations on dev; retrieval also on test)

| change | effect | |
|---|---|---|
| asking the user for missing facts | turning it off: task pass 0.684 -> 0.456 (-0.23, p 0.002) | the main lever |
| data tools covering the records | accounts, refunds, subscriptions, invoices, API keys, feature flags, historical `as_of` lookups; the course's tools reach none of these (its tables cap record evidence at 0.79) | |
| retrieval: dense -> BM25 -> hybrid -> 4B embeddings -> LLM reranker | **test:** recall@10 0.361 -> 0.406 -> 0.549 -> 0.586 -> **0.638** for one search (+0.28, CI [+0.23, +0.32]) | real for one search, hybrid fusion is the big step; inside the loop document evidence is level with the course's repeated searches (0.649 vs 0.620) |
| one search tool vs one per document type | +0.09, inside run-to-run noise (0.05) | not a measured win |
| bouncing answers that cite unseen ids (course's `disputed()`) | would fire on 1 of 60 answers, a false positive | not built |

## How it is measured

- **Tune on dev, report on test once.** The benchmark's split (dev 60 / test 240) is
  grouped by scenario family, so no template appears in both; its hash is pinned in a
  test. The test run followed a plan written before any test result existed.
- **Grade against hidden labels, never the agent's own claims.** Deterministic graders
  (gold chunk ids by character span, gold records per selector, instance facts, `as_of`,
  clarification categories) plus the LLM judge with verbatim-quote checks. Every grader
  has an oracle that passes, a null that fails and a wrong answer that fails, inside
  `make verify`.
- **The judge is chosen by agreement with hand labels** (166 assertions, labelled blind).
- **No ground truth reaches the agent**: the audit journal, gold ids and labels are kept
  out of every prompt and tool output, checked by leakage tests on all 300 questions.
- **Infra failures are not scores**: API errors are rerun, never graded.

## Limitations

- **Dev favoured this agent** (we tuned there): its lead over the course's best stage was
  +0.19 on dev and +0.10 on test. Quote the test number.
- **The judge agrees with hand labels only fairly** (Cohen's kappa 0.36 on required
  conclusions; it is stricter than the human labeller). The gap to the course is several
  times the judge's own trial-to-trial noise, but absolute pass rates depend on it.
- **One trial per system on test.** On dev, two runs of the same agent differed by 0.05.
- **The data tools were built to the benchmark's declared tool and operation set**
  (all 535 gold lookups reproduced); the course's tools were not.
- **The course's s4 is penalised by its own protocol**: on 17 of 24 dev single-turn
  questions its whole reply is a request for ids, which the course's runner grades as the
  answer. Compare s4 on questions needing records (0.324 vs 0.521).
- **Weak question types:** multi-source (0.49) and tool-use diagnostic (0.42).
- **The test split is now spent** (agent and retrieval each run once): any further
  change needs a new held-out set before it can be claimed. On test, retrieval gains were
  smaller than on dev (the reranker +0.05 vs +0.11).

## Architecture

```
question ─> agent loop (raw DeepSeek API, our own ~300 lines, no framework)
              ├─ search_* (5 doc types) ─> BM25 (jieba) + Qwen3-Embedding-4B ─RRF─> Haiku rerank
              ├─ data tools ─> policy.toml: read-only, account + environment bound in SQL
              ├─ ask_user ─> the dataset's simulated user
              └─ submit_answer (answer + cited handles) ─> answer record ─> graders, judge
```

Python 3.12, `mypy --strict`, ruff; Postgres + pgvector for vectors; every model reply
cached by request; traces per question. Layout and commands: [CLAUDE.md](CLAUDE.md).

## Reproduce

Requires the FlowPay dataset from the KamaAgenticRAG course; it is not in this repo and is
never committed. Runs on a 4-vCPU / 8 GB Linux VM with Ollama.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                    # once
unzip flowpay-dataset.zip 'flowpay-dataset/*' -d ~
mkdir -p ~/.yfr && cp .env.example ~/.yfr/.env    # ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
make sync && make verify && make pg-up && make data-check
uv run yfr retrieval --retriever hybrid --embed-model qwen3-embedding:4b --rerank haiku
uv run yfr answer --system agent --tool-shape per-type \
    --retriever hybrid --embed-model qwen3-embedding:4b --rerank haiku   # dev, paid
uv run yfr grade --records <run>/records.jsonl --name agent --judge haiku
uv run yfr compare <grade run A> <grade run B>                       # paired, with CIs
```

The course's stages run in the course's own checkout and venv via
`scripts/course_stages.py`, then `yfr import-course` (see `docs/ROADMAP.md` "S4e").

## Data and licence

The FlowPay dataset is course material distributed to course members; it has no licence
file. Publishing results is allowed per the course terms; the data, the benchmark
questions and the documents are not redistributed here. The course code is MIT and is
imported at run time, never copied (see `VENDORED.md`).
