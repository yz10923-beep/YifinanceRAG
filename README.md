# YifinanceRAG

A RAG + agent support assistant for FlowPay, a simulated payments platform, **measured
rather than demoed**: retrieval ablations (dense vs BM25 vs hybrid vs reranker) scored
against gold chunk ids, a calibrated LLM judge, and the KamaAgenticRAG course's reference
stages as baselines on the same chat model. Plan and status: [docs/ROADMAP.md](docs/ROADMAP.md).

Requires the FlowPay dataset from the KamaAgenticRAG course (程序员Carl, 卡码网). The
dataset is not in this repo and is never committed.

## Setup (Linux VM)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # once
unzip flowpay-dataset.zip 'flowpay-dataset/*' -d ~  # ~/flowpay-dataset/, skips __MACOSX
mkdir -p ~/.yfr && cp .env.example ~/.yfr/.env    # then fill in ANTHROPIC_API_KEY
make sync
make verify
make pg-up                                        # pgvector on 127.0.0.1:5432
make data-check                                   # verify the dataset: hashes, counts, leakage
```

## Retrieval eval (S1, no LLM, no cost)

```bash
uv run yfr retrieval --retriever bm25            # dev split; also random, oracle, wrong-oracle
```

Each run writes `evals/results/retrieval/<run>/summary.json` (metrics per question type,
plus the release, split and chunk-set hashes and the git commit) and `table.md`. Run
outputs are machine-local (gitignored); the runs a report cites are committed as copies
under `evals/records/`.

Dense baselines (S1b) need Ollama with the course's embedding model on the VM:

```bash
make pg-up && make course-index && make pg-test   # the course's own index, checked
uv run yfr retrieval --retriever course-dense     # the course's index, the course's queries
uv run yfr retrieval --retriever dense-809        # same model, the benchmark's chunking
```
