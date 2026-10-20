# S3 step 6: the one-call RAG baseline on dev (VM, 2026-10-17)

Answers: `yfr answer --retriever hybrid --embed-model qwen3-embedding:4b --rerank haiku`
(run `20261017T101359Z_onecall-hybrid-4b+rerank_dev`; deepseek-v4-flash, thinking off),
graded three times on the same 60 records. Rows as Yi sent them.

| measure (all 60) | judge Sonnet 5.5 | judge Haiku 5.5 | Haiku, trial 1 |
|---|---:|---:|---:|
| judge_assertions | 0.508 | 0.508 | 0.475 |
| judge_task | 0.383 | 0.383 | 0.333 |
| judge_task, single-turn (24) | 0.667 | 0.667 | 0.625 |
| judge_task, progressive (36) | 0.194 | 0.194 | 0.139 |
| judge cost (60 answers) | $0.2580 | $0.0179 | $0.0176 |
| incomplete / unverified | 0 / 0 | 0 / 1 | 0 / 2 |

Deterministic (same for all three): doc_evidence 0.714, instance_facts 0.208,
cited_grounded 1.000, cited_gold 0.168, course_task 0.383, course_evidence 0.533.

Against pasting the same top 10 with no chat model (verbose-null, S2 config):

| | one-call (DeepSeek) | verbose-null |
|---|---:|---:|
| judge_task, Haiku | 0.383 | 0.367 |
| judge_task, Sonnet | 0.383 | 0.350 |
| course_task | 0.383 | 0.567 |
| instance_facts | 0.208 | 0.255 |

## Re-scored after the ellipsis-quote fix (VM, 2026-10-20, Haiku, cached, $0)

Run `20261020T095244Z_onecall-hybrid-4b+rerank_dev` (commit 93b913c). This is S4's bar.

| rows | judge_assertions | judge_task |
|---|---:|---:|
| all (60) | 0.517 | 0.400 |
| single-turn (24) | 0.750 | 0.708 |
| progressive (36) | 0.361 | 0.194 |

Only multi_document changed (judge_task 0.400 -> 0.500): the one answer whose
"supported" verdict quoted two pieces joined by "..." now counts.
