# S3 step 5: judged selftest on dev (VM, 2026-10-17)

`yfr grade --selftest --judge haiku|sonnet --retriever hybrid --embed-model qwen3-embedding:4b
--rerank haiku` at commit 3a80bb6; both ended `selftest on dev: ok`. Rows are the `all`
row (60 dev questions) of each system's table, as Yi sent them. Thresholds: ROADMAP S3
step 5 (fixed before these runs).

| system | measure | Haiku 5.5 | Sonnet 5.5 | threshold | course scorer |
|---|---|---:|---:|---|---:|
| oracle | judge_task | 1.000 | 0.950 | >= 0.90 | 1.000 (course_task) |
| null | judge_assertions | 0.000 | 0.000 | <= 0.05 | 0.000 |
| wrong | judge_task | 0.000 | 0.000 | <= 0.20 | 0.000 |
| forbidden | judge_forbidden_clear | 0.000 | 0.050 | <= 0.10 | 0.000 (verbatim claim) |
| rule-only | judge_task | 1.000 | 0.983 | reported | 1.000 |
| verbose-null | judge_task | 0.367 | 0.308 | reported | 0.567 |
| verbose-null, single-turn (24) | judge_task | 0.667 | 0.500 | reported | 0.750 |
| verbose-null, progressive (36) | judge_task | 0.167 | 0.206 | reported | 0.444 |

| judge | calls (cached) | incomplete | unverified quotes | tokens in / out | cost |
|---|---|---:|---:|---|---:|
| claude-haiku-5-5 | 336 (24) | 0 | 0 | 362,890 / 65,276 | $0.0689 |
| claude-sonnet-5-5 | 336 (24) | 9 | 18 | 362,890 / 41,581 | $1.1416 |

Sonnet's oracle misses are all on progressive questions (judge_task 0.917 there).

## Re-scored from the cache after the parser fix (VM, 2026-10-17, commit 8c5327a)

Quotes compared after NFKC; the last JSON verdict in a reply is final. 0 calls, 360
cached, $0 for both. Both `selftest on dev: ok`.

| system | measure | Haiku 5.5 | Sonnet 5.5 | course scorer |
|---|---|---:|---:|---:|
| oracle | judge_task | 1.000 | 1.000 | 1.000 |
| null | judge_assertions | 0.000 | 0.000 | 0.000 |
| wrong | judge_task | 0.000 | 0.000 | 0.000 |
| forbidden | judge_forbidden_clear | 0.000 | 0.000 | 0.000 |
| rule-only | judge_task | 1.000 | 1.000 | 1.000 |
| verbose-null | judge_task | 0.367 | 0.350 | 0.567 |
| verbose-null, single-turn | judge_task | 0.667 | 0.542 | 0.750 |
| verbose-null, progressive | judge_task | 0.167 | 0.222 | 0.444 |

| judge | incomplete | unverified quotes |
|---|---:|---:|
| claude-haiku-5-5 | 0 | 0 |
| claude-sonnet-5-5 | 0 (was 9) | 0 (was 18) |
