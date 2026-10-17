# S2 dev: Haiku 5.5 listwise reranker over hybrid-4b top 20 (VM, 2026-10-17, commit 3125900; `yfr compare`)

A = hybrid-4b (20261017T070140Z). Trial 0 = 20261017T072231Z (46 replies cached by the first,
noisy run at 0ff23ce + 14 new calls; same prompts). Trial 1 = 20261017T072441Z (`--trial 1`,
all 60 asked again). Only trial 0 has summary.json here; both runs' cases.jsonl are on the VM.

```
rerank: 14 calls, 46 cached, 1 fallbacks, 45300 in / 5683 out tokens, $0.0074
(trial 0; the calls cached from the first run are not in its $)
rerank: 60 calls, 0 cached, 0 fallbacks, 187044 in / 19819 out tokens, $0.0286
(trial 1: the full price of one dev run)
```

## hybrid-4b+rerank (trial 0) vs hybrid-4b

recall@10: B = hybrid-4b+rerank vs A = hybrid-4b (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.714 | 0.786 | +0.071 | [+0.000, +0.214] | 1/0/13 | 1.000 |
| multi_document | 10 | 0.650 | 0.750 | +0.100 | [+0.000, +0.250] | 2/0/8 | 0.500 |
| multi_hop | 12 | 0.625 | 0.708 | +0.083 | [+0.000, +0.208] | 2/0/10 | 0.500 |
| temporal_versioned | 5 | 0.547 | 0.663 | +0.117 | [+0.000, +0.250] | 2/0/3 | 0.500 |
| multi_source | 14 | 0.506 | 0.702 | +0.196 | [+0.071, +0.339] | 7/1/6 | 0.070 |
| tool_use_diagnostic | 5 | 0.483 | 0.533 | +0.050 | [-0.150, +0.300] | 1/1/3 | 1.000 |
| mode=progressive | 36 | 0.548 | 0.675 | +0.127 | [+0.056, +0.204] | 12/2/22 | 0.013 |
| mode=single | 24 | 0.688 | 0.771 | +0.083 | [+0.000, +0.188] | 3/0/21 | 0.250 |
| all | 60 | 0.604 | 0.714 | +0.110 | [+0.054, +0.171] | 15/2/43 | 0.002 |

mrr@10: B = hybrid-4b+rerank vs A = hybrid-4b (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.271 | 0.270 | -0.001 | [-0.201, +0.198] | 6/4/4 | 0.754 |
| multi_document | 10 | 0.473 | 0.513 | +0.041 | [-0.302, +0.379] | 5/4/1 | 1.000 |
| multi_hop | 12 | 0.317 | 0.573 | +0.256 | [-0.029, +0.539] | 7/5/0 | 0.774 |
| temporal_versioned | 5 | 0.567 | 0.490 | -0.077 | [-0.427, +0.273] | 2/3/0 | 1.000 |
| multi_source | 14 | 0.364 | 0.449 | +0.084 | [-0.111, +0.269] | 7/4/3 | 0.549 |
| tool_use_diagnostic | 5 | 0.309 | 0.479 | +0.170 | [-0.351, +0.691] | 3/2/0 | 1.000 |
| mode=progressive | 36 | 0.369 | 0.500 | +0.131 | [-0.029, +0.288] | 19/14/3 | 0.487 |
| mode=single | 24 | 0.355 | 0.371 | +0.017 | [-0.170, +0.203] | 11/8/5 | 0.648 |
| all | 60 | 0.363 | 0.449 | +0.085 | [-0.032, +0.205] | 30/22/8 | 0.332 |

recall@20: B = hybrid-4b+rerank vs A = hybrid-4b (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.786 | 0.786 | +0.000 | [+0.000, +0.000] | 0/0/14 | 1.000 |
| multi_document | 10 | 0.800 | 0.800 | +0.000 | [+0.000, +0.000] | 0/0/10 | 1.000 |
| multi_hop | 12 | 0.750 | 0.750 | +0.000 | [+0.000, +0.000] | 0/0/12 | 1.000 |
| temporal_versioned | 5 | 0.730 | 0.730 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| multi_source | 14 | 0.756 | 0.756 | +0.000 | [+0.000, +0.000] | 0/0/14 | 1.000 |
| tool_use_diagnostic | 5 | 0.683 | 0.683 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| mode=progressive | 36 | 0.740 | 0.740 | +0.000 | [+0.000, +0.000] | 0/0/36 | 1.000 |
| mode=single | 24 | 0.792 | 0.792 | +0.000 | [+0.000, +0.000] | 0/0/24 | 1.000 |
| all | 60 | 0.761 | 0.761 | +0.000 | [+0.000, +0.000] | 0/0/60 | 1.000 |

## Noise: trial 1 (B) vs trial 0 (A), same prompts asked again

recall@10: B = hybrid-4b+rerank vs A = hybrid-4b+rerank (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.786 | 0.786 | +0.000 | [+0.000, +0.000] | 0/0/14 | 1.000 |
| multi_document | 10 | 0.750 | 0.750 | +0.000 | [+0.000, +0.000] | 0/0/10 | 1.000 |
| multi_hop | 12 | 0.708 | 0.667 | -0.042 | [-0.125, +0.000] | 0/1/11 | 1.000 |
| temporal_versioned | 5 | 0.663 | 0.663 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| multi_source | 14 | 0.702 | 0.667 | -0.036 | [-0.107, +0.000] | 0/1/13 | 1.000 |
| tool_use_diagnostic | 5 | 0.533 | 0.533 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| mode=progressive | 36 | 0.675 | 0.648 | -0.028 | [-0.069, +0.000] | 0/2/34 | 0.500 |
| mode=single | 24 | 0.771 | 0.771 | +0.000 | [+0.000, +0.000] | 0/0/24 | 1.000 |
| all | 60 | 0.714 | 0.697 | -0.017 | [-0.042, +0.000] | 0/2/58 | 0.500 |

mrr@10: B = hybrid-4b+rerank vs A = hybrid-4b+rerank (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.270 | 0.259 | -0.011 | [-0.040, +0.013] | 1/3/10 | 0.625 |
| multi_document | 10 | 0.513 | 0.467 | -0.047 | [-0.227, +0.130] | 2/4/4 | 0.688 |
| multi_hop | 12 | 0.573 | 0.481 | -0.093 | [-0.235, +0.009] | 2/3/7 | 1.000 |
| temporal_versioned | 5 | 0.490 | 0.490 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| multi_source | 14 | 0.449 | 0.408 | -0.041 | [-0.123, +0.013] | 2/3/9 | 1.000 |
| tool_use_diagnostic | 5 | 0.479 | 0.479 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| mode=progressive | 36 | 0.500 | 0.453 | -0.047 | [-0.105, -0.001] | 4/6/26 | 0.754 |
| mode=single | 24 | 0.371 | 0.346 | -0.026 | [-0.106, +0.049] | 3/7/14 | 0.344 |
| all | 60 | 0.449 | 0.410 | -0.038 | [-0.084, +0.003] | 7/13/40 | 0.263 |

recall@20: B = hybrid-4b+rerank vs A = hybrid-4b+rerank (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.786 | 0.786 | +0.000 | [+0.000, +0.000] | 0/0/14 | 1.000 |
| multi_document | 10 | 0.800 | 0.800 | +0.000 | [+0.000, +0.000] | 0/0/10 | 1.000 |
| multi_hop | 12 | 0.750 | 0.750 | +0.000 | [+0.000, +0.000] | 0/0/12 | 1.000 |
| temporal_versioned | 5 | 0.730 | 0.730 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| multi_source | 14 | 0.756 | 0.756 | +0.000 | [+0.000, +0.000] | 0/0/14 | 1.000 |
| tool_use_diagnostic | 5 | 0.683 | 0.683 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| mode=progressive | 36 | 0.740 | 0.740 | +0.000 | [+0.000, +0.000] | 0/0/36 | 1.000 |
| mode=single | 24 | 0.792 | 0.792 | +0.000 | [+0.000, +0.000] | 0/0/24 | 1.000 |
| all | 60 | 0.761 | 0.761 | +0.000 | [+0.000, +0.000] | 0/0/60 | 1.000 |
