# S4e: the course's stages s1-s4 on dev, graded by our graders (2026-10-23)

Run by Yi on the VM: `scripts/course_stages.py` in the course's checkout (`fdeda6b`, its
venv, its knowledge index and config: `deepseek-v4-flash`, thinking off), then
`yfr import-course` and `yfr grade --judge haiku`. Compared with our S5 config
(`agent-per-type`: per-type search tools, ask_user, hybrid-4b + Haiku rerank).

| system (dev, 60) | judge_task | single | progressive | instance_facts | record_evidence | doc_evidence |
|---|---:|---:|---:|---:|---:|---:|
| course-s1 | 0.300 | 0.708 | 0.028 | 0.200 | 0.000 | 0.306 |
| course-s2 | 0.550 | 0.875 | 0.333 | 0.274 | 0.000 | 0.603 |
| course-s3 | 0.350 | 0.667 | 0.139 | 0.265 | 0.019 | 0.417 |
| course-s4 | 0.183 | 0.167 | 0.194 | 0.630 | 0.322 | 0.333 |
| **agent-per-type (ours)** | **0.741** | **0.958** | **0.588** | **0.921** | **0.958** | **0.744** |

Not comparable (course tool vocabulary / no as-of): tools, operations, negative_found,
as_of. Checks: course-s1 doc evidence 0.306 vs the course index's recall@10 0.293 measured
in S1 (the span mapping holds); the course's tables cap record evidence at 0.787 on dev
(22 of 36 cases fully readable), so most of course-s4's gap is behaviour, not coverage.

## Grade tables

course-s1 on dev: 60 cases
| type | n | doc_evidence | record_evidence | negative_found | id_recall | tools | operations | instance_facts | as_of | clarification | cited_grounded | cited_gold | course_task | course_evidence | judge_assertions | judge_forbidden_clear | judge_task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_retrieval | 14 | 0.286 | - | - | 0.286 | 0.714 | - | - | - | 1.000 | - | - | 0.857 | 0.286 | 0.857 | 1.000 | 0.857 |
| multi_document | 10 | 0.500 | - | - | 0.500 | 0.900 | - | - | - | 1.000 | - | - | 0.500 | 0.200 | 0.650 | 1.000 | 0.500 |
| multi_hop | 12 | 0.292 | 0.000 | 0.000 | 0.109 | 0.364 | 0.000 | 0.152 | - | 0.000 | - | - | 0.167 | 0.083 | 0.250 | 1.000 | 0.000 |
| temporal_versioned | 5 | 0.333 | 0.000 | 0.000 | 0.170 | 0.470 | 0.000 | 0.320 | - | 0.167 | - | - | 0.600 | 0.000 | 0.400 | 1.000 | 0.200 |
| multi_source | 14 | 0.238 | 0.000 | 0.000 | 0.102 | 0.385 | 0.000 | 0.210 | - | 0.057 | - | - | 0.357 | 0.071 | 0.214 | 1.000 | 0.000 |
| tool_use_diagnostic | 5 | 0.167 | 0.000 | 0.000 | 0.087 | 0.487 | 0.000 | 0.167 | - | 0.000 | - | - | 0.200 | 0.000 | 0.200 | 1.000 | 0.000 |
| mode=progressive | 36 | 0.259 | 0.000 | 0.000 | 0.112 | 0.404 | 0.000 | 0.200 | - | 0.045 | - | - | 0.306 | 0.056 | 0.250 | 1.000 | 0.028 |
| mode=single | 24 | 0.375 | - | - | 0.375 | 0.792 | - | - | - | 1.000 | - | - | 0.708 | 0.250 | 0.771 | 1.000 | 0.708 |
| all | 60 | 0.306 | 0.000 | 0.000 | 0.217 | 0.559 | 0.000 | 0.200 | - | 0.427 | - | - | 0.467 | 0.133 | 0.458 | 1.000 | 0.300 |
judge claude-haiku-5-5: 60 calls, 0 cached, 0 incomplete, 1 unverified quotes, 101035 in / 18195 out tokens, $0.0192

written to /root/YifinanceRAG/evals/results/grade/20261023T220731Z_course-s1_dev
course-s2 on dev: 60 cases
| type | n | doc_evidence | record_evidence | negative_found | id_recall | tools | operations | instance_facts | as_of | clarification | cited_grounded | cited_gold | course_task | course_evidence | judge_assertions | judge_forbidden_clear | judge_task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_retrieval | 14 | 0.643 | - | - | 0.643 | 0.857 | - | - | - | 1.000 | - | - | 0.929 | 0.643 | 1.000 | 1.000 | 1.000 |
| multi_document | 10 | 0.700 | - | - | 0.700 | 0.950 | - | - | - | 1.000 | - | - | 0.700 | 0.500 | 0.750 | 1.000 | 0.700 |
| multi_hop | 12 | 0.542 | 0.000 | 0.000 | 0.226 | 0.489 | 0.000 | 0.322 | - | 0.200 | - | - | 0.417 | 0.083 | 0.417 | 1.000 | 0.250 |
| temporal_versioned | 5 | 0.567 | 0.000 | 0.000 | 0.284 | 0.480 | 0.000 | 0.291 | - | 0.000 | - | - | 0.600 | 0.200 | 0.500 | 1.000 | 0.200 |
| multi_source | 14 | 0.560 | 0.000 | 0.000 | 0.247 | 0.508 | 0.000 | 0.241 | - | 0.057 | - | - | 0.357 | 0.214 | 0.500 | 0.929 | 0.357 |
| tool_use_diagnostic | 5 | 0.600 | 0.000 | 0.000 | 0.320 | 0.620 | 0.000 | 0.230 | - | 0.167 | - | - | 0.200 | 0.200 | 0.700 | 1.000 | 0.600 |
| mode=progressive | 36 | 0.560 | 0.000 | 0.000 | 0.255 | 0.513 | 0.000 | 0.274 | - | 0.112 | - | - | 0.389 | 0.167 | 0.500 | 0.972 | 0.333 |
| mode=single | 24 | 0.667 | - | - | 0.667 | 0.896 | - | - | - | 1.000 | - | - | 0.833 | 0.583 | 0.896 | 1.000 | 0.875 |
| all | 60 | 0.603 | 0.000 | 0.000 | 0.420 | 0.666 | 0.000 | 0.274 | - | 0.467 | - | - | 0.567 | 0.333 | 0.658 | 0.983 | 0.550 |
judge claude-haiku-5-5: 60 calls, 0 cached, 0 incomplete, 1 unverified quotes, 121287 in / 22054 out tokens, $0.0232

written to /root/YifinanceRAG/evals/results/grade/20261023T220941Z_course-s2_dev
course-s3 on dev: 60 cases
| type | n | doc_evidence | record_evidence | negative_found | id_recall | tools | operations | instance_facts | as_of | clarification | cited_grounded | cited_gold | course_task | course_evidence | judge_assertions | judge_forbidden_clear | judge_task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_retrieval | 14 | 0.571 | - | - | 0.571 | 0.714 | - | - | - | 1.000 | - | - | 0.857 | 0.571 | 0.857 | 1.000 | 0.857 |
| multi_document | 10 | 0.600 | - | - | 0.600 | 0.850 | - | - | - | 1.000 | - | - | 0.400 | 0.300 | 0.600 | 1.000 | 0.400 |
| multi_hop | 12 | 0.250 | 0.000 | 0.000 | 0.093 | 0.297 | 0.000 | 0.245 | 0.000 | 0.067 | - | - | 0.250 | 0.083 | 0.250 | 1.000 | 0.167 |
| temporal_versioned | 5 | 0.467 | 0.133 | 0.000 | 0.262 | 0.400 | 0.000 | 0.422 | 0.000 | 0.233 | - | - | 0.200 | 0.400 | 0.300 | 1.000 | 0.200 |
| multi_source | 14 | 0.238 | 0.000 | 0.000 | 0.105 | 0.311 | 0.000 | 0.229 | 0.000 | 0.000 | - | - | 0.214 | 0.143 | 0.214 | 1.000 | 0.071 |
| tool_use_diagnostic | 5 | 0.467 | 0.000 | 0.000 | 0.250 | 0.447 | 0.000 | 0.253 | 0.000 | 0.000 | - | - | 0.200 | 0.000 | 0.200 | 1.000 | 0.200 |
| mode=progressive | 36 | 0.306 | 0.019 | 0.000 | 0.143 | 0.338 | 0.000 | 0.265 | 0.000 | 0.055 | - | - | 0.222 | 0.139 | 0.236 | 1.000 | 0.139 |
| mode=single | 24 | 0.583 | - | - | 0.583 | 0.771 | - | - | - | 1.000 | - | - | 0.667 | 0.458 | 0.750 | 1.000 | 0.667 |
| all | 60 | 0.417 | 0.019 | 0.000 | 0.319 | 0.511 | 0.000 | 0.265 | 0.000 | 0.433 | - | - | 0.400 | 0.267 | 0.442 | 1.000 | 0.350 |
judge claude-haiku-5-5: 60 calls, 0 cached, 0 incomplete, 2 unverified quotes, 90429 in / 15154 out tokens, $0.0166

written to /root/YifinanceRAG/evals/results/grade/20261023T221124Z_course-s3_dev
course-s4 on dev: 60 cases
| type | n | doc_evidence | record_evidence | negative_found | id_recall | tools | operations | instance_facts | as_of | clarification | cited_grounded | cited_gold | course_task | course_evidence | judge_assertions | judge_forbidden_clear | judge_task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_retrieval | 14 | 0.214 | - | - | 0.214 | 0.214 | - | - | - | 1.000 | - | - | 0.214 | 0.214 | 0.214 | 1.000 | 0.214 |
| multi_document | 10 | 0.250 | - | - | 0.250 | 0.400 | - | - | - | 1.000 | - | - | 0.200 | 0.100 | 0.200 | 1.000 | 0.100 |
| multi_hop | 12 | 0.375 | 0.312 | 0.000 | 0.309 | 0.389 | 0.000 | 0.533 | 0.000 | 0.378 | - | - | 0.250 | 0.083 | 0.208 | 1.000 | 0.000 |
| temporal_versioned | 5 | 0.517 | 0.400 | 0.000 | 0.379 | 0.480 | 0.000 | 0.636 | 0.000 | 0.333 | - | - | 0.200 | 0.400 | 0.500 | 1.000 | 0.200 |
| multi_source | 14 | 0.393 | 0.292 | 0.000 | 0.311 | 0.368 | 0.000 | 0.681 | 0.000 | 0.481 | - | - | 0.286 | 0.214 | 0.429 | 1.000 | 0.214 |
| tool_use_diagnostic | 5 | 0.383 | 0.350 | 0.000 | 0.363 | 0.620 | 0.000 | 0.715 | 0.000 | 0.267 | - | - | 0.200 | 0.000 | 0.800 | 1.000 | 0.600 |
| mode=progressive | 36 | 0.403 | 0.322 | 0.000 | 0.327 | 0.425 | 0.000 | 0.630 | 0.000 | 0.396 | - | - | 0.250 | 0.167 | 0.417 | 1.000 | 0.194 |
| mode=single | 24 | 0.229 | - | - | 0.229 | 0.292 | - | - | - | 1.000 | - | - | 0.208 | 0.167 | 0.208 | 1.000 | 0.167 |
| all | 60 | 0.333 | 0.322 | 0.000 | 0.288 | 0.372 | 0.000 | 0.630 | 0.000 | 0.638 | - | - | 0.233 | 0.167 | 0.333 | 1.000 | 0.183 |
judge claude-haiku-5-5: 60 calls, 0 cached, 0 incomplete, 1 unverified quotes, 87120 in / 17588 out tokens, $0.0175

written to /root/YifinanceRAG/evals/results/grade/20261023T221320Z_course-s4_dev
## Paired: agent-per-type (B) vs each course stage (A)

judge_task: B = agent-per-type vs A = course-s1 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.857 | 1.000 | +0.143 | [+0.000, +0.357] | 2/0/12 | 0.500 |
| multi_document | 10 | 0.500 | 0.900 | +0.400 | [+0.100, +0.700] | 4/0/6 | 0.125 |
| multi_hop | 12 | 0.000 | 0.583 | +0.583 | [+0.333, +0.833] | 7/0/5 | 0.016 |
| temporal_versioned | 5 | 0.200 | 0.600 | +0.400 | [+0.000, +0.800] | 2/0/3 | 0.500 |
| multi_source | 14 | 0.000 | 0.643 | +0.643 | [+0.357, +0.857] | 9/0/5 | 0.004 |
| tool_use_diagnostic | 3 | 0.000 | 0.333 | +0.333 | [+0.000, +1.000] | 1/0/2 | 1.000 |
| mode=progressive | 34 | 0.029 | 0.588 | +0.559 | [+0.382, +0.735] | 19/0/15 | 0.000 |
| mode=single | 24 | 0.708 | 0.958 | +0.250 | [+0.083, +0.417] | 6/0/18 | 0.031 |
| all | 58 | 0.310 | 0.741 | +0.431 | [+0.310, +0.569] | 25/0/33 | 0.000 |

instance_facts: B = agent-per-type vs A = course-s1 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.152 | 0.949 | +0.796 | [+0.701, +0.878] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.320 | 0.964 | +0.644 | [+0.446, +0.781] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.210 | 0.858 | +0.648 | [+0.478, +0.783] | 13/1/0 | 0.002 |
| tool_use_diagnostic | 5 | 0.167 | 0.989 | +0.822 | [+0.764, +0.879] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.200 | 0.921 | +0.721 | [+0.640, +0.790] | 35/1/0 | 0.000 |
| all | 36 | 0.200 | 0.921 | +0.721 | [+0.640, +0.790] | 35/1/0 | 0.000 |

record_evidence: B = agent-per-type vs A = course-s1 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.000 | 0.893 | +0.893 | [+0.714, +1.000] | 13/0/1 | 0.000 |
| tool_use_diagnostic | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.000 | 0.958 | +0.958 | [+0.889, +1.000] | 35/0/1 | 0.000 |
| all | 36 | 0.000 | 0.958 | +0.958 | [+0.889, +1.000] | 35/0/1 | 0.000 |

doc_evidence: B = agent-per-type vs A = course-s1 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.286 | 0.857 | +0.571 | [+0.286, +0.786] | 8/0/6 | 0.008 |
| multi_document | 10 | 0.500 | 0.750 | +0.250 | [+0.050, +0.450] | 6/1/3 | 0.125 |
| multi_hop | 12 | 0.292 | 0.667 | +0.375 | [+0.042, +0.667] | 7/2/3 | 0.180 |
| temporal_versioned | 5 | 0.333 | 0.563 | +0.230 | [+0.050, +0.410] | 3/0/2 | 0.250 |
| multi_source | 14 | 0.238 | 0.744 | +0.506 | [+0.345, +0.679] | 12/0/2 | 0.000 |
| tool_use_diagnostic | 5 | 0.167 | 0.783 | +0.617 | [+0.533, +0.700] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.259 | 0.699 | +0.439 | [+0.306, +0.565] | 27/2/7 | 0.000 |
| mode=single | 24 | 0.375 | 0.812 | +0.438 | [+0.250, +0.625] | 14/1/9 | 0.001 |
| all | 60 | 0.306 | 0.744 | +0.439 | [+0.332, +0.545] | 41/3/16 | 0.000 |

judge_task: B = agent-per-type vs A = course-s2 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 0/0/14 | 1.000 |
| multi_document | 10 | 0.700 | 0.900 | +0.200 | [+0.000, +0.500] | 2/0/8 | 0.500 |
| multi_hop | 12 | 0.250 | 0.583 | +0.333 | [+0.000, +0.667] | 5/1/6 | 0.219 |
| temporal_versioned | 5 | 0.200 | 0.600 | +0.400 | [+0.000, +0.800] | 2/0/3 | 0.500 |
| multi_source | 14 | 0.357 | 0.643 | +0.286 | [+0.000, +0.571] | 5/1/8 | 0.219 |
| tool_use_diagnostic | 3 | 0.667 | 0.333 | -0.333 | [-1.000, +0.000] | 0/1/2 | 1.000 |
| mode=progressive | 34 | 0.324 | 0.588 | +0.265 | [+0.059, +0.471] | 12/3/19 | 0.035 |
| mode=single | 24 | 0.875 | 0.958 | +0.083 | [+0.000, +0.208] | 2/0/22 | 0.500 |
| all | 58 | 0.552 | 0.741 | +0.190 | [+0.052, +0.328] | 14/3/41 | 0.013 |

instance_facts: B = agent-per-type vs A = course-s2 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.322 | 0.949 | +0.626 | [+0.509, +0.736] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.291 | 0.964 | +0.672 | [+0.606, +0.733] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.241 | 0.858 | +0.617 | [+0.427, +0.772] | 13/1/0 | 0.002 |
| tool_use_diagnostic | 5 | 0.230 | 0.989 | +0.759 | [+0.653, +0.844] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.274 | 0.921 | +0.647 | [+0.562, +0.722] | 35/1/0 | 0.000 |
| all | 36 | 0.274 | 0.921 | +0.647 | [+0.562, +0.722] | 35/1/0 | 0.000 |

record_evidence: B = agent-per-type vs A = course-s2 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.000 | 0.893 | +0.893 | [+0.714, +1.000] | 13/0/1 | 0.000 |
| tool_use_diagnostic | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.000 | 0.958 | +0.958 | [+0.889, +1.000] | 35/0/1 | 0.000 |
| all | 36 | 0.000 | 0.958 | +0.958 | [+0.889, +1.000] | 35/0/1 | 0.000 |

doc_evidence: B = agent-per-type vs A = course-s2 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.643 | 0.857 | +0.214 | [-0.071, +0.500] | 4/1/9 | 0.375 |
| multi_document | 10 | 0.700 | 0.750 | +0.050 | [-0.100, +0.200] | 2/1/7 | 1.000 |
| multi_hop | 12 | 0.542 | 0.667 | +0.125 | [-0.083, +0.333] | 5/2/5 | 0.453 |
| temporal_versioned | 5 | 0.567 | 0.563 | -0.003 | [-0.267, +0.260] | 2/2/1 | 1.000 |
| multi_source | 14 | 0.560 | 0.744 | +0.185 | [+0.000, +0.351] | 9/2/3 | 0.065 |
| tool_use_diagnostic | 5 | 0.600 | 0.783 | +0.183 | [-0.200, +0.517] | 3/1/1 | 0.625 |
| mode=progressive | 36 | 0.560 | 0.699 | +0.138 | [+0.019, +0.252] | 19/7/10 | 0.029 |
| mode=single | 24 | 0.667 | 0.812 | +0.146 | [-0.042, +0.333] | 6/2/16 | 0.289 |
| all | 60 | 0.603 | 0.744 | +0.141 | [+0.041, +0.243] | 25/9/26 | 0.009 |

judge_task: B = agent-per-type vs A = course-s3 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.857 | 1.000 | +0.143 | [+0.000, +0.357] | 2/0/12 | 0.500 |
| multi_document | 10 | 0.400 | 0.900 | +0.500 | [+0.200, +0.800] | 5/0/5 | 0.062 |
| multi_hop | 12 | 0.167 | 0.583 | +0.417 | [+0.000, +0.750] | 6/1/5 | 0.125 |
| temporal_versioned | 5 | 0.200 | 0.600 | +0.400 | [+0.000, +0.800] | 2/0/3 | 0.500 |
| multi_source | 14 | 0.071 | 0.643 | +0.571 | [+0.286, +0.786] | 8/0/6 | 0.008 |
| tool_use_diagnostic | 3 | 0.000 | 0.333 | +0.333 | [+0.000, +1.000] | 1/0/2 | 1.000 |
| mode=progressive | 34 | 0.118 | 0.588 | +0.471 | [+0.294, +0.647] | 17/1/16 | 0.000 |
| mode=single | 24 | 0.667 | 0.958 | +0.292 | [+0.125, +0.458] | 7/0/17 | 0.016 |
| all | 58 | 0.345 | 0.741 | +0.397 | [+0.259, +0.534] | 24/1/33 | 0.000 |

instance_facts: B = agent-per-type vs A = course-s3 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.245 | 0.949 | +0.704 | [+0.545, +0.824] | 11/0/1 | 0.001 |
| temporal_versioned | 5 | 0.422 | 0.964 | +0.542 | [+0.270, +0.708] | 4/0/1 | 0.125 |
| multi_source | 14 | 0.229 | 0.858 | +0.629 | [+0.481, +0.764] | 13/0/1 | 0.000 |
| tool_use_diagnostic | 5 | 0.253 | 0.989 | +0.736 | [+0.500, +0.956] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.265 | 0.921 | +0.656 | [+0.563, +0.741] | 33/0/3 | 0.000 |
| all | 36 | 0.265 | 0.921 | +0.656 | [+0.563, +0.741] | 33/0/3 | 0.000 |

record_evidence: B = agent-per-type vs A = course-s3 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.133 | 1.000 | +0.867 | [+0.600, +1.000] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.000 | 0.893 | +0.893 | [+0.714, +1.000] | 13/0/1 | 0.000 |
| tool_use_diagnostic | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.019 | 0.958 | +0.940 | [+0.861, +1.000] | 35/0/1 | 0.000 |
| all | 36 | 0.019 | 0.958 | +0.940 | [+0.861, +1.000] | 35/0/1 | 0.000 |

doc_evidence: B = agent-per-type vs A = course-s3 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.571 | 0.857 | +0.286 | [+0.000, +0.571] | 5/1/8 | 0.219 |
| multi_document | 10 | 0.600 | 0.750 | +0.150 | [-0.050, +0.350] | 4/1/5 | 0.375 |
| multi_hop | 12 | 0.250 | 0.667 | +0.417 | [+0.125, +0.708] | 6/1/5 | 0.125 |
| temporal_versioned | 5 | 0.467 | 0.563 | +0.097 | [-0.320, +0.470] | 2/1/2 | 1.000 |
| multi_source | 14 | 0.238 | 0.744 | +0.506 | [+0.304, +0.708] | 10/0/4 | 0.002 |
| tool_use_diagnostic | 5 | 0.467 | 0.783 | +0.317 | [+0.067, +0.567] | 3/0/2 | 0.250 |
| mode=progressive | 36 | 0.306 | 0.699 | +0.393 | [+0.240, +0.544] | 21/2/13 | 0.000 |
| mode=single | 24 | 0.583 | 0.812 | +0.229 | [+0.042, +0.438] | 9/2/13 | 0.065 |
| all | 60 | 0.417 | 0.744 | +0.327 | [+0.206, +0.448] | 30/4/26 | 0.000 |

judge_task: B = agent-per-type vs A = course-s4 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.214 | 1.000 | +0.786 | [+0.571, +1.000] | 11/0/3 | 0.001 |
| multi_document | 10 | 0.100 | 0.900 | +0.800 | [+0.500, +1.000] | 8/0/2 | 0.008 |
| multi_hop | 12 | 0.000 | 0.583 | +0.583 | [+0.333, +0.833] | 7/0/5 | 0.016 |
| temporal_versioned | 5 | 0.200 | 0.600 | +0.400 | [+0.000, +0.800] | 2/0/3 | 0.500 |
| multi_source | 14 | 0.214 | 0.643 | +0.429 | [+0.214, +0.714] | 6/0/8 | 0.031 |
| tool_use_diagnostic | 3 | 0.333 | 0.333 | +0.000 | [+0.000, +0.000] | 0/0/3 | 1.000 |
| mode=progressive | 34 | 0.147 | 0.588 | +0.441 | [+0.294, +0.618] | 15/0/19 | 0.000 |
| mode=single | 24 | 0.167 | 0.958 | +0.792 | [+0.625, +0.958] | 19/0/5 | 0.000 |
| all | 58 | 0.155 | 0.741 | +0.586 | [+0.466, +0.707] | 34/0/24 | 0.000 |

instance_facts: B = agent-per-type vs A = course-s4 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.533 | 0.949 | +0.415 | [+0.219, +0.613] | 10/0/2 | 0.002 |
| temporal_versioned | 5 | 0.636 | 0.964 | +0.328 | [+0.109, +0.547] | 4/0/1 | 0.125 |
| multi_source | 14 | 0.681 | 0.858 | +0.176 | [-0.036, +0.364] | 10/1/3 | 0.012 |
| tool_use_diagnostic | 5 | 0.715 | 0.989 | +0.274 | [+0.085, +0.584] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.630 | 0.921 | +0.291 | [+0.171, +0.406] | 29/1/6 | 0.000 |
| all | 36 | 0.630 | 0.921 | +0.291 | [+0.171, +0.406] | 29/1/6 | 0.000 |

record_evidence: B = agent-per-type vs A = course-s4 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.312 | 1.000 | +0.688 | [+0.465, +0.889] | 11/0/1 | 0.001 |
| temporal_versioned | 5 | 0.400 | 1.000 | +0.600 | [+0.400, +0.800] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.292 | 0.893 | +0.601 | [+0.286, +0.851] | 13/1/0 | 0.002 |
| tool_use_diagnostic | 5 | 0.350 | 1.000 | +0.650 | [+0.250, +1.000] | 4/0/1 | 0.125 |
| mode=progressive | 36 | 0.322 | 0.958 | +0.637 | [+0.479, +0.775] | 33/1/2 | 0.000 |
| all | 36 | 0.322 | 0.958 | +0.637 | [+0.479, +0.775] | 33/1/2 | 0.000 |

doc_evidence: B = agent-per-type vs A = course-s4 (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.214 | 0.857 | +0.643 | [+0.357, +0.857] | 9/0/5 | 0.004 |
| multi_document | 10 | 0.250 | 0.750 | +0.500 | [+0.250, +0.750] | 7/0/3 | 0.016 |
| multi_hop | 12 | 0.375 | 0.667 | +0.292 | [+0.083, +0.500] | 7/1/4 | 0.070 |
| temporal_versioned | 5 | 0.517 | 0.563 | +0.047 | [-0.320, +0.360] | 2/1/2 | 1.000 |
| multi_source | 14 | 0.393 | 0.744 | +0.351 | [+0.125, +0.577] | 9/2/3 | 0.065 |
| tool_use_diagnostic | 5 | 0.383 | 0.783 | +0.400 | [+0.167, +0.633] | 4/0/1 | 0.125 |
| mode=progressive | 36 | 0.403 | 0.699 | +0.296 | [+0.160, +0.428] | 22/4/10 | 0.001 |
| mode=single | 24 | 0.229 | 0.812 | +0.583 | [+0.417, +0.750] | 16/0/8 | 0.000 |
| all | 60 | 0.333 | 0.744 | +0.411 | [+0.295, +0.524] | 38/4/18 | 0.000 |