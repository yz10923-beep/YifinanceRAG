# S4d: the two A/Bs on dev (2026-10-23)

Run by Yi on the VM, each against the S4c run (`agent-single`), trial 0, Haiku judge.
Both use hybrid-4b + Haiku rerank and the prompt rendered from the tool set (S4c's
prompt unchanged, hash pinned). Paired tables drop cases the judge left incomplete.

DeepSeek bill for the day (usage export, all hours off-peak): 487 requests, 654,951
cache-miss / 1,743,744 cache-hit input and 237,735 output tokens, $0.2461 for both runs
plus the 5-question smoke run. Our price table reproduces each billed hour exactly
(`tests/unit/test_pricing.py`).

## Grade tables

agent-single-noask on dev: 60 cases
| type | n | doc_evidence | record_evidence | negative_found | id_recall | tools | operations | instance_facts | as_of | clarification | cited_grounded | cited_gold | course_task | course_evidence | judge_assertions | judge_forbidden_clear | judge_task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_retrieval | 14 | 1.000 | - | - | 1.000 | 1.000 | - | - | - | 1.000 | 1.000 | 0.078 | 0.786 | 1.000 | 0.786 | 1.000 | 0.786 |
| multi_document | 10 | 0.850 | - | - | 0.850 | 1.000 | - | - | - | 1.000 | 1.000 | 0.135 | 0.600 | 0.700 | 0.800 | 1.000 | 0.600 |
| multi_hop | 12 | 0.708 | 0.000 | 0.000 | 0.284 | 0.575 | 0.042 | 0.293 | 0.000 | 0.000 | 1.000 | 0.065 | 0.417 | 0.417 | 0.409 | 1.000 | 0.182 |
| temporal_versioned | 5 | 0.803 | 0.000 | 0.000 | 0.438 | 0.520 | 0.000 | 0.277 | 0.000 | 0.000 | 1.000 | 0.118 | 0.600 | 0.400 | 0.400 | 1.000 | 0.200 |
| multi_source | 14 | 0.774 | 0.036 | 0.000 | 0.347 | 0.588 | 0.060 | 0.320 | 0.000 | 0.000 | 1.000 | 0.098 | 0.500 | 0.571 | 0.462 | 1.000 | 0.231 |
| tool_use_diagnostic | 5 | 0.950 | 0.000 | 0.000 | 0.509 | 0.620 | 0.000 | 0.290 | - | 0.000 | 1.000 | 0.165 | 0.400 | 0.800 | 0.800 | 1.000 | 0.800 |
| mode=progressive | 36 | 0.781 | 0.014 | 0.000 | 0.361 | 0.579 | 0.037 | 0.301 | 0.000 | 0.000 | 1.000 | 0.099 | 0.472 | 0.528 | 0.485 | 1.000 | 0.294 |
| mode=single | 24 | 0.938 | - | - | 0.938 | 1.000 | - | - | - | 1.000 | 1.000 | 0.102 | 0.708 | 0.875 | 0.792 | 1.000 | 0.708 |
| all | 60 | 0.843 | 0.014 | 0.000 | 0.592 | 0.747 | 0.037 | 0.301 | 0.000 | 0.400 | 1.000 | 0.100 | 0.567 | 0.667 | 0.612 | 1.000 | 0.466 |
judge claude-haiku-5-5: 60 calls, 0 cached, 2 incomplete, 2 unverified quotes, 120726 in / 22946 out tokens, $0.0235

agent-per-type on dev: 60 answers, stops {'submitted': 60}; 255 model calls (20 cached), 321392 in / 123964 out tokens, unknown (no DeepSeek price recorded)
| type | n | doc_evidence | record_evidence | negative_found | id_recall | tools | operations | instance_facts | as_of | clarification | cited_grounded | cited_gold | course_task | course_evidence | judge_assertions | judge_forbidden_clear | judge_task |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| simple_retrieval | 14 | 0.857 | - | - | 0.857 | 0.857 | - | - | - | 1.000 | 1.000 | 0.060 | 0.929 | 0.857 | 1.000 | 1.000 | 1.000 |
| multi_document | 10 | 0.750 | - | - | 0.750 | 0.950 | - | - | - | 1.000 | 1.000 | 0.099 | 0.600 | 0.600 | 0.950 | 1.000 | 0.900 |
| multi_hop | 12 | 0.667 | 1.000 | 1.000 | 0.864 | 0.958 | 0.972 | 0.949 | 0.872 | 0.756 | 1.000 | 0.221 | 0.417 | 0.500 | 0.792 | 1.000 | 0.583 |
| temporal_versioned | 5 | 0.563 | 1.000 | 1.000 | 0.739 | 1.000 | 0.933 | 0.964 | 0.900 | 0.600 | 1.000 | 0.227 | 0.200 | 0.200 | 0.800 | 1.000 | 0.600 |
| multi_source | 14 | 0.744 | 0.893 | 0.800 | 0.812 | 0.917 | 0.929 | 0.858 | 0.764 | 0.679 | 1.000 | 0.234 | 0.571 | 0.500 | 0.750 | 1.000 | 0.643 |
| tool_use_diagnostic | 5 | 0.783 | 1.000 | 1.000 | 0.885 | 0.850 | 1.000 | 0.989 | 0.865 | 0.733 | 1.000 | 0.234 | 0.800 | 0.400 | 0.667 | 1.000 | 0.333 |
| mode=progressive | 36 | 0.699 | 0.958 | 0.909 | 0.830 | 0.933 | 0.954 | 0.921 | 0.835 | 0.701 | 1.000 | 0.229 | 0.500 | 0.444 | 0.765 | 1.000 | 0.588 |
| mode=single | 24 | 0.812 | - | - | 0.812 | 0.896 | - | - | - | 1.000 | 1.000 | 0.076 | 0.792 | 0.750 | 0.979 | 1.000 | 0.958 |
| all | 60 | 0.744 | 0.958 | 0.909 | 0.823 | 0.918 | 0.954 | 0.921 | 0.835 | 0.821 | 1.000 | 0.166 | 0.617 | 0.567 | 0.853 | 1.000 | 0.741 |
judge claude-haiku-5-5: 60 calls, 0 cached, 2 incomplete, 0 unverified quotes, 129184 in / 17161 out tokens, $0.0215
## Paired against agent-single (S4c)

judge_task: B = agent-single-noask vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.929 | 0.786 | -0.143 | [-0.357, +0.000] | 0/2/12 | 0.500 |
| multi_document | 10 | 0.700 | 0.600 | -0.100 | [-0.400, +0.200] | 1/2/7 | 1.000 |
| multi_hop | 11 | 0.818 | 0.182 | -0.636 | [-0.909, -0.364] | 0/7/4 | 0.016 |
| temporal_versioned | 5 | 0.400 | 0.200 | -0.200 | [-0.600, +0.000] | 0/1/4 | 1.000 |
| multi_source | 13 | 0.385 | 0.231 | -0.154 | [-0.462, +0.154] | 1/3/9 | 0.625 |
| tool_use_diagnostic | 4 | 0.750 | 0.750 | +0.000 | [+0.000, +0.000] | 0/0/4 | 1.000 |
| mode=progressive | 33 | 0.576 | 0.273 | -0.303 | [-0.485, -0.121] | 1/11/21 | 0.006 |
| mode=single | 24 | 0.833 | 0.708 | -0.125 | [-0.292, +0.042] | 1/4/19 | 0.375 |
| all | 57 | 0.684 | 0.456 | -0.228 | [-0.351, -0.105] | 2/15/40 | 0.002 |

instance_facts: B = agent-single-noask vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.951 | 0.293 | -0.659 | [-0.732, -0.582] | 0/12/0 | 0.000 |
| temporal_versioned | 5 | 0.950 | 0.277 | -0.672 | [-0.748, -0.597] | 0/5/0 | 0.062 |
| multi_source | 14 | 0.890 | 0.320 | -0.570 | [-0.685, -0.457] | 0/14/0 | 0.000 |
| tool_use_diagnostic | 5 | 1.000 | 0.290 | -0.710 | [-0.800, -0.639] | 0/5/0 | 0.062 |
| mode=progressive | 36 | 0.934 | 0.301 | -0.633 | [-0.687, -0.575] | 0/36/0 | 0.000 |
| all | 36 | 0.934 | 0.301 | -0.633 | [-0.687, -0.575] | 0/36/0 | 0.000 |

record_evidence: B = agent-single-noask vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] | 0/12/0 | 0.000 |
| temporal_versioned | 5 | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] | 0/5/0 | 0.062 |
| multi_source | 14 | 0.929 | 0.036 | -0.893 | [-1.000, -0.714] | 0/13/1 | 0.000 |
| tool_use_diagnostic | 5 | 1.000 | 0.000 | -1.000 | [-1.000, -1.000] | 0/5/0 | 0.062 |
| mode=progressive | 36 | 0.972 | 0.014 | -0.958 | [-1.000, -0.889] | 0/35/1 | 0.000 |
| all | 36 | 0.972 | 0.014 | -0.958 | [-1.000, -0.889] | 0/35/1 | 0.000 |

doc_evidence: B = agent-single-noask vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.857 | 1.000 | +0.143 | [+0.000, +0.357] | 2/0/12 | 0.500 |
| multi_document | 10 | 0.800 | 0.850 | +0.050 | [-0.100, +0.200] | 2/1/7 | 1.000 |
| multi_hop | 12 | 0.708 | 0.708 | +0.000 | [-0.125, +0.125] | 1/1/10 | 1.000 |
| temporal_versioned | 5 | 0.657 | 0.803 | +0.147 | [+0.000, +0.307] | 2/0/3 | 0.500 |
| multi_source | 14 | 0.613 | 0.774 | +0.161 | [-0.018, +0.357] | 5/1/8 | 0.219 |
| tool_use_diagnostic | 5 | 0.667 | 0.950 | +0.283 | [+0.067, +0.517] | 3/0/2 | 0.250 |
| mode=progressive | 36 | 0.658 | 0.781 | +0.122 | [+0.025, +0.222] | 11/2/23 | 0.022 |
| mode=single | 24 | 0.833 | 0.938 | +0.104 | [-0.021, +0.250] | 4/1/19 | 0.375 |
| all | 60 | 0.728 | 0.843 | +0.115 | [+0.040, +0.194] | 15/3/42 | 0.008 |

judge_task: B = agent-per-type vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.929 | 1.000 | +0.071 | [+0.000, +0.214] | 1/0/13 | 1.000 |
| multi_document | 10 | 0.700 | 0.900 | +0.200 | [+0.000, +0.500] | 2/0/8 | 0.500 |
| multi_hop | 11 | 0.818 | 0.545 | -0.273 | [-0.545, +0.000] | 0/3/8 | 0.250 |
| temporal_versioned | 5 | 0.400 | 0.600 | +0.200 | [+0.000, +0.600] | 1/0/4 | 1.000 |
| multi_source | 14 | 0.357 | 0.643 | +0.286 | [+0.071, +0.500] | 4/0/10 | 0.125 |
| tool_use_diagnostic | 2 | 0.500 | 0.500 | +0.000 | [+0.000, +0.000] | 0/0/2 | 1.000 |
| mode=progressive | 32 | 0.531 | 0.594 | +0.062 | [-0.094, +0.219] | 5/3/24 | 0.727 |
| mode=single | 24 | 0.833 | 0.958 | +0.125 | [+0.000, +0.250] | 3/0/21 | 0.250 |
| all | 56 | 0.661 | 0.750 | +0.089 | [-0.018, +0.196] | 8/3/45 | 0.227 |

instance_facts: B = agent-per-type vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.951 | 0.949 | -0.003 | [-0.025, +0.018] | 2/2/8 | 1.000 |
| temporal_versioned | 5 | 0.950 | 0.964 | +0.014 | [+0.000, +0.043] | 1/0/4 | 1.000 |
| multi_source | 14 | 0.890 | 0.858 | -0.032 | [-0.218, +0.131] | 3/2/9 | 1.000 |
| tool_use_diagnostic | 5 | 1.000 | 0.989 | -0.011 | [-0.033, +0.000] | 0/1/4 | 1.000 |
| mode=progressive | 36 | 0.934 | 0.921 | -0.013 | [-0.087, +0.050] | 6/5/25 | 1.000 |
| all | 36 | 0.934 | 0.921 | -0.013 | [-0.087, +0.050] | 6/5/25 | 1.000 |

record_evidence: B = agent-per-type vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 0/0/12 | 1.000 |
| temporal_versioned | 5 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| multi_source | 14 | 0.929 | 0.893 | -0.036 | [-0.250, +0.179] | 1/2/11 | 1.000 |
| tool_use_diagnostic | 5 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| mode=progressive | 36 | 0.972 | 0.958 | -0.014 | [-0.097, +0.069] | 1/2/33 | 1.000 |
| all | 36 | 0.972 | 0.958 | -0.014 | [-0.097, +0.069] | 1/2/33 | 1.000 |

doc_evidence: B = agent-per-type vs A = agent-single (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.857 | 0.857 | +0.000 | [-0.286, +0.286] | 2/2/10 | 1.000 |
| multi_document | 10 | 0.800 | 0.750 | -0.050 | [-0.200, +0.100] | 1/2/7 | 1.000 |
| multi_hop | 12 | 0.708 | 0.667 | -0.042 | [-0.250, +0.167] | 1/3/8 | 0.625 |
| temporal_versioned | 5 | 0.657 | 0.563 | -0.093 | [-0.267, +0.080] | 1/2/2 | 1.000 |
| multi_source | 14 | 0.613 | 0.744 | +0.131 | [-0.071, +0.339] | 6/2/6 | 0.289 |
| tool_use_diagnostic | 5 | 0.667 | 0.783 | +0.117 | [-0.233, +0.500] | 2/1/2 | 1.000 |
| mode=progressive | 36 | 0.658 | 0.699 | +0.040 | [-0.080, +0.165] | 10/8/18 | 0.815 |
| mode=single | 24 | 0.833 | 0.812 | -0.021 | [-0.188, +0.167] | 3/4/17 | 1.000 |
| all | 60 | 0.728 | 0.744 | +0.016 | [-0.087, +0.119] | 13/12/35 | 1.000 |
