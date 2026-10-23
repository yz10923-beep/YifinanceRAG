# S4c: the first agent run on dev (2026-10-23)

Run by Yi on the VM: `yfr answer --system agent` (single `search_documents` tool over
hybrid-4b + Haiku rerank, ask_user on, max 16 steps, trial 0, git 4ed1bd4), graded with the
Haiku judge. 60 dev questions, all stopped at `submit_answer`.

Usage: 254 model calls (17 cached), 240,393 input / 116,807 output / 813,440 cache-read
tokens; DeepSeek cost unknown (price not recorded yet). Rerank 127 calls, $0.0515.
Judge $0.021, 2 incomplete, 5 unverified quotes. 4.5 steps a question (max 8), 21 s.

Agent vs the one-call bar (Haiku judge; instance facts after the thousands-separator fix):

| metric | one-call | agent |
|---|---:|---:|
| judge_task | 0.400 | 0.672 |
| judge_task, single-turn | 0.708 | 0.833 |
| judge_task, progressive | 0.194 | 0.559 |
| instance_facts | 0.208 | 0.934 |
| record_evidence | 0.000 | 0.972 |
| course_task | 0.383 | 0.600 |

Paired judge_task (VM, Yi, 2026-10-23: `yfr compare
evals/results/grade/20261020T095244Z_onecall-hybrid-4b+rerank_dev
evals/results/grade/20261023T005613Z_agent-single_dev`; 58 cases, the 2 the judge left
incomplete in either run drop out):

judge_task: B = agent-single vs A = onecall-hybrid-4b+rerank (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.857 | 0.929 | +0.071 | [-0.143, +0.286] | 2/1/11 | 1.000 |
| multi_document | 10 | 0.500 | 0.700 | +0.200 | [-0.200, +0.600] | 3/1/6 | 0.625 |
| multi_hop | 11 | 0.182 | 0.818 | +0.636 | [+0.364, +0.909] | 7/0/4 | 0.016 |
| temporal_versioned | 5 | 0.200 | 0.400 | +0.200 | [-0.400, +0.800] | 2/1/2 | 1.000 |
| multi_source | 14 | 0.286 | 0.357 | +0.071 | [-0.143, +0.286] | 2/1/11 | 1.000 |
| tool_use_diagnostic | 4 | 0.000 | 0.750 | +0.750 | [+0.250, +1.000] | 3/0/1 | 0.250 |
| mode=progressive | 34 | 0.206 | 0.559 | +0.353 | [+0.147, +0.529] | 14/2/18 | 0.004 |
| mode=single | 24 | 0.708 | 0.833 | +0.125 | [-0.083, +0.333] | 5/2/17 | 0.453 |
| all | 58 | 0.414 | 0.672 | +0.259 | [+0.103, +0.397] | 19/4/35 | 0.003 |

The deterministic tables below were recomputed in the container from the two runs'
records; the VM's output is identical.

instance_facts: B = agent-single vs A = onecall (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.203 | 0.951 | +0.748 | [+0.656, +0.830] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.258 | 0.950 | +0.691 | [+0.606, +0.777] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.213 | 0.890 | +0.676 | [+0.554, +0.786] | 14/0/0 | 0.000 |
| tool_use_diagnostic | 5 | 0.158 | 1.000 | +0.842 | [+0.747, +0.942] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.208 | 0.934 | +0.725 | [+0.663, +0.783] | 36/0/0 | 0.000 |
| all | 36 | 0.208 | 0.934 | +0.725 | [+0.663, +0.783] | 36/0/0 | 0.000 |

record_evidence: B = agent-single vs A = onecall (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| multi_hop | 12 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 12/0/0 | 0.000 |
| temporal_versioned | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| multi_source | 14 | 0.000 | 0.929 | +0.929 | [+0.786, +1.000] | 13/0/1 | 0.000 |
| tool_use_diagnostic | 5 | 0.000 | 1.000 | +1.000 | [+1.000, +1.000] | 5/0/0 | 0.062 |
| mode=progressive | 36 | 0.000 | 0.972 | +0.972 | [+0.917, +1.000] | 35/0/1 | 0.000 |
| all | 36 | 0.000 | 0.972 | +0.972 | [+0.917, +1.000] | 35/0/1 | 0.000 |

doc_evidence: B = agent-single vs A = onecall (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.786 | 0.857 | +0.071 | [-0.143, +0.286] | 2/1/11 | 1.000 |
| multi_document | 10 | 0.750 | 0.800 | +0.050 | [-0.100, +0.200] | 2/1/7 | 1.000 |
| multi_hop | 12 | 0.708 | 0.708 | +0.000 | [-0.208, +0.167] | 2/1/9 | 1.000 |
| temporal_versioned | 5 | 0.663 | 0.657 | -0.007 | [-0.227, +0.260] | 1/2/2 | 1.000 |
| multi_source | 14 | 0.702 | 0.613 | -0.089 | [-0.304, +0.143] | 2/5/7 | 0.453 |
| tool_use_diagnostic | 5 | 0.533 | 0.667 | +0.133 | [-0.133, +0.350] | 3/1/1 | 0.625 |
| mode=progressive | 36 | 0.675 | 0.658 | -0.017 | [-0.142, +0.104] | 8/9/19 | 1.000 |
| mode=single | 24 | 0.771 | 0.833 | +0.062 | [-0.083, +0.229] | 4/2/18 | 0.688 |
| all | 60 | 0.714 | 0.728 | +0.015 | [-0.082, +0.114] | 12/11/37 | 1.000 |

course_task: B = agent-single vs A = onecall (B wins / losses / ties per case)
| type | n | A | B | B-A | 95% CI | W/L/T | sign p |
|---|---:|---:|---:|---:|---|---|---:|
| simple_retrieval | 14 | 0.714 | 0.857 | +0.143 | [-0.143, +0.429] | 3/1/10 | 0.625 |
| multi_document | 10 | 0.500 | 0.500 | +0.000 | [-0.300, +0.300] | 1/1/8 | 1.000 |
| multi_hop | 12 | 0.250 | 0.583 | +0.333 | [+0.083, +0.583] | 4/0/8 | 0.125 |
| temporal_versioned | 5 | 0.200 | 0.200 | +0.000 | [+0.000, +0.000] | 0/0/5 | 1.000 |
| multi_source | 14 | 0.214 | 0.571 | +0.357 | [+0.143, +0.643] | 5/0/9 | 0.062 |
| tool_use_diagnostic | 5 | 0.200 | 0.600 | +0.400 | [+0.000, +0.800] | 2/0/3 | 0.500 |
| mode=progressive | 36 | 0.222 | 0.528 | +0.306 | [+0.167, +0.472] | 11/0/25 | 0.001 |
| mode=single | 24 | 0.625 | 0.708 | +0.083 | [-0.125, +0.292] | 4/2/18 | 0.688 |
| all | 60 | 0.383 | 0.600 | +0.217 | [+0.100, +0.333] | 15/2/43 | 0.002 |

