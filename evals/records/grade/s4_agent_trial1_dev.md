# S4: second agent trial on dev (2026-10-23, VM, Yi)

`yfr answer --system agent --tool-shape per-type --answer-trial 1` (the S5 config, new
DeepSeek replies; retrieval and rerank replies cached, so only the agent varies), graded
with Haiku, paired against trial 0 (`agent-per-type`).

| | trial 0 | trial 1 | B-A (paired) |
|---|---:|---:|---|
| judge_task (58 paired) | 0.741 | 0.690 | -0.052 [-0.17, +0.07], 5/8/45, p 0.58 |
| judge_task single / progressive | 0.958 / 0.588 | 0.875 / 0.559 | -0.083 / -0.029 |
| instance_facts | 0.921 | 0.920 | -0.001 |
| record_evidence | 0.958 | 0.972 | +0.014 |
| doc_evidence | 0.744 | 0.784 | +0.039 |

Trial 1: 272 model calls, 344,348 in / 140,399 out tokens, 59 submitted + 1 text stop;
cost printed as unknown (DeepSeek reports the served model as `deepseek-flash`, which the
price table lacked; fixed). At the billed rates about $0.14 for the run.
