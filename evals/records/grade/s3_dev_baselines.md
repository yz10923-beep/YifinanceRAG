# S3 dev baselines (no LLM)

Each baseline graded on the 60 dev questions by `yfr grade --system ...` (commit in each
`summary.json`). A cell is the mean over the cases a grader applies to; `-` = applies to none.
verbose-null = BM25's top 10 chunks for the opener pasted as the answer.

| system | rows | n | doc_evidence | record_evidence | negative_found | tools | operations | instance_facts | as_of | clarification | course_task | course_evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | mode=progressive | 36 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| oracle | mode=single | 24 | 1.000 | - | - | 1.000 | - | - | - | 1.000 | 1.000 | 1.000 |
| oracle | all | 60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| null | mode=progressive | 36 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.000 | 0.000 | 0.000 |
| null | mode=single | 24 | 0.000 | - | - | 0.000 | - | - | - | 1.000 | 0.000 | 0.000 |
| null | all | 60 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - | 0.400 | 0.000 | 0.000 |
| wrong | mode=progressive | 36 | 0.025 | 0.000 | 0.000 | 0.748 | 0.243 | 0.224 | 0.000 | 0.981 | 0.000 | 0.000 |
| wrong | mode=single | 24 | 0.000 | - | - | 0.271 | - | - | - | 1.000 | 0.000 | 0.000 |
| wrong | all | 60 | 0.015 | 0.000 | 0.000 | 0.557 | 0.243 | 0.224 | 0.000 | 0.989 | 0.000 | 0.000 |
| rule-only | mode=progressive | 36 | 1.000 | 0.000 | 0.000 | 0.531 | 0.000 | 0.119 | - | 0.000 | 1.000 | 1.000 |
| rule-only | mode=single | 24 | 1.000 | - | - | 1.000 | - | - | - | 1.000 | 1.000 | 1.000 |
| rule-only | all | 60 | 1.000 | 0.000 | 0.000 | 0.719 | 0.000 | 0.119 | - | 0.400 | 1.000 | 1.000 |
| verbose-null-bm25 | mode=progressive | 36 | 0.315 | 0.000 | 0.000 | 0.466 | 0.000 | 0.173 | - | 0.000 | 0.306 | 0.139 |
| verbose-null-bm25 | mode=single | 24 | 0.667 | - | - | 0.938 | - | - | - | 1.000 | 0.625 | 0.583 |
| verbose-null-bm25 | all | 60 | 0.456 | 0.000 | 0.000 | 0.654 | 0.000 | 0.173 | - | 0.400 | 0.433 | 0.317 |
