# S3 step 6: judge vs Yi's labels on the one-call dev answers (VM, 2026-10-20)

`yfr agree evals/records/grade/s3_labels_dev.tsv evals/results/grade/*onecall*`: 60 answers,
166 items (106 required, 60 forbidden), labelled by Yi blind to the judges. Pairs are
(Yi, judge); kappa = Cohen's.

| judge | level | n | agree | kappa | Yi yes | judge yes |
|---|---|---:|---:|---:|---:|---:|
| Haiku 5.5, trial 0 | required | 106 | 0.670 | 0.361 | 76 | 49 |
| | forbidden | 60 | 0.983 | 0.000 | 1 | 0 |
| | all items | 166 | 0.783 | 0.553 | 77 | 49 |
| | task | 60 | 0.600 | 0.263 | 41 | 23 |
| Haiku 5.5, trial 1 | required | 106 | 0.651 | 0.345 | 76 | 45 |
| | forbidden | 60 | 0.983 | 0.000 | 1 | 0 |
| | all items | 166 | 0.771 | 0.526 | 77 | 45 |
| | task | 60 | 0.617 | 0.317 | 41 | 20 |
| Sonnet 5.5, trial 0 | required | 106 | 0.689 | 0.397 | 76 | 49 |
| | forbidden | 60 | 0.983 | 0.000 | 1 | 0 |
| | all items | 166 | 0.795 | 0.578 | 77 | 49 |
| | task | 60 | 0.633 | 0.324 | 41 | 23 |

Disagreements: Haiku 36 / 38, Sonnet 34; no incomplete verdicts.
