# E-011A-TOP128-GEOM-DIAG RESULT_CARD

Status: **COMPLETED / COMPRESSION_ORACLE_READY**

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A-TOP128-GEOM-DIAG/run-01`

Watchdog:

- Job: `20260810_123618_dcf9fad1`
- Status: `succeeded`, return code `0`
- Log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260810_123618_dcf9fad1.log`
- Report JSON: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_123618_dcf9fad1.json`
- Report text: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260810_123618_dcf9fad1.txt`

Frozen identity:

- Spec SHA256: `b0709e151e1efd83be9af0d8767144457fd0795643a53bbd07e7b3a3feafb582`
- CURRENT_STATE: `0f480f9ea2d8e80a03ca26df57ba019e049d74371f6fc44c3fd1d59f5a830fe9`
- EXPERIMENT_INDEX: `82de2a1ca3dfa98f9d0aa4bb20ac97680baea6c10bffda10f5b243b51fd73721`
- TASKS: `90208c9dd7553a78007a048967d7c1e21ee3e3ddc321ea236612896ebd42763d`
- DECISION_LOG: `09f679a7edbfa8ca1c5e62d705c3e0176c44bd765c0fcec932664e302824feb4`

Parity / isolation:

- E-011A top128 parity: PASS
- Reproduced top128 mean/P10/P50/P90: `68.710 / 34 / 66 / 106`
- Reproduced E-011A top128 oracle AEPE: `14.249`
- Candidate output before label/oracle join: PASS
- Baseline/E-011A output read-only isolation: PASS
- Reviewer: APPROVE

Dominant buckets:

- `COMPRESSION_ORACLE_READY`: `401/404` pairs (`99.26%`)
- `RANKING_TAIL_LIMITED`: `77/404` pairs (`19.06%`)
- `CANDIDATE_NOISE_LIMITED`: `47/404` pairs (`11.63%`)
- `COUNT_LIMITED`: `35/404` pairs (`8.66%`)
- `COVERAGE_LIMITED`: `5/404` pairs (`1.24%`)
- `NON_AFFINE_OR_SOURCE_SET_LIMITED`: `3/404` pairs (`0.74%`)

Oracle summaries:

| Oracle | mean AEPE | P50 / P90 / P95 AEPE | PLR / valid |
|---|---:|---|---|
| all-hard2px-LS | 4.155 | 0.299 / 0.533 / 0.617 | 99.26% / 99.26% |
| rank64-hard2px-LS | 4.258 | 0.376 / 0.743 / 0.882 | 99.26% / 99.26% |
| spatial-balanced-K64 candidate | 4.183 | 0.324 / 0.549 / 0.616 | 99.26% / 99.26% |
| condition-filtered candidate | 4.174 | 0.316 / 0.549 / 0.635 | 99.26% / 99.26% |
| GT-endpoint affine | 3.802 | 0.00004 / 0.00006 / 0.00007 | 99.26% / 99.26% |
| spatial-balanced GT-endpoint | 3.802 | 0.00004 / 0.00006 / 0.00006 | 99.26% / 99.26% |
| condition-filtered GT-endpoint | 3.802 | 0.00004 / 0.00006 / 0.00007 | 99.26% / 99.26% |

Rank distribution of top128 correct proposals:

- top32: `12,561 / 27,759` (`45.25%`)
- top64: `18,535 / 27,759` (`66.77%`)
- top96: `23,425 / 27,759` (`84.39%`)
- top128: `27,759 / 27,759` (`100%`)

Interpretation:

- E-011A top128 is geometrically compressible under GT-guided source/candidate selection for almost all dev pairs.
- The earlier poor E-011A top128 oracle AEPE is not evidence that the source set is non-affine; only `3/404` pairs remain source-set/non-affine limited under GT endpoint oracle.
- The practical barrier is ranking/selection: too many correct correspondences remain in ranks 65-128, and candidate noise affects a smaller but nontrivial subset.
- This supports a next method that keeps top128 evidence but learns/derives a geometry-aware compression or re-ranking mechanism before top64/topK, rather than expanding proposals or abandoning raw patch evidence.

Artifacts:

- Local summary: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A-TOP128-GEOM-DIAG/geom_summary.json`
- Local bucket counts: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A-TOP128-GEOM-DIAG/geom_bucket_counts.json`
- Local reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A-TOP128-GEOM-DIAG/reviewer_conclusion.md`
- Local manifest: `/Users/shihao/Documents/图像匹配之二/research/results/E-011A-TOP128-GEOM-DIAG/RUN_MANIFEST.json`
- Remote pair metrics: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-011A-TOP128-GEOM-DIAG/run-01/formal/geom_pair_metrics.jsonl`
