# E-006A Result Card

Status: `COMPLETED / P0_FAIL / TRAINING_NOT_STARTED`

Method: `M001 v0.9 Region-Guided Confidence-Growing Sparse Affine Feedback`

## Decision

The corrected formal 404-pair P0 capacity gate failed. Per the frozen experiment order, E-006A stopped before training, G1, and G2. No rescue version, descriptor tuning, proposal-grid expansion, or E-006B was started.

The first watchdog result (`p0-gate.json`) is superseded and excluded from interpretation because independent review found that it used a GT-fitted single basin and did not execute the frozen B8/512-token region path. The corrected `p0-gate-r2.json` is the only formal result.

## Frozen Identity

- CURRENT_STATE: `b02349dbc8dbf8342cd88e6ddc3266bff00e85aeadee679dfb2c1edb55d481e8`
- METHOD_SPEC: `b794d83a9f56cfb91ade47c1b42227042575f7fd0760b85716db8ad7dffe3ea7`
- METHOD_DELTA: `a76ae60b661e6c815d506e75244f7e0c796356c98db89a5b42a324fde145bb1f`
- INNOVATION_CONTRACT: `3fdd63e141bdbeffc0931fd10cd8d4ed3fdf468537fd4d0f1053999b7a3c9d24`
- E-006A: `d6afdb39b2c8938dc01758db7b3950ccc2958fd54d674b4c3cfeecfedb46210f`
- REVIEW_PACKET: `e6dbce50112c43958350fcedb6c15a7d1b61891223e29a477e78fccd3a966864`
- Split manifest: `e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3`
- GDROS checkpoint: `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- E-006A source manifest: `a26f9619ad395bdbc1d45b3ec09ae256d603512042681259efdc8aac4e486d2a`

## Automatic And Review Gates

- Automatic tests: `12/12 PASS`.
- Method/coordinate/candidate review: `APPROVE`.
- Protocol/provenance/budget/evaluator review: `APPROVE`.
- Live maximum score tensor: `262,144` elements.
- Live traces: global `4 x [1,256,1024]`, region `8 x [1,32,512]`, local `64 x [1,32,81]`.
- No hidden `4096x4096`, dense state/GRU, fixed K128/Kmax, inference GT, or baseline fallback was found.
- Full-image evaluator passed a nontrivial differential against frozen E-005A metrics.

## Formal P0

| Metric | Result | Gate |
|---|---:|---:|
| Oracle AEPE | 0.190768 px | PASS, <=1.6 |
| PairLocalReady | 100.0% | PASS, >=99% |
| Valid pairs | 100.0% | PASS, >=99% |
| Mean 2 px-correct proposals | 66.210 / 256 | **FAIL**, >=140 |
| Exact formal dev set | 404 pairs | PASS |
| Live budget / no hidden N2 | PASS | PASS |

Correct-proposal distribution: min `14`, P10 `33.3`, P50 `64`, P90 `97.7`, max `153`; only `0.743%` of pairs reached 140. Mean in-bounds proposals were `215.32`, and only `30.16%` of in-bounds proposals were 2 px-correct. AEPE P50/P95 were `0.17096/0.34950 px`.

## Interpretation Boundary

The sparse candidates that are correct are geometrically sufficient: oracle affine AEPE, PLR, condition, and spatial validity all pass. The failure is candidate-regeneration coverage: the frozen global top8 + region top8 + B8 local top4 space exposes too few 2 px-correct proposal endpoints for the required confidence-growing training protocol.

This result rejects the frozen E-006A P0 candidate-space realization. It does not by itself reject the broader sparse-observation/low-dimensional-affine-feedback claim or prove that recurrence and robust rejection cannot work with a different, separately approved observation mechanism.

## Artifacts

- Formal gate: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A/run-01/p0-gate-r2.json` (`9b5afebf...64d5`)
- Per-pair rows: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A/run-01/p0-pairs-r2.jsonl` (`70dfb864...8c01`)
- Watchdog job: `20260806_211153_4d64e53a`
- Watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260806_211153_4d64e53a.log` (`2e254198...090`)
- Watchdog job JSON: `/mnt/sda/zhushihao/experiment_watchdog/state/jobs/20260806_211153_4d64e53a.json`
- Method review: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A/run-01/reviews/method-review.md`
- Protocol review: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A/run-01/reviews/protocol-review.md`
- Source: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e006a`
- Run manifest: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A/run-01/RUN_MANIFEST.json`
