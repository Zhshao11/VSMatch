# E-003A Result Card

Status: `COMPLETED / G0 PASS / G1 FAIL / G2 NOT RUN`

Method: `M001 v0.6 / Learned Sparse Geometry Selector`

Decision: stop and return to the ACTIVE method task. Do not start E-003B, E-003C, resource G2, descriptor
training, local refinement, budget reduction or a rescue variant.

## Frozen identity

- Source closure SHA: `daaf3fea395180560109d74154bb1078e50958cbaacb337738fe9c1c86518482`
- Formal config SHA: `325e2059da71b9c802670a89b58a20304b014c758da0ea8db1e0b62c152c5737`
- Split manifest SHA: `e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3`
- Cache manifest SHA: `980c2d856653bb18080d53d16ef5bfb5ae0313b40a5e7476072cd98d9eae73da`
- Cache aggregate SHA: `863ad1c0bc49075eef2d3f27e182ca6f5f37f5572eb6e539931295a8454ee930`
- Official GDROS checkpoint SHA: `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Train/dev: `1,607 / 404` pairs; decoded-optical group overlap `0`; exact OS-testing overlap `0`.

## G0

- `16/16` automatic tests PASS, including real-checkpoint frozen/no-grad and NaN/Inf abstention.
- 2-pair online/cache exact differential PASS; 2,011-pair cache validation v2 PASS.
- Two independent final reviews: `APPROVE / APPROVE`.
- Runtime provenance negative test correctly rejected an old unbound checkpoint.
- Observed complexity smoke: four `[1,128,1024]` score chunks totaling `128x4096`; selector `128x8`;
  no hidden N2 materialization.

## Training

| Variant / seed | Best epoch | Best dev grid error | Runtime | Peak allocated | Checkpoint SHA |
|---|---:|---:|---:|---:|---|
| main / 20260805 | 97 | 11.0970 px | 155.28 s | 212.66 MiB | `555e89c8...9e43a` |
| main / 20260806 | 98 | 11.6980 px | 160.98 s | 212.66 MiB | `32b5f4e0...82aa` |
| main / 20260807 | 100 | 11.5722 px | 156.72 s | 212.66 MiB | `8d9d0853...b830e` |
| unary / 20260805 | 74 | 18.4230 px | 129.62 s | 180.77 MiB | `f37294b0...2cfa` |

The three main seeds fail the 8 px development target consistently; the result is not a seed anomaly.
Geometry feedback is useful relative to unary, but it is insufficient.

## G1 Main Result

| Check | Required | Observed | Result |
|---|---:|---:|---|
| AEPE | <=3.0 px | 10.6808 px | FAIL |
| PairLocalReady | >=90% | 67.0792% | FAIL |
| Grid mean | <=8.0 px | 11.0970 px | FAIL |
| Non-abstention coverage | >=90% | 0.7426% | FAIL |
| 90%-coverage risk reduction | >=20% | 8.7197% | FAIL |
| Finite or explicit abstention | required | all pairs | PASS |

Additional main metrics: CMR@1/2/5 `0 / 0 / 12.3762%`; P50/P95/P99
`9.6509 / 22.6478 / 29.3037 px`; AEPE bootstrap 95% CI `[10.1696, 11.2400]`.
Confidence AUROC is `0.7891`, average precision `0.3923`; it contains ranking signal, but the fixed 5 px
threshold rejects almost every pair and 90%-coverage risk falls only from `10.6808` to `9.7495 px`.

## Controls And Audit

| Variant | AEPE | Grid mean | PairLocalReady | CMR@5 |
|---|---:|---:|---:|---:|
| learned-main-T2 | 10.6808 | 11.0970 | 67.08% | 12.38% |
| learned-unary-only | 17.7924 | 18.4230 | 29.95% | 1.49% |
| frozen E-002B C8 | 16.2327 | 17.0134 | 77.23% | 32.92% |
| nonlearned soft-EM | 133.7937 | 142.0526 | 0% | 0% |
| corrected top8 GT oracle | 2.8410 | 2.9589 | 100% | 95.54% |

The original G1 JSON oracle row is invalid because it included `recoverable=false` anchors. A standalone
post-output audit corrected it and received independent `APPROVE`; learned-main gate metrics are unaffected.
Only `34.9242%` of anchors are recoverable on average (median 43/128), yet the corrected oracle reaches
2.84 px. The dominant failure is therefore candidate selection/dustbin discrimination and geometric
consensus, not candidate-set capacity. Main improves AEPE and the C8 long tail, but loses CMR@5 and
PairLocalReady, indicating moderate-error regression toward globally plausible but insufficiently precise
affines rather than reliable inlier selection.

Hard groups include large displacement (`>=64 px`: AEPE `11.4054`), rotation (`>=10 deg`: `12.1766`) and
black-border fraction (`>=10%`: `11.2888`). These are diagnostic only and were not used for selection.

## Resources

G2 was not run because G1 failed. The pre-gate worker smoke cannot be promoted to a formal efficiency result;
it only confirmed the measurement path (selector p50 about `3.10 ms`, slim p50 about `11.09 ms`, slim warm
allocated `281.53 MiB`, same-process speedup about `10.46x`). Selector parameters: `18,692`; reported linear
operations: `21.38M FLOPs` excluding elementwise operations and the 3x3 solves.

## Absolute Artifacts

- Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01`
- G1 JSON: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/evaluation/g1-main-20260805/g1.json`
- Per-pair CSV: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/evaluation/g1-main-20260805/per_pair.csv`
- Corrected oracle: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/audits/corrected-oracle/result.json`
- Source provenance: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/gates/provenance-final-v3.json`
- Cache gate: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/cache-validation-v2/cache_gate.json`
- Reviews: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/reviews`
- Training: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-003A/run-01/training`
- G1 watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260805_150732_15cc8068.log`
- G1 watchdog report: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260805_150732_15cc8068.json`

## Method Decision Needed

The next design should target explicit sparse inlier/dustbin discrimination and stronger cross-anchor
consensus while preserving O(TNK). The current width-64 per-candidate encoder plus one residual update is
too weak when roughly two thirds of anchors are unrecoverable. This is a new method decision; E-003A does
not authorize an automatic rescue experiment.
