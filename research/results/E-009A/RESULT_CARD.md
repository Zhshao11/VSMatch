# E-009A Result Card

Status: `COMPLETED / LEARNED_TOP64_FAIL`

## Identity

- Method: `M001 v1.2`
- Experiment: `E-009A Learned Region-Conditioned Top64 Retriever`
- Run: `run-01`
- Remote run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-009A/run-01`
- Watchdog job: `20260807_194529_905f27db`, `succeeded`, GPU 2
- Implementation source SHA256: `1eb43a36f88c69b661eaef2be3ba8655eef1f21ca78abf0167595395f32e8bf6`

## Protocol

- Fixed E-008A S0 source proposals and S1 matched-region union.
- Replaced only S2 top64 ranker with a trainable raw/structure retriever.
- Learned retriever ignores frozen GDROS feature dimensions and scores before top64.
- GT is used only for train labels and post-output evaluation.
- No OS424/confirm, no full-image blind 4096 search, no hidden 4096x4096 tensor.

## Training

- Train split: 1607 records, 4 epochs, 1599 used pairs/epoch.
- Samples: 153405 positive groups/epoch, 4908960 scored train tokens/epoch.
- Loss: sampled in-union cross entropy, 1 positive + 31 negatives.
- Parameters: 16065.
- Time: 1055.83 s.
- Peak allocated: 29564416 bytes.

## G1 Main Result

Frozen S2 top64 baseline:

- Mean correct: `50.0149`
- P10/P50/P90: `26 / 48 / 78`
- Base rate 2px: `0.05161%`
- Nearest P50/P90: `15.490 / 75.309 px`

Learned top64:

- Mean correct: `34.7450`
- P10/P50/P90: `15 / 33 / 55.7`
- Base rate 2px: `0.03585%`
- Nearest P50/P90: `27.634 / 102.911 px`

G1 checks: all learned-vs-frozen checks failed except `formal_exact_404`.

## TopR Curve

| R | mean correct | P10/P50/P90 | base 2px | nearest P50/P90 |
|---|---:|---|---:|---|
| 4 | 4.619 | 1 / 4 / 9 | 0.07626% | 106.662 / 296.660 |
| 8 | 8.042 | 2 / 7 / 15.7 | 0.06639% | 79.394 / 239.498 |
| 16 | 13.510 | 5 / 12 / 24.7 | 0.05576% | 57.605 / 184.396 |
| 32 | 21.864 | 9 / 21 / 36 | 0.04512% | 40.685 / 138.163 |
| 64 | 34.745 | 15 / 33 / 55.7 | 0.03585% | 27.634 / 102.911 |
| 128 | 55.394 | 26 / 53.5 / 85 | 0.02858% | 17.642 / 79.482 |

Note: top128 mean correct exceeds frozen top64 mean, but E-009A G1 is fixed to learned top64, so this remains FAIL.

## Resource

- Candidate generation p50/p95: `244.33 / 292.81 ms`
- Max target union tokens: `1648`
- Max live score elements: `25344`
- Peak allocated eval bytes: `11097600`

## Artifacts

- Summary: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-009A/run-01/formal/topr_summary.json`
- Pair metrics: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-009A/run-01/formal/topr_pair_metrics.jsonl`
- Train report: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-009A/run-01/formal/train.json`
- Checkpoint: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-009A/run-01/models/seed-20260818.pt`
- Watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260807_194529_905f27db.log`
- Watchdog report: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260807_194529_905f27db.json`
- Reviewer: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-009A/run-01/review/review-r1.md`

## Decision

Stop per protocol. Do not launch E-009B, E-008B, rescue, OS424/confirm, or full sparse affine feedback.
