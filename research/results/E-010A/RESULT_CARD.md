# E-010A Result Card

Status: `COMPLETED / STRONG_RETRIEVER_FAIL`

## Identity

- Method: `M001 v1.3`
- Experiment: `E-010A Strong Cross-modal Region-local Retriever`
- Run: `run-01`
- Remote run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-010A/run-01`
- Watchdog job: `20260807_214707_aa6436f3`, `succeeded`, GPU 2
- Implementation source SHA256: `24a99d9d351e8fe00d55330bdd36010063381952864130a37ea358bb240d3719`

## Protocol

- Fixed E-008A S0 source proposals and S1 matched-region union.
- Frozen S2 used only for parity/baseline, not teacher or training target.
- Learned retriever uses raw/structure/patch tail only; GDROS frozen feature prefix and frozen scores are excluded.
- Main loss is full S1-union Gaussian listwise retrieval plus current-model hard-negative auxiliary.
- GT is used only for train labels and post-output evaluation.
- No OS424/confirm, no hidden 4096x4096, no full-image blind search, no fallback.

## Training

- Train split: 1607 records, 4 epochs, 1599 used pairs/epoch.
- Proposal groups: 102336/epoch.
- Hard-negative tokens: 122112348/epoch.
- Loss: `7.3208 -> 7.1563`.
- Parameters: 134914; descriptor dim: 96.
- Time: 2713.64 s; peak train allocated: 542887424 bytes.

## G1 Main Result

Frozen S2 top64 baseline:

- Mean correct: `50.0149`
- P10/P50/P90: `26 / 48 / 78`
- Base rate 2px: `0.05161%`
- Nearest P50/P90: `15.490 / 75.309 px`

Strong top64:

- Mean correct: `37.0767`
- P10/P50/P90: `16 / 36 / 60`
- Base rate 2px: `0.03826%`
- Nearest P50/P90: `25.331 / 99.603 px`
- Oracle AEPE / PLR: `48.541 px / 90.59%`

G1 checks: all strong-vs-frozen checks failed except `formal_exact_404`.

## TopR Curve

| R | mean correct | P10/P50/P90 | base 2px | nearest P50/P90 | oracle AEPE | PLR |
|---|---:|---|---:|---|---:|---:|
| 4 | 5.087 | 1 / 5 / 10 | 0.08398% | 99.858 / 280.404 | 490.480 | 4.21% |
| 8 | 9.040 | 3 / 8 / 17 | 0.07462% | 73.413 / 224.274 | 384.157 | 25.00% |
| 16 | 15.052 | 5 / 14 / 27 | 0.06213% | 52.848 / 173.056 | 228.428 | 55.45% |
| 32 | 23.661 | 10 / 22 / 38 | 0.04883% | 37.265 / 130.413 | 109.392 | 78.71% |
| 64 | 37.077 | 16 / 36 / 60 | 0.03826% | 25.331 / 99.603 | 48.541 | 90.59% |
| 128 | 58.131 | 29 / 55 / 89 | 0.02999% | 16.376 / 78.946 | 23.138 | 95.54% |

Note: top128 improved over E-009A top128 but cannot rescue top64 failure by protocol.

## Resource

- Eval p50/p95: `322.16 / 388.22 ms`
- E-009A eval p50/p95 reference: `244.33 / 292.81 ms`
- Max target union tokens: `1648`
- Max live score elements: `25344`
- Peak eval allocated bytes: `13053440`

## Artifacts

- Summary: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-010A/run-01/formal/topr_summary.json`
- Pair metrics: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-010A/run-01/formal/topr_pair_metrics.jsonl`
- Train report: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-010A/run-01/formal/train.json`
- Checkpoint: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-010A/run-01/models/seed-20260819.pt`
- Watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260807_214707_aa6436f3.log`
- Watchdog report: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260807_214707_aa6436f3.json`
- Reviewer: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-010A/run-01/review/review-r1.md`

## Decision

Stop per protocol. Do not launch E-010B, E-009B, E-008B, rescue, OS424/confirm, or full sparse affine feedback.
