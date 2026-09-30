# E-001 Equivalent-Data Diagnostic Run 01

Date: `2026-08-04`

Method: `M001 v0.2`

Status: `DIAGNOSTIC PASSED`; formal gate is not applicable until E-000 publishes and
validates the independently extracted official archive.

## Scope

- Dataset: `/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official`
- Pairs: `424/424`
- Checkpoint SHA256:
  `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Primary metric: public GDROS component error, all `512 x 512` pixels, no valid mask
- Diagnostic metric: standard two-dimensional vector EPE
- Exact pool: row-major `[10000:210000]`
- Random seeds: `1103, 2207, 3301, 4409, 5519`
- Accuracy watchdog job: `20260804_013217_6498a01b`, physical GPU 3
- Benchmark watchdog job: `20260804_013231_be2806bf`, physical GPU 4

Both watchdog jobs succeeded. The accuracy run records `formal_run=false`,
`formal_gate_applicable=false`, and `accuracy_gate_pass=null` for every candidate.

## Compression Effect

| Exact-public-pool strategy | AEPE | Delta AEPE | Delta CMR@1 | Vector EPE delta |
| --- | ---: | ---: | ---: | ---: |
| Public LSR, 200k | 1.461230 | - | - | - |
| K64 linspace | 1.466231 | +0.005000 | +0.708 pp | +0.006827 |
| K64 stratified | 1.463490 | +0.002259 | -0.472 pp | +0.001917 |
| K64 random, 5-seed mean | 1.466884 | +0.005654 +/- 0.004379 | -0.283 +/- 0.690 pp | +0.008898 |
| K128 linspace | 1.461851 | +0.000620 | +0.708 pp | +0.000841 |
| K128 stratified | 1.462331 | +0.001100 | +0.236 pp | +0.001263 |
| K128 random, 5-seed mean | 1.463576 | +0.002345 +/- 0.002227 | -0.283 +/- 0.275 pp | +0.004119 |

All K64/K128 exact-pool trials fall inside the numerical accuracy limits proposed in the
method specification on this equivalent materialization. The worst random seed has
Delta AEPE `+0.011975 px` and Delta CMR@1 `-0.943 pp`. This is evidence for sparse affine
sufficiency, not a formal gate result.

The public reference has P95/P99 `3.247386/4.597962 px` and failure rate
`2/424 = 0.471698%`. K128 random has mean P95/P99 `3.246551/4.566851 px` and the same
mean failure rate.

## Coverage Effect

At K128, full-grid-edge has Delta AEPE `+0.002145 px` and Delta CMR@1 `+0.236 pp`
against the 200k reference; full-stratified-center has `+0.001095 px` and `-0.708 pp`.
These means conceal large per-pair cancellation:

- Full-grid-edge K128: 130 pairs regress by more than `0.05 px`; range
  `-0.433` to `+0.783 px`.
- Full-stratified-center K128: 99 pairs regress by more than `0.05 px`; range
  `-0.277` to `+0.600 px`.
- Exact public-stratified K128: one pair regresses by more than `0.05 px`; maximum
  `+0.0595 px`.

Therefore the current diagnostic supports stable compression inside the public pool. It
does not support a general benefit from changing coverage. The two reference failures,
`000225_opt` and `000244_opt`, remain above 5 px under every strategy.

## Protocol And Oracle Checks

- Public generalized 200k parity: maximum dense-flow error `1.22e-4 px`; maximum
  primary pair-error delta `1.91e-6 px`.
- 45 sampling designs are rank 3: 35 exact-pool and 10 full-image diagnostic designs.
- All 45 GT affine oracle variants have AEPE from `1.60e-5` to `3.54e-5 px` and zero
  failures.
- `per_pair.csv` contains exactly `424 x 91 = 38,584` rows with no duplicate or missing
  result/pair combinations.
- K64 per-pair five-seed range reaches `0.452 px`; K128 reduces it to `0.274 px`.
  K128 is the safer default for E-002 planning.

## Resource Result

All measurements use batch 1, FP32, warm-up 5, repeat 30 and CUDA synchronization.

- Generalized 200k, K64 and K128 postprocess latency:
  `0.391/0.392/0.393 ms`; incremental peak is `7.061 MiB` for all three.
- Generalized 200k, K64 and K128 end-to-end latency:
  `81.908/81.924/81.902 ms`; peak allocated is `316.298 MiB` for all three.
- Standalone model forward is `85.901 +/- 5.133 ms`, slower than the end-to-end cases,
  demonstrating run-order and clock noise. Millisecond subtraction is not valid here.

Reducing only final LSR points has no measurable end-to-end latency or memory benefit,
because CorrBlock and all 32 GRU iterations remain unchanged. It must not be presented as
the lightweight method contribution.

## Artifacts And Review

- Accuracy metrics:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/equivalent-diagnostic/run-01/accuracy/metrics.json`
- Per-pair CSV:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/equivalent-diagnostic/run-01/accuracy/per_pair.csv`
- Conditions CSV:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/equivalent-diagnostic/run-01/accuracy/conditions.csv`
- Benchmark JSON:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-001/equivalent-diagnostic/run-01/benchmark/benchmark.json`
- Watchdog reports:
  `/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_013217_6498a01b.json` and
  `/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_013231_be2806bf.json`
- Watchdog logs:
  `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_013217_6498a01b.log`
  and
  `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_013231_be2806bf.log`

Independent code and metrics review found no implementation blocker. Remaining limits are
the non-formal dataset provenance, the benchmark shared resident state, missing semantic
labels for occlusion/speckle/repeated texture/black-border grouping, and the coverage
mean-cancellation noted above.
