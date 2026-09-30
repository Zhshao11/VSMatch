# E-000 Equivalent-Data Precheck

Date: `2026-08-04`

Status: `DIAGNOSTIC PASSED`; this is not the formal direct-archive E-000 run.

## Scope

- Dataset root: `/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official`
- Pairs: `424/424`
- Checkpoint SHA256:
  `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Watchdog job: `20260804_003152_c09566f7`, physical GPU 3
- Batch size: 1, 32 GDROS iterations, FP32
- Evaluation: all `512 x 512` pixels, no valid mask, black borders included

## Results

- Public-code AEPE: `1.4612304443`
- Pair public-error standard deviation: `0.8622916814`
- Public-code legacy variance: `0.7435469438`
- CMR@1/2/3/4/5: `33.7264/81.6038/92.6887/98.1132/99.5283%`
- Pair public-error P50/P95/P99: `1.23038/3.24739/4.59796 px`
- Pair public-error >= 5 px: `2/424 = 0.4717%`
- Wall-clock for the complete 424-pair loop: `45.5331 s`
- Peak allocated CUDA memory: `315.8472 MiB`

The public evaluator labels the primary quantity AEPE, but at batch 1 its
`sum(..., dim=0)` reduces the batch axis. It therefore averages absolute x/y flow-component
errors rather than Euclidean vector EPE. E-001 preserves this public metric as its primary
paper-comparison value and reports mathematical vector EPE separately.

## Artifacts

- Metrics:
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-000/run-01/equivalent-precheck/metrics.json`
- Metrics SHA256:
  `fcafa4d9b077b2b7e88e43e7efe2bfd638f6a4af2b1007961f8f056efb6cc6db`
- Watchdog log:
  `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260804_003152_c09566f7.log`
- Watchdog job record:
  `/mnt/sda/zhushihao/experiment_watchdog/state/jobs/20260804_003152_c09566f7.json`
- Watchdog report:
  `/mnt/sda/zhushihao/experiment_watchdog/reports/20260804_003152_c09566f7.json`

## Remaining E-000 Gate

The official local ZIP is still transferring into remote staging. E-000 remains
`IMPLEMENTING` until the complete archive has the expected size and SHA256, passes
`unzip -tq`, is independently extracted, produces 424 aligned triples, and the formal
watchdog evaluation matches this precheck.
