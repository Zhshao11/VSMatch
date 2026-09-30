# LuckyR1 recovery and result consolidation

Status: `CURRENT_MODEL_ABC_COST_GATE_COMPLETE_CLEAN_DAG_RUNNING`

## Result

The dedicated watchdog remains healthy. The persistent orchestrator was repaired and restarted as PID `2324644`; advancing heartbeats were verified at `09:36:57`, `09:37:13`, `09:37:45`, and `10:31:47`. Its previous crash was an uncaught `FileNotFoundError` while writing a missing `collisions/` directory. Startup now creates that directory.

Historical-M4 v3 G0 job `20260915_093624_331c5869` succeeded. The E074-B F2/F3 evidence graph was repaired without unfreezing base parameters. B passed `6/6` and C passed `21/21` trainable gradient tensors as present, finite, and nonzero. Peak reserved memory was approximately `14.8 GB`.

## Current-model exploratory 404

These are pair-level metrics on the historical official-404 view, not clean model-unseen results.

| Arm | Status | AEPE | CMR@1 | CMR@2 | CMR@5 | CMR@10 | CMR@16 |
|---|---|---:|---:|---:|---:|---:|---:|
| Protected M4 | control | 2.3613 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |
| A step500 | cost stop | 6.1013 | 25.7426% | 54.7030% | 71.2871% | 79.9505% | 86.8812% |
| B step500 | no-signal stop | 2.3617415 | 56.6832% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |
| C step500 | no-signal stop | 2.3615115 | 56.4356% | 78.9604% | 88.8614% | 94.8020% | 97.0297% |

B and C each have 404 prediction tensors, 404 manifest rows, and 404 metric-join rows. Predictions were materialized before the GT join. Neither arm has a positive continuation signal, so both were stopped after step500.

## Clean lineage

The simultaneous B/C rc137 event remains unattributed. Both jobs reached recorded step117000 and ended at `2026-09-14 16:31:54+08:00`. Accessible kernel/system journals contain no OOM, GPU Xid, reboot, or shutdown evidence. The shared timestamp supports a common event, but not an actor claim.

Cost control selected B step80k and C step40k under the predeclared clean-201 ordering; no resume to 180k was scheduled. `bc_reduce` job `20260915_093640_13a1739d` succeeded. Clean M4 raw job `20260915_093657_b59ea46b` is running and had reached step6000 at the last check. A clean-v3 full-M4 G0 now gates the subsequent clean Lucky A/B/C jobs.

The protected M4 SHA remains `55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b`. WHU remains stopped.
