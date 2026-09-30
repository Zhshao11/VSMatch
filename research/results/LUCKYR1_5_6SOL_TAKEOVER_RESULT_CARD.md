# GDROS SparseGeo LuckyR1 5.6sol takeover

Status: `CLEAN_F2_TRAINING_RAM_CACHE_COMPLETE_EXPLORATORY_A_STOPPED`

## Completed in this takeover

- Re-audited the current OS best, E075D diagnosis, LuckyR1 design, parent lineage cards, repaired foundations, constructor smoke, dedicated watchdog, and orchestrator state.
- Invalidated the historical-E046-contaminated repaired F2/F3 path for clean model-unseen claims.
- Added split-safe foundation evaluation that persists prediction tensors and a manifest before a separate GT metric join.
- Added source/checkpoint audit, clean B/C training, B/C reducer, M4 raw control, M4 reducer, real matcher-facing LuckyR1 A/B/C, method reducer, and untouched official-404 finalizer.
- Added per-attempt JSON and Markdown result cards, signal/error capture, pair-level AEPE and CMR@1/2/5/10/16, and checkpoint cost gates.
- Preserved the current-model exploratory track separately and retained protected E075A-M4 checkpoint SHA `55d6e6cd7e849c5c25723004ef6b7a7c79b8cd55e2665951257cafb07b86d81b` read-only.

## Model-facing methods

- A: real CorrBlock candidates are converted to an explicit absolute-target belief with unmatched mass; local geometric compatibility and coordinate-consistent previous belief reweight the correlation before BasicUpdateBlock.
- B: within-modality feature relations are compared after target offsets are transformed by the current flow Jacobian; the resulting evidence reweights the real CorrBlock tensor before BasicUpdateBlock.
- C: relation evidence and geometric belief are applied jointly in the recurrent matcher.

## Scheduler state

The source/checkpoint audit succeeded first, followed by post-audit G0 job `20260913_194750_4cfcc693`. Real model-facing A/B/C forward/backward smoke job `20260913_195310_57d5340e` also succeeded: all three paths produced finite `[1,2,512,512]` flows and finite nonzero evidence gradients through the actual CorrBlock and BasicUpdateBlock.

An external user's processes later appeared on GPU0/GPU4/GPU6/GPU7 after or during watchdog allocation. LuckyR1 yielded its affected jobs without touching the external processes. F2/F3 interruption cards retain last steps 4078/3818; exploratory B/C retain steps 2/5. Replacement jobs are queued with `candidate_gpus=null`.

Current-model exploratory A completed 500 real training steps and a full 404-pair evaluation. Its AEPE is `6.1013`; CMR@1/2/5/10/16 is `25.7426/54.7030/71.2871/79.9505/86.8812%`. This is substantially below the protected M4 control, so A is stopped by the checkpoint cost gate. All 404 prediction tensors and their manifest were persisted before the separate GT join.

The historical M4/B/C/E073/E072/E046 checkpoints are now copied and SHA-verified in `/dev/shm` by stdlib-only CPU job `20260913_205044_6d4869f7`. Future exploratory B/C constructors use this cache and preserve source provenance.

Clean F2 job `20260913_203510_3576ef8f` has priority20, was assigned uncontended GPU1, and entered real training with a `TRAIN_STEP_1` card. Clean F3 is queued. Cached exploratory B/C retries are also queued at lower priority. Persistent orchestrator PID `2631514` retains GPU process-group collision monitoring, refreshes removed tasks to their actual terminal status, and records per-job process/CPU-tick liveness telemetry without cancelling live work.

Long GPU training nodes allow up to eight attempts with checkpoint resume, so external process races do not exhaust the clean DAG. CPU reducers remain limited to two attempts.

Foundation retries now use an atomically replaced rolling checkpoint every 6000 steps plus a sidecar pointer. If interruption occurs after a formal checkpoint is written but before its 201-pair evaluation finishes, the next attempt completes that missing evaluation before resuming training.

The old shared-watchdog F2 attempt is retained as failed and the old queued F3 attempt is retained as cancelled. Neither can enter the formal lineage. WHU has not been started.
