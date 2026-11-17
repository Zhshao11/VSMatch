# E100A: frozen OS streamlined C/student on WHU and UBCv2

Requested 2026-11-16. Evaluate the exact E099 compact C state, without training,
checkpoint selection, threshold tuning or changing the protected full M4.

## Fixed scope

- Same OS-trained C weights as E099; compact SHA256
  `c2d4f79b45ce190df48f52f588e40a7654fa6cce0a32da1bc0c32020b35636ad`.
- One C/F3 matcher; no B or C teachers constructed. Two feature frontends,
  two four-level correlation pyramids, frozen DINO and descriptor adapters.
- 512x512, RGB float32 0..255, optical source/SAR target, batch1, 32 steps,
  both TF32 flags off, no autocast, one terminal official
  `LSR(1,512,512,flow,200000)`, no internal LSR.
- WHU: same author-public 700 pairs used by E092B, archive/input parity.
  Previous GDROS comparator uses the released **WHU** checkpoint.
- UBCv2: same official raw1625 source pairs and existing independent geometry
  seeds 901/902/903 used by E092CR. Previous GDROS comparator uses released
  **OS** weights. These groups are an independent reproduction, not authors'
  screened1447 or exact unpublished transforms/preprocessing.
- Official-code metric convention: full-image per-pair component absolute
  mean, strict per-pair error<threshold CMR. The public RMSE-named field is
  population variance across pair errors; standard vector EPE retained separately.
- Previous accepted GDROS/full-M4 accuracy is reused on identical groups;
  this task requests fresh C inference, not new GDROS timing.

## Execution and evidence

Entry: `remote_stage/os/e100a_streamlined_c_crossdataset.py`.
Stage with E092B/E092CR/E093A and E099 model dependencies in the configured
`.codex_tmp` directory, from the migrated project root. Run `predict`, then
`join`, both with the same `--output-dir` argument. The output directory must
be absent for fresh prediction. Existing input groups are never regenerated.

Each prediction is saved before GT metric joining, with pair IDs, image
hashes, prediction hash, and explicit no-GT flag. Source and compact-state
identity are checked; protected student is strictly loaded then the compact
state is strictly loaded. Dense states/predictions remain outside Git.

`remote_stage/os/verify_e100a_results.py --run RUN --provenance` independently
rehashes all predictions/inputs/GT, scores using the exact public Torch
expression, checks joined statistics, WHU archive and official loader input
parity, and all three UBC generated transform/input audits. With `--group`
it can audit a completed group while others are still running.

Runtime is diagnostic only because zzvr GPUs are shared; this is not a new
latency benchmark. At launch GPU4 was selected with ~30 GiB free and 0%
utilization. No existing jobs were stopped.

Historical training audit is a separate read-only CPU task and does not
change weights or historical files. See
[distillation and training audit](../../docs/C_STUDENT_DISTILLATION_AND_TRAINING.md)
and the result bundle's training audit JSON.

## Status

COMPLETE. All 5,575 fresh predictions and all four independent group audits
passed; final joined cards agree with independent official-formula scoring.
WHU mean 2.7420633925 px, CMR@1/2/5 63.5714/80.8571/89.1429%.
UBCv2 mean over three groups 4.1170854818 ± 0.1596963754 px (ddof=0),
CMR@1/2/5 46.7487/61.7026/72.5949%.
See [complete report](../results/E100A_STREAMLINED_C_WHU_UBCV2_20261116/REPORT.md).
