# Reproducibility Guide

The E095A verified inference path is now named **M4-streamlined** (2026-11-16).
Use `remote_stage/os/m4_streamlined.py:M4Streamlined`; its constructor and
forward are inherited unchanged from the frozen `CleanM4` implementation.
The default retains four internal fits and one terminal LSR. Naming, compact
state identity, input/output contract and example integration are in
[`docs/M4_STREAMLINED.md`](../docs/M4_STREAMLINED.md).

## What is reproducible here

This repository preserves method code, deterministic split rules, experiment
DAG definitions, evaluation contracts, and compact result summaries. It does
not include licensed/downloaded datasets, upstream repositories, checkpoints,
prediction tensors, or remote scheduler state.

The current WHU chain is:

1. `e081a_whu_our_native_train.py`: train E081A-M4 from the official
   RAFT-Sintel initialization.
2. `remote_stage/e081a_m4_isomorphic_diagnostic.py`: frozen 700-pair diagnosis.
3. E088A/E089A/E090A/E090B: local-candidate and representation diagnostics.
4. `remote_stage/e090d_recurrent_evidence_injection.py`: inject structured
   9x9 local evidence into the final recurrent updates through an identity-safe
   residual adapter.
5. `remote_stage/e091a_heldout_generalization.py`: train on 200 source groups
   (1600 pairs), evaluate every epoch on 100 disjoint source groups (800 pairs),
   and apply the preregistered dev gate without opening WHU 700-test.

The protected OS chain is preserved under `remote_stage/os/` as E072 -> E073 ->
E074 -> E075A-M4. `remote_stage/os/SOURCE_MANIFEST.json` records the SHA256 of
each exact zzvr source file before the portable environment-root wrapper was
added locally.

## Required external assets

- Upstream GDROS source at `$SPARSEGEO_ROOT/external_repos/E068A_GDROS_official`
  or the equivalent deployment location expected by the E081A wrapper.
- The migrated experiment module directory at
  `$SPARSEGEO_ROOT/image_matching_part2_migrated/.codex_tmp`.
- WHU-OPT-SAR: 5600 training pairs and 700 testing pairs.
- RAFT-Sintel initialization and the E081A-M4 checkpoint. Checkpoint hashes are
  recorded in result cards; weights are intentionally excluded from Git.

## Environment variables

| Variable | Purpose | Historical default |
| --- | --- | --- |
| `SPARSEGEO_ROOT` | Remote workspace containing code, datasets, and runs | `/mnt/sda/zhushihao/VS_Match` |
| `SPARSEGEO_WHU_TRAIN` | WHU training source directory | zzvr WHU training path |
| `SPARSEGEO_WHU_TEST` | WHU testing source directory | zzvr WHU testing path |
| `SPARSEGEO_PYTHON` | Python used by generated watchdog tasks | zzvr xoftr5090 environment |
| `SPARSEGEO_MIGRATED_ROOT` | Working directory used by watchdog tasks | `$SPARSEGEO_ROOT/image_matching_part2_migrated` |

Example:

```bash
export SPARSEGEO_ROOT=/path/to/VS_Match
export SPARSEGEO_WHU_TRAIN=/path/to/WHU/training/whu_opt_sar512
export SPARSEGEO_WHU_TEST=/path/to/WHU/testing/testing
```

## Data contract

Each dataset item resolves to an optical image, a SAR image, a dense two-channel
flow in target-minus-source convention, and a valid-pixel mask. E081A/E090D
operate on 512x512 image pairs. E091A groups the 5600 WHU training pairs into
700 source groups of eight variants, selects 200 groups for training by a
deterministic SHA256 ordering, and preserves the predeclared 100-group dev set.
The train/dev group intersection must be empty.

## Metrics and leakage control

The paragraph below describes the historical vector-EPE workflows. For the
2026-11-15 E092A/B/C-R comparisons, use the separate conventions and full-image
protocol in the following section; do not reuse a historical mask or compare
vector-EPE numbers with GDROS public-component numbers.

For each pair, compute mean dense EPE over valid pixels first. AEPE is the mean
of those pair-level values. `CMR@t` is the percentage of pairs whose pair-level
EPE is below `t` for `t in {1,2,5,10,16}`.

Prediction tensors and manifests must be fully materialized before a separate
GT join computes metrics. GT may supervise training and labeled diagnostic
oracles, but it must not select inference candidates, gates, thresholds, or
test checkpoints. E091A checkpoint selection is fixed as:

```text
CMR@1 -> AEPE -> CMR@2 -> CMR@5
```

## Frozen three-dataset comparison: E092A/B/C-R

These evaluations perform no training or fine-tuning. M4 is always the OS
E075A step80K checkpoint, including on WHU and UBCv2. GDROS uses its official
OS checkpoint on OS/UBCv2 and its official WHU checkpoint on WHU. Checkpoint
hashes and the full M4 dependency chain are checked before inference.

- OS: all 424 pairs of one author-public test group (E092A).
- WHU: all 700 pairs of one author-public test group (E092B).
- UBCv2: all 1625 pairs from the official raw RGB/SAR test archive, with
  independent geometry seeds 901/902/903 (E092C-R). This is not a replay of
  the paper's screened 1447 pairs or its undisclosed realized transforms.

All use 512x512 inputs, 32 iterations for both primary outputs, official LSR
with 200000 points, and the entire image including padded borders, without
a valid-pixel mask. M4 iteration 24 is a preregistered supplement only.
Predictions are produced without loading GT, saved with hash manifests, and
then joined with GT in a separate CPU stage.

Two metrics are retained independently. For a pair, public-component error is
`mean((abs(dx)+abs(dy))/2)`; standard vector EPE is
`mean(sqrt(dx**2+dy**2))`. CMR is the percentage of **pairs** with their
respective pair error strictly `<t`. The public code's field named RMSE calls
`np.var`; result cards retain that variance and also provide the actual
standard deviation. Success-subset AEPE@t averages only pairs below t and
must always be read alongside CMR@t. It is different from the all-pair mean.

Exact entry points and frozen DAGs:

- `remote_stage/os/e092a_official_eval.py`, `e092a_plan.json`
- `remote_stage/os/e092b_whu_eval.py`, `e092b_plan.json`
- `remote_stage/ubcv2/e092cr_build_groups.py`, `e092cr_ubcv2_eval.py`,
  `e092cr_plan.json`

The eval scripts use `SPARSEGEO_ROOT`; place the external assets in the
documented deployment layout and stage the OS dependency modules under
`image_matching_part2_migrated/.codex_tmp` as required by their imports.
The checked-in DAGs preserve the actual zzvr deployment paths and interpreter
locations: adapt those paths for another host before dispatching. A clone
alone does not supply the checkpoints, datasets, upstream GDROS code, or GPU
environment.

UBCv2 generation takes extracted `<raw-root>/rgb/<stem>.tif` and
`<raw-root>/sar/<stem>.tif`; output includes optical/SAR PNGs, Middlebury `.flo`
GT and per-pair affine metadata. See the E092C-R experiment contract for the
frozen log-SAR mapping, origin-centered affine convention, exact parameter
ranges and OpenCV version audit. The raw native registration is assumed;
the synthetic GT describes the additional affine transform.

Compact exported UBCv2 results can be checked without data, weights, NumPy,
Torch or a GPU:

```bash
python research/remote_stage/ubcv2/verify_e092cr_results.py \
  --run-dir research/results/E092CR_UBCV2_RAW1625_PAPER_STYLE_20261115/run \
  --seeds 901 902 903 --protocol --aggregate
```

This independently recomputes CSV statistics and checks the compact manifest
hash/ID chain. It does not re-read the full prediction arrays retained on
zzvr; their integrity checks belong to the separately recorded CPU join.
All three groups and the aggregate card must be complete for this command.

## Running the current held-out workflow

The exact remote DAG is recorded in `remote_stage/luckyr1/lucky_plan.json`.
To append E091A tasks to a copy of that plan:

```bash
python research/remote_stage/e091a_append_plan.py /path/to/lucky_plan.json
```

The stage entry supports:

```bash
python research/remote_stage/e091a_heldout_generalization.py --mode prepare
python research/remote_stage/e091a_heldout_generalization.py --mode train --phase candidate --epoch 1
python research/remote_stage/e091a_heldout_generalization.py --mode infer --phase candidate --epoch 1 --shard 0
python research/remote_stage/e091a_heldout_generalization.py --mode join --phase candidate --epoch 1
python research/remote_stage/e091a_heldout_generalization.py --mode select
```

The full DAG runs four candidate epochs and three recurrent epochs, with four
dev inference shards followed by a GT join after each epoch. GPU jobs must use
watchdog allocation with `candidate_gpus=null`; code sees the assigned device as
logical `cuda:0`.

## Current limitation

E091A did not pass held-out validation. Its best dev checkpoint improved
CMR@1, but the AEPE confidence interval crossed zero and shuffled evidence was
indistinguishable from main evidence. Therefore the repository does not claim
an improved final WHU model, and the WHU 700-test result was not produced.

## E094A: final-only M4 fusion, 129 to 5 LSR calls

The fixed OS step80K checkpoint passed a full 424-pair paired equivalence
check on 2026-11-16. The independent inference function
`remote_stage/os/m4_final_only_lsr.py:forward_final_only` keeps all recurrent
iterations and computes only the requested final B/M4 gates. Its raw return
still needs one terminal official LSR; four internal fits plus that terminal
fit give five calls. This path is for eval, not sequence-supervised training.

Read `experiments/E-094A-M4-FINAL-ONLY-LSR-424.md` for execution conditions,
and `results/E094A_M4_FINAL_ONLY_LSR_424_20261116/REPORT.md` for accuracy,
paired shared-GPU timing, exact output equivalence and historical-cache
differences. The same remote teacher setting is preserved in both paths.
The historical E092A main table and protected checkpoint remain distinct
from this implementation optimization and its explicitly fixed TF32 settings.

Compact statistics can be independently checked without GPU, data or weights:

```bash
python3 research/remote_stage/os/verify_e094a_results.py \
  research/results/E094A_M4_FINAL_ONLY_LSR_424_20261116
```

The verifier accepts either original CSV records or their committed JSON-row
equivalents. Dense predictions and full runtime/source snapshots stay outside
Git, on the experiment host and in the designated outputs directory.

## E095A: remove inference redundancy without changing predictions

`remote_stage/os/m4_clean_deploy.py:CleanM4` extracts the active inference
graph from the strictly loaded protected M4. It removes inactive storage and
discarded computation and conditionally reuses identical frozen DINO and
descriptor-adapter outputs within each image pair. All active recurrent
updates and five official LSR fits remain. Its raw return still requires the
terminal official LSR. This is not the sequence-supervised training path.

The full official OS424 paired test passed on 2026-11-16: raw and final
outputs were bitwise equal for all pairs, and all 848 saved predictions
passed independent SHA256 verification. Final timings, exact source hashes,
per-pair comparisons, coverage, metrics and JSON row records are in
`results/E095A_M4_CLEAN_DEPLOY_424_20261116/`. See its `REPORT.md` and
`experiments/E-095A-M4-CLEAN-DEPLOY-424.md` for input formats, execution,
the shared-GPU timing limitation and the current legacy-construction
dependency of the compact checkpoint.

```bash
python3 research/remote_stage/os/verify_e095a_results.py \
  research/results/E095A_M4_CLEAN_DEPLOY_424_20261116
```

For a new full run, stage `e095a_clean_deploy_eval.py`, `m4_clean_deploy.py`
and the existing E094/E092 helper chain in `.codex_tmp/` on the experiment
host, run from the migrated project root, and choose a fresh output directory.
The final validation run was `E-095A-M4-CLEAN-DEPLOY-424-20261116-r2`.
Large predictions and the 1.392 GB compact state remain outside Git.

## E096A: fresh paired official GDROS versus clean M4

`remote_stage/os/e096a_gdros_clean_m4_eval.py` reruns both models on all
official OS424 pairs in alternating order on one GPU, using the frozen
E092A strict loaders, official GDROS test_mode and E095A CleanM4. It records
actual LSR counts (1 vs 5), checkpoints/source hashes, GPU load snapshots,
per-pair times, saved predictions and post-prediction CPU metric joins.
It checks current M4 predictions against the validated E095A archive.

The 2026-11-16 run completed and passed independent verification of all
848 prediction-file hashes and compact statistics. M4 matched E095A on
424/424 pairs bitwise. See `experiments/E-096A-GDROS-CLEAN-M4-OS424.md`
for settings and `results/E096A_GDROS_CLEAN_M4_OS424_20261116/REPORT.md`
for fresh accuracy/latency and the shared-GPU limits. GDROS is the original
official implementation; no inference cleanup is applied to that baseline.

```bash
python3 research/remote_stage/os/verify_e096a_results.py \
  research/results/E096A_GDROS_CLEAN_M4_OS424_20261116
```

The remote final run is `E-096A-GDROS-CLEAN-M4-OS424-20261116` under the
existing experiments run root. Dense NPY predictions remain on zzvr.

## E097A: fixed-weight LSR slot ablations

`remote_stage/os/m4_lsr_ablation.py:SelectedLSR` wraps the unmodified
CleanM4 LSR callback. Its policies are all slots 1..5, only slot 5, and
only slots 4/5. Skipped fits return raw flow, producing zero residual
channels; they are never counted as executions of official LSR. Terminal
slot 5 stays enabled. This is a gate-input distribution ablation, not an
equivalence optimization, and no weights are retrained.

`remote_stage/os/e097a_lsr_slot_ablation_eval.py` completed all OS424 pairs
for each policy on 2026-11-16, cycling all six timing orders on shared GPU7.
The five-fit control reproduced E096A exactly. Full statistics and all
1,272 dense-prediction hashes passed independent verification. Default
inference remains five fits; see `experiments/E-097A-M4-LSR-SLOT-ABLATION-OS424.md`
for the explicit substitution and `results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116/REPORT.md`
for metrics, timings, output changes and interpretation limits.

```bash
python3 research/remote_stage/os/verify_e097a_results.py \
  research/results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116
```

The remote run is `E-097A-M4-LSR-SLOT-ABLATION-OS424-20261116`. Compact
JSON records are in the research result directory; dense predictions remain
on zzvr outside Git. Add `--verify-predictions` on that host to rehash them.

## E098A: component cost and historical branch cache re-audit

`remote_stage/os/e098a_reaudit_cached_branches.py` validates E093A identities,
four manifests and 1,696 saved B/C/AVG/M4 hashes, requires bitwise M4 equality
against accepted E096A on all 424 pairs, then recomputes GT metrics. It uses
new audit outputs and preserves the old failed E093A join. It does not claim
fresh branch inference.

`remote_stage/os/e098a_component_cost.py` records fine call counts;
`e098b_coarse_component_cost.py` measures fewer, nested CUDA-event ranges
on ten fixed pairs, three warmups and three repeats, alternating methods.
The fine protocol perturbs time; use coarse spans for cost interpretation.
Stream spans include scheduling/CPU gaps. Matching is an inclusive residual
bucket, not isolated GRU time. All profiled predictions match accepted
E096A; instrumentation changes no output values and is restored afterward.

Completed on 2026-11-16. See `experiments/E-098A-M4-COST-CONTRIBUTION-AUDIT.md`
and `results/E098A_M4_COST_CONTRIBUTION_20261116/REPORT.md` for exact scope,
cost shares, cache accuracy, historical status and attribution limits.

```bash
python3 research/remote_stage/os/verify_e098a_results.py \
  research/results/E098A_M4_COST_CONTRIBUTION_20261116
```

The remote root is `E-098A-M4-COST-CONTRIBUTION-20261116` under the existing
experiments run directory. Source snapshots/full local evidence are in
`outputs/m4_cost_contribution_20261116/`; dense caches remain on zzvr.
