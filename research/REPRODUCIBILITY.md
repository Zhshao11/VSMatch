# Reproducibility Guide

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
