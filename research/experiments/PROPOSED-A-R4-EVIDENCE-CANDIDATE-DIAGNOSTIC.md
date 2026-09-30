# Proposed A R4: Evidence and Candidate-Generation Diagnostic

Revision: `1`

Status: `IMPLEMENTATION_READY / CPU_DIAGNOSTIC_ONLY / NO_TRAINING`

This document supersedes no previous result. It is a new diagnostic stage
after `PROPOSED-A-D1-CORRECTED-R3`. Existing R3 evidence, manifests, run
directories, checkpoints, and result cards are read-only.

## Purpose

R3 showed that an affine residual family can represent useful corrections, but
the current generated pool did not contain them. R4 isolates where the loss
occurs:

1. whether the stored cross-modal features contain a usable response at the
   true target;
2. whether the existing edge endpoints can witness a useful correction when GT
   only selects an existing edge;
3. whether candidate generation fails because displacement K-means does not
   represent one affine explanation.

R4 does not train a selector, change the backbone, or make a generalization
claim about the OS long tail.

## Frozen Inputs and Isolation

- Reuse the immutable R2 evidence bundles under
  `PROPOSED-A-D1-EVIDENCE-REBUILD-R2/evidence/{os,whu}`.
- Use the released GDROS checkpoint and feature arrays already recorded in the
  R2 manifest.
- Do not regenerate labels, alter feature arrays, or use test-set selection.
- The 32 pairs overlap checkpoint training and are mechanism diagnostics only.
- Do not overwrite `PROPOSED-A-D1-CORRECTED-R3` or any E075/E080/E084-E087
  artifact.
- No optimizer, backward pass, checkpoint update, or training is permitted.

## Implementation

The CPU diagnostic implementation is:

`image_matching_part2_migrated/.codex_tmp/proposed_a_r4_diagnostics.py`

It computes:

- cosine score: L2-normalized source/target dot product;
- GDROS-native score: `source dot target / sqrt(D)`;
- full-score true-target rank and top-k target-neighborhood recall;
- GT-assisted witness that chooses an endpoint from the existing edge set and
  then fits geometry, without replacing any endpoint by a GT coordinate;
- displacement-K-means pool;
- deterministic seed plus per-source geometry-conditioned soft-assignment pool.

The generated pool is an exploratory diagnostic, not a deployed matcher. It
keeps the same coarse flow, feature-grid coordinate convention, 12-mode budget,
and 16,384-edge budget used in R3.

## Required Runs

Run once for each frozen bundle:

```bash
python .codex_tmp/proposed_a_r4_diagnostics.py \
  --bundle <run-root>/evidence/os \
  --topk 4 8 16 \
  --max-edges 16384 \
  --output <new-run-root>/r4_os.json

python .codex_tmp/proposed_a_r4_diagnostics.py \
  --bundle <run-root>/evidence/whu \
  --topk 4 8 16 \
  --max-edges 16384 \
  --output <new-run-root>/r4_whu.json
```

The experiment engineer may use the pinned project environment and CPU
workers. GPU training is explicitly disallowed in this stage. If a GPU is
used only for an equivalent inference-free tensor operation, record it as
resource metadata and do not call it a training result.

## Required Metrics

For both score definitions and each `K in {4,8,16}` report:

- true-target score rank mean/P50/P90;
- nearest target-grid distance mean/P90;
- recall of a target within 1 and 2 feature-grid cells;
- candidate edge count and source/target spatial coverage;
- GT-assisted existing-edge witness AEPE/CMR@1/2/5;
- K-means pool oracle AEPE/CMR@1/2/5;
- geometry-conditioned pool oracle AEPE/CMR@1/2/5;
- mode count and duplicate-mode count;
- raw and post-LSR output metrics when an output operator is applicable.

The score control is diagnostic. If native dot improves the evidence metrics,
the result must be labeled `EVIDENCE_DEFINITION_CORRECTION`, not a method gain.

## Interpretation and Stop Rules

### Case A: Native or cosine full correlation has high true-target recall, but
the current edge policy does not retain it

The bottleneck is candidate sparsification/edge organization. Proceed only to
an edge-support design with fixed budgets. Do not train a selector on the old
pool.

### Case B: Existing edges contain useful endpoints under the GT witness, but
the K-means pool oracle does not

The bottleneck is geometric hypothesis generation. Compare the
geometry-conditioned pool and, only if its oracle is higher, design a held-out
no-GT mode score. Do not use GT to produce deployed endpoints.

### Case C: Full correlation and the existing edges both lack target-near
responses

The frozen feature evidence is insufficient. Stop this geometry-only branch
and prepare a separately reviewed, small frozen-backbone correspondence-cost
learning experiment. Do not start large-scale pretraining automatically.

### Case D: Neither generator improves the existing pool

Record `R4_GENERATOR_NO_GAIN`; do not train a selector or change the backbone
based on this diagnostic.

## Required Artifacts

Create a new run root, for example:

`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/PROPOSED-A-R4-EVIDENCE-CANDIDATE-DIAG/run-01`

Required files:

- `RESULT_CARD.md` and `RESULT_CARD.json`;
- `RUN_MANIFEST.json`;
- `source_sha256.json`;
- `r4_os.json` and `r4_whu.json`;
- `score_comparison.json`;
- `candidate_pool_manifest.json`;
- `resource_report.json`;
- stdout/stderr logs;
- reviewer conclusion covering protocol, no-GT separation, and candidate
  generation implementation.

Any exception, OOM, external kill, queue cancellation, or partial run must
still produce a `RESULT_CARD` identifying the cause and the last completed
stage. Existing queues must remain unchanged.

## Handoff Boundary

The output of R4 is a decision about the next mechanism branch. It is not
authorization for training. Training can be considered only after a generated
pool demonstrates improved reachability and the evidence source is not
ambiguous.
