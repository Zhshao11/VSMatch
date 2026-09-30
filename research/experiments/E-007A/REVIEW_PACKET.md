# E-007A Review Packet

Experiment: `E-007A`

Method: `M001 v1.0 / Observation-Evidence Sparse Affine Feedback`

## Read First

- `research/CURRENT_STATE.md`
- `research/METHOD_DELTA.md`
- `research/INNOVATION_CONTRACT.yaml`
- `research/experiments/E-007A.md`
- `research/results/E-006A-SELECT-DIAG/RESULT_CARD.md`

Do not reread full history unless a red flag appears.

## Core Question

Can stronger patch, structure, or descriptor evidence make E-006A candidates selectable at useful purity?

## Frozen Baseline

- E-006A-SELECT-DIAG main D2 topK-32 precision: `2.259%`.
- E-006A-SELECT-DIAG main D2 affine AEPE: `63.516 px`.
- E-006A-SELECT-DIAG main D2 CMR@2: `0%`.

## High-Risk Checks

- Reuse E-006A corrected candidate generator; do not interpret invalid `p0-gate.json`.
- Candidate feature dumps must be materialized before label join.
- raw truth_flow may create train labels and metrics only; it must not enter inference features.
- OS424, confirm and datum labels are forbidden.
- GDROS backbone and FeatureTransformer stay frozen no-grad.
- Descriptor adapter may be trained, but only on train split and only for candidate scoring; no dense retrieval.
- No proposal-grid expansion, dense flow, dense GRU, hidden `4096x4096`, RANSAC, IRLS or baseline fallback.

## Required Evidence

- Patch extraction coordinate and padding tests.
- Structure channel determinism tests.
- Descriptor positive/negative mining and split-isolation tests.
- P/S/D/all-evidence tables for AUROC/AP/ECE, topK precision/recall, AEPE/CMR/PLR/valid.
- Resource timings and memory.
- Ablations: score-only, raw-patch-only, feature-patch-only, structure-only, descriptor-only, all-evidence.
- Interpretation category: `STRONG_OBSERVATION_GAIN`, `USEFUL_OBSERVATION_GAIN`,
  `WEAK_OBSERVATION_GAIN`, or `NO_USEFUL_OBSERVATION_GAIN`.
