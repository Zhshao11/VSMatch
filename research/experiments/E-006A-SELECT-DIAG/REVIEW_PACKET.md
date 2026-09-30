# E-006A-SELECT-DIAG Review Packet

Experiment: `E-006A-SELECT-DIAG`

## Read First

- `research/CURRENT_STATE.md`
- `research/experiments/E-006A-SELECT-DIAG.md`
- `research/results/E-006A/RESULT_CARD.md`

Do not reread full history unless a red flag appears.

## Core Question

Can no-GT features select high-purity inliers from E-006A's existing candidate space, or are the correct candidates
too weakly distinguishable?

## High-Risk Checks

- Must reuse corrected E-006A candidate generator semantics; do not interpret invalid `p0-gate.json`.
- No proposal-grid expansion, descriptor training, dense flow, dense GRU, RANSAC, IRLS or baseline fallback.
- Candidate features must be materialized before raw-flow labels are joined.
- raw truth_flow may create labels and metrics only; it must not enter inference features.
- Endpoint labels `<=1/2/5 px` are based only on discrete candidate distance to the raw-flow endpoint; do not
  conjoin with in-bounds. Report in-bounds separately.
- Geometry-condition proxy is an independent feature dimension, bound as dim 26; do not mix determinant into
  basin_score.
- Feature normalization and any learned selector must fit train split only.
- Dev is evaluation only; OS424/confirm are forbidden.

## Required Evidence

- Candidate generator parity and source/checkpoint/split SHA.
- Feature dump schema with explicit no-GT audit.
- Label base rate and endpoint thresholds.
- Separate in-bounds base rate and endpoint-by-in-bounds groups.
- D1 handcrafted selector table.
- D2 learned selector AUROC/AP/ECE/topK precision/recall.
- Affine AEPE/CMR/PLR/valid for topK and threshold strategies.
- Ablations: no region, no basin residual, no local stats, score-only.
- Result interpretation category: `SELECTABLE`, `PARTIALLY_SELECTABLE`, or `NOT_SELECTABLE_WITH_CURRENT_FEATURES`.
- `PARTIALLY_SELECTABLE` binding: compare dev `full-main global topK-32`; D1 is the best hand-crafted
  selector by affine AEPE on that track; E-006A no-GT score proxy is `D1 cosine/topK-32`; D2 must improve
  AUROC or AP by `>=0.03` and reduce affine AEPE by `>=0.25 px`.
