# E-008A Review Packet

Experiment: `E-008A`

Method: `M001 v1.1 / Region-to-Point Cross-modal Candidate Generator`

## Read First

- `research/CURRENT_STATE.md`
- `research/METHOD_DELTA.md`
- `research/INNOVATION_CONTRACT.yaml`
- `research/CANDIDATE_GENERATION_BASELINE.md`
- `research/experiments/E-008A.md`

Do not reread full history unless a red flag appears.

## Core Question

Does the new region-to-point candidate generator produce more and more selectable 2px-correct candidates than
the frozen E-006A corrected candidate generator?

## Frozen Baseline

- mean 2px-correct proposals: `66.210/256`
- P10/P50/P90 correct proposals: `33/64/98`
- 2px candidate base rate: `0.1343%`
- P@32: `2.259%`
- AP: `0.01386`
- best observed selector affine AEPE: `36.0761 px`
- oracle affine AEPE: `0.190768 px`

## High-Risk Checks

- This experiment must change candidate generation, not merely add a verifier on E-006A candidates.
- No full-image blind target search as the main path.
- No hidden `4096x4096`, dense flow state, dense GRU, RANSAC, IRLS or fallback.
- Feature dumps must be materialized before label join.
- raw truth_flow may create train labels and metrics only; it must not enter inference features.
- OS424, confirm and datum labels are forbidden.
- Dynamic proposal count must be reported; batch padding must not become a semantic cap.

## Required Evidence

- Region scale and matched target-region trace.
- Dynamic source proposal count distribution.
- Candidate source type counts and labels.
- Old baseline parity or frozen SHA check.
- P0 coverage metrics and P1 selectability metrics.
- Oracle and no-GT affine metrics.
- Resource timing and memory.
- Reviewer conclusion and result category.
