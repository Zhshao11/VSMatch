# E-006A Review Packet

Method: `M001 v0.9`

Experiment: `E-006A`

## Read First

- `research/CURRENT_STATE.md`
- `research/METHOD_DELTA.md`
- `research/INNOVATION_CONTRACT.yaml`
- `research/experiments/E-006A.md`

Do not reread full history unless a red flag appears.

## Core Claim

Replace GDROS dense flow state and dense GRU with recursive feedback between sparse cross-modal observations
and multiple low-dimensional affine states. v0.9 realizes this with region-guided multiple basins, confidence-growing
active points, affine-guided candidate regeneration and cross-region hard rejection.

## High-Risk Semantics

- No fixed K128 main path and no artificial Kmax.
- M256 is a finite observation grid, not an active point cap.
- Final target coordinate must be hard selected candidate plus local offset, not top-k expectation.
- Local offset is clipped to `[-4,4] px`; supervision applies only when discrete candidate center is within `8 px`.
- Confidence target is 2px GT endpoint correctness, not self-consensus with current H.
- Active confidence thresholds are fixed to `0.90/0.85/0.80/0.75` for rounds 1..4.
- Leave-region-out validation must exclude the source region being validated.
- Region-balanced weights must prevent one region from dominating final affine.
- No dense flow state, dense GRU, `4096x4096` hidden correlation, RANSAC/hypothesis enumeration, descriptor training or baseline fallback.

## Required Evidence

- Region score trace: `64x64` and top-L=8.
- Candidate source trace for global, region-restricted and affine-local candidates.
- Active/unresolved/rejected counts for all rounds.
- Assertion/trace proving no K128/Kmax clamp.
- B=8 basin states and final basin selection.
- Leave-region-out residuals with excluded region id.
- GT/self inlier ratios at 1/2/5 px.
- Full-image no-mask metric parity.
- Gradient, toy overfit, checkpoint load.
- Synchronized resource benchmark with baseline parity.

## Stop

If P0 fails, do not train. If G1 or G2 fails, write `RESULT_CARD.md` and stop.
