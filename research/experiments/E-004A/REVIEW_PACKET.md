# E-004A P0 Targeted Review Packet

Review status: `APPROVE AFTER TARGETED FIX REREVIEW`

## Frozen scope

- Stage: proposal/cache/oracle/evaluator only. Heads, training, replacement and resource code are forbidden.
- Frozen experiment SHA: `6110948722ff841257a938e59ac8e7682666491944b57c75eadfefb42b9ccdc9`.
- BIND-001: uniform-region-K128 uses only `(2,2)` and `(6,6)` in every region.
- Source manifest SHA: `2e1b64ad31c41a455bebd68b92948c9c97e17a6c02cadab1eedbec16c3bbe801`.

## Review targets

Remote module:
`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/e004a`

Local method and mapping:

- `/Users/shihao/Documents/图像匹配之二/research/experiments/E-004A.md`
- `/Users/shihao/Documents/图像匹配之二/research/experiments/E-004A/P0_IMPLEMENTATION_MAP.md`
- `/Users/shihao/Documents/图像匹配之二/research/experiments/E-004A/RUN_MANIFEST.json`

Evidence:

- Unit tests: 18 passed, one real-checkpoint test skipped by default.
- Real-checkpoint no-grad integration: 1 passed on `cuda:2`.
- Smoke build: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01/gates/p0-smoke-build.json`.
- Exact online/cache differential: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-004A/run-01/gates/p0-smoke-parity.json`.

## Required inspection

Read the actual implementation and call paths. Verify:

1. Proposal order `(ry,rx,dy,dx)`, region identity and BIND-001 exactness.
2. M256 exact cosine top8 uses the frozen stable tie-break in four `256x1024` chunks, never `4096x4096`.
3. Optical source to SAR target; raw flow is target-source; pixel/token scales and in-bounds/radius are exact.
4. Candidate cache has no GT/datum payload and labels are separate.
5. Oracle allocation is recoverable-first, nearest-error, fixed-ID; K128 and every region 1..3.
6. GT-nearest top8 correspondence, >=12, >=3 quadrants, FP32 ordinary LS, no ridge, and `cond(X^T X)<=20`.
7. Dense 512x512 no-mask AEPE/CMR/PairLocalReady metrics match the frozen public evaluator.
8. Fixed-public and uniform controls use the same feature/candidate/label/evaluator protocol.
9. Gate aggregation and invalid-pair denominators implement all P0 thresholds exactly.
10. No OS424/test access, consensus/hypothesis, replacement, head, training, GT leakage or baseline fallback.

Return `APPROVE` or `CHANGES_REQUIRED`, with file/line evidence. A negative P0 result cannot be accepted until this review is `APPROVE`.

## Round-1 fixes for targeted rereview

- Removed the transitive import of `e003a.evaluate_selector`; `evaluator.py` now contains only the frozen
  analytical affine-flow and full-image metric equations. A runtime import test rejects consensus,
  triple-bank and reranker modules.
- Candidate and label files now bind pair/split, current source manifest, checkpoint, optical/SAR SHA and
  truth-flow SHA. Schema, proposal identity, finiteness and GT/datum separation are validated both when a
  cache is reused and before evaluation.
- The final evaluator output computes and embeds SHA256 for every candidate/label file and the complete
  404-pair cache manifest.
- Both 2-pair smoke artifacts were rebuilt under source manifest
  `2e1b64ad31c41a455bebd68b92948c9c97e17a6c02cadab1eedbec16c3bbe801`.
