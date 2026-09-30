# E-010A Targeted Review R1

Conclusion: `APPROVE_WITH_RESULT_FAIL`

## Scope

Reviewed E-010A implementation against `research/experiments/E-010A.md` and `REVIEW_PACKET.md`.

## Findings

- No blocking implementation mismatch found.
- `StrongRegionLocalRetriever` reads only `source_point/target_point[:, 32:50]`, the raw/structure/local tail from E-008A observations. Unit test verifies changing GDROS frozen prefix `0:32` leaves scores unchanged.
- Frozen S2 scores and region scores are not used by the learned model or loss. Frozen S2 is recomputed only as baseline parity.
- `score_union_for_region()` scores complete S1 matched-region unions before topR/top64 selection; it is not a verifier after frozen top64.
- Training loss is full-union Gaussian listwise retrieval over the complete S1 union for each selected proposal, plus current-model hard-negative auxiliary.
- Formal train trace records 102336 proposal groups/epoch and 122112348 hard-negative tokens/epoch.
- GT enters only through train labels and post-output dev metrics; candidate outputs are materialized before label/eval join.
- Formal resource trace reports max union `1648`, max live score elements `25344`, no hidden `4096x4096`, and no full-image blind search.

## Residual Risk

- Training uses 64 recoverable proposals per pair rather than every proposal. The per-proposal loss is full-union/listwise, but proposal sampling may still limit learning.
- Inputs are the E-008A cached raw/structure/local tail, not freshly loaded image patches. This respects the no-GDROS-feature constraint, but may underuse the spec's optional raw 512x512 patch capacity.

Reviewer decision: implementation is valid for interpreting the formal G1 failure.
