# E-009A Targeted Review R1

Conclusion: `APPROVE_WITH_RESULT_FAIL`

## Scope

Reviewed E-009A implementation against `research/experiments/E-009A.md` and `REVIEW_PACKET.md`.

## Findings

- No blocking implementation mismatch found.
- `LearnedRegionTopKRetriever` uses only `source_point/target_point[:, 32:50]`, the raw/structure tail from E-008A observations. Unit test confirms changing frozen GDROS feature dimensions `0:32` does not change learned score.
- `learned_topr_for_observation()` reconstructs S1 matched-region union and applies learned scoring before topR/top64. It does not rank inside frozen S2 top64.
- Training labels are derived from train raw `truth_flow` only; dev labels are loaded after candidate outputs for evaluation.
- Formal resource trace reports max union `1648` and max live score elements `25344`; no hidden `4096x4096` or full-image blind search.
- Frozen S2 parity was recomputed from E-008A observations and matches E-008A-STAGE-DIAG: mean `50.0149`, P10/P50/P90 `26/48/78`, base rate `0.05161%`.

## Residual Risk

- Training uses sampled negatives rather than full-union listwise CE. This is allowed by the spec as sampled/listwise training, but it may under-train hard in-union discrimination.
- Only one formal seed was run because E-009A did not request multi-seed and G1 failed decisively.

Reviewer decision: implementation is valid for interpreting the formal G1 failure.
