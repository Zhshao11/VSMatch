# E-008A Result Interpretation Review

Verdict: `CHANGES_REQUIRED`

This is a read-only review of the final result interpretation. No experiment source or result artifact was
modified.

## Verified

- The local `p0-gate.json` and `p0-pairs.jsonl` are byte-identical to the formal remote artifacts. The per-pair
  file contains exactly 404 records.
- The result-card P0 table, coverage metrics, source-track ablations, and resource values agree with
  `p0-gate.json`. Independent aggregation of the 404 pair records confirms the reported means, quantiles,
  candidate-weighted base rates, hard-oracle/nearest-oracle AEPE, valid/PairLocalReady rate, and in-bounds rate.
- The frozen E-006A comparison values and baseline-card SHA256 match
  `research/CANDIDATE_GENERATION_BASELINE.md`.
- `P0 FAIL` is correct: eight quality/coverage checks fail and only the no-hidden-N2, no-blind-search, and exact
  404-pair checks pass. The frozen protocol therefore requires `P1 NOT_RUN_P0_FAIL`; the result card observes
  that boundary and does not report unexecuted P1 metrics.
- Both seed checkpoint hashes, parameter count, 341,327 samples, 48,761 positives, 1,599/1,607 used train pairs,
  training times, and train peak allocation agree with the two formal model reports.
- The watchdog report records all 404 pairs and return code 2 after emitting the formal `P0 FAIL`; this is the
  evaluator's expected method-gate exit, not an infrastructure failure.
- Provenance and audit claims are consistent with the gate, run manifest, candidate traces, and approved
  candidate/evaluator/protocol reviews. The reported source SHA, split SHA, frozen GDROS checkpoint, selector
  checkpoint, cache counts, search-budget maxima, and artifact paths are internally consistent.
- The source ablation numbers support that region-only has slightly higher candidate purity than the frozen old
  aggregate base rate, while descriptor and patch/structure tracks are lower and the deduplicated union remains
  below baseline. They also support the broader conclusion that proposal quantity alone did not solve candidate
  coverage. The card correctly states that P1 selectability was not evaluated.

## Required Correction

The sentence stating that "matched-region search misses most correct target locations" is more specific than the
available diagnostics support. P0 measures the final generated candidate sets, but does not report GT target-region
membership or stage-wise recall before and after matched-region union, within-region top-64 pooling, per-source
top-4 selection, and final deduplication. It therefore cannot localize the misses to region matching itself.

Replace the first and last causal statements in the Source Ablation interpretation with:

> The dynamic generator produces many more source proposals than the old M256 generator, but its final generated
> candidate sets contain far fewer 2px-correct proposals. Region-only candidates have slightly higher 2px purity
> than the old aggregate baseline, while the descriptor and patch/structure tracks are lower and the full union
> remains below baseline. The observed failure is a candidate-coverage/region-to-point-retrieval failure despite
> ample proposal count; this P0 diagnostic does not isolate whether the misses arise from target-region selection,
> within-region pool construction, or final candidate ranking/deduplication. This experiment does not evaluate P1
> selectability because P0 did not pass.

No metric, gate, resource, provenance, or stop-decision correction is otherwise required.
