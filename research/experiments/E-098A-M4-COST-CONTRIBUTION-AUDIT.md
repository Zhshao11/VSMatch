# E098A: locate M4 cost and existing CMR@1 evidence

2026-11-16 user question: after removing four LSR fits saves only tens of
milliseconds, where does M4's extra latency come from, and which parts
explain its CMR@1 improvement over GDROS?

No training, checkpoint edits or architecture changes. Two complementary
audits are kept separate from full-424 timing benchmark E096A.

## Cached branch evidence re-audit

`remote_stage/os/e098a_reaudit_cached_branches.py` verifies E093A preflight,
strict-load/checkpoint identity metadata, all four 32-step manifests and
1,696 saved B/C/AVG/M4 prediction hashes. It compares every saved M4 output
against accepted E096A, requiring exact equality on all 424 pairs, then
recomputes all branch metrics from saved predictions with the same official
GT and component/vector definitions.

E093A's historical join failed its older E092A regression check. That failure
card is preserved. The present audit uses a new directory and the accepted
fixed-backend E096A control; it is not a silent historical-status rewrite or
a claim of fresh branch inference. All four branch variants use effective
parameters extracted from the protected M4 step80K checkpoint. AVG fuses
raw B/C equally before the terminal LSR.

## Focused component timing

Ten fixed uniformly spaced OS pair indices (`linspace(0,423,10,dtype=int)`),
three warmups per model, three measured repeats each; same shared GPU5,
FP32, batch=1, 512x512, each matcher 32 iterations, both TF32 flags off.
Uninstrumented synchronized wall times are recorded separately, then
CUDA-event ranges measure instrumented spans. Model order alternates across
samples/repeats. Predictions must reproduce the accepted E096A archive and
instrumentation must change no output elements.

Fine audit: `remote_stage/os/e098a_component_cost.py`. It attaches ranges
to recurrent submodules and correlation samplers, among others, and records
actual module counts. Its many event insertions perturb latency; results
are retained as diagnostic evidence, not authoritative component percentages.

Coarse audit: `remote_stage/os/e098b_coarse_component_cost.py`. It brackets
whole matchers and a small set of nested feature/DINO/adapter/context/corr
construction modules. Iterative matching and related operations are the
inclusive matcher span minus those nested spans. This group includes lookup,
GRU/motion/flow updates, mask/upsampling, coordinate/image arithmetic and
other matcher-local overhead. Final gates and LSR are separate. Categories
are nonoverlapping after subtraction; their sum plus residual equals the
whole instrumented stream span. This reduces per-iteration event overhead.

DINO is bracketed at `forward_features`, because this frontend bypasses
`Module.__call__` and ordinary forward hooks would miss it. Instrumentation
is removed/restored after measurement. Sources are hashed before/after.

CUDA events measure elapsed stream spans, including scheduling and CPU
gaps; they are not sums of active kernel time. Percentages refer to each
instrumented run's span total and must not be multiplied by E096A wall
latencies. Plain-vs-instrumented phases also experience shared-GPU drift.

## Interpretation limits

- E097A fixed-weight ablations measure gate-input substitution, not retrained
  architectural necessity. CMR@1 and CMR@5 are cumulative thresholds, not
  measurements of error changes exclusively inside the >5 px subset.
- E094A removed intermediate gate/feature computations together with LSR;
  its full speedup cannot be attributed to LSR alone.
- Branch outputs can localize high precision to the existing C matching
  package and conditional benefit of B/C fusion. They cannot isolate DINO,
  dual correlations, distillation or training-budget effects.
- Historical 404-pair vector-EPE ablations remain distinct from current
  official 424-pair component-error tables.

Remote root:
`/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-098A-M4-COST-CONTRIBUTION-20261116/`

The cache audit, fine profile and coarse profile use distinct subdirectories.
Compact final evidence belongs in `research/results/E098A_M4_COST_CONTRIBUTION_20261116/`;
full local records/source snapshots in `outputs/m4_cost_contribution_20261116/`.

## Final status: COMPLETE

Both profile protocols and the new cache audit completed. All 1,696 cached
predictions passed hash checks, cached M4 matched E096A bitwise on all 424
pairs, and instrumentation caused zero output change. Independent offline
aggregate/count/parity verification passed.

Coarse M4 diagnostic shares: matching and related 60.57%, DINO 17.78%,
feature frontends 12.23%, all LSR 5.45%. The first three explain 88.7% of
extra diagnostic elapsed time versus GDROS. Cached C-only CMR@1 is 69.5755%
versus full M4 72.4057%; this localizes precision to the matching package
without isolating individual training/module causes. No defaults changed.
Full interpretation: `results/E098A_M4_COST_CONTRIBUTION_20261116/REPORT.md`.
