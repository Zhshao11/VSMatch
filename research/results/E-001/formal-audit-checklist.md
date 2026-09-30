# E-001 Formal Result Audit Checklist

Method: `M001 v0.2`

This checklist was prepared by the independent E-001 implementation reviewer. Any failed hard
gate blocks promotion to E-002.

## Run identity and artifacts

- Accuracy and benchmark watchdog jobs both succeed without traceback, OOM, or skipped samples.
- Accuracy contains metrics, run config, per-pair CSV, conditions CSV, command, environment, and
  baseline dependency diff.
- Benchmark contains benchmark JSON/config, command, environment, and dependency diff.
- Method spec is `M001 v0.2` and checkpoint SHA is
  `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`.
- Reviewed implementation hashes remain evaluator `a16fe758...a226e`, benchmark
  `3dbcb88f...0473`, and tests `39e00d83...b3d0`.

## Official data binding

- `formal_run=true`, `testing`, `512x512`, `max_samples=null`, and exactly 424 pairs.
- Arguments, loader binding, published root, extracted root, and validated root all resolve to
  `/mnt/sda/zhushihao/VS_Match/datasets/extracted/gdros_official_e000_archive`.
- Provenance is the root `E000_PROVENANCE.json`, not a diagnostic record.
- Dataset binding passes and loader opt/SAR/flow paths run exactly from pair 000000 to 000423.
- Provenance validation requires and passes dataset-root binding, archive rehash, published-root
  rehash, and archive-member rehash.
- Archive size/SHA/member counts remain `1,973,635,350`, `e9c29d64...ad81218`, 2126 entries,
  2120 files, and 6 directories.
- All 1272 evaluation inputs match under the pinned GDROS loader-content semantics. Raw-byte
  matches are optical/SAR/FLO `0/424/424`, so this must not be described as all file bytes being
  identical.
- Accuracy and benchmark post-run manifests equal their initial published manifest.

## Accuracy protocol and parity

- `formal_gate_applicable=true`; evaluated and dataset pair counts both equal 424.
- Reference pool is row-major `[10000:210000]`; evaluation uses all 512x512 pixels without a
  valid mask, including black borders.
- GT is never used for selection; vector EPE and all oracle rows remain diagnostic.
- Full-public-pool parity is at most `0.002 px`; expected dense maximum is about `1.22e-4 px` and
  pair-error maximum about `1.91e-6 px`.
- Results contain 91 conditions: one reference, 45 predicted, and 45 GT oracle.
- Conditions contain 45 rank-3 designs; per-pair CSV contains `424*91=38,584` data rows with no
  duplicate or missing pair/result combination.
- Expected baseline is AEPE about `1.461230482`, CMR@1/2/5
  `33.7264/81.6038/99.5283%`, P95/P99 about `3.247386/4.597962 px`, and failure rate
  `0.471698%`.

## Exact-pool decision gate

- There are exactly 14 K64/K128 candidates: linspace, stratified, and five fixed random seeds at
  each K. Every candidate is non-GT, `predicted_sparse`, and `exact-public-pool`.
- No `full_*` or `gt_oracle__*` result appears in the decision list.
- Independently recompute `AEPE delta <= 0.05 px` and `CMR@1 delta >= -1.0 pp`.
- At least one candidate must pass. Reference expectations are K64 stratified
  `+0.002259 px/-0.472 pp` and K128 stratified `+0.001100 px/+0.236 pp`.
- Random seeds are exactly `1103,2207,3301,4409,5519`.
- Expected random AEPE delta mean/std is about `0.005654/0.004379 px` at K64 and
  `0.002345/0.002227 px` at K128. Expected CMR@1 ranges are about `1.887/0.708 pp`.

## Coverage and tail audit

- Coverage effects remain a separate ten-row diagnostic and never enter the decision gate.
- At K128, expected aggregate full-grid versus exact-stratified delta is about
  `+0.001044 px/0 pp`; full-stratified-center is about `-0.000005 px/-0.943 pp`.
- Recompute pairwise cancellation. Expected counts for delta above `+0.05` and below `-0.05` are
  130/138 for full-grid K128, 99/107 for full-stratified-center K128, and 1/0 for exact
  stratified K128.
- Oracle results remain `gt_affine_oracle`/`gt-oracle-diagnostic`, have near-zero error and zero
  failure, and never count toward passage.
- Recheck P95, P99, and failure rate for primary candidates. For every result, failure rate must
  equal `100 - CMR@5` under the strict pair-error threshold.

## Benchmark interpretation

- Formal provenance, binding, and post-run validation pass in benchmark output.
- FP32, batch 1, 5 warm-ups, 30 repeats, and CUDA synchronization before and after each repeat.
- Pure K comparison uses generalized-public-200k versus public-stratified K64/K128 only.
- End-to-end comparison uses cases that all return a CPU dense flow. Original LSR's CPU return and
  different generator exclude it from a pure K comparison.
- Record mean/std/P50/P95 latency, throughput, and base/peak/incremental allocated memory.
- Do not treat incremental peak as independent deployment memory or subtract separately timed
  means to derive component latency.
- Preserve the scope conclusion: E-001 retains the full correlation pyramid and 32 GRU updates,
  so it can prove final affine-fit point redundancy but cannot claim end-to-end acceleration.

## Formal run-01 audit outcome

- Accuracy watchdog `20260804_040321_a1ab77de`: `succeeded`, GPU 2.
- Benchmark watchdog `20260804_040325_fa3cd863`: `succeeded`, GPU 3.
- Two independent audits and the experiment task's structured CSV/JSON recomputation: `APPROVE`.
- Exact-public K64/K128 decision candidates: `14/14 PASS`.
- Full report: `research/results/E-001/formal-official-archive-run-01.md`.
