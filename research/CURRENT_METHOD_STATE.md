# Current Method State

> 2026-11-16：本页已从 2026-09 历史快照更新为 E092–E098 的当前状态。
> 精简推理版正式命名为 **M4-streamlined**，入口见 [版本存档](../docs/M4_STREAMLINED.md)。
> 方法训练设计仍参考 [METHOD_DESIGN](../docs/METHOD_DESIGN.md)，早期评测与消融见
> [RESULTS_OS](../docs/RESULTS_OS.md) 和 [EVALUATION_PROTOCOL](../docs/EVALUATION_PROTOCOL.md)；
> 后续固定数值设置下的推理等价验证、LSR 消融与耗时诊断以本页链接的 E095–E098 为准。

Updated: 2026-11-16 (Asia/Shanghai)

Latest cost/accuracy audit (E098A, no training): on the shared-GPU ten-pair
coarse diagnostic, iterative matching and related operations, DINO, and
feature frontends account for about 60.6%, 17.8%, and 12.2% of CleanM4's
instrumented elapsed stream span; all five LSR fits account for 5.45%.
These are diagnostic spans, not active-kernel sums or a replacement for
E096A full-424 wall timing. The first three categories explain about 88.7%
of extra diagnostic time versus official GDROS.

E098A separately revalidated all 1,696 historical B/C/AVG/M4 cached
predictions and reproduced accepted E096A M4 on 424/424 pairs bitwise.
Official-component CMR@1 is B 58.4906%, C/student 69.5755%, equal B/C
69.3396%, full M4 72.4057% (GDROS E096A: 33.7264%). Most of the precision
advantage is already present in C as a package; these fixed-weight branch
comparisons do not isolate DINO, dual correlations or training effects.
E093A's original failed join remains preserved; this is a new successful
cache audit, not fresh B/C inference or a rewritten historical status.
Default M4 remains five LSR fits; no candidate replaced it. See
`results/E098A_M4_COST_CONTRIBUTION_20261116/REPORT.md` before proposing
compression or attributing CMR@1 to one module.

This file is the canonical starting point for a new researcher or Codex task.
Result cards and source snapshots remain authoritative for exact metrics and
hashes; this page states which results are current and which experiments did
not replace them.

Protocol correction audited on 2026-11-15: read
`results/OS_GDROS_PROTOCOL_AUDIT_20261115.md` before comparing OS with GDROS.
Historical M4 uses standard vector EPE on 404 pairs; GDROS's public result uses
component absolute error on 424 pairs. They must not be compared directly.
Reaggregation on identical 404 pair IDs with vector EPE gives GDROS/M4 AEPE
`2.296983 / 2.361299`, CMR@1 `9.6535 / 56.6832%`, CMR@2
`52.9703 / 78.9604%`, and CMR@5 `94.3069 / 88.8614%`.
The initial audit did not find a full 424-pair public-component M4 result.
E092A subsequently completed that fixed-weight evaluation on 2026-11-15:

| Fixed model/output | Public mean error | CMR@1 | CMR@2 | CMR@5 |
| --- | ---: | ---: | ---: | ---: |
| GDROS released, iter32 post-LSR | 1.461486 | 33.7264% | 81.6038% | 99.5283% |
| M4 step80K, iter32 post-LSR (preregistered main) | 1.474430 | 72.4057% | 84.9057% | 94.1038% |
| M4 step80K, iter24 post-LSR (historical supplement) | 1.445683 | 71.6981% | 84.9057% | 93.6321% |

All rows use all 424 official pairs, whole-image/no-mask evaluation, and the
public component-error convention. Main-table iterations were fixed to 32
before evaluation. M4-24 remains a separately declared historical deployment
point; no new checkpoint or iteration was selected from these test results.
Prediction tensors were saved before a separate CPU GT join. Both historical
replay checks passed. See `results/OS_OFFICIAL424_E092A_SUMMARY_20261115.md`
and `results/E092A_OS_OFFICIAL424_M4_GDROS_EVAL_20261115/RESULT_CARD.json`.

E092B then completed fixed **OS-M4** cross-dataset inference on all 700 pairs
of the author-public WHU test group. GDROS used the official WHU checkpoint;
M4 used the exact OS step80K checkpoint above. No training or fine-tuning was
performed. Both main outputs are iter32 post-LSR, whole-image/no-mask:

| Fixed model / training source | Public mean error | CMR@1 | CMR@2 | CMR@5 |
| --- | ---: | ---: | ---: | ---: |
| GDROS released / WHU | 0.867463 | 71.4286% | 97.2857% | 99.7143% |
| OS E075A-M4 step80K / OS | 2.642175 | 65.1429% | 81.8571% | 89.1429% |

This is not the native-WHU E081A-M4 model listed below. Under standard vector
EPE the corresponding means are 1.371480 / 4.164684; the two metric
conventions must remain separate. See
`results/WHU_OFFICIAL700_E092B_SUMMARY_20261115.md`. The author-public OS and
WHU groups are single groups, not the paper's three-seed mean plus/minus std.

UBCv2 E092C-R completed all three preregistered geometry groups 901/902/903
on the full official raw 1625-pair test archive. Both models kept OS weights;
no training or fine-tuning was performed. Main outputs are iter32 post-LSR,
whole-image/no-mask. Across-group mean +/- std below uses ddof=0:

| Fixed model / training source | Public mean error | CMR@2 | CMR@3 | CMR@5 |
| --- | ---: | ---: | ---: | ---: |
| GDROS released / OS | 22.387892 +/- 0.396615 | 0.9231 +/- 0.0502% | 2.6667 +/- 0.1535% | 10.5231 +/- 0.5435% |
| OS E075A-M4 step80K / OS | 3.995178 +/- 0.147440 | 61.8667 +/- 0.7407% | 68.0000 +/- 0.6571% | 73.6821 +/- 0.5535% |

Standard vector-EPE means are 35.648434 +/- 0.651923 (GDROS) and
6.272793 +/- 0.231645 (M4), kept separate from public-component error.
This is an **independent generalization test** under the paper's affine
bounds, using a frozen SAR mapping and the same raw images across geometry
groups. It is not the authors' screened 1447-pair split, exact transforms,
preprocessing, or a native-UBCv2-trained GDROS result. The original E092C
exact-replay blocker and archive-count correction remain historical.
All 15 DAG tasks succeeded. Independent CSV/manifest/statistics and
cross-group aggregation checks passed. See
`results/UBCV2_RAW1625_E092CR_SUMMARY_20261115.md` and
`results/THREE_DATASET_FROZEN_OS_M4_COMPARISON_20261115.md` for the complete
comparison, limitations, hashes and reproducibility entries.
These evaluation results did not create or select a new checkpoint, and
do not replace the separately protected OS and native-WHU methods below.

Architecture and historical ablation evidence were audited from the exact
E072/E073/E074/E075 source chain on 2026-11-15. Read
[the architecture and ablation report](results/M4_ARCHITECTURE_ABLATION_AUDIT_20261115/REPORT.md)
for rendered diagrams, source hashes and the scope of each historical control.
The retained legacy source snapshot executes five matcher instances (eight
CorrBlock pyramids), including two C teachers that do not feed its returned
student prediction. The actual remote E074/E075 interfaces used for recent
timing already default to `c_skip_teachers=True`, executing three matchers
(five CorrBlock pyramids). E094A snapshots and hashes record that distinction;
do not attribute removal of C teachers to the E094A LSR optimization.
LSR supplies fusion features and the final affine projection; it is not fed
back into the matcher recurrent coordinate state. These are static code-audit
findings, not measured speedups or newly completed architecture ablations.

## Completed inference optimization: E094A

On 2026-11-16, the protected OS-M4 step80K checkpoint completed a full paired
424-image evaluation of original 129-call inference versus final-only fusion
with 5 LSR calls. Both paths keep 32 GRU iterations, identical weights,
the same C-teacher setting, final gates and terminal official LSR.
Only unused intermediate outer gates and their LSR features are omitted.

All 424 raw and post-LSR flow pairs are elementwise equal (maximum difference
zero); all 424 corresponding final NPY hashes are also identical. Public
component mean is 1.474417970 for both, with CMR@1/2/5 of
72.405660 / 84.905660 / 94.103774 percent. No training took place.

Paired alternating-order measurements on one shared RTX 5090 give
2482.721 versus 1364.588 ms/pair: 1.819x speedup, 45.037 percent lower mean
latency. These are shared-GPU measurements, not exclusive deployment times
or LSR-only attribution. Do not compare their absolute times with the earlier
GPU6 quick benchmark.

The current original and final-only outputs differ numerically from archived
E092A tensors (largest component difference 0.523743 px); the historical mean
1.474429713 and CMR@1/2/5 remain separately documented. Old evaluation did not
explicitly fix all current backend settings. This cross-run difference is
already present in the current 129-call control and is not caused by pruning.
No new checkpoint or test-selected method replaces the protected weights.

Report: [E094A full verification](results/E094A_M4_FINAL_ONLY_LSR_424_20261116/REPORT.md).

## Completed inference cleanup: E095A

On 2026-11-16 the separate inference-only `remote_stage/os/m4_clean_deploy.py`
passed the full official OS424 paired check against E094A's five-LSR path:
all 424 raw and post-LSR predictions were bitwise equal (maximum difference
zero). The current reference also matched all E094A archived final5 outputs.
The protected step80K checkpoint and scientific best-method selection stay
the same; this is an equivalent deployment path without training.

CleanM4 removes unused module storage, ignored nested context computation,
nonfinal mask/upsampling outputs, diagnostic transfers and repeated lookup
grid construction. Frozen DINO and descriptor adapters are shared only after
exact state/config equality checks. Three matchers with 32 GRU iterations
each, five four-level correlation pyramids, active learned gates/projections,
and five official LSR calls are retained. All 540 parameterized leaf modules
in the clean graph executed in the coverage pass.

Registered parameters, including frozen and unused historical weights, fall
from 1,692,064,147 to 347,975,753 (79.43%). This is not a claim about trainable
parameter or FLOP reduction. In the final paired shared-GPU6 run, mean forward
plus terminal-LSR time was 621.689 ms vs 340.450 ms (1.826x, 45.24% lower).
These timings cannot be combined directly with previous runs on other GPUs.

Public component mean remains 1.4744179696 px, CMR@1/2/5 remains
72.40566038/84.90566038/94.10377358%; 25 pairs still have error >=5 px.
Vector EPE remains 2.3679955441 px. Cleanup does not address this error tail.

The compact state stays on zzvr in the E095A final `-r2` run. Construction
currently extracts modules from the protected legacy architecture; an
independent cold-start factory is not implemented. Training still uses the
historical sequence-output path.

Report: [E095A full verification](results/E095A_M4_CLEAN_DEPLOY_424_20261116/REPORT.md).

## Latest paired GDROS comparison: E096A

On 2026-11-16 both official released OS GDROS and clean M4 were rerun on all
424 official OS pairs, 32 iterations per matcher, FP32, both TF32 flags off,
batch=1, same shared RTX 5090 GPU3, alternating order per pair. Actual LSR
counts were GDROS=1 and M4=5. This comparison uses unchanged official GDROS
code and the validated inference-clean M4; no GDROS cleanup was applied.

| Metric | GDROS | Clean M4 |
|---|---:|---:|
| Public component mean / px | 1.461660750 | 1.474417970 |
| CMR@1 / % | 33.72641509 | 72.40566038 |
| CMR@2 / % | 81.60377358 | 84.90566038 |
| CMR@5 / % | 99.52830189 | 94.10377358 |
| Failures with per-pair error >=5 px | 2 | 25 |
| Vector EPE / px | 2.321225131 | 2.367995544 |
| Mean forward plus terminal LSR / ms | 221.343 | 571.037 |

M4 latency is 2.580x GDROS in this paired shared-GPU run. Earlier GPU6 M4
timings belong to a different load interval and should remain separate.
All 424 current M4 predictions match E095A bitwise. GDROS CMR values match
the historical E092A run; its mean differs by +0.000175244 px. Keep the
fresh run's values and original history distinct; no isolated attribution
of this small numerical difference was performed.

M4 has lower per-pair public error on 331/424 pairs, but its larger error
tail prevents a better global mean. Independent verification recomputed
compact statistics and all 848 prediction-file SHA256 hashes successfully.

Report: [E096A full comparison](results/E096A_GDROS_CLEAN_M4_OS424_20261116/REPORT.md).

## Latest LSR-input ablation: E097A

On 2026-11-16 all OS424 pairs were rerun with fixed M4 weights, keeping
only logical slot 5 (`final1`), only slots 4/5 (`last45`), and all five fits
as the paired control. Missing fits use identity flow, making the respective
geometry residuals zero while preserving gate dimensions and weights.
All three 32-step matchers remain. This changes learned-gate input
distributions and is not an exact redundancy deletion or a retrained ablation.

| Metric | Five fits | Only slot 5 | Only slots 4/5 |
|---|---:|---:|---:|
| Public component mean / px | 1.474417970 | 1.510789589 | 1.482544107 |
| CMR@1 / % | 72.40566038 | 71.46226415 | 70.51886792 |
| CMR@2 / % | 84.90566038 | 84.66981132 | 85.14150943 |
| CMR@5 / % | 94.10377358 | 93.63207547 | 93.63207547 |
| Per-pair error >=5 px | 25 | 27 | 27 |
| Mean forward plus terminal LSR / ms | 354.466 | 337.404 | 343.778 |

This paired shared-RTX5090 GPU7 run observed 4.81% and 3.02% latency
reductions respectively. The full control matches E096A bitwise on all
424 pairs, but both ablations change every final prediction. Independent
statistics and all 1,272 prediction-file hashes passed verification.

Default inference remains CleanM4 with five fits: the tested reductions
lose CMR@1/@5 and increase mean error for modest total-latency savings.
Keeping C's fit improves mean error relative to final-only but does not
improve all accuracy thresholds. These frozen-input substitutions do not
establish what a gate retrained without those channels could achieve.

Report: [E097A LSR ablation](results/E097A_M4_LSR_SLOT_ABLATION_OS424_20261116/REPORT.md).
Inference helper: `remote_stage/os/m4_final_only_lsr.py`.
Runner and independent verifier: `remote_stage/os/e094a_final_only_lsr_eval.py`
and `remote_stage/os/verify_e094a_results.py`.

## Active frozen-weight ablation: E093A

E093A was dispatched to the existing experiment-engineering chat and its
dedicated zzvr watchdog/orchestrator started on 2026-11-15 at 16:55 +08:00.
At the independent check on 2026-11-16 at 08:29 +08:00, preflight and
eight-pair parity had passed; all tested original/deployment raw/post output
differences were zero. All four variants' 424-pair predictions at 24/32
iterations completed at 01:15. CPU join failed at 01:18 because the M4
cross-experiment prediction regression against E092A exceeded its preset
tolerance. The engineering chat has been asked to diagnose and repair this
without relaxing tolerance or overwriting predictions. Profiles have not
started and were then **cancelled by the user on 2026-11-16**. Independent
remote verification confirmed all five profile tasks are `cancelled` and
removed from the active DAG; the final aggregate now depends only on
`join_v2` and `historical_replay`. No timing/memory results are required.
Read the [preliminary status](results/E093A_OS_BC_FROZEN_ABLATION_PROFILE_20261115/PRELIMINARY_STATUS_20261116.md)
for independently reaggregated CSV metrics, which are not accepted final results.
This is an active experiment, **not a completed result**.

The four outputs are the current M4 checkpoint's B branch, C.student branch,
fixed raw-flow average, and trained M4 region-gate fusion, all followed by
the same official LSR. Main results use 32 iterations; 24 is supplemental.
No training or feature-side ablation is included. Each deployment profile
uses an independent process on the same physical GPU, 10 warmups and
64 pairs repeated three times, recording synchronized wall-clock latency
and allocated/reserved memory baselines and peaks. Original M4 with redundant
C teachers is a separate reference; equivalence must pass before evaluation.

Read [the execution protocol](experiments/E-093A-OS-BC-FROZEN-ABLATION-AND-PROFILE.md).
Entry: `remote_stage/os/e093a_bc_ablation_profile.py`.
Remote run: `GDROS/experiments/sparsegeo/runs/E-093A-OS-BC-FROZEN-ABLATION-AND-PROFILE`
under the zzvr project root. Scheduler: `/mnt/sda/zhushihao/experiment_watchdog_e093a`.
The engineering chat continues the DAG and compact-result synchronization;
do not treat this startup record as evidence of accuracy or speed gains.

## Current protected methods

### OS: E075A-M4 Region Gate

- Checkpoint: `M4_step80000.pt`
- Inference output: `iter24_post_lsr`
- Pair-level metrics on the historical 404-pair protocol:
  - AEPE: `2.3613`
  - CMR@1/2/5/10/16: `56.6832 / 78.9604 / 88.8614 / 94.8020 / 97.0297%`
- Main advantage: higher CMR@1/2 in the same-protocol E092A comparison above.
- Main weakness: model-unseen generalization and the error tail beyond 5 px.
- Source: `remote_stage/os/e075a_preserve_high_precision_recover_c2c5.py`
- Detailed snapshot: `results/CURRENT_BEST_OS_METHOD_20260913.md`

E075A-M4 remains the current OS method. LuckyR1, E075 diagnostics, and later
candidate generators did not produce a valid replacement.

### WHU: E081A-M4

- Checkpoint: `M4_step120000.pt`
- Pair-level metrics on the 700-pair WHU test protocol:
  - AEPE: `1.9055`
  - CMR@1/2/5/10/16: `28.0000 / 72.0000 / 98.1429 / 99.5714 / 99.5714%`
- Main strength: nearly all pairs are within 5 px.
- Main weakness: fine localization below 1-2 px.
- Source: `../e081a_whu_our_native_train.py`
- Diagnostic: `results/E081A_M4_ISOMORPHIC_DIAGNOSTIC_20260921/RESULT_CARD.md`

E081A-M4 remains the current WHU method. No E088-E091 experiment replaced it.

## WHU held-out training experiment: E091A (historical)

E090D first demonstrated an identity-safe late recurrent residual path on
1/4/8 training pairs. E091A then tested that path at a meaningful scale:

- Train: 200 source groups / 1600 pairs.
- Dev: 100 disjoint source groups / 800 pairs.
- Candidate presentations: 6400.
- Recurrent presentations: 4800.
- WHU 700-test: sealed and unused.

The selected recurrent epoch 3 changed dev AEPE from `2.4803` to `2.4655` and
CMR@1 from `22.50%` to `24.75%`. It did not pass the preregistered gate:

- CMR@1 gain was `+2.25 pp`, below `+3 pp`.
- AEPE gain was `0.0148 px`, below `0.05 px`.
- AEPE improvement 95% CI crossed zero.
- Shuffled evidence matched the main evidence, so evidence-specific benefit was
  not established.

Status: `STOP_HELDOUT_GATE_FAILED`. Do not describe E091A as the best method or
evaluate it on WHU 700-test without a new method decision.

## Closed directions

- LuckyR1 A/B/C: `NO_GAIN` under current-model and clean-lineage evaluation.
- Frozen stride-8 local candidate scoring: below random candidate selection.
- Lightweight high-resolution post-hoc descriptor: failed actionability.
- Tested relation descriptor: failed progressive overfit.
- E090D post-hoc evidence at E091 scale: small non-specific dev gain.

See `FAILED_METHODS.md`, `DECISION_LOG.md`, and individual result cards before
reopening a direction.

## Recommended next research question

The next WHU proposal should explain why shuffled structured evidence performs
like real evidence. A justified direction must separate useful cross-modal
evidence from a generic learned flow prior, for example by enforcing
evidence-conditional counterfactuals or updating the shared feature/correlation
representation rather than adding another terminal selector.

For OS, retain the separate diagnosis: the dominant issue is B/C-shared
generalization failure and the >=5 px tail, not the WHU fine-localization gap.
Do not transfer a WHU local head to OS without a clean OS lineage experiment.

## Before running anything

1. Read `REPRODUCIBILITY.md`.
2. Run `python scripts/preflight.py --scope all`.
3. Verify checkpoint hashes from `CHECKPOINTS.json`.
4. Keep prediction-before-GT evaluation and pair-level metrics.
5. Use a new experiment ID and preserve all terminal result cards.
