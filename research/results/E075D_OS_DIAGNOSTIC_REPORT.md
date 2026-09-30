# OS Dataset Diagnostic Report

## Scope

This report consolidates E075B, E075C, E075D-R1/R2/R3 for the frozen E075A-M4 OS model. Historical test comparisons are exploratory evidence only. Path hashes, split audits, and hook parity are protocol facts.

## Frozen model and protocol

- Primary model: E075A-M4, `M4_step80000.pt`.
- Exact path: frozen B/C branches -> 16-channel 4x4 region gate -> fused flow -> iter12/24/32 -> LSR.
- Primary formal metric: pair-level mean EPE; `CMR@t` means pair EPE `< t`.
- Training-origin screening: 2,011 official-training pairs, all `MODEL_SEEN_DIAGNOSTIC_ONLY`.
- Model-unseen OS validation split: unavailable.

## Evidence by experiment

### E075B: random training-only dev check

The 16-pair subset had no final `>=5 px` failure. M4 iter24 post-LSR had pair-level CMR@5 = 100%. This was insufficient for tail attribution.

### E075C: full training-origin failure discovery

All 2,011 training pairs were screened:

| Pair stratum | Count |
|---|---:|
| `<1 px` | 1,656 |
| `1-2 px` | 332 |
| `2-5 px` | 23 |
| `>=5 px` | 0 |

Training-origin aggregate: AEPE `0.6405`, CMR@1 `82.35%`, CMR@2 `98.86%`, CMR@5 `100%`. The worst pair AEPE was `4.023 px`.

The immutable E075A test materialization reproduced the formal result exactly: AEPE `2.3613`, CMR@1/2/5/10/16 = `56.68/78.96/88.86/94.80/97.03%`, with 45 final `>=5 px` failures.

### E075D-R1/R2: exact branch and initial evidence checks

Among the 45 historical failures, exact iter24 post-LSR recomputation found:

- B and C both `>=5 px`: `41/45 = 91.11%`.
- At least one branch `<5 px`: `4/45 = 8.89%`.
- iter12 post-LSR success followed by iter24 failure: `0`.
- iter24 pre-LSR success followed by post-LSR failure: `0`.

Using the actual raw CorrBlock score `dot/sqrt(256)` with continuous GT neighborhoods, failure samples had weaker but not absent initial support. At radius 5 px and top-32 targets, token-level recall was approximately:

| Matcher | Failures | Controls |
|---|---:|---:|
| B_F2 | 1.22% | 1.44% |
| B_F3 | 1.74% | 1.97% |
| C_student_F3 | 1.89% | 2.30% |

These are token-level statistics, not pair-level evidence sufficiency.

### E075D-R3: source-level recurrent capture

Hook capture was bitwise-parity equivalent on six deterministic parity pairs; output max/mean absolute difference was `0.0` with identical AEPE and threshold decisions.

At iter24, the continuous GT target was inside at least one observed pyramid query support for essentially all valid source tokens in both failures and controls. Level-0 support was slightly lower for failures:

- B_F2: failures `97.74%`, controls `99.65%`.
- B_F3: failures `97.27%`, controls `99.60%`.
- C_student_F3: failures `97.78%`, controls `99.78%`.

Thus GT being outside the recurrent query window is not the dominant explanation.

The reduced update statistics do not show a clean failure/control separator. At iter24, failure-group away-update fractions were about `49.2-51.6%`, while controls were `55.3-60.8%`; mean distance changes were small and not consistently in one direction. The card reports 33 of the 41 both-branch-failure pairs as having an away-majority, but this remains descriptive and does not prove causality.

Token-level correct-basin fractions at iter24 were much lower for failures than controls. For C_student_F3, the `<5 px` fraction was `15.38%` in failures versus `82.22%` in controls; `<2 px` was `2.11%` versus `41.75%`; `<1 px` was `0.48%` versus `15.72%`. These are token-level fractions, and token identity was not retained well enough to prove basin entry/exit per pair.

## Final diagnosis

### Established

1. The OS failure is a generalization failure: the complete training-origin population has no final `>=5 px` tail, while the historical test view has 45/404 final failures.
2. The dominant failure population is shared by both B and C: 41/45 failures remain bad in both branches at exact iter24 post-LSR.
3. M4 fusion can only be the primary target for the remaining 4/45 branch-available cases; it cannot explain the dominant 41/45 failures.
4. No evidence supports LSR regression or a generic iter12->iter24 failure as the main cause.
5. Initial raw correlation evidence is weaker on failures, but not absent; the observed difference is modest and sparse.
6. GT is generally inside the observed recurrent query support, so the fixed radius-4 support window is not the dominant failure cause.

### Still not proven

- Whether the recurrent update uses the in-support evidence incorrectly at the token level.
- Whether correct basin entry is followed by escape, because online reduction did not preserve token identities.
- Whether the remaining error is fine localization rather than basin selection.
- A model-unseen validation estimate: no predeclared OS split excluded from E075A/B/C training is available.

## Decision boundary

The first justified shared OS/WHU module is **not yet uniquely identified**. The strongest current hypothesis is a shared B/C feature-correlation and recurrent interaction generalization problem, not M4 gating or LSR. Before changing that module, run the same failure/evidence/query diagnostic on WHU and identify the intersection of OS and WHU failure signatures. Any later improvement must be retrained and evaluated on a predeclared model-unseen split or an untouched final test set.

## Source cards

- E075C: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-075C-M4-FINAL-FAILURE-DISCOVERY-AND-TRAJECTORY-ATTRIBUTION/run-01/RESULT_CARD.md`
- E075D-R2: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-075D-BRANCH-EVIDENCE-VS-DECODER-DIAG/run-r2/RESULT_CARD.md`
- E075D-R3: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-075D-BRANCH-EVIDENCE-VS-DECODER-DIAG/run-r3/RESULT_CARD.md`
