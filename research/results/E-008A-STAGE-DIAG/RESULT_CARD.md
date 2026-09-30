# E-008A-STAGE-DIAG Result Card

Status: `COMPLETED / MIXED_LOSS_DIAGNOSED`

## Decision

E-008A loses the largest absolute amount of recoverable 2 px coverage in the matched-region
`top64` prefilter. However, its conditional loss (74.57%) is only 8.61 percentage points larger
than the subsequent `top4` ranking loss (65.97%). The frozen `<10 pp` rule therefore requires a
mixed diagnosis rather than a single-stage attribution.

Region containment is a smaller contributor. Target-ID dedup changes candidate multiplicity but
does not remove any proposal that had a 2 px candidate.

## Funnel

| Stage | Mean 2px-correct proposals | P10 / P50 / P90 | Conditional retention | 2px candidate base rate |
|---|---:|---:|---:|---:|
| S0 global stride-8 capacity | 239.351 | 188.3 / 241.5 / 288.0 | - | 15.8069% |
| S1 matched-region union | 196.686 | 138 / 200.5 / 250.7 | 82.174% | 0.01080% |
| S2 matched-region top64 | 50.015 | 26 / 48 / 78 | 25.429% | 0.05161% |
| S4 three-branch top4 union | 17.022 | 7 / 15 / 29 | 34.034% vs S2 | 0.09939% |
| S5 frozen post-dedup final | 17.022 | 7 / 15 / 29 | 100.000% vs S4 | 0.09828% |

S3 branch retention from S2: region 17.163%, descriptor 12.328%, patch/structure 6.617%,
diagnostic full-score 9.893%. The full-score top4 is diagnostic only; it is not injected into S4.

S1 in-bounds containment: 4x4 69.289%, 8x8 61.147%, 16x16 31.758%, combined 82.147%.

Absolute loss as a fraction of S0 correct proposals: region 17.826%, top64 61.278%, top4
13.784%, dedup 0.000%.

## Parity And Audit

- Formal dev pairs: 404.
- Frozen P0 parity: mean 17.022277, P10 7, P50 15, base rate 0.0982774%; all checks PASS.
- Candidate IDs are recomputed and checked against frozen no-GT traces before label loading.
- No hidden 4096x4096, blind full-image search, fallback, training, OS424, or confirm access.
- Tests: 6 passed; 2-pair S5 parity PASS; formal subset rejection PASS.
- Independent review: R1 findings fixed; R2 `APPROVE`.

## Artifacts

- Remote run: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A-STAGE-DIAG/run-01`
- Summary: `formal/stage_summary.json`
- Pair metrics: `formal/stage_pair_metrics.jsonl`
- Proposal metrics: `formal/stage_proposal_metrics.npz`
- Reviewer: `review/review-r2.md`
- Watchdog job: `20260807_170823_e46768d6`
