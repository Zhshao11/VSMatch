# E-005A-INLIER-DIAG Result Card

## Decision

- Status: `COMPLETED / DIAGNOSTIC PASS / WAITING_USER`.
- Scope: frozen M001 v0.8 checkpoints and 404-pair isolated dev split; read-only, no training, no G2, no OS424/confirm, no method/configuration change.
- Watchdog job: `20260806_162435_151fc93f`, succeeded with return code 0.
- Both independent reviews ended `APPROVE`; automated tests ended `11 passed`.

## Main Finding

The E-005A reliability weights mostly identify endpoints that agree with the model's own incorrect affine, not endpoints that agree with raw ground truth. At 2 px on final active K128:

| Ratio | Raw | E-005A weighted |
|---|---:|---:|
| GT endpoint inlier | `14.70%` | `22.20%` |
| Self-consensus inlier | `47.41%` | `73.33%` |

For comparison, frozen GDROS on the same M256 source positions has `72.40%` GT endpoint inliers and `96.75%` self-consensus inliers at 2 px. The current recurrent model has learned a strong internal consensus signal, but it often converges to the wrong cross-modal affine basin.

## GDROS Point Inliers

Pooled micro ratios; each track also has 404-pair macro mean/median/P5/P95 in `summary.json`.

| Track | Basis | 1 px | 2 px | 5 px | 8 px | 16 px |
|---|---|---:|---:|---:|---:|---:|
| public-200k | GT | `38.12%` | `75.43%` | `98.80%` | `99.92%` | `100%` |
| public-200k | self | `90.30%` | `99.00%` | `99.99%` | `100%` | `100%` |
| full-512 | GT | `35.16%` | `72.37%` | `98.22%` | `99.86%` | `100%` |
| full-512 | self | `82.29%` | `96.51%` | `99.94%` | `100%` | `100%` |
| matched-M256 | GT | `35.10%` | `72.40%` | `98.28%` | `99.87%` | `100%` |
| matched-M256 | self | `82.36%` | `96.75%` | `99.97%` | `100%` | `100%` |

The public fitting pool has the highest self-consensus, as expected from the frozen ordinary-LS path. Full-image and matched-M256 GT ratios are nearly identical, so M256 location coverage does not explain the E-005A failure.

## E-005A Point Inliers

Pooled micro ratios:

| Track | Basis/weighting | 1 px | 2 px | 5 px | 8 px | 16 px |
|---|---|---:|---:|---:|---:|---:|
| M256 | GT raw | `2.27%` | `7.68%` | `25.34%` | `34.62%` | `45.64%` |
| M256 | GT weighted | `6.52%` | `22.05%` | `70.21%` | `90.29%` | `99.12%` |
| M256 | self raw | `11.54%` | `24.39%` | `33.90%` | `38.17%` | `46.57%` |
| M256 | self weighted | `34.33%` | `72.04%` | `95.50%` | `99.30%` | `99.99%` |
| active K128 | GT raw | `4.35%` | `14.70%` | `47.90%` | `64.51%` | `80.56%` |
| active K128 | GT weighted | `6.56%` | `22.20%` | `70.23%` | `90.25%` | `99.09%` |
| active K128 | self raw | `22.69%` | `47.41%` | `64.58%` | `71.39%` | `82.29%` |
| active K128 | self weighted | `35.34%` | `73.33%` | `95.86%` | `99.34%` | `99.99%` |

Historical top8-nearest-within-16-px availability is reported separately: M256 mean `83.50/256`, final active mean `79.76/128`. It is not treated as final endpoint correctness.

## K x Rho Construction

- All 84 equal/E-005A-weight curve cells were present; each used 32 deterministic repetitions per eligible pair.
- Eligibility was `99.26%` for the reported cells, so none was greyed for falling below 90%.
- Raw curve CSV contains `1,085,952` rows and was reaggregated to verify AEPE, CMR@2, P95, condition, eligibility and sampling CI exactly.
- Equal-weight results only meet both dense tolerances at `rho=1.00`. The Pareto-minimum passing cell is `(K=16, rho=1.00)`, AEPE `0.7837 px`, CMR@2 `100%`.
- At `(K=256, rho=0.95)`, AEPE is `1.9384 px` and CMR@2 is `60.18%`, still below the required `76.46%` CMR@2.
- At `(K=128, rho=0.95)`, equal-weight AEPE/CMR@2 is `2.3804 px / 46.93%`; E-005A-weight is worse at `3.7259 px / 43.71%`.

The controlled construction shows that increasing K cannot compensate for the present high-impact outliers. Near-perfect inlier purity is required under ordinary LS, and the current E-005A weights do not provide that purity.

## Oracle Decomposition

| Final active K128 variant | Valid | AEPE | CMR@2 |
|---|---:|---:|---:|
| predicted coordinates + predicted weights | `404/404` | `4.6343` | `7.92%` |
| GT coordinates + predicted weights | `404/404` | `0.000036` | `100%` |
| predicted coordinates + GT binary weights | `156/404` | `1.1861` | `98.72%` |
| GT coordinates + GT binary weights | `156/404` | `0.000038` | `100%` |

The GT-binary rows are conditional on the preregistered `>=12` inliers, `>=3` quadrants and condition `<=20` gate. Only `38.61%` of pairs qualify. For those pairs, removing false-positive weights is sufficient to recover good affine accuracy; for the remaining `61.39%`, the final active endpoints lack enough correct, spatially usable support. Coordinate/candidate correctness is therefore the primary bottleneck, with false-positive weight contamination as a secondary bottleneck on the eligible subset.

## What This Establishes

- GDROS raw flow is both mostly GT-correct and extremely affine-self-consistent; public-200k ordinary LS is not failing because it fits a small minority of arbitrary points.
- E-005A global recurrence and affine feedback do form a coherent consensus, but the consensus is commonly wrong in GT coordinates.
- The current learned weights emphasize that wrong consensus and are not a robust substitute for high-purity inlier selection.
- A future method decision should target GT-correlated cross-modal coordinate recovery and explicit false-positive rejection before changing K or adding more ordinary-LS points.

This diagnostic does not test a new trainable method, efficiency, OS424 generalization, or a rescue configuration. It does not authorize E-005B.

## Frozen Identity

- CURRENT_STATE: `19851ba9123915a7ca47ddc0944b36b753f94fcfe9deddfaa17cda9a54bfd883`
- Diagnostic spec: `a9bcfb533f98212e4357d96b79e1e9ddaa886672c3a18aea874d5c0572cc52e3`
- E-005A RESULT_CARD: `e2716e03aa384436cc71bd72061e48fa29d414b1b44884ba2f72e7991ee1feea`
- E-005A RUN_MANIFEST: `faf88bf566e5f217b4706d56d460d502489bf18a639bc12afa47af7d10fcac54`
- Diagnostic source manifest: `52874aeb0dbe6d0006d732b3e875b5b32cce39bec3bdee766f2cd2fe8073fc95`
- Summary JSON: `027b76c717974235f0d0decfe594417b33a13f85c29a8c95c8463574ff4cdd95`
- Gate JSON: `5f9172e21e5de69cddc7f39f833744e8bc9245f779a23c3a9ab4b9010a73aad6`
- Raw curve CSV: `86ecff591346e8509a551e0b32a9e4df5823b848f2397f298dc2cfa6526de122`

## Artifacts

- Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01`
- Summary: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/summary.json`
- Gate: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/gate.json`
- Point CSV: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/point_inlier_raw.csv`
- Curve CSV: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/k_rho_raw.csv`
- Oracle CSV: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/oracle_decomposition_raw.csv`
- Recoverable CSV: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/recoverable_raw.csv`
- Plot manifest: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic/plot_manifest.json`
- PNG/PDF figures: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/diagnostic`; visually verified PDF copies are under `/Users/shihao/Documents/图像匹配之二/output/pdf/E-005A-INLIER-DIAG`.
- Definition review: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/reviews/definition-coordinate-sampling.md`
- Final evaluator review: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A-INLIER-DIAG/run-01/reviews/evaluator-aggregation-plot-r3.md`
- Watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260806_162435_151fc93f.log`

## Stop

The diagnostic is complete and waiting for the active method task. No training, G2, E-005B, rescue, checkpoint modification or further experiment was started.
