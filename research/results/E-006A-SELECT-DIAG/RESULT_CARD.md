# E-006A-SELECT-DIAG Result Card

Status: `COMPLETED / WAITING_USER`

Interpretation: `NOT_SELECTABLE_WITH_CURRENT_FEATURES`

## Result

The corrected E-006A candidate space is not selectable at useful purity with the frozen 26 no-GT features
and <=100k selector family. The main learned selector ranks candidates better than handcrafted scores, but
does not produce a geometrically usable top-32 set.

| Dev main track | Value |
|---|---:|
| Candidate AUROC / AP / ECE | 0.9171 / 0.01386 / 0.3133 |
| Global topK-32 precision / recall | 2.259% / 7.778% |
| Affine AEPE | 63.516 px |
| CMR@2 / CMR@5 | 0.000% / 0.990% |
| Valid pairs / PairLocalReady | 93.812% / 11.881% |

The second seed is consistent: AUROC 0.9188, AP 0.01405, top-32 precision 2.390%, AEPE 66.530 px,
CMR@2 0%. The failure is not seed-specific.

## Candidate Evidence

- Dev candidate base rates: <=1 px 0.0338%, <=2 px 0.1343%, <=5 px 0.7577%.
- 82.155% of candidates belong to in-bounds source endpoints. Their 2 px base rate is 0.1622%; the
  out-of-bounds group's independent 2 px rate is 0.00561%.
- D1 cosine proxy: AUROC 0.7482, AP 0.00476, top-32 precision 1.230%, affine AEPE 119.853 px.
- Best D1 by frozen affine criterion is `d1_combined`: AUROC 0.8377, AP 0.00805, precision 1.740%,
  affine AEPE 61.633 px.
- D2 improves the bound candidate metric by 0.0794, but its affine AEPE is 1.883 px worse than best D1.
  It therefore fails the frozen `PARTIALLY_SELECTABLE` gate as well as the hard NOT thresholds.

## Oracle Decomposition

- D0 GT-candidate ranking has effectively perfect candidate AUROC/AP and top-32 recall 99.973%, but only
  29.03% top-32 precision when aggregated across all proposals.
- Fitting all 256 per-proposal GT-nearest candidates gives AEPE 22.725 px because incorrect proposals still
  contaminate ordinary LS.
- D0 hard <=2 px inliers have 98.89% selected precision. Only 27.97% of pairs meet the frozen fit-validity
  requirements, but those valid pairs achieve AEPE 0.653 px.

This isolates the bottleneck: sufficiently pure coordinates solve the affine accurately, while current no-GT
features cannot recover enough of them at high purity. High AUROC is misleading under the 0.134% positive rate.

## Ablations

| Dev global topK-32 | AUROC | AP | Precision | AEPE px | CMR@2 |
|---|---:|---:|---:|---:|---:|
| Full main | 0.9171 | 0.01386 | 2.259% | 63.516 | 0.000% |
| No region | 0.9037 | 0.01241 | 2.305% | 56.478 | 0.000% |
| No basin residual | 0.9045 | 0.01283 | 2.058% | 59.819 | 0.000% |
| No local stats | 0.9117 | 0.01264 | 2.197% | 38.148 | 0.000% |
| Score only | 0.8052 | 0.00740 | 1.524% | 88.893 | 0.000% |

No ablation is promoted: all remain far outside the frozen useful-accuracy gates.

## Scope

This diagnostic disproves selectability with the current E-006A corrected candidates, 26 frozen no-GT
features, and tiny pointwise selector. It does not disprove stronger patch/descriptor evidence, a changed
observation space, multi-basin robust consensus, or other methods not tested here. It does not authorize
E-006B, descriptor training, proposal expansion, or a rescue variant.

## Verification

- Frozen identities and corrected candidate source/checkpoint/split SHA: PASS.
- Automatic tests: 19/19 PASS.
- Feature reviewer, evaluator reviewer, and both incremental repair reviews: APPROVE.
- Formal data: 1607 train + 404 dev; 2011 feature dumps, 2011 label joins, 6 exact checkpoints.
- Baseline unchanged; old invalid P0, OS424, and confirm were not accessed.
- Preserved failures: `cache-formal-v1` non-finite masked local scores; `evaluation-v2` D1 grid broadcasting.

## Absolute Artifacts

- Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01`
- Final cache: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/cache-formal-v2`
- Models: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/model-v2`
- Summary: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/evaluation-v3/summary.json`
- Per-pair raw rows (18,180): `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/evaluation-v3/per-pair.jsonl`
- Manifests: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/manifests`
- Final watchdog log: `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260807_000857_ec3fb033.log`
- Run manifest: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/RUN_MANIFEST.json`
- Gate JSON: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/GATE.json`
- Reviews: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/FEATURE_REVIEW.md`,
  `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-006A-SELECT-DIAG/run-01/EVALUATOR_REVIEW.md`
