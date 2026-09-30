# E-014A RESULT_CARD

Status: **COMPLETED / NO_SIGNAL_FOUND**

Date: `2026-08-11`

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-014A/run-01`

Watchdog job: `20260811_000028_e2c8a870` / `succeeded` / GPU `3`

## Gate outcome

- P0: **PASS**
  - E-011A top128 parity: PASS
  - Candidate count: `78,303,360`
  - Correct@2px proposals mean/P10/P50/P90: `68.710 / 34 / 66 / 106`
  - Oracle affine AEPE: `14.249`
  - Sanity controls: PASS, including optical-source -> SAR-target direction, stride-8 token/pixel conversion, patch center/radius/padding, same-image identity, known-shift, gradient/RIFT/structure tensor toys, shuffled target, swapped direction, and random score control.
- Reviewer feature implementation: **APPROVE**, no blocking finding.
- Reviewer evaluator/protocol: **APPROVE**, no blocking finding.

## Main result

Main diagnosis: **NO_SIGNAL_FOUND**

Best non-reference signal:

- `census_rank_self_similarity/local_self_similarity`
- Dev AUROC: `0.531545`
- Dev AP: `0.000443655`
- Best selection: `region_quota=8`
  - correct mean: `0.212871`
  - precision mean: `0.000866667`
  - coverage pass: `100%`

Reference tracks:

- E-013A pre-feedback endpoint score: AUROC `0.776095`, AP `0.00274588`, best correct mean `1.22030`, best precision `0.00533358`
- E-013A final score: AUROC `0.502858`, AP `0.000487640`
- E-011A raw score: AP `0.00125910`, AUROC `0.637741`
- Random: AP `0.000353054`, AUROC `0.498586`

Interpretation: the tested raw patch, log-SAR/normalized patch, gradient/edge, RIFT/HOPC-style, census/rank/self-similarity, structure tensor/cornerness, E-011A raw score, and E-013A score tracks do not provide a usable endpoint-correctness ranking signal on E-011A top128 candidates. Because P0 and both reviewers passed, this negative result is not attributed to implementation/protocol failure.

## Artifacts

- Local result dir: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A`
- Remote formal dir: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-014A/run-01/formal`
- Signal summary: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/signal_summary.json`
- Signal table: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/signal_table.csv`
- TopK/quota table: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/topk_quota_table.csv`
- P0 sanity report: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/p0_sanity_report.json`
- Feature reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/feature_reviewer_conclusion.md`
- Evaluator/protocol reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/evaluator_reviewer_conclusion.md`
- RUN_MANIFEST: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/RUN_MANIFEST.json`
- Watchdog report: `/Users/shihao/Documents/图像匹配之二/research/results/E-014A/watchdog_report.json`

## Stop condition

Per E-014A protocol, no E-014B, training, descriptor retraining, complete sparse affine feedback, OS424/confirm, or rescue experiment was started.
