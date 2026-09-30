# E-015A RESULT_CARD

Status: **COMPLETED / MATCHER_CANDIDATE_FOUND**

Date: `2026-08-11`

Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-015A/run-01`

Watchdog job: `20260811_100830_13c60853` / `succeeded` / GPU `4`

## Gate outcome

- P0: **PASS**
  - E-011A top128 parity: PASS.
  - Regenerated E-011A candidate sample parity: PASS on 2 dev pairs.
  - Candidate-before-label materialization: PASS; score and expanded-candidate digests are written before label join.
  - DINOv2-L local checkpoint load: PASS; feature nonconstant, known-shift, shuffled target order, swapped direction, and 37x37 grid mapping PASS.
  - No OS424/confirm/datum, training, finetune, checkpoint write, or geometry feedback.
- Feature/model reviewer: **APPROVE**.
- Evaluator/protocol reviewer: **APPROVE**.

## Main result

Main diagnosis: **MATCHER_CANDIDATE_FOUND**

Available strong track:

- `foundation_dense/dinov2_vitl14_local`
- Checkpoint SHA: `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`

Top128 reranking:

- Best non-reference signal: `fusion/zscore_available_pretrained`
- Dev AUROC/AP: `0.801631 / 0.00166453`
- Best selection: global topK-256, correct mean `0.9901`, precision `0.4030%`
- Reference E-013A pre-feedback: AUROC/AP `0.776095 / 0.00274588`, best correct mean `1.2203`, precision `0.5334%`
- Interpretation: reranking is not stronger than E-013A pre-feedback and does not qualify as STRONG/WEAK signal.

Expanded / independent candidate generation:

- Track: `foundation_dense/dinov2_vitl14_local/full4096_top128`
- Search: full 4096 SAR target tokens per source proposal; **resource-unqualified diagnostic**
- top128 correct@2px proposals mean/P10/P50/P90: `115.943 / 65.6 / 115 / 164`
- candidate base rate @2px: `0.05982%`
- nearest distance P50/P90: `7.283 / 54.623 px`
- oracle affine AEPE: `12.903 px`
- PLR/valid: `97.525% / 97.525%`

Interpretation: DINOv2-L full-target retrieval provides much stronger candidate acquisition than E-011A top128 (`68.710` mean, P50 `66`) and E-006A quantity line (`66.210` mean, P50 `64`). However oracle affine remains poor (`12.903 px`), so the next issue is geometry-aware compression/distribution/ranking of the much richer candidate set, not merely obtaining enough candidates.

## Resource

- DINOv2-L per-pair p50/p95/max time: `98.8 / 177.3 / 265.9 ms`
- Peak allocated p50/p95/max: `2.842 / 2.884 / 2.932 GB`
- Max live score elements: `6,545,408`
- Full-image blind search: `true`
- Hidden 4096x4096 matrix: `false`

## Artifacts

- Local result dir: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A`
- Remote formal dir: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-015A/run-01/formal`
- Feature signal summary: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/feature_signal_summary.json`
- Feature manifest: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/feature_manifest.json`
- Expanded candidate table: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/expanded_candidate_table.csv`
- Top128 rerank table: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/top128_rerank_table.csv`
- P0 integration report: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/p0_integration_report.json`
- Feature reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/feature_reviewer_conclusion.md`
- Evaluator reviewer: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/evaluator_reviewer_conclusion.md`
- Watchdog report: `/Users/shihao/Documents/图像匹配之二/research/results/E-015A/watchdog_report.json`

## Stop condition

Per protocol, no E-015B, descriptor training, full sparse affine feedback, OS424/confirm, or rescue experiment was started.
