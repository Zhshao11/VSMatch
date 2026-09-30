# E-008A Result Card

## Identity

- Method: `M001 v1.1 / E-008A Region-to-Point Cross-modal Candidate Generator`
- Final implementation source SHA256: `b8e4dd9f4fce7bf55f2f8ff5a875052e214248c2052ac0de5cb3fb6398ba04a5`
- Split: frozen group-isolated `1607 train / 404 dev`
- GDROS checkpoint SHA256: `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Seed-20260816 checkpoint SHA256: `46830fd161d90d9c83793e3ca0245e43bcf3245cf93a5bbcaf0e04c6cfd6de6b`
- Seed-20260817 checkpoint SHA256: `f2e5d2b78ac0c41118620ad2dedfd72fc3d3530c28d57bbf051e1e22636f020c`
- Frozen old-generator baseline identity SHA256: `eecff3522b8b9b26ffa196cdb559858ad9039291360bf050e791e1a1c13bd197`
- Test result: `13 passed`, two unrelated timm deprecation warnings.

## Gate Result

`P0 FAIL`. Per the frozen protocol, P1 was not run.

| Metric | E-006A baseline | E-008A | Gate | Result |
|---|---:|---:|---:|---|
| Mean 2px-correct proposals | 66.210 | 17.022 | >66.210 | FAIL |
| Correct proposals P10 | 33 | 7 | >33 | FAIL |
| Correct proposals P50 | 64 | 15 | >64 | FAIL |
| Candidate 2px base rate | 0.1343% | 0.09828% | >0.1343% | FAIL |
| Hard-2px oracle AEPE | 0.1908 px | 189.229 px | <=0.5 px | FAIL |
| PairLocalReady | not used | 63.119% | >=99% | FAIL |
| Valid affine | not used | 63.119% | >=99% | FAIL |
| Hidden 4096x4096 | no | no | no | PASS |
| Full-image blind search | no | no | no | PASS |

Additional coverage metrics:

- Candidate base rate: 1px `0.02522%`, 2px `0.09828%`, 5px `0.56474%`.
- Proposal count P10/P50/P90/max: `1485.3 / 1515 / 1544 / 1598`.
- Candidate count P10/P50/P90: `16980.2 / 17323.5 / 17678.0`.
- In-bounds proposals: `81.131%`.
- GT-nearest-per-proposal oracle AEPE: `37.848 px`.

## Source Ablation

| Track/source | Mean correct proposals | 2px base rate | Hard oracle AEPE |
|---|---:|---:|---:|
| Dynamic-source top1 | 3.005 | 0.19845% | 508.206 px |
| Region top4 | 8.584 | 0.14173% | 398.098 px |
| Descriptor top4 | 6.166 | 0.10180% | 470.235 px |
| Patch/structure top4 | 3.309 | 0.05464% | 504.412 px |
| Full deduplicated union | 17.022 | 0.09828% | 189.229 px |

The dynamic generator produces many more source proposals than the old M256 generator, but the final generated candidate set recovers far fewer correct proposals. Region-only candidates have slightly higher 2px purity than the old aggregate baseline, while the learned descriptor and patch/structure additions dilute full-union purity. The failure is therefore final candidate coverage, not insufficient proposal count. Because P0 did not record stage-wise GT recall, it cannot localize the loss among region selection, within-region top64 retrieval, or final top4 ranking/deduplication. This experiment does not evaluate P1 selectability because P0 did not pass.

## Training And Resource

- Trainable parameters: `28,290`.
- Formal training samples: `341,327`, including `48,761` positives from `1,599/1,607` train pairs.
- Seed training time: `8.425 s` and `7.508 s`; peak allocated memory `37,075,456 bytes` each.
- Candidate generation p50/p95: `573.39 / 655.87 ms`.
- Selector p50/p95: `453.91 / 531.61 ms`.
- Candidate generation peak allocated: `305,848,320 bytes`; selector peak allocated: `58,313,728 bytes`.
- Actual maximum target-region union: `1,648` tokens; maximum live score tensor: `25,344` elements.

## Audit

- Candidate outputs and 404 per-pair traces were materialized before GT label access.
- Candidate traces bind observation content, implementation source, and actual learned checkpoint SHA.
- Raw truth flow is joined only in FP32 label/evaluation paths; datum, OS424, and confirm were not accessed.
- Formal cache counts: observations `1607/404`, labels `1607/404`, P0 candidate traces `404`.
- Candidate reviewer: `APPROVE`, `research/experiments/E-008A/reviews/candidate-review-r3.md`.
- Evaluator reviewer: `APPROVE`, `research/experiments/E-008A/reviews/evaluator-review-r3.md`.
- Protocol reviewer: `APPROVE`, `research/experiments/E-008A/reviews/protocol-review-r4-addendum.md`.

## Artifacts

- Remote run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A/run-01`
- Formal cache: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A/run-01/cache-formal-b8e4dd9f`
- Checkpoints/reports: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A/run-01/models`
- P0 gate/per-pair: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A/run-01/gates`
- Candidate traces: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A/run-01/candidate-traces/p0`
- Observation/label manifests: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-008A/run-01/manifests`
- P0 watchdog report: `/mnt/sda/zhushihao/experiment_watchdog/reports/20260807_160756_1ba31964.json`
- Local gate: `/Users/shihao/Documents/图像匹配之二/research/results/E-008A/p0-gate.json`
- Local per-pair metrics: `/Users/shihao/Documents/图像匹配之二/research/results/E-008A/p0-pairs.jsonl`

## Decision Boundary

Stop at `P0 FAIL`. Do not run P1, E-008B, full sparse affine feedback, OS424/confirm, or any rescue version without a new user-approved method decision.
