# E-005A Result Card

## Decision

- Experiment: `E-005A`, M001 v0.8 Sparse Recurrent Affine Consensus
- Status: `COMPLETED / P0 PASS / G0 PASS / G1 FAIL / G2 NOT RUN`
- Stop reason: the frozen full model failed the preregistered dev-accuracy gate. Per protocol, no G2 resource run or rescue experiment was started.
- Innovation contract: structurally satisfied. Both independent reviewers approved the implementation, sparse-complexity boundary, evaluator, and provenance. This does not establish the requested accuracy-efficiency objective because G1 failed and G2 was not eligible.

## Frozen Identity

- `CURRENT_STATE.md`: `b4d260b4aa16301624c8ec39acc4faff8fd9b4228aa7e0a4f273e19978fe9fa8` (`BIND-STATE-001`)
- `METHOD_SPEC.md`: `2e43d633f65994875fd95631630c5cdd0caf6c549ee59d90ec486bb25df35723`
- `METHOD_DELTA.md`: `f22425ceb59034e5f22169d99f59fefd399d05cd3c2f6d0df67d650adff70893`
- `INNOVATION_CONTRACT.yaml`: `572e4dcd7eb374115b121866e45f2866dc0439b63d0030eaf04bbf3344216245`
- `E-005A.md`: `30e162c9d2ebf95c2ccda6d897a5eb809074d4d050d7738b66706112df58a6a1`
- E-005A source manifest: `ec2596a30e65f30277989ffd6507188431d60964e4471cba8b6a282cb58865f4`
- Feature-cache manifest: `4af9d555254962d484246c04c8ff9ec1f66d12168b5e6a2770d21bb143a723a8`
- Split manifest: `e8c990888123896af425923970215eb1a9ec8aca89eae0421998d2bdc13a99d3` (`1,607/404` group-isolated train/dev)
- GDROS checkpoint: `80fd768e2d5f2f3887ff8af62b165e43e073e6cec187544c542b6d4b591ba683`
- Main checkpoint (`20260808`): `edb62904adf41a52644fd6ac2923a626d59f7f7e568ba8bbe01e51383b078f1f`
- Review checkpoint (`20260809`): `3594e1118823c16de26f758c38c4cd58faffec0ee7b979206b8e18cb95e82d5f`

## Qualification

- Automated tests: `18 passed`; post-G1 audit reran the same suite with `18 passed`.
- G0: PASS. Exact score layout was four physical `[1,256,1024]` chunks for one logical `256x4096` search (`1,048,576` elements), `204,800` local samples, and no hidden N2.
- P0 on all 404 dev pairs: PASS; AEPE `0.147608 px`, PairLocalReady `100%`, valid affine `100%`.
- Reviews: method Round 3 `APPROVE`; evaluator/resource Round 4 `APPROVE`.
- Dense baseline parity: PASS with official maximum absolute difference `0.0`.

## Training

| Seed | Best epoch | Epochs | Best dev grid error |
|---|---:|---:|---:|
| `20260808` (main) | 31 | 40 | `4.792370 px` |
| `20260809` (review) | 41 | 50 | `4.716619 px` |

Both runs used the frozen feature-cache and source manifests. Checkpoints were selected by the preregistered training rule; no metric-dependent retuning was performed.

## G1 Accuracy

| Track | AEPE | CMR@2 | PairLocalReady | Valid | Grid mean |
|---|---:|---:|---:|---:|---:|
| Dense GDROS dev | `1.459719` | `79.455%` | `100%` | `100%` | `1.499447` |
| E-005A main seed | `4.634304` | `7.921%` | `96.535%` | `100%` | `4.792369` |
| E-005A review seed | `4.559264` | `11.139%` | `96.287%` | `100%` | `4.716619` |

The main-seed limits were AEPE `<=2.0 px`, CMR@2 `>=76.455%`, PairLocalReady `>=99%`, valid `>=99%`, and grid mean `<=2.5 px`. Only valid and all-finite passed. The review seed failed in the same direction and was not used to replace the main result.

Main-seed full-model round means:

| Round | Candidate accuracy | Recoverable active / 128 | Replacement | Condition | ESS | Dustbin Brier |
|---|---:|---:|---:|---:|---:|---:|
| 1 | `55.77%` | `62.42` | `0.00%` | `5.620` | `98.94` | `0.2359` |
| 2 | `81.25%` | `77.03` | `25.00%` | `5.165` | `87.26` | `0.1006` |
| 3 | `90.86%` | `78.69` | `6.67%` | `4.916` | `86.82` | `0.0442` |
| 4 | `90.37%` | `78.77` | `3.07%` | `4.834` | `86.70` | `0.0510` |

The recurrent path substantially improves candidate selection and calibration, but the selected active set saturates at about 79 recoverable anchors and does not produce a sufficiently accurate affine estimate.

## Ablation Evidence

Main-seed AEPE: full `4.6343`, M256-all-weighted `4.6805`, fixed-K128/no-reactivation `5.7359`, no sparse local correlation `6.3523`, no affine feedback `23.9357`, no global consensus tokens `36.1657`, and uniform-K128/top8/no-recurrence `138.5782` px. The global tokens, affine feedback, recurrence, local evidence, and dynamic active set all contribute, but their frozen combination remains below the G1 target.

This experiment disproves the sufficiency of the current v0.8 training/objective and active-set realization on the isolated dev protocol. It does not disprove the frozen GDROS top8 candidate pool, whose P0 oracle remains strong, or all sparse recurrent-consensus formulations.

## Resource Decision

- G2: `NOT RUN (INELIGIBLE AFTER G1 FAIL)`.
- No latency, speedup, or memory claim is made for E-005A.

## Artifacts

- Run root: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01`
- Run manifest: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/provenance/RUN_MANIFEST.json`
- Implementation map: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/provenance/IMPLEMENTATION_MAP.md`
- Third-party provenance: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/provenance/third-party-validation.json`
- G0: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/gates/g0-automatic-r4.json`
- P0 and rows: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/gates/p0-r5.json`, `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/gates/p0-r5-per-pair.jsonl`
- Dense baseline: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/gates/dense-baseline-dev.json`
- G1 and per-pair rows: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/gates/g1-dev.json`, `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/gates/g1-dev-per-pair.jsonl`
- Reviews: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/reviews/method-round3.md`, `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/reviews/eval-resource-round4.md`
- Main training: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/train/seed-20260808`
- Review training: `/mnt/sda/zhushihao/VS_Match/GDROS/experiments/sparsegeo/runs/E-005A/run-01/train/seed-20260809`
- Watchdog baseline: job `20260806_145821_85f65bf4`, log `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260806_145821_85f65bf4.log`
- Watchdog G1: job `20260806_150154_3a08e590`, log `/mnt/sda/zhushihao/experiment_watchdog/state/job_logs/20260806_150154_3a08e590.log`; exit code `2` denotes the emitted gate FAIL, not an infrastructure failure.
- Watchdog training: jobs `20260806_142829_8d2da668` and `20260806_142829_fdefed49`.

## Stop

E-005A is complete and waiting for method-level analysis. No E-005B, rescue, descriptor training, external confirmation, or local refinement was started.
